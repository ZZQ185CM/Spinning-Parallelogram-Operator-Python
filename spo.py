"""
SPO (Spinning Parallelogram Operator) main algorithm
"""

import numpy as np
from PIL import Image
import os
import sys
import time
from full_to_epi import full_to_epi
from depth_integration import depth_integration

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.ndimage import convolve as cp_convolve
    GPU_AVAILABLE = True
    print("GPU acceleration enabled (CuPy detected)")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    from scipy.ndimage import convolve
    print("GPU not available, using CPU mode")

def SPO(
    filepath_input,
    filepath_output,
    scale,
    number_of_bins,
    nD,
    sigma,
    guided_filter_radius=10,
    guided_filter_eps=0.0001,
    use_gpu=True,
):
    """
    Main SPO algorithm for light field depth estimation
    
    Args:
        filepath_input: Input file path
        filepath_output: Output file path
        scale: Window width parameter
        number_of_bins: Number of histogram bins
        nD: Number of depth labels
        sigma: Sigma parameter for reliability
        guided_filter_radius: Guided filter radius
        guided_filter_eps: Guided filter regularization term
        use_gpu: Whether to use GPU acceleration (default: True)
    """
    
    # Determine which array library to use
    if use_gpu and GPU_AVAILABLE:
        xp = cp
        print("Using GPU for computation")
    else:
        xp = np
        if use_gpu and not GPU_AVAILABLE:
            print("GPU requested but not available, falling back to CPU")
        else:
            print("Using CPU for computation")
    
    # Load depth optimization parameters
    depth_opt_path = os.path.join(filepath_input, 'depth_opt.py')
    if not os.path.exists(depth_opt_path):
        # Create default depth_opt.py if it doesn't exist
        print(f"Warning: {depth_opt_path} not found. Using default parameters.")
        opts = {
            'Dmin': -2.2 * 4,
            'Dmax': 1.4 * 4,
            'NumView': 9,
        }
    else:
        # Load parameters from depth_opt.py
        import importlib.util
        spec = importlib.util.spec_from_file_location("depth_opt", depth_opt_path)
        depth_opt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(depth_opt)
        opts = depth_opt.opts
    
    # Light field image preprocessing
    print("Loading light field image...")
    lf_path = os.path.join(filepath_input, 'lf.png')
    lf = np.array(Image.open(lf_path)).astype(np.float64)
    
    print("Converting to EPI representation...")
    img_h_RGB, img_v_RGB, img_view, _ = full_to_epi(lf, opts['NumView'])
    
    # Depth setting for the operator
    height, width, nB = img_view.shape
    Dstep = (opts['Dmax'] - opts['Dmin']) * 1.0 / (nD - 1)  # Depth interval
    MatchNum = int(np.ceil(max(abs(opts['Dmax']), abs(opts['Dmin']))) + 5)
    r_test = xp.arange(-MatchNum, MatchNum + 1, dtype=xp.float64)
    u = xp.arange(1, opts['NumView'] + 1)
    
    # Initialize weight arrays
    w_h = xp.zeros((opts['NumView'], MatchNum * 2 + 1, nD))
    w_v = xp.zeros((MatchNum * 2 + 1, opts['NumView'], nD))
    
    print("Computing depth-dependent weights...")
    for depth in range(nD):
        shift = (opts['Dmin'] + depth * Dstep) - \
                (opts['Dmin'] + depth * Dstep) * 2 / ((opts['NumView'] - 1) * 1.0) * (u - 1)
        for v in range(opts['NumView']):
            w = r_test - shift[v]
            w_h[v, :, depth] = w * xp.exp(-w * w / 2 / scale)
        w_v[:, :, depth] = w_h[:, :, depth].T
    
    # Bins extraction
    print("Extracting histogram bins...")
    # Transfer data to GPU if using GPU
    img_h_RGB_xp = xp.asarray(img_h_RGB)
    img_v_RGB_xp = xp.asarray(img_v_RGB)
    
    histogram_h = xp.zeros((height * opts['NumView'], width, number_of_bins, nB), dtype=bool)
    histogram_v = xp.zeros((height, width * opts['NumView'], number_of_bins, nB), dtype=bool)
    filter_h = xp.zeros((height, width, nD))
    filter_v = xp.zeros((height, width, nD))
    bins = 256 / number_of_bins
    
    for bin_idx in range(number_of_bins):
        for b in range(nB):
            histogram_h[:, :, bin_idx, b] = \
                (img_h_RGB_xp[:, :, b] > bin_idx * bins + 1) & \
                (img_h_RGB_xp[:, :, b] <= bin_idx * bins + bins)
            histogram_v[:, :, bin_idx, b] = \
                (img_v_RGB_xp[:, :, b] > bin_idx * bins + 1) & \
                (img_v_RGB_xp[:, :, b] <= bin_idx * bins + bins)
    
    # Cost volume generation
    print("Generating cost volume...")
    
    # Choose convolution function based on backend
    if use_gpu and GPU_AVAILABLE:
        conv_func = cp_convolve
    else:
        conv_func = convolve
    
    overall_start = time.time()
    for depth in range(nD):
        iter_start = time.time()
        
        for bin_idx in range(number_of_bins):
            for b in range(nB):
                hist_h = xp.zeros((height, width))
                hist_v = xp.zeros((height, width))
                
                for v in range(opts['NumView']):
                    # Horizontal processing
                    histogram_view = histogram_h[v::opts['NumView'], :, bin_idx, b]
                    cal_view = conv_func(histogram_view.astype(xp.float32), 
                                       w_h[v, :, depth].reshape(1, -1), 
                                       mode='constant')
                    hist_h += cal_view
                    
                    # Vertical processing
                    histogram_view = histogram_v[:, v::opts['NumView'], bin_idx, b]
                    cal_view = conv_func(histogram_view.astype(xp.float32), 
                                       w_v[:, v, depth].reshape(-1, 1), 
                                       mode='constant')
                    hist_v += cal_view
                
                filter_h[:, :, depth] += hist_h ** 2
                filter_v[:, :, depth] += hist_v ** 2
        
        # Progress bar
        iter_elapsed = time.time() - iter_start
        overall_elapsed = time.time() - overall_start
        progress = (depth + 1) / nD
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(
            f"\r进度 [{bar}] {depth + 1}/{nD} ({progress * 100:5.1f}%) | 本轮 {iter_elapsed:.2f}s | 总计 {overall_elapsed:.1f}s"
        )
        sys.stdout.flush()
    
    print()  # New line after progress bar
    
    # Transfer results back to CPU if using GPU
    if use_gpu and GPU_AVAILABLE:
        filter_h = cp.asnumpy(filter_h)
        filter_v = cp.asnumpy(filter_v)
    
    # Debug: Check filter values
    print(f"\nDebug Info:")
    print(f"filter_h shape: {filter_h.shape}, range: [{filter_h.min():.3f}, {filter_h.max():.3f}]")
    print(f"filter_v shape: {filter_v.shape}, range: [{filter_v.min():.3f}, {filter_v.max():.3f}]")
    
    # Check upper region
    upper_h = filter_h[:100, :, :]
    upper_v = filter_v[:100, :, :]
    print(f"Upper region filter_h range: [{upper_h.min():.3f}, {upper_h.max():.3f}]")
    print(f"Upper region filter_v range: [{upper_v.min():.3f}, {upper_v.max():.3f}]")
    
    # Depth optimization
    print("\nPerforming depth integration and optimization...")
    depth_integration(
        filepath_output,
        img_view,
        filter_h,
        filter_v,
        nD,
        sigma,
        guided_filter_radius=guided_filter_radius,
        guided_filter_eps=guided_filter_eps,
        use_gpu=use_gpu,
    )
