"""
Depth integration and optimization module
"""

import numpy as np
from PIL import Image
import os
import sys
import time
from optimization.guidedfilter_color import guidedfilter_color

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

def depth_integration(
    filepath_output,
    img_view,
    filter_h,
    filter_v,
    nD,
    sigma,
    guided_filter_radius=10,
    guided_filter_eps=0.0001,
    use_gpu=True,
):
    """
    Perform depth integration and optimization
    
    Args:
        filepath_output: Output file path
        img_view: Center view image
        filter_h: Horizontal filter response
        filter_v: Vertical filter response
        nD: Number of depth labels
        sigma: Sigma parameter for reliability calculation
        guided_filter_radius: Guided filter radius
        guided_filter_eps: Guided filter regularization term
        use_gpu: Whether to use GPU acceleration (default: True)
    """
    
    # Determine which array library to use
    if use_gpu and GPU_AVAILABLE:
        xp = cp
    else:
        xp = np
    
    # Initial depth estimation results
    print("Computing initial depth estimation...")
    # Transfer data to GPU if using GPU
    filter_h_xp = xp.asarray(filter_h)
    filter_v_xp = xp.asarray(filter_v)
    
    h_max = xp.max(filter_h_xp, axis=2)
    h_avg = xp.mean(filter_h_xp, axis=2)
    c1 = xp.exp(-(h_avg / (h_max + 1e-10)) / sigma)
    
    v_max = xp.max(filter_v_xp, axis=2)
    v_avg = xp.mean(filter_v_xp, axis=2)
    c2 = xp.exp(-(v_avg / (v_max + 1e-10)) / sigma)
    
    # Calculate reliability weights
    reliable_h = c1 / (c1 + c2)
    reliable_v = c2 / (c1 + c2)
    
    # Debug: Check for NaN and zero divisions
    if use_gpu and GPU_AVAILABLE:
        nan_count_h = cp.sum(cp.isnan(reliable_h))
        nan_count_v = cp.sum(cp.isnan(reliable_v))
        zero_sum = cp.sum(c1 + c2 == 0)
    else:
        nan_count_h = np.sum(np.isnan(reliable_h))
        nan_count_v = np.sum(np.isnan(reliable_v))
        zero_sum = np.sum(c1 + c2 == 0)
    
    print(f"Debug: NaN in reliable_h: {int(nan_count_h)}, in reliable_v: {int(nan_count_v)}")
    print(f"Debug: Pixels where c1+c2=0: {int(zero_sum)}")
    
    # Handle NaN values
    reliable_h = xp.nan_to_num(reliable_h, nan=0.0)
    reliable_v = xp.nan_to_num(reliable_v, nan=0.0)
    
    # Debug: Check reliability values
    if use_gpu and GPU_AVAILABLE:
        rel_h_cpu = cp.asnumpy(reliable_h)
        rel_v_cpu = cp.asnumpy(reliable_v)
    else:
        rel_h_cpu = reliable_h
        rel_v_cpu = reliable_v
    
    print(f"Debug: reliable_h range: [{rel_h_cpu.min():.3f}, {rel_h_cpu.max():.3f}]")
    print(f"Debug: reliable_v range: [{rel_v_cpu.min():.3f}, {rel_v_cpu.max():.3f}]")
    
    # Expand reliability to match depth dimensions
    reliable_h = xp.repeat(reliable_h[:, :, xp.newaxis], nD, axis=2)
    reliable_v = xp.repeat(reliable_v[:, :, xp.newaxis], nD, axis=2)
    
    # Combine horizontal and vertical cost volumes
    sumC = reliable_h * filter_h_xp + reliable_v * filter_v_xp
    
    # Debug: Check sumC values
    if use_gpu and GPU_AVAILABLE:
        sumC_cpu = cp.asnumpy(sumC)
    else:
        sumC_cpu = sumC
    
    print(f"Debug: sumC shape: {sumC_cpu.shape}")
    print(f"Debug: sumC range: [{sumC_cpu.min():.3f}, {sumC_cpu.max():.3f}]")
    
    # Check if any depth layers are all zeros
    for d in range(min(3, nD)):
        layer_sum = np.sum(np.abs(sumC_cpu[:, :, d]))
        print(f"Debug: sumC layer {d} sum: {layer_sum:.3f}")
    
    # Get initial depth map
    labels_max = xp.argmax(sumC, axis=2)
    # Transfer back to CPU for saving
    if use_gpu and GPU_AVAILABLE:
        labels_max = cp.asnumpy(labels_max)
    # Map depth values to grayscale (inverted for proper visualization)
    # Near (small index) -> bright (white), far (large index) -> dark (black)
    save_img = np.uint8(np.clip(255 - (256 / nD) * labels_max, 0, 255))
    
    # Save initial depth map
    output_path = os.path.join(filepath_output, 'depth_initial.bmp')
    Image.fromarray(save_img).save(output_path)
    print(f"Initial depth map saved to: {output_path}")
    
    # Matching cost filtering calculation
    print("Applying guided filter to cost volume...")
    filter_start = time.time()
    for d in range(nD):
        # Transfer current slice to CPU for guided filter
        if use_gpu and GPU_AVAILABLE:
            p = cp.asnumpy(sumC[:, :, d])
        else:
            p = sumC[:, :, d]
        
        q = guidedfilter_color(
            img_view.astype(np.float64),
            p,
            guided_filter_radius,
            guided_filter_eps,
            use_gpu=use_gpu,
        )
        
        # Transfer result back to GPU if needed
        if use_gpu and GPU_AVAILABLE:
            sumC[:, :, d] = cp.asarray(q)
        else:
            sumC[:, :, d] = q
        
        # Progress bar
        filter_elapsed = time.time() - filter_start
        progress = (d + 1) / nD
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(
            f"\r过滤进度 [{bar}] {d + 1}/{nD} ({progress * 100:5.1f}%) | 总计 {filter_elapsed:.1f}s"
        )
        sys.stdout.flush()
    
    print()  # New line after progress bar
    
    # Get filtered depth map
    sumD = xp.argmax(sumC, axis=2)
    # Transfer back to CPU for saving
    if use_gpu and GPU_AVAILABLE:
        sumD = cp.asnumpy(sumD)
    # Map depth values to grayscale (inverted for proper visualization)
    # Near (small index) -> bright (white), far (large index) -> dark (black)
    save_img = np.uint8(np.clip(255 - (256 / nD) * sumD, 0, 255))
    
    # Save filtered depth map
    output_path = os.path.join(filepath_output, 'depth_filtering.bmp')
    Image.fromarray(save_img).save(output_path)
    print(f"Filtered depth map saved to: {output_path}")
    
    # Note: Multi-label optimization via Graph-cuts is commented out
    # This would require additional dependencies like PyMaxflow or gco-python
    # The code structure is preserved for future implementation
    
    """
    # Multi-label optimization via Graph-cuts (requires gco-python or similar)
    # E2 = np.max(sumC) - sumC
    # Ic = img_view
    # 
    # param = {
    #     'data': 5,
    #     'smooth': 3,
    #     'neigh': 0.009
    # }
    # E3 = GraphCuts(E2, Ic, param)
    # 
    # save_img = np.uint8((256 / np.max(E3)) * E3)
    # output_path = os.path.join(filepath_output, 'depth_global.bmp')
    # Image.fromarray(save_img).save(output_path)
    """
    
    print("Depth integration completed!")
