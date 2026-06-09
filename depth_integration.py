"""
Depth integration and optimization module
"""

import numpy as np
from PIL import Image
import os
import sys
import time
from optimization.guidedfilter_color import guidedfilter_color_precompute, guidedfilter_color_runfilter

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
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
    
    if use_gpu and GPU_AVAILABLE:
        nan_count_h = int(cp.asnumpy(cp.sum(cp.isnan(reliable_h))))
        nan_count_v = int(cp.asnumpy(cp.sum(cp.isnan(reliable_v))))
        zero_sum = int(cp.asnumpy(cp.sum(c1 + c2 == 0)))
    else:
        nan_count_h = int(np.sum(np.isnan(reliable_h)))
        nan_count_v = int(np.sum(np.isnan(reliable_v)))
        zero_sum = int(np.sum(c1 + c2 == 0))
    
    print(f"Debug: NaN in reliable_h: {int(nan_count_h)}, in reliable_v: {int(nan_count_v)}")
    print(f"Debug: Pixels where c1+c2=0: {int(zero_sum)}")
    
    # Handle NaN values
    reliable_h = xp.nan_to_num(reliable_h, nan=0.0)
    reliable_v = xp.nan_to_num(reliable_v, nan=0.0)
    
    if use_gpu and GPU_AVAILABLE:
        rel_h_min = float(cp.asnumpy(cp.min(reliable_h)))
        rel_h_max = float(cp.asnumpy(cp.max(reliable_h)))
        rel_v_min = float(cp.asnumpy(cp.min(reliable_v)))
        rel_v_max = float(cp.asnumpy(cp.max(reliable_v)))
    else:
        rel_h_min = float(np.min(reliable_h))
        rel_h_max = float(np.max(reliable_h))
        rel_v_min = float(np.min(reliable_v))
        rel_v_max = float(np.max(reliable_v))
    
    print(f"Debug: reliable_h range: [{rel_h_min:.3f}, {rel_h_max:.3f}]")
    print(f"Debug: reliable_v range: [{rel_v_min:.3f}, {rel_v_max:.3f}]")
    
    # Expand reliability to match depth dimensions
    reliable_h = xp.repeat(reliable_h[:, :, xp.newaxis], nD, axis=2)
    reliable_v = xp.repeat(reliable_v[:, :, xp.newaxis], nD, axis=2)
    
    # Combine horizontal and vertical cost volumes
    sumC = reliable_h * filter_h_xp + reliable_v * filter_v_xp
    
    if use_gpu and GPU_AVAILABLE:
        sumC_min = float(cp.asnumpy(cp.min(sumC)))
        sumC_max = float(cp.asnumpy(cp.max(sumC)))
    else:
        sumC_min = float(np.min(sumC))
        sumC_max = float(np.max(sumC))
    
    print(f"Debug: sumC shape: {sumC.shape}")
    print(f"Debug: sumC range: [{sumC_min:.3f}, {sumC_max:.3f}]")
    
    # Check if any depth layers are all zeros
    for d in range(min(3, nD)):
        if use_gpu and GPU_AVAILABLE:
            layer_sum = float(cp.asnumpy(cp.sum(cp.abs(sumC[:, :, d]))))
        else:
            layer_sum = float(np.sum(np.abs(sumC[:, :, d])))
        print(f"Debug: sumC layer {d} sum: {layer_sum:.3f}")
    
    # Get initial depth map
    labels_max = xp.argmax(sumC, axis=2)
    # Transfer back to CPU for saving
    if use_gpu and GPU_AVAILABLE:
        labels_max = cp.asnumpy(labels_max)
    # Match MATLAB: lower label (negative disparity/far) -> dark, higher label (positive disparity/near) -> bright.
    save_img = np.uint8(np.clip((256 / nD) * labels_max, 0, 255))
    
    # Save initial depth map
    output_path = os.path.join(filepath_output, 'depth_initial.bmp')
    Image.fromarray(save_img).save(output_path)
    print(f"Initial depth map saved to: {output_path}")
    
    # Matching cost filtering calculation
    print("Applying guided filter to cost volume...")
    filter_start = time.time()

    gf_precomputed = guidedfilter_color_precompute(
        img_view.astype(np.float64),
        guided_filter_radius,
        guided_filter_eps,
        use_gpu=use_gpu,
    )

    if use_gpu and GPU_AVAILABLE:
        depth_chunk_size = min(nD, 16)
        for chunk_start in range(0, nD, depth_chunk_size):
            chunk_end = min(chunk_start + depth_chunk_size, nD)
            # Shape: (height, width, chunk_depth). Each depth slice is filtered independently.
            sumC[:, :, chunk_start:chunk_end] = guidedfilter_color_runfilter(
                gf_precomputed,
                sumC[:, :, chunk_start:chunk_end],
                return_cpu=False,
            )

            filter_elapsed = time.time() - filter_start
            progress = chunk_end / nD
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '#' * filled_length + '-' * (bar_length - filled_length)
            sys.stdout.write(
                f"\r过滤进度 [{bar}] {chunk_end}/{nD} ({progress * 100:5.1f}%) | 总计 {filter_elapsed:.1f}s"
            )
            sys.stdout.flush()
    else:
        for d in range(nD):
            sumC[:, :, d] = guidedfilter_color_runfilter(
                gf_precomputed,
                sumC[:, :, d],
                return_cpu=False,
            )

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
    # Match MATLAB: lower label (negative disparity/far) -> dark, higher label (positive disparity/near) -> bright.
    save_img = np.uint8(np.clip((256 / nD) * sumD, 0, 255))
    
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
