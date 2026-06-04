"""
Weighted median filter with guided filter weights
"""

import numpy as np
from .guidedfilter_color import guidedfilter_color

def weighted_median_filter(disp_in, img_guide, vec_disps, r, epsilon=0.01):
    """
    Weighted median filter with guided filter weights
    
    Args:
        disp_in: Input 1-channel discrete disparity map
        img_guide: Input guidance image (3-channel RGB)
        vec_disps: Vector of disparities in consideration
        r: Local window radius for guided filter weights
        epsilon: Regularization parameter for guided filter weights
        
    Returns:
        disp_out: Filtered disparity map
    """
    
    # Normalize image to [0, 1]
    if img_guide.dtype == np.uint8:
        img_guide = img_guide.astype(np.float64) / 255.0
    
    disp_out = np.zeros_like(disp_in, dtype=disp_in.dtype)
    img_accum = np.zeros(disp_in.shape, dtype=np.float64)
    
    for d_idx, d_val in enumerate(vec_disps):
        print(f'{d_idx + 1} of {len(vec_disps)}')
        
        # Create binary map for current disparity
        img_binary = (disp_in == d_val).astype(np.float64)
        
        # Apply guided filter
        img_filtered = guidedfilter_color(img_guide, img_binary, r, epsilon)
        
        # Accumulation to find median disparity for each pixel
        img_accum += img_filtered
        idx_selected = (img_accum > 0.5) & (disp_out == 0)
        disp_out[idx_selected] = d_idx + 1
    
    return disp_out
