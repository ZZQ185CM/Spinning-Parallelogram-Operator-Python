"""
O(1) time box filtering using cumulative sum
"""

import numpy as np

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

def boxfilter(im_src, r, use_gpu=False):
    """
    O(1) time box filtering using cumulative sum
    
    Args:
        im_src: Input image (numpy or cupy array)
        r: Box filter radius
        use_gpu: Whether to use GPU acceleration (default: False)
        
    Returns:
        im_dst: Filtered image
        
    Definition: im_dst(x, y) = sum(sum(im_src(x-r:x+r, y-r:y+r)))
    Running time independent of r
    """
    # Determine which array library to use
    if use_gpu and GPU_AVAILABLE:
        xp = cp
        im_src = cp.asarray(im_src)
    else:
        xp = np
        if hasattr(im_src, 'get'):  # Check if it's a CuPy array
            im_src = im_src.get()
    
    hei, wid = im_src.shape
    im_dst = xp.zeros_like(im_src)
    
    # Cumulative sum over Y axis
    im_cum = xp.cumsum(im_src, axis=0)
    
    # Difference over Y axis
    im_dst[0:r+1, :] = im_cum[r:2*r+1, :]
    im_dst[r+1:hei-r, :] = im_cum[2*r+1:hei, :] - im_cum[0:hei-2*r-1, :]
    im_dst[hei-r:hei, :] = xp.tile(im_cum[hei-1:hei, :], (r, 1)) - im_cum[hei-2*r:hei-r, :]
    
    # Cumulative sum over X axis
    im_cum = xp.cumsum(im_dst, axis=1)
    
    # Difference over X axis
    im_dst[:, 0:r+1] = im_cum[:, r:2*r+1]
    im_dst[:, r+1:wid-r] = im_cum[:, 2*r+1:wid] - im_cum[:, 0:wid-2*r-1]
    im_dst[:, wid-r:wid] = xp.tile(im_cum[:, wid-1:wid], (1, r)) - im_cum[:, wid-2*r:wid-r]
    
    return im_dst
