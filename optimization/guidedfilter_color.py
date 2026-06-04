"""
O(1) time implementation of guided filter using a color image as the guidance
Reference: He, Kaiming, Jian Sun, and Xiaoou Tang. "Guided image filtering." 
           IEEE transactions on pattern analysis and machine intelligence (2013).
"""

import numpy as np
from .boxfilter import boxfilter

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

def guidedfilter_color(I, p, r, eps, use_gpu=False):
    """
    Guided filter using a color (RGB) image as guidance
    
    Args:
        I: Guidance image (should be a color RGB image) - (height, width, 3)
        p: Filtering input image (should be a gray-scale/single channel image) - (height, width)
        r: Local window radius
        eps: Regularization parameter
        use_gpu: Whether to use GPU acceleration (default: False)
        
    Returns:
        q: Filtered output image
    """
    # Determine which array library to use
    if use_gpu and GPU_AVAILABLE:
        xp = cp
        I = cp.asarray(I)
        p = cp.asarray(p)
    else:
        xp = np
        # Convert from CuPy to NumPy if needed
        if hasattr(I, 'get'):
            I = I.get()
        if hasattr(p, 'get'):
            p = p.get()
    hei, wid = p.shape
    N = boxfilter(xp.ones((hei, wid)), r, use_gpu=use_gpu)  # the size of each local patch
    
    # Mean of guidance image for each channel
    mean_I_r = boxfilter(I[:, :, 0], r, use_gpu=use_gpu) / N
    mean_I_g = boxfilter(I[:, :, 1], r, use_gpu=use_gpu) / N
    mean_I_b = boxfilter(I[:, :, 2], r, use_gpu=use_gpu) / N
    
    # Mean of input image p
    mean_p = boxfilter(p, r, use_gpu=use_gpu) / N
    
    # Mean of I*p for each channel
    mean_Ip_r = boxfilter(I[:, :, 0] * p, r, use_gpu=use_gpu) / N
    mean_Ip_g = boxfilter(I[:, :, 1] * p, r, use_gpu=use_gpu) / N
    mean_Ip_b = boxfilter(I[:, :, 2] * p, r, use_gpu=use_gpu) / N
    
    # Covariance of (I, p) in each local patch
    cov_Ip_r = mean_Ip_r - mean_I_r * mean_p
    cov_Ip_g = mean_Ip_g - mean_I_g * mean_p
    cov_Ip_b = mean_Ip_b - mean_I_b * mean_p
    
    # Variance of I in each local patch: the matrix Sigma
    # Sigma is a 3x3 symmetric matrix:
    #     rr, rg, rb
    #     rg, gg, gb
    #     rb, gb, bb
    var_I_rr = boxfilter(I[:, :, 0] * I[:, :, 0], r, use_gpu=use_gpu) / N - mean_I_r * mean_I_r
    var_I_rg = boxfilter(I[:, :, 0] * I[:, :, 1], r, use_gpu=use_gpu) / N - mean_I_r * mean_I_g
    var_I_rb = boxfilter(I[:, :, 0] * I[:, :, 2], r, use_gpu=use_gpu) / N - mean_I_r * mean_I_b
    var_I_gg = boxfilter(I[:, :, 1] * I[:, :, 1], r, use_gpu=use_gpu) / N - mean_I_g * mean_I_g
    var_I_gb = boxfilter(I[:, :, 1] * I[:, :, 2], r, use_gpu=use_gpu) / N - mean_I_g * mean_I_b
    var_I_bb = boxfilter(I[:, :, 2] * I[:, :, 2], r, use_gpu=use_gpu) / N - mean_I_b * mean_I_b
    
    # Calculate coefficient a for each pixel
    # Vectorized version for GPU acceleration
    a = xp.zeros((hei, wid, 3))
    
    # Stack variance matrices for batch processing
    # Shape: (hei, wid, 3, 3)
    Sigma = xp.stack([
        xp.stack([var_I_rr, var_I_rg, var_I_rb], axis=2),
        xp.stack([var_I_rg, var_I_gg, var_I_gb], axis=2),
        xp.stack([var_I_rb, var_I_gb, var_I_bb], axis=2)
    ], axis=2)
    
    # Add regularization term
    eye_3 = xp.eye(3)
    Sigma = Sigma + eps * eye_3[xp.newaxis, xp.newaxis, :, :]
    
    # Stack covariance vectors
    # Shape: (hei, wid, 3)
    cov_Ip = xp.stack([cov_Ip_r, cov_Ip_g, cov_Ip_b], axis=2)
    
    # Batch matrix inversion and multiplication
    # For each pixel: a = cov_Ip @ inv(Sigma)
    if use_gpu and GPU_AVAILABLE:
        # Use batched GPU operations
        Sigma_flat = Sigma.reshape(-1, 3, 3)
        cov_Ip_flat = cov_Ip.reshape(-1, 3, 1)
        Sigma_inv_flat = cp.linalg.inv(Sigma_flat)
        a_flat = cp.matmul(Sigma_inv_flat, cov_Ip_flat).squeeze(-1)
        a = a_flat.reshape(hei, wid, 3)
    else:
        # CPU fallback - vectorized numpy operations
        for y in range(hei):
            for x in range(wid):
                Sigma_pixel = Sigma[y, x, :, :]
                cov_Ip_pixel = cov_Ip[y, x, :]
                a[y, x, :] = cov_Ip_pixel @ np.linalg.inv(Sigma_pixel)
    
    # Eqn. (15) in the paper
    b = mean_p - a[:, :, 0] * mean_I_r - a[:, :, 1] * mean_I_g - a[:, :, 2] * mean_I_b
    
    # Eqn. (16) in the paper
    q = (boxfilter(a[:, :, 0], r, use_gpu=use_gpu) * I[:, :, 0] + 
         boxfilter(a[:, :, 1], r, use_gpu=use_gpu) * I[:, :, 1] + 
         boxfilter(a[:, :, 2], r, use_gpu=use_gpu) * I[:, :, 2] + 
         boxfilter(b, r, use_gpu=use_gpu)) / N
    
    # Convert back to CPU if needed
    if use_gpu and GPU_AVAILABLE:
        q = cp.asnumpy(q)
    
    return q
