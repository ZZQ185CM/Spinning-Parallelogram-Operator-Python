"""
Convert full light field image to EPI (Epipolar Plane Image) representation
"""

import numpy as np

def full_to_epi(img_RGB, num_view):
    """
    Convert light field image to EPI images
    
    Args:
        img_RGB: Light field image (height, width, channels)
        num_view: Number of views in each dimension
        
    Returns:
        img_h: Horizontal EPI image
        img_v: Vertical EPI image
        img_view: Center view image
        focus_img: All-in-focus image
    """
    mid_view = round(num_view / 2)
    height, width, nB = img_RGB.shape
    
    # Create horizontal EPI
    img_h = np.zeros((height, width // num_view, nB), dtype=img_RGB.dtype)
    for i in range(height // num_view):
        for j in range(width // num_view):
            img_h[i*num_view:(i+1)*num_view, j, :] = \
                img_RGB[mid_view + i*num_view, j*num_view:(j+1)*num_view, :]
    
    # Create vertical EPI
    img_v = np.zeros((height // num_view, width, nB), dtype=img_RGB.dtype)
    for i in range(height // num_view):
        for j in range(width // num_view):
            img_v[i, j*num_view:(j+1)*num_view, :] = \
                img_RGB[i*num_view:(i+1)*num_view, mid_view + j*num_view, :]
    
    # Extract center view
    img_view = img_RGB[mid_view::num_view, mid_view::num_view, :]
    
    # Create all-in-focus image
    focus_img = np.zeros((height // num_view, width // num_view, nB), dtype=np.float64)
    for viewh in range(num_view):
        for viewv in range(num_view):
            for b in range(nB):
                focus_img[:, :, b] += img_RGB[viewh::num_view, viewv::num_view, b].astype(np.float64)
    
    focus_img = focus_img / (num_view * num_view)
    
    return img_h, img_v, img_view, focus_img
