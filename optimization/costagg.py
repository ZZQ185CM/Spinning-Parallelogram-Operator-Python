"""
Fast cost-volume filtering for visual correspondence
Reference: C Rhemann et al., "Fast cost-volume filtering for visual correspondence and beyond", CVPR 2011
"""

import numpy as np
from .guidedfilter_color import guidedfilter_color

def cost_agg(E1, Ic, param):
    """
    Cost aggregation using guided filter
    
    Args:
        E1: Cost volume (height, width, num_labels)
        Ic: Center view image (RGB)
        param: Dictionary with parameters
               - r: window radius for guided filter
               - eps: regularization parameter
               
    Returns:
        dispVol1: Filtered cost volume
    """
    
    r = param['r']
    eps = param['eps']
    num_labels = E1.shape[2]
    im_cen = Ic
    
    dispVol1 = np.zeros_like(E1)
    
    # Process each depth label
    for d in range(num_labels):
        p1 = E1[:, :, d]
        q1 = guidedfilter_color(im_cen, p1, r, eps)
        dispVol1[:, :, d] = q1
        print(f'CostAgg.. cost slice {d+1}/{num_labels}')
    
    return dispVol1
