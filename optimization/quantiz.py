"""
Quantization function - produces a quantization index and quantized output value
Translated from MATLAB quantiz function
"""

import numpy as np

def quantiz(sig, partition, codebook=None):
    """
    Produce a quantization index and a quantized output value.
    
    Args:
        sig: Input signal (array)
        partition: Strict ascending ordered vector that specifies the boundaries
        codebook: Optional codebook for output values
        
    Returns:
        indx: Quantization index (0 to N-1)
        quantv: Quantized output values (if codebook provided)
        distor: Distortion value (if codebook provided)
    """
    
    # Input validation
    if not isinstance(sig, np.ndarray):
        sig = np.array(sig)
    
    if not isinstance(partition, np.ndarray):
        partition = np.array(partition)
    
    if len(sig) == 0 or not np.isrealobj(sig):
        raise ValueError("Invalid signal: must be non-empty and real")
    
    if len(partition) == 0 or not np.isrealobj(partition):
        raise ValueError("Invalid partition: must be non-empty and real")
    
    if not np.all(np.diff(partition) > 0):
        raise ValueError("Invalid partition: must be strictly ascending")
    
    # Compute index
    original_shape = sig.shape
    indx = np.zeros_like(sig, dtype=int)
    
    for i in range(len(partition)):
        indx += (sig > partition[i]).astype(int)
    
    if codebook is None:
        return indx
    
    # Compute quantized values
    if not isinstance(codebook, np.ndarray):
        codebook = np.array(codebook)
    
    if len(codebook) == 0 or not np.isrealobj(codebook):
        raise ValueError("Invalid codebook: must be non-empty and real")
    
    if len(codebook) != len(partition) + 1:
        raise ValueError(f"Invalid codebook length: expected {len(partition)+1}, got {len(codebook)}")
    
    quantv = codebook[indx]
    
    # Compute distortion
    distor = 0
    for i in range(len(codebook)):
        mask = (indx == i)
        if np.any(mask):
            distor += np.sum((sig[mask] - codebook[i]) ** 2)
    distor = distor / sig.size
    
    return indx, quantv, distor
