"""
References:
-----------
P. T. Choi and L. M. Lui,  
Fast Disk Conformal Parameterization of Simply-Connected Open Surfaces, 
Journal of Scientific Computing, 65(3), pp. 1065-1090, 2015.
"""

import numpy as np

def face_area(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Compute the area of every face of a 3D triangle mesh.
    
    Arguments:
    ----------
    v: nv x 3 numpy array
       vertex coordinates
    f: nf x 3 numpy array
       triangular connectivity of mesh, 0-indexed
    
    Returns:
    --------
    fa : nf x 1 numpy array
         area of each face
    """
    # Convert to 0-based indexing if needed
    f_0 = f - 1 if f.min() == 1 else f
    
    # Compute edge vectors
    v12 = v[f_0[:, 1]] - v[f_0[:, 0]]
    v23 = v[f_0[:, 2]] - v[f_0[:, 1]]
    v31 = v[f_0[:, 0]] - v[f_0[:, 2]]
    
    # Compute edge lengths
    a = np.sqrt(np.sum(v12**2, axis=1))
    b = np.sqrt(np.sum(v23**2, axis=1))
    c = np.sqrt(np.sum(v31**2, axis=1))
    
    # Use Heron's formula
    s = (a + b + c) / 2
    fa = np.sqrt(s * (s - a) * (s - b) * (s - c))
    
    # Handle degenerate cases (numerical precision issues)
    fa = np.nan_to_num(fa, nan=0.0, posinf=0.0, neginf=0.0)
    
    return fa