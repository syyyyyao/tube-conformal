import numpy as np

def conformal_bend_major(tube:np.ndarray, R:float) -> np.ndarray:
    """
    Conformal bending of a tube along its major axis.
    
    Parameters:
    tube (np.ndarray): An Nx3 array representing the vertices of the tube.
    R (float): The radius of the bending. Larger R results in gentler bends.
    
    Returns:
    np.ndarray: An Nx3 array representing the vertices of the bent tube.
    """
    # Extract the z-coordinates and center them
    z = tube[:, 2]
    z_centered = z - (np.max(z) + np.min(z)) / 2
    
    # Compute the bending angles
    theta = 2 * np.arctan2(np.sqrt(R + 1) * np.sin(np.sqrt(R**2 - 1) / 2 * z_centered),
                           np.sqrt(R - 1) * np.cos(np.sqrt(R**2 - 1) / 2 * z_centered))
    
    # Compute the new coordinates
    phi = np.arctan2(tube[:, 1], tube[:, 0])
    
    tube_bent = np.column_stack([
        (R + np.cos(theta)) * np.cos(phi),
        np.sin(theta),
        (R + np.cos(theta)) * np.sin(phi)
    ])
    
    return tube_bent

def conformal_bend_minor(tube:np.ndarray, R:float) -> np.ndarray:
    """
    Conformal bending of a tube along its minor axis.
    
    Parameters:
    tube (np.ndarray): An Nx3 array representing the vertices of the tube.
    R (float): The radius of the bending. Larger R results in gentler bends.
    
    Returns:
    np.ndarray: An Nx3 array representing the vertices of the bent tube.
    """
    u = np.arctan2(tube[:,1], tube[:,0])

    theta = 2 * np.arctan2(np.sqrt(R+1) * np.sin(0.5*u),
                           np.sqrt(R-1) * np.cos(0.5*u))


    phi = tube[:,2] / np.sqrt(R**2 - 1)

    tube_bent = np.column_stack([
        (R + np.cos(theta)) * np.cos(phi),
        np.sin(theta),
        (R + np.cos(theta)) * np.sin(phi)
    ])
    
    return tube_bent