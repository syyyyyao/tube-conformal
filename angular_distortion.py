import numpy as np

def angular_distortion(v: np.ndarray, f: np.ndarray, vmap: np.ndarray) -> np.ndarray:
    """
    Compute the angle distortion (in degree) of a mapping.
    
    Arguments:
    ----------
    v: nv x 3 or nv x 2 numpy array
       vertex coordinates
    f: nf x 3 numpy array
       triangular connectivity of mesh, 0-indexed
    vmap: nv x 3 or nv x 2 numpy array
          vertex coordinates of the mapping
    
    Returns:
    --------
    distortion: 3*nf x 1 numpy array
                angle distortion
    """
    
    f1 = f[:, 0]
    f2 = f[:, 1]
    f3 = f[:, 2]

    if np.size(v,1) == 2:
        v = np.hstack([v, np.zeros((len(v),1))])
        
    if np.size(vmap,1) == 2:
        vmap = np.hstack([vmap, np.zeros((len(vmap),1))])

    # calculate angles on v
    
    a3 = v[f1,:] - v[f3,:]
    b3 = v[f2,:] - v[f3,:]
    a1 = v[f2,:] - v[f1,:]
    b1 = v[f3,:] - v[f1,:]
    a2 = v[f3,:] - v[f2,:]
    b2 = v[f1,:] - v[f2,:]
    
    vcos1 = (a1[:,0]*b1[:,0]+a1[:,1]*b1[:,1]+a1[:,2]*b1[:,2])/np.sqrt(a1[:,0]**2+a1[:,1]**2+a1[:,2]**2)/np.sqrt(b1[:,0]**2+b1[:,1]**2+b1[:,2]**2)
    vcos2 = (a2[:,0]*b2[:,0]+a2[:,1]*b2[:,1]+a2[:,2]*b2[:,2])/np.sqrt(a2[:,0]**2+a2[:,1]**2+a2[:,2]**2)/np.sqrt(b2[:,0]**2+b2[:,1]**2+b2[:,2]**2)
    vcos3 = (a3[:,0]*b3[:,0]+a3[:,1]*b3[:,1]+a3[:,2]*b3[:,2])/np.sqrt(a3[:,0]**2+a3[:,1]**2+a3[:,2]**2)/np.sqrt(b3[:,0]**2+b3[:,1]**2+b3[:,2]**2)

    # calculate angles on vmap
    c3 = vmap[f1,:] - vmap[f3,:]
    d3 = vmap[f2,:] - vmap[f3,:]
    c1 = vmap[f2,:] - vmap[f1,:]
    d1 = vmap[f3,:] - vmap[f1,:]
    c2 = vmap[f3,:] - vmap[f2,:]
    d2 = vmap[f1,:] - vmap[f2,:]
    
    mapcos1 = (c1[:,0]*d1[:,0]+c1[:,1]*d1[:,1]+c1[:,2]*d1[:,2])/np.sqrt(c1[:,0]**2+c1[:,1]**2+c1[:,2]**2)/np.sqrt(d1[:,0]**2+d1[:,1]**2+d1[:,2]**2)
    mapcos2 = (c2[:,0]*d2[:,0]+c2[:,1]*d2[:,1]+c2[:,2]*d2[:,2])/np.sqrt(c2[:,0]**2+c2[:,1]**2+c2[:,2]**2)/np.sqrt(d2[:,0]**2+d2[:,1]**2+d2[:,2]**2)
    mapcos3 = (c3[:,0]*d3[:,0]+c3[:,1]*d3[:,1]+c3[:,2]*d3[:,2])/np.sqrt(c3[:,0]**2+c3[:,1]**2+c3[:,2]**2)/np.sqrt(d3[:,0]**2+d3[:,1]**2+d3[:,2]**2)

    # calculate the angle difference
    angular_distortion = np.hstack((np.arccos(mapcos1) - np.arccos(vcos1), np.arccos(mapcos2) - np.arccos(vcos2), np.arccos(mapcos3) - np.arccos(vcos3))) * 180 / np.pi
    
    return angular_distortion