import numpy as np
from scipy.sparse.linalg import spsolve

from .meshboundaries import meshboundaries
from .beltrami_coefficient import beltrami_coefficient
from .generalized_laplacian import generalized_laplacian


def tube_conformal_map(tube0: np.ndarray, f: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    
    """
    # get boundaries
    boundary_loops = meshboundaries(f)
    if len(boundary_loops) != 2:
        raise ValueError(f"Expected 2 boundary loops for a tube, found {len(boundary_loops)}")
    
    # map tube to annulus
    annulus0 = np.column_stack([ np.exp(tube0[:,2]) * tube0[:,0], np.exp(tube0[:,2]) * tube0[:,1] ])
    bd = np.concatenate([boundary_loops[0], boundary_loops[1]])

    # construct laplacian matrix
    mu = beltrami_coefficient(annulus0, f, v)
    laplacian_mat = generalized_laplacian(annulus0, f, mu).tolil()

    # fix boundaries
    laplacian_mat[bd, :] = 0
    laplacian_mat[bd, bd] = 1
    laplacian_mat = laplacian_mat.tocsr() # back to csr matrix

    x_fixed = np.zeros(len(v))
    y_fixed = np.zeros(len(v))
    x_fixed[bd] = annulus0[bd, 0]
    y_fixed[bd] = annulus0[bd, 1]

    # solve annulus coordinates
    x_ann = spsolve(laplacian_mat, x_fixed)
    y_ann = spsolve(laplacian_mat, y_fixed)

    # map to a tube
    l_ann = np.sqrt(x_ann**2 + y_ann**2)
    tube = np.column_stack([x_ann/l_ann, y_ann/l_ann, np.log(l_ann)])
    return tube