import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .meshboundaries import meshboundaries

def ring_smooth(v_raw: np.ndarray, f_ext: np.ndarray, smooth_weight: float) -> np.ndarray:
    boundary_loops = meshboundaries(f_ext)
    v_smoothed = v_raw.copy()
    
    for boundary in boundary_loops:
        ring_raw = v_raw[boundary]
        n = len(ring_raw)
        # check number of rings and smooth weight
        if n < 3 or smooth_weight == 0.0:
            return ring_raw
        
        # ring smoothing
        cyc_laplacian = _cycle_laplacian(n)
        lhs = (sparse.eye(n, format="csr") + smooth_weight * cyc_laplacian).tocsc()
        ring_smoothed = spsolve(lhs, ring_raw)
        v_smoothed[boundary] = ring_smoothed

    return v_smoothed


def _cycle_laplacian(n: int) -> sparse.csr_matrix:
    ii = np.arange(n, dtype=np.int64)
    jj_prev = (ii - 1) % n
    jj_next = (ii + 1) % n

    rows = np.concatenate([ii, ii, ii])
    cols = np.concatenate([ii, jj_prev, jj_next])
    vals = np.concatenate([2.0 * np.ones(n), -np.ones(n), -np.ones(n)])
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))