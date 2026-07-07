import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize, minimize_scalar

from .meshboundaries import meshboundaries
from .beltrami_coefficient import beltrami_coefficient
from .cotangent_laplacian import cotangent_laplacian
from .generalized_laplacian import generalized_laplacian


def parallelogram_conformal_map(v: np.ndarray, f: np.ndarray, corner: np.ndarray) -> np.ndarray:
    # check dimension of v
    if v.shape[1] == 2:
        v = np.column_stack([v, np.zeros(len(v))])

    # find boundaries
    boundary_loops = meshboundaries(f)

    if len(boundary_loops) != 1:
        raise ValueError(f"Expected 1 boundary loops for an annulus, found {len(boundary_loops)}")
    
    bd_index = boundary_loops[0]


    # rearrange the order of boundaries
    id0 = np.where(bd_index == corner[0])[0][0]
    if id0.size == 0:
        raise ValueError("corner[0] is not on the boundary")
    bd_index = np.concatenate([bd_index[id0:], bd_index[:id0]])

    # fix corner index
    corner1 = 0
    corner2 = int(np.where(bd_index == corner[1])[0][0])
    corner3 = int(np.where(bd_index == corner[2])[0][0])
    corner4 = int(np.where(bd_index == corner[3])[0][0])


    # compute harmonic disk
    nv = len(v)
    bd_len = np.linalg.norm(v[bd_index] - v[np.roll(bd_index, -1)], axis=1)
    partial_edge_sum = np.concatenate([[0.0], np.cumsum(bd_len[:-1])])
    theta = 2.0 * np.pi * partial_edge_sum / np.maximum(bd_len.sum(), np.finfo(float).eps)
    bd = np.exp(1j * theta)

    m = cotangent_laplacian(v, f).tolil()
    m[bd_index, :] = 0
    m[bd_index, bd_index] = 1

    c = np.zeros(nv, dtype=complex)
    c[bd_index] = bd
    z = spsolve(m.tocsr(), c)
    disk = np.column_stack([np.real(z), np.imag(z)])
    

    # map disk to parallelogram
    mu = beltrami_coefficient(disk, f, v)
    ax = generalized_laplacian(disk, f, mu).tolil()
    ay = ax.copy()

    bx = np.zeros(nv, dtype=float)
    by = np.zeros(nv, dtype=float)

    seam0 = bd_index[corner1 : corner2 + 1]
    boundary1 = bd_index[corner2 : corner3 + 1]
    seam1 = bd_index[corner3 : corner4 + 1]
    boundary0 = np.concatenate([bd_index[corner4:], bd_index[: corner1 + 1]])

    if len(seam0) != len(seam1):
        raise RuntimeError("Seam boundary segments must have equal length")
    
    ax[seam0, :] = 0
    for r, b, t in zip(seam0, seam0, seam1[::-1]):
        ax[r, b] += 1
        ax[r, t] += -1
        
    x_fixed = np.unique(np.concatenate([boundary0, boundary1]))
    ax[x_fixed, :] = 0
    ax[x_fixed, x_fixed] = 1
    bx[boundary1] = 1.0
    base_height = spsolve(ax.tocsr(), bx)

    y_fixed = np.unique(np.concatenate([seam1, seam0]))
    ay[y_fixed, :] = 0
    ay[y_fixed, y_fixed] = 1
    by[seam1] = 2.0 * np.pi
    base_theta = spsolve(ay.tocsr(), by)

    def objective(params):
        width, shift = params
        if width <= np.finfo(float).eps:
            return np.inf
        para = np.column_stack([width * base_height, base_theta + shift * base_height])
        mu_para = beltrami_coefficient(para, f, v)
        return float(np.sum(np.abs(mu_para) ** 2))

    width_init = minimize_scalar(
        lambda width: objective((width, 0.0)),
        method="Bounded",
        bounds=(0.0, 10.0),
    ).x
    init_energy = objective((width_init, 0.0))

    opt = minimize(
        objective,
        x0=np.array([width_init, 0.0]),
        method="Powell",
        bounds=((0.01, 100.0), (-2.0 * np.pi, 2.0 * np.pi)),
        options={"xtol": 1e-4, "ftol": 1e-4, "maxiter": 120},
    )

    if np.isfinite(opt.fun) and opt.fun < init_energy:
        width_opt, shift_opt = opt.x
    else:
        width_opt, shift_opt = width_init, 0.0

    height = width_opt * base_height
    height = height - np.max(height)
    theta = base_theta + shift_opt * base_height

    return np.column_stack([height, theta])
