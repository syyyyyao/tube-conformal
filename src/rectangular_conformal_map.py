import numpy as np
import trimesh
import networkx as nx
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize_scalar

from .meshboundaries import meshboundaries
from .beltrami_coefficient import beltrami_coefficient
from .cotangent_laplacian import cotangent_laplacian
from .generalized_laplacian import generalized_laplacian
from .face_area import face_area


def rectangular_conformal_map(v: np.ndarray, f: np.ndarray, corner: np.ndarray) -> np.ndarray:
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
    

    # map disk to rectangle
    mu = beltrami_coefficient(disk, f, v)
    ax = generalized_laplacian(disk, f, mu).tolil()
    ay = ax.copy()

    bx = np.zeros(nv, dtype=float)
    by = np.zeros(nv, dtype=float)

    left = bd_index[corner1 : corner2 + 1]
    top = bd_index[corner2 : corner3 + 1]
    right = bd_index[corner3 : corner4 + 1]
    bottom = np.concatenate([bd_index[corner4:], bd_index[: corner1 + 1]])

    if len(left) != len(right):
        raise RuntimeError("Left and right boundary segments must have equal length")
    
    ay[left, :] = 0
    for r, b, t in zip(left, left, right[::-1]):
        ay[r, b] += 1
        ay[r, t] += -1


    x_fixed = np.unique(np.concatenate([left, right]))
    ax[x_fixed, :] = 0
    ax[x_fixed, x_fixed] = 1
    bx[right] = 2.0 * np.pi
    rec_x = spsolve(ax.tocsr(), bx)

    y_fixed = np.unique(np.concatenate([top, bottom]))
    ay[y_fixed, :] = 0
    ay[y_fixed, y_fixed] = 1


    def objective(x_len):
        by = np.zeros(nv, dtype=float)
        by[top] = x_len
        rec_y = spsolve(ay.tocsr(), by)
        rec = np.column_stack([rec_x, rec_y])

        return np.sum(np.abs(beltrami_coefficient(rec, f, v)) ** 2)

    by_opt = minimize_scalar(objective, method="Bounded", bounds=(0,10)).x
    by[top] = by_opt
    rec_y_opt = spsolve(ay.tocsr(), by)

    return np.column_stack([rec_x, rec_y_opt])
