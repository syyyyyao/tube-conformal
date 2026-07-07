import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve

from .meshboundaries import meshboundaries
from .beltrami_coefficient import beltrami_coefficient
from .cut_path_finder import cut_path_finder
from .generalized_laplacian import generalized_laplacian


def tube_conformal_map(
    tube0: np.ndarray,
    f: np.ndarray,
    v: np.ndarray,
    seam_strip_width: float = 0.05,
) -> np.ndarray:
    """
    Apply a quasi-conformal correction only near the cut seam of the tube map.

    Outside the seam strip, the input tube map is preserved.
    """

    # get boundaries
    boundary_loops = meshboundaries(f)
    if len(boundary_loops) != 2:
        raise ValueError(f"Expected 2 boundary loops for a tube, found {len(boundary_loops)}")

    # map tube to annulus
    annulus0 = np.column_stack([np.exp(tube0[:, 2]) * tube0[:, 0], np.exp(tube0[:, 2]) * tube0[:, 1]])

    cut_path = cut_path_finder(v, f, boundary_loops)
    strip_vertex_mask = _cut_path_strip_mask(tube0, f, cut_path, seam_strip_width)
    strip_face_mask = np.all(strip_vertex_mask[f], axis=1)

    if not np.any(strip_face_mask):
        return tube0
    
    local_vertices, local_faces = np.unique(f[strip_face_mask].reshape(-1), return_inverse=True)
    local_faces = local_faces.reshape(-1, 3)

    if len(local_vertices) == 0 or len(local_faces) == 0:
        return tube0
    local_annulus0 = annulus0[local_vertices]
    local_v = v[local_vertices]

    local_boundary_loops = meshboundaries(local_faces)
    if len(local_boundary_loops) == 0:
        return tube0

    local_bd = np.unique(np.concatenate(local_boundary_loops))
    free_mask = np.ones(len(local_vertices), dtype=bool)
    free_mask[local_bd] = False

    if not np.any(free_mask):
        return tube0

    mu = beltrami_coefficient(local_annulus0, local_faces, local_v)
    laplacian_mat = generalized_laplacian(local_annulus0, local_faces, mu).tolil()

    laplacian_mat[local_bd, :] = 0
    laplacian_mat[local_bd, local_bd] = 1
    laplacian_mat = laplacian_mat.tocsr()

    x_fixed = np.zeros(len(local_vertices))
    y_fixed = np.zeros(len(local_vertices))
    x_fixed[local_bd] = local_annulus0[local_bd, 0]
    y_fixed[local_bd] = local_annulus0[local_bd, 1]

    x_local = spsolve(laplacian_mat, x_fixed)
    y_local = spsolve(laplacian_mat, y_fixed)

    annulus = annulus0.copy()
    free_vertices = local_vertices[free_mask]
    annulus[free_vertices, 0] = x_local[free_mask]
    annulus[free_vertices, 1] = y_local[free_mask]

    radius = np.sqrt(annulus[:, 0] ** 2 + annulus[:, 1] ** 2)

    return np.column_stack([annulus[:, 0] / radius, annulus[:, 1] / radius, np.log(radius)])


def _cut_path_strip_mask(
    tube: np.ndarray,
    f: np.ndarray,
    cut_path: np.ndarray,
    seam_strip_width: float,
) -> np.ndarray:
    if len(cut_path) == 0:
        return np.zeros(len(tube), dtype=bool)

    edges = np.sort(
        np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]),
        axis=1,
    )
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(tube[edges[:, 0]] - tube[edges[:, 1]], axis=1)

    source = len(tube)
    eps_edges = np.full(len(cut_path), np.finfo(float).eps)
    rows = np.concatenate([edges[:, 0], edges[:, 1], np.full(len(cut_path), source), cut_path])
    cols = np.concatenate([edges[:, 1], edges[:, 0], cut_path, np.full(len(cut_path), source)])
    data = np.concatenate([lengths, lengths, eps_edges, eps_edges])
    graph = sparse.csr_matrix((data, (rows, cols)), shape=(len(tube) + 1, len(tube) + 1))

    dist = dijkstra(graph, directed=False, indices=source)[: len(tube)]
    threshold = 2.0 * np.pi * seam_strip_width
    return dist <= threshold
