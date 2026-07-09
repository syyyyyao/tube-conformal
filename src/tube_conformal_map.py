import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve

from .meshboundaries import meshboundaries
from .beltrami_coefficient import beltrami_coefficient
from .cut_path_finder import cut_path_finder
from .generalized_laplacian import generalized_laplacian
from .parallelogram_conformal_map import parallelogram_conformal_map
from .slice_mesh import slice_mesh


def tube_conformal_map(
    v: np.ndarray,
    f: np.ndarray,
    seam_strip_width: float = 0.10,
) -> np.ndarray:
    """
    Compute the full three-stage conformal tubular parameterization.

    The pipeline is:
    1. initial tube parameterization,
    2. cut seam correction,
    3. cut-open interior refinement.
    """

    tube0 = initial_tube(v, f)
    tube_corrected = seam_correction(tube0, f, v, seam_strip_width)
    return interior_refinement(tube_corrected, f, v)


def initial_tube(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Compute the initial tubular parameterization stage.
    """

    v, f = _as_mesh_arrays(v, f)
    boundary_loops = _tube_boundary_loops(f)
    cut_path = cut_path_finder(v, f, boundary_loops)

    if len(cut_path) < 2:
        raise RuntimeError("Cut path must contain at least two vertices")

    v_sliced, f_sliced = slice_mesh(v, f, cut_path)
    corner = np.array(
        [cut_path[0], cut_path[-1], len(v) + len(cut_path) - 1, len(v)],
        dtype=np.int64,
    )

    para_sliced = parallelogram_conformal_map(v_sliced, f_sliced, corner)
    para = para_sliced[: len(v)]
    return np.column_stack([np.cos(para[:, 1]), np.sin(para[:, 1]), para[:, 0]])


def seam_correction(
    tube0: np.ndarray,
    f: np.ndarray,
    v: np.ndarray,
    seam_strip_width: float = 0.10,
) -> np.ndarray:
    """
    Correct a local annular strip around the cut seam.
    """

    v, f = _as_mesh_arrays(v, f)
    tube0 = np.asarray(tube0, dtype=float)
    if len(tube0) != len(v):
        raise ValueError("tube0 and v must have the same number of vertices")

    boundary_loops = _tube_boundary_loops(f)
    cut_path = cut_path_finder(v, f, boundary_loops)

    annulus0 = np.column_stack(
        [np.exp(tube0[:, 2]) * tube0[:, 0], np.exp(tube0[:, 2]) * tube0[:, 1]]
    )

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


def interior_refinement(
    tube_corrected: np.ndarray,
    f: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Cut open the corrected tube, fix the cut-surface boundary, and improve
    interior vertices with one generalized Laplacian solve.
    """

    v, f = _as_mesh_arrays(v, f)
    tube_corrected = np.asarray(tube_corrected, dtype=float)
    if len(tube_corrected) != len(v):
        raise ValueError("tube_corrected and v must have the same number of vertices")

    boundary_loops = _tube_boundary_loops(f)
    cut_path = cut_path_finder(v, f, boundary_loops)

    v_sliced, f_sliced = slice_mesh(v, f, cut_path)
    tube_sliced, f_sliced_tube = slice_mesh(tube_corrected, f, cut_path)
    if not np.array_equal(f_sliced, f_sliced_tube):
        raise RuntimeError("Surface and tube slicing produced different connectivity")

    boundary_loops_sliced = meshboundaries(f_sliced)
    if len(boundary_loops_sliced) == 0:
        return tube_corrected

    boundary_vertices = np.unique(np.concatenate(boundary_loops_sliced))
    free_mask = np.ones(len(v_sliced), dtype=bool)
    free_mask[boundary_vertices] = False
    if not np.any(free_mask):
        return tube_corrected

    para = _tube_to_unwrapped_parallelogram(tube_sliced, f_sliced)
    mu = beltrami_coefficient(para, f_sliced, v_sliced)
    laplacian_mat = generalized_laplacian(para, f_sliced, mu).tolil()
    laplacian_mat[boundary_vertices, :] = 0
    laplacian_mat[boundary_vertices, boundary_vertices] = 1
    laplacian_mat = laplacian_mat.tocsr()

    x_fixed = np.zeros(len(v_sliced))
    y_fixed = np.zeros(len(v_sliced))
    x_fixed[boundary_vertices] = para[boundary_vertices, 0]
    y_fixed[boundary_vertices] = para[boundary_vertices, 1]

    x = spsolve(laplacian_mat, x_fixed)
    y = spsolve(laplacian_mat, y_fixed)

    theta = y[: len(v)]
    return np.column_stack([np.cos(theta), np.sin(theta), x[: len(v)]])


def _as_mesh_arrays(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] not in (2, 3):
        raise ValueError("v must have shape (n_vertices, 2) or (n_vertices, 3)")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("The input mesh is not triangulated")
    return v, f


def _tube_boundary_loops(f: np.ndarray) -> list[np.ndarray]:
    boundary_loops = meshboundaries(f)
    if len(boundary_loops) != 2:
        raise ValueError(f"Expected 2 boundary loops for a tube, found {len(boundary_loops)}")
    return boundary_loops


def _tube_to_unwrapped_parallelogram(tube: np.ndarray, f: np.ndarray) -> np.ndarray:
    theta_mod = np.arctan2(tube[:, 1], tube[:, 0])
    theta = np.full(len(tube), np.nan)

    edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    adjacency: list[list[int]] = [[] for _ in range(len(tube))]
    for i, j in edges:
        i = int(i)
        j = int(j)
        adjacency[i].append(j)
        adjacency[j].append(i)

    vertices = np.unique(f.reshape(-1))
    for root in vertices:
        root = int(root)
        if np.isfinite(theta[root]):
            continue
        theta[root] = theta_mod[root]
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if np.isfinite(theta[neighbor]):
                    continue
                delta = (theta_mod[neighbor] - theta_mod[current] + np.pi) % (2.0 * np.pi) - np.pi
                theta[neighbor] = theta[current] + delta
                stack.append(neighbor)

    theta = np.where(np.isfinite(theta), theta, theta_mod)
    return np.column_stack([tube[:, 2], theta])


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
