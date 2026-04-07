import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .meshboundaries import meshboundaries


_LAYER_DECAY = 0.85
_MAX_STEP_RETRIES = 8
_SMOOTH_RING_WEIGHT = 0.25
_NORMAL_BLEND = 0.15
_INTERSECTION_EPS = 1e-8
_FACE_AREA_EPS = 1e-14
_PLANAR_REL_TOL = 1e-8


def extend_mesh(
    v: np.ndarray,
    f: np.ndarray,
    n_layers: int,
    smooth_ring_weight: float = _SMOOTH_RING_WEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extend a manifold annulus mesh by adding n_layers of strips on both boundaries.

    The input must be a triangulated surface with exactly two boundary loops.
    Original vertices are preserved as a prefix in the output:
    v_ext[:len(v)] == v.
    """
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=np.int64)
    input_dim = v.shape[1] if v.ndim == 2 else -1

    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("f must have shape (nf, 3)")
    if v.ndim != 2 or input_dim not in (2, 3):
        raise ValueError("v must have shape (nv, 2) or (nv, 3)")
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    if smooth_ring_weight < 0.0:
        raise ValueError("smooth_ring_weight must be >= 0")

    if input_dim == 2:
        # Keep the 2D API while using the same 3D geometric predicates internally.
        v = np.column_stack([v, np.zeros(len(v), dtype=float)])
        normal_blend = 0.0
        planar_mode = True
        plane_origin = np.zeros(3, dtype=float)
        plane_normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        planar_mode, plane_origin, plane_normal = _detect_plane(v)
        normal_blend = 0.0 if planar_mode else _NORMAL_BLEND

    boundary_loops = meshboundaries(f)
    if len(boundary_loops) != 2:
        raise ValueError(f"Expected annulus topology (2 boundary loops), found {len(boundary_loops)}")

    v_work = v.copy()
    f_work = f.copy()

    for loop_id, boundary in enumerate(boundary_loops):
        current_ring = np.asarray(boundary, dtype=np.int64)

        for layer_idx in range(n_layers):
            directions, base_steps = _outward_directions_and_steps(
                v_work,
                f_work,
                current_ring,
                normal_blend=normal_blend,
                forced_normal=plane_normal if planar_mode else None,
            )
            base_steps = np.maximum(base_steps * (_LAYER_DECAY**layer_idx), _INTERSECTION_EPS)

            accepted = False
            for retry in range(_MAX_STEP_RETRIES):
                steps = base_steps * (0.5**retry)
                ring_raw = v_work[current_ring] + directions * steps[:, None]
                ring_smoothed = _smooth_ring(ring_raw, smooth_ring_weight)
                if planar_mode:
                    ring_smoothed = _project_to_plane(ring_smoothed, plane_origin, plane_normal)

                new_start = len(v_work)
                new_ring = np.arange(new_start, new_start + len(current_ring), dtype=np.int64)
                stitched = _stitch_rings(current_ring, new_ring)

                v_candidate = np.vstack([v_work, ring_smoothed])
                if _has_degenerate_faces(v_candidate, stitched):
                    continue
                if _has_overlaps(v_candidate, f_work, stitched):
                    continue

                v_work = v_candidate
                f_work = np.vstack([f_work, stitched])
                current_ring = new_ring
                accepted = True
                break

            if not accepted:
                raise RuntimeError(
                    f"Failed to extend boundary {loop_id} at layer {layer_idx}: "
                    "overlap-free step not found."
                )

    if input_dim == 2:
        return v_work[:, :2], f_work
    return v_work, f_work


def _outward_directions_and_steps(
    v: np.ndarray,
    f: np.ndarray,
    ring: np.ndarray,
    normal_blend: float,
    forced_normal: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    adjacency = _vertex_adjacency(len(v), f)
    normals = _vertex_normals(v, f) if forced_normal is None else None
    ring_set = set(int(x) for x in ring)

    dirs = np.zeros((len(ring), 3), dtype=float)
    steps = np.zeros(len(ring), dtype=float)
    inward_units = np.zeros((len(ring), 3), dtype=float)

    for i, vid in enumerate(ring):
        prev_vid = int(ring[(i - 1) % len(ring)])
        next_vid = int(ring[(i + 1) % len(ring)])
        tangent = v[next_vid] - v[prev_vid]
        tan_norm = np.linalg.norm(tangent)
        if tan_norm < _INTERSECTION_EPS:
            tangent = v[next_vid] - v[int(vid)]
            tan_norm = np.linalg.norm(tangent)
        if tan_norm < _INTERSECTION_EPS:
            tangent = v[int(vid)] - v[prev_vid]
            tan_norm = np.linalg.norm(tangent)
        tangent = tangent / max(tan_norm, _INTERSECTION_EPS)

        interior_neighbors = [n for n in adjacency[int(vid)] if n not in ring_set]
        if not interior_neighbors:
            raise RuntimeError(f"Boundary vertex {int(vid)} has no interior neighbor")

        interior_pts = v[np.asarray(interior_neighbors, dtype=np.int64)]
        inward = (interior_pts - v[int(vid)]).mean(axis=0)
        inward_norm = np.linalg.norm(inward)
        if inward_norm < _INTERSECTION_EPS:
            inward = interior_pts[0] - v[int(vid)]
            inward_norm = np.linalg.norm(inward)
        inward_unit = inward / max(inward_norm, _INTERSECTION_EPS)
        inward_units[i] = inward_unit

        if forced_normal is None:
            normal = normals[int(vid)]
            normal_norm = np.linalg.norm(normal)
            if normal_norm < _INTERSECTION_EPS:
                neigh_normals = normals[np.asarray(interior_neighbors, dtype=np.int64)]
                normal = neigh_normals.mean(axis=0)
                normal_norm = np.linalg.norm(normal)
            normal = normal / max(normal_norm, _INTERSECTION_EPS)
        else:
            normal = forced_normal

        # Candidate outward direction from boundary frame.
        binormal = np.cross(tangent, normal)
        binormal_norm = np.linalg.norm(binormal)
        if binormal_norm < _INTERSECTION_EPS:
            binormal = np.cross(tangent, inward_unit)
            binormal_norm = np.linalg.norm(binormal)
        if binormal_norm < _INTERSECTION_EPS:
            binormal = -inward_unit
            binormal_norm = np.linalg.norm(binormal)
        binormal = binormal / max(binormal_norm, _INTERSECTION_EPS)

        # Flip to outward using local interior direction.
        if np.dot(binormal, inward_unit) > 0.0:
            binormal = -binormal

        direction = (1.0 - normal_blend) * binormal + normal_blend * normal
        direction_norm = np.linalg.norm(direction)
        if direction_norm < _INTERSECTION_EPS:
            direction = binormal
            direction_norm = np.linalg.norm(direction)
        direction = direction / max(direction_norm, _INTERSECTION_EPS)

        if np.dot(direction, inward_unit) > 0.0:
            direction = -direction

        dirs[i] = direction
        steps[i] = np.linalg.norm(interior_pts - v[int(vid)], axis=1).mean()

    # Smooth direction field along the ring to reduce kinks.
    dirs = 0.5 * dirs + 0.25 * np.roll(dirs, -1, axis=0) + 0.25 * np.roll(dirs, 1, axis=0)
    dirs = _normalize_rows(dirs)
    for i in range(len(dirs)):
        if np.dot(dirs[i], inward_units[i]) > 0.0:
            dirs[i] = -dirs[i]
    dirs = _normalize_rows(dirs)
    steps = np.maximum(steps, _INTERSECTION_EPS)
    return dirs, steps


def _smooth_ring(
    ring_raw: np.ndarray,
    smooth_ring_weight: float,
) -> np.ndarray:
    n = len(ring_raw)
    if n < 3 or smooth_ring_weight == 0.0:
        return ring_raw

    lap = _cycle_laplacian(n)
    lhs = (sparse.eye(n, format="csr") + smooth_ring_weight * lap).tocsc()

    out = np.empty_like(ring_raw)
    for dim in range(3):
        out[:, dim] = spsolve(lhs, ring_raw[:, dim])
    return out


def _cycle_laplacian(n: int) -> sparse.csr_matrix:
    ii = np.arange(n, dtype=np.int64)
    jj_prev = (ii - 1) % n
    jj_next = (ii + 1) % n

    rows = np.concatenate([ii, ii, ii])
    cols = np.concatenate([ii, jj_prev, jj_next])
    vals = np.concatenate([2.0 * np.ones(n), -np.ones(n), -np.ones(n)])
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def _stitch_rings(prev_ring: np.ndarray, new_ring: np.ndarray) -> np.ndarray:
    n = len(prev_ring)
    i = np.arange(n, dtype=np.int64)
    i_next = (i + 1) % n

    tri_a = np.column_stack([prev_ring[i], new_ring[i], prev_ring[i_next]])
    tri_b = np.column_stack([prev_ring[i_next], new_ring[i], new_ring[i_next]])
    return np.vstack([tri_a, tri_b]).astype(np.int64)


def _vertex_adjacency(nv: int, f: np.ndarray) -> list[list[int]]:
    adj = [set() for _ in range(nv)]
    for a, b, c in f:
        a = int(a)
        b = int(b)
        c = int(c)
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    return [sorted(list(x)) for x in adj]


def _vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    tri = v[f]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(v)
    np.add.at(normals, f[:, 0], face_normals)
    np.add.at(normals, f[:, 1], face_normals)
    np.add.at(normals, f[:, 2], face_normals)
    return _normalize_rows(normals)


def _detect_plane(v: np.ndarray) -> tuple[bool, np.ndarray, np.ndarray]:
    origin = v.mean(axis=0)
    centered = v - origin

    if len(v) < 3:
        return False, origin, np.array([0.0, 0.0, 1.0], dtype=float)

    _, svals, vh = np.linalg.svd(centered, full_matrices=False)
    if len(svals) < 3:
        return False, origin, np.array([0.0, 0.0, 1.0], dtype=float)

    normal = vh[-1]
    normal = normal / max(np.linalg.norm(normal), _INTERSECTION_EPS)
    scale = max(svals[0], _INTERSECTION_EPS)
    is_planar = bool(svals[-1] <= _PLANAR_REL_TOL * scale)
    return is_planar, origin, normal


def _project_to_plane(points: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    rel = points - origin[None, :]
    dist = rel @ normal
    return points - dist[:, None] * normal[None, :]


def _has_degenerate_faces(v: np.ndarray, f: np.ndarray) -> bool:
    tri = v[f]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return bool(np.any(area2 <= _FACE_AREA_EPS))


def _has_overlaps(v_all: np.ndarray, f_existing: np.ndarray, f_new: np.ndarray) -> bool:
    tri_existing = v_all[f_existing]
    tri_new = v_all[f_new]
    min_existing = tri_existing.min(axis=1)
    max_existing = tri_existing.max(axis=1)
    min_new = tri_new.min(axis=1)
    max_new = tri_new.max(axis=1)

    # New-vs-existing intersections
    for i in range(len(f_new)):
        aabb_mask = np.all(min_new[i] <= max_existing + _INTERSECTION_EPS, axis=1) & np.all(
            max_new[i] >= min_existing - _INTERSECTION_EPS, axis=1
        )
        hits = np.where(aabb_mask)[0]
        for j in hits:
            if _shares_vertex(f_new[i], f_existing[j]):
                continue
            if _triangles_intersect(tri_new[i], tri_existing[j]):
                return True

    # New-vs-new intersections
    for i in range(len(f_new)):
        aabb_mask = np.all(min_new[i] <= max_new + _INTERSECTION_EPS, axis=1) & np.all(
            max_new[i] >= min_new - _INTERSECTION_EPS, axis=1
        )
        hits = np.where(aabb_mask)[0]
        for j in hits:
            if j <= i:
                continue
            if _shares_vertex(f_new[i], f_new[j]):
                continue
            if _triangles_intersect(tri_new[i], tri_new[j]):
                return True
    return False


def _shares_vertex(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(len(set(int(x) for x in a).intersection(int(x) for x in b)) > 0)


def _triangles_intersect(tri_a: np.ndarray, tri_b: np.ndarray) -> bool:
    if _coplanar(tri_a, tri_b):
        return _coplanar_triangles_intersect(tri_a, tri_b)

    edges_a = ((tri_a[0], tri_a[1]), (tri_a[1], tri_a[2]), (tri_a[2], tri_a[0]))
    edges_b = ((tri_b[0], tri_b[1]), (tri_b[1], tri_b[2]), (tri_b[2], tri_b[0]))

    for p0, p1 in edges_a:
        if _segment_intersects_triangle(p0, p1, tri_b):
            return True
    for p0, p1 in edges_b:
        if _segment_intersects_triangle(p0, p1, tri_a):
            return True

    return False


def _coplanar(tri_a: np.ndarray, tri_b: np.ndarray) -> bool:
    na = np.cross(tri_a[1] - tri_a[0], tri_a[2] - tri_a[0])
    nb = np.cross(tri_b[1] - tri_b[0], tri_b[2] - tri_b[0])
    na_norm = np.linalg.norm(na)
    nb_norm = np.linalg.norm(nb)
    if na_norm <= _FACE_AREA_EPS or nb_norm <= _FACE_AREA_EPS:
        return False

    if np.linalg.norm(np.cross(na, nb)) > _INTERSECTION_EPS * na_norm * nb_norm:
        return False

    na_unit = na / na_norm
    plane_dist = abs(np.dot(tri_b[0] - tri_a[0], na_unit))
    return bool(plane_dist <= _INTERSECTION_EPS)


def _segment_intersects_triangle(p0: np.ndarray, p1: np.ndarray, tri: np.ndarray) -> bool:
    v0, v1, v2 = tri
    direction = p1 - p0
    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(direction, edge2)
    a = float(np.dot(edge1, h))
    if abs(a) <= _INTERSECTION_EPS:
        return False

    inv_a = 1.0 / a
    s = p0 - v0
    u = inv_a * float(np.dot(s, h))
    if u < -_INTERSECTION_EPS or u > 1.0 + _INTERSECTION_EPS:
        return False

    q = np.cross(s, edge1)
    v = inv_a * float(np.dot(direction, q))
    if v < -_INTERSECTION_EPS or u + v > 1.0 + _INTERSECTION_EPS:
        return False

    t = inv_a * float(np.dot(edge2, q))
    return bool(-_INTERSECTION_EPS <= t <= 1.0 + _INTERSECTION_EPS)


def _coplanar_triangles_intersect(tri_a: np.ndarray, tri_b: np.ndarray) -> bool:
    normal = np.cross(tri_a[1] - tri_a[0], tri_a[2] - tri_a[0])
    axis = int(np.argmax(np.abs(normal)))
    keep = [0, 1, 2]
    keep.remove(axis)

    a2 = tri_a[:, keep]
    b2 = tri_b[:, keep]

    edges = ((0, 1), (1, 2), (2, 0))
    for i0, i1 in edges:
        for j0, j1 in edges:
            if _segments_intersect_2d(a2[i0], a2[i1], b2[j0], b2[j1]):
                return True

    if _point_in_triangle_2d(a2[0], b2) or _point_in_triangle_2d(b2[0], a2):
        return True
    return False


def _segments_intersect_2d(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    o1 = _orient2d(p1, p2, q1)
    o2 = _orient2d(p1, p2, q2)
    o3 = _orient2d(q1, q2, p1)
    o4 = _orient2d(q1, q2, p2)

    if (o1 * o2 < -_INTERSECTION_EPS) and (o3 * o4 < -_INTERSECTION_EPS):
        return True

    if abs(o1) <= _INTERSECTION_EPS and _on_segment_2d(p1, q1, p2):
        return True
    if abs(o2) <= _INTERSECTION_EPS and _on_segment_2d(p1, q2, p2):
        return True
    if abs(o3) <= _INTERSECTION_EPS and _on_segment_2d(q1, p1, q2):
        return True
    if abs(o4) <= _INTERSECTION_EPS and _on_segment_2d(q1, p2, q2):
        return True

    return False


def _orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment_2d(a: np.ndarray, p: np.ndarray, b: np.ndarray) -> bool:
    return bool(
        min(a[0], b[0]) - _INTERSECTION_EPS <= p[0] <= max(a[0], b[0]) + _INTERSECTION_EPS
        and min(a[1], b[1]) - _INTERSECTION_EPS <= p[1] <= max(a[1], b[1]) + _INTERSECTION_EPS
    )


def _point_in_triangle_2d(p: np.ndarray, tri: np.ndarray) -> bool:
    a, b, c = tri
    o1 = _orient2d(a, b, p)
    o2 = _orient2d(b, c, p)
    o3 = _orient2d(c, a, p)
    has_neg = (o1 < -_INTERSECTION_EPS) or (o2 < -_INTERSECTION_EPS) or (o3 < -_INTERSECTION_EPS)
    has_pos = (o1 > _INTERSECTION_EPS) or (o2 > _INTERSECTION_EPS) or (o3 > _INTERSECTION_EPS)
    return not (has_neg and has_pos)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, _INTERSECTION_EPS)
    return x / n
