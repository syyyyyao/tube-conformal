import numpy as np

def _ordered_boundary_loops(f: np.ndarray) -> list[np.ndarray]:
    """
    Extract ordered boundary loops from a triangular mesh.
    """
    faces = np.asarray(f, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("f must have shape (nf, 3)")

    directed = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )
    undirected = np.sort(directed, axis=1)

    max_vid = int(faces.max()) + 1
    keys = undirected[:, 0] * max_vid + undirected[:, 1]
    unique_keys, counts = np.unique(keys, return_counts=True)
    key_to_count = dict(zip(unique_keys.tolist(), counts.tolist()))
    is_boundary = np.array([key_to_count[k] == 1 for k in keys], dtype=bool)
    boundary_edges = directed[is_boundary]

    if boundary_edges.size == 0:
        return []

    start_to_edges: dict[int, list[int]] = {}
    for idx, (u, _) in enumerate(boundary_edges):
        u = int(u)
        if u not in start_to_edges:
            start_to_edges[u] = []
        start_to_edges[u].append(idx)

    used = np.zeros(len(boundary_edges), dtype=bool)
    loops: list[np.ndarray] = []

    for edge_idx in range(len(boundary_edges)):
        if used[edge_idx]:
            continue

        u0, v0 = boundary_edges[edge_idx]
        used[edge_idx] = True
        loop = [int(u0), int(v0)]
        cur = int(v0)

        while cur != int(u0):
            candidates = [idx for idx in start_to_edges.get(cur, []) if not used[idx]]
            if not candidates:
                raise RuntimeError("Failed to reconstruct a closed boundary loop")

            nxt_edge = candidates[0]
            used[nxt_edge] = True
            _, nxt = boundary_edges[nxt_edge]
            loop.append(int(nxt))
            cur = int(nxt)

        loops.append(np.asarray(loop[:-1], dtype=np.int64))

    loops.sort(key=len, reverse=True)
    return loops


def slice_mesh(v: np.ndarray, f: np.ndarray, slice_path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice an annulus mesh into a simply-connected mesh.
    """
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=np.int64)
    slice_path = np.asarray(slice_path, dtype=np.int64)

    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("The input mesh is not triangulated!")
    if slice_path.ndim != 1 or slice_path.size < 2:
        raise ValueError("slice_path must be a 1D array with at least 2 vertices")
    if len(v) == 0 or len(f) == 0:
        raise ValueError("v and f must be non-empty")
    if slice_path.min() < 0 or slice_path.max() >= len(v):
        raise ValueError("slice_path contains vertex indices outside v")

    boundary_loops = _ordered_boundary_loops(f)
    if len(boundary_loops) == 0:
        raise ValueError("Expected at least one boundary loop")

    start_vid = int(slice_path[0])
    bd = None
    for loop in boundary_loops:
        if np.any(loop == start_vid):
            bd = loop
            break
    if bd is None:
        bd = boundary_loops[0]

    nv_ori = len(v)
    nf_ori = len(f)

    centroid = np.mean(v[bd], axis=0)
    v_aug = np.vstack([v, centroid])
    center_idx = nv_ori

    f_aug = np.vstack(
        [
            f,
            np.column_stack(
                [np.roll(bd, -1), bd, np.full(len(bd), center_idx, dtype=np.int64)]
            ),
        ]
    )
    slice_path_aug = np.concatenate([[center_idx], slice_path])

    nv = len(v_aug)
    nf = len(f_aug)
    np_path = len(slice_path_aug)

    v_sliced = np.vstack([v_aug, v_aug[slice_path_aug[1:]]])
    e = np.vstack([f_aug[:, [0, 1]], f_aug[:, [1, 2]], f_aug[:, [2, 0]]])
    f_sliced = f_aug.copy()

    for i in range(1, np_path):
        pid = int(slice_path_aug[i])
        prev = int(slice_path_aug[i - 1])

        dir2 = np.where((e[:, 1] == prev) & (e[:, 0] == pid))[0]
        if dir2.size == 0:
            raise RuntimeError("Failed to locate initial slice face")

        fid = int(dir2[0] % nf)
        id_in_face = np.where(f_aug[fid] == pid)[0]
        if id_in_face.size == 0:
            raise RuntimeError("Slice vertex is not in located face")
        id_in_face = int(id_in_face[0])

        f_sliced[fid, id_in_face] = nv + i - 1

        flag = True
        while flag:
            if id_in_face == 0:
                nxt = np.where((e[:, 0] == f_aug[fid, 0]) & (e[:, 1] == f_aug[fid, 2]))[0]
            elif id_in_face == 1:
                nxt = np.where((e[:, 0] == f_aug[fid, 1]) & (e[:, 1] == f_aug[fid, 0]))[0]
            else:
                nxt = np.where((e[:, 0] == f_aug[fid, 2]) & (e[:, 1] == f_aug[fid, 1]))[0]

            if nxt.size == 0:
                break

            fid = int(nxt[0] % nf)
            id_in_face = np.where(f_aug[fid] == pid)[0]
            if id_in_face.size == 0:
                break
            id_in_face = int(id_in_face[0])

            f_sliced[fid, id_in_face] = nv + i - 1
            flag = int(np.isin(f_aug[fid], slice_path_aug).sum()) == 1

    v_sliced = np.delete(v_sliced, center_idx, axis=0)
    f_sliced = f_sliced[:nf_ori]

    shift_mask = f_sliced >= nv_ori + 1
    f_sliced[shift_mask] -= 1

    return v_sliced, f_sliced
