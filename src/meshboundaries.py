from __future__ import annotations

from collections import defaultdict

import numpy as np


def meshboundaries(f: np.ndarray) -> list[np.ndarray]:
    """
    Extract ordered boundary loops from a triangular mesh.
    Returns a list of loops sorted by descending length.
    """
    faces = np.asarray(f, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("f must have shape (nf, 3)")

    directed = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)

    # Count undirected occurrences. A boundary edge appears exactly once.
    keys = undirected[:, 0] * (faces.max() + 1) + undirected[:, 1]
    unique_keys, counts = np.unique(keys, return_counts=True)
    key_to_count = dict(zip(unique_keys.tolist(), counts.tolist()))

    is_boundary = np.array([key_to_count[k] == 1 for k in keys], dtype=bool)
    boundary_edges = directed[is_boundary]

    if boundary_edges.size == 0:
        return [], np.array([], dtype=np.int64)

    start_to_edges: dict[int, list[int]] = defaultdict(list)
    for idx, (u, _) in enumerate(boundary_edges):
        start_to_edges[int(u)].append(idx)

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

    loops.sort(key=lambda bd: len(bd), reverse=True)
    return loops
