import numpy as np

from .meshboundaries import meshboundaries

def raw_extension(v: np.ndarray, f: np.ndarray, normal_blend: float) -> tuple[np.ndarray, np.ndarray]:
    # check dimensions
    if v.shape[1] == 2:
        v = np.column_stack([v, np.zeros(len(v), dtype=float)])
        normal_blend = 0.0
    elif v.shape[1] != 3:
        raise ValueError("v should have shape (n, 2) or (n, 3)")
    
    # get boundary loops
    boundary_loops = meshboundaries(f)
    if len(boundary_loops) != 2:
        raise ValueError(f"Expected annulus topology (2 boundary loops), found {len(boundary_loops)}")
    
    v_raw = v.copy()
    f_ext = f.copy()

    # extension on each boundary loop
    for boundary in boundary_loops:
        directions, steps = _outward_directions_and_steps(v_raw, f_ext, boundary, normal_blend) # get directions and step sizes
        ring_raw = v_raw[boundary] + directions * steps.reshape(-1, 1) # compute raw ring positions

        # get new vertex and connectivity
        new_start = len(v_raw)
        v_raw = np.vstack([v_raw, ring_raw])

        new_ring = np.arange(new_start, new_start + len(boundary), dtype=np.int64)
        stitched = _stitch_rings(boundary, new_ring)
        f_ext = np.vstack([f_ext, stitched])
    
    return v_raw, f_ext


def _outward_directions_and_steps(
    v: np.ndarray,
    f: np.ndarray,
    ring: np.ndarray,
    normal_blend: float,
    ) -> tuple[np.ndarray, np.ndarray]:
    
    # get adjacency and normals
    adjacency = _vertex_adjacency(len(v), f)
    normals = _vertex_normals(v, f)

    # initial setup
    dirs = np.zeros((len(ring), 3), dtype=float)
    steps = np.zeros(len(ring), dtype=float)
    inward_units = np.zeros((len(ring), 3), dtype=float)

    for i, vid in enumerate(ring):
        prev_vid = int(ring[(i - 1) % len(ring)])
        next_vid = int(ring[(i + 1) % len(ring)])

        # compute tangent direction
        tangent = v[next_vid] - v[prev_vid]
        tan_norm = np.linalg.norm(tangent)
        tangent = tangent / tan_norm

        # check interior points
        interior_neighbors = [n for n in adjacency[int(vid)] if n not in ring]
        if not interior_neighbors:
            raise RuntimeError(f"Boundary vertex {int(vid)} has no interior neighbor")
        
        # compute inward direction
        interior_pts = v[interior_neighbors]
        inward = (interior_pts - v[int(vid)]).mean(axis=0)
        inward_norm = np.linalg.norm(inward)
        inward_units[i] = inward / inward_norm
        
        # compute outward direction
        normal = normals[vid]
        normal_norm = np.linalg.norm(normal)
        normal = normal / normal_norm

        # compute binormal direction
        binormal = np.cross(tangent, normal)
        binormal_norm = np.linalg.norm(binormal)
        binormal = binormal / binormal_norm

        # flip to outward if binormal points inward
        if np.dot(binormal, inward_units[i]) > 0.0:
            binormal = -binormal

        # compute the extension direction
        direction = (1.0 - normal_blend) * binormal + normal_blend * normal
        direction_norm = np.linalg.norm(direction)
        direction = direction / direction_norm

        # update directions and step sizes
        dirs[i] = direction
        steps[i] = np.linalg.norm(interior_pts - v[int(vid)], axis=1).mean()

    return dirs, steps


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
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)

def _stitch_rings(prev_ring: np.ndarray, new_ring: np.ndarray) -> np.ndarray:
    n = len(prev_ring)
    i = np.arange(n, dtype=np.int64)
    i_next = (i + 1) % n

    tri_a = np.column_stack([prev_ring[i], new_ring[i], prev_ring[i_next]])
    tri_b = np.column_stack([prev_ring[i_next], new_ring[i], new_ring[i_next]])
    return np.vstack([tri_a, tri_b]).astype(np.int64)