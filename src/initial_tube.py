import numpy as np
import trimesh
import networkx as nx

from .meshboundaries import meshboundaries
from .slice_mesh import slice_mesh
from .rectangular_conformal_map import rectangular_conformal_map


def initial_tube(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    
    """
    # check input
    if f.shape[1] != 3:
        raise ValueError("The input mesh is not triangulated!")

    # find boundaries
    
    boundary_loops = meshboundaries(f)
    
    if len(boundary_loops) != 2:
        raise ValueError(f"Expected 2 boundary loops for a tube, found {len(boundary_loops)}")
        
    bd1 = list(boundary_loops[0])
    bd2 = list(boundary_loops[1])

    # determine the outer and inner boundaries
    l_bd1 = np.float32(np.linalg.norm(v[bd1] - v[np.roll(bd1, -1)], axis=1).sum())
    l_bd2 = np.float32(np.linalg.norm(v[bd2] - v[np.roll(bd2, -1)], axis=1).sum())
    if l_bd1 >= l_bd2:
        outer_bd, inner_bd = bd1, bd2
    else:
        outer_bd, inner_bd = bd2, bd1
    

    # search shortest path with Dijkstra
    # initialize graph
    G = nx.Graph()
    
    # add all edges with their lengths as weights
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    edges = mesh.edges_unique
    lengths = mesh.edges_unique_length
    weighted_edges = [(e[0], e[1], l) for e, l in zip(edges, lengths)]
    G.add_weighted_edges_from(weighted_edges)

    # add virtual nodes
    G.add_node('Virtual_Source')
    G.add_node('Virtual_Target')

    # connect source to all nodes in Boundary A (weight = 0)
    for node in outer_bd:
        G.add_edge('Virtual_Source', node, weight=0)
        
    # connect all nodes in Boundary B to target (weight = 0)
    for node in inner_bd:
        G.add_edge(node, 'Virtual_Target', weight=0)

    # run Dijkstra
    full_path = nx.dijkstra_path(G, 'Virtual_Source', 'Virtual_Target', weight='weight')
    slice_path = full_path[1:-1]


    # cut the surface to disk
    v_sliced, f_sliced = slice_mesh(v, f, slice_path)

    corner = np.array(
        [slice_path[0], slice_path[-1], len(v) + len(slice_path) - 1, len(v)],
        dtype=np.int64,
    )

    # map the sliced surface to a square
    rect_sliced = rectangular_conformal_map(v_sliced, f_sliced, corner)

    # map to a tube
    rect = rect_sliced[:len(v)]

    return np.column_stack([np.cos(rect[:, 0]), np.sin(rect[:, 0]), rect[:, 1]])
