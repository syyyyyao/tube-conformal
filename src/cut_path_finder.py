import numpy as np
import trimesh
import networkx as nx

from .meshboundaries import meshboundaries


def cut_path_finder(
    v: np.ndarray,
    f: np.ndarray,
    boundary_loops: list[np.ndarray] | None = None,
) -> np.ndarray:
    if boundary_loops is None:
        boundary_loops = meshboundaries(f)

    if len(boundary_loops) != 2:
        raise ValueError(f"Expected 2 boundary loops for a tube, found {len(boundary_loops)}")

    bd1 = list(boundary_loops[0])
    bd2 = list(boundary_loops[1])

    l_bd1 = np.float32(np.linalg.norm(v[bd1] - v[np.roll(bd1, -1)], axis=1).sum())
    l_bd2 = np.float32(np.linalg.norm(v[bd2] - v[np.roll(bd2, -1)], axis=1).sum())
    if l_bd1 >= l_bd2:
        outer_bd, inner_bd = bd1, bd2
    else:
        outer_bd, inner_bd = bd2, bd1

    graph = nx.Graph()

    mesh = trimesh.Trimesh(vertices=v, faces=f)
    edges = mesh.edges_unique
    lengths = mesh.edges_unique_length
    weighted_edges = [(e[0], e[1], length) for e, length in zip(edges, lengths)]
    graph.add_weighted_edges_from(weighted_edges)

    graph.add_node("Virtual_Source")
    graph.add_node("Virtual_Target")

    for node in outer_bd:
        graph.add_edge("Virtual_Source", node, weight=0)
    for node in inner_bd:
        graph.add_edge(node, "Virtual_Target", weight=0)

    full_path = nx.dijkstra_path(graph, "Virtual_Source", "Virtual_Target", weight="weight")
    return np.asarray(full_path[1:-1], dtype=np.int64)
