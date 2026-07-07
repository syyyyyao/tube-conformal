import numpy as np

from .meshboundaries import meshboundaries
from .slice_mesh import slice_mesh
from .rectangular_conformal_map import rectangular_conformal_map
from .cut_path_finder import cut_path_finder


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
        
    slice_path = cut_path_finder(v, f, boundary_loops)


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

    return np.column_stack([np.cos(rect[:, 1]), np.sin(rect[:, 1]), rect[:, 0]])
