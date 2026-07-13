import numpy as np
from scipy.optimize import minimize, minimize_scalar


def conformal_bend_major(
    tube: np.ndarray,
    R: float | None = None,
    f: np.ndarray | None = None,
    v: np.ndarray | None = None,
) -> np.ndarray:
    """Bend a unit tube in its major (axial) direction.

    When ``R`` is given, this retains the original, centred bending map.  When
    ``R`` is ``None``, ``f`` and ``v`` are required and both the bending radius
    and the initial axial position are chosen by minimizing squared normalized
    log-area distortion relative to ``v``.

    Parameters
    ----------
    tube : (n, 3) ndarray
        Unit tubular coordinates ``(cos(phi), sin(phi), z)``.
    R : float, optional
        Bending radius greater than one. If omitted, it is optimized
        automatically within the non-overlapping major-bend interval.
    f : (m, 3) ndarray, optional
        Triangle connectivity, required for automatic optimization.
    v : (n, 2) or (n, 3) ndarray, optional
        Reference surface, required for automatic optimization.
    """
    tube = _validate_tube(tube)
    z = tube[:, 2]
    z_center = 0.5 * (np.max(z) + np.min(z))
    phi = np.arctan2(tube[:, 1], tube[:, 0])

    def bend(radius: float, phase: float) -> np.ndarray:
        axial_angle = (
            0.5 * np.sqrt(radius**2 - 1.0) * (z - z_center) + phase
        )
        theta = 2.0 * np.arctan2(
            np.sqrt(radius + 1.0) * np.sin(axial_angle),
            np.sqrt(radius - 1.0) * np.cos(axial_angle),
        )
        return np.column_stack(
            [
                (radius + np.cos(theta)) * np.cos(phi),
                np.sin(theta),
                (radius + np.cos(theta)) * np.sin(phi),
            ]
        )

    if R is not None:
        return bend(_validate_radius(R), phase=0.0)

    f, v = _validate_optimization_mesh(tube, f, v)
    height = float(np.ptp(z))
    if height <= np.finfo(float).eps:
        raise ValueError("Cannot optimize a major bend for a tube with zero height")

    # A major bend must not traverse more than one full toroidal turn.
    R_max = float(np.sqrt(1.0 + (2.0 * np.pi / height) ** 2))
    radius_bounds = _open_radius_bounds(1.0, R_max)

    def objective(params: tuple[float, float] | np.ndarray) -> float:
        radius, phase = params
        if not radius_bounds[0] <= radius <= radius_bounds[1]:
            return np.inf
        return _area_distortion_energy(v, f, bend(radius, phase))

    radius_init = minimize_scalar(
        lambda radius: objective((radius, 0.0)),
        method="bounded",
        bounds=radius_bounds,
        options={"xatol": 1e-4},
    ).x
    radius_opt, phase_opt = _optimize_radius_and_phase(
        objective,
        radius_init,
        radius_bounds,
        phase_bounds=(-0.5 * np.pi, 0.5 * np.pi),
    )
    return bend(radius_opt, phase_opt)


def conformal_bend_minor(
    tube: np.ndarray,
    R: float | None = None,
    f: np.ndarray | None = None,
    v: np.ndarray | None = None,
) -> np.ndarray:
    """Bend a unit tube in its minor (circumferential) direction.

    When ``R`` is given, this retains the original bending map. When ``R`` is
    ``None``, ``f`` and ``v`` are required and both the bending radius and the
    initial circumferential position are chosen by minimizing squared
    normalized log-area distortion relative to ``v``.
    """
    tube = _validate_tube(tube)
    u = np.arctan2(tube[:, 1], tube[:, 0])
    z = tube[:, 2]

    def bend(radius: float, phase: float) -> np.ndarray:
        u_shifted = u + phase
        theta = 2.0 * np.arctan2(
            np.sqrt(radius + 1.0) * np.sin(0.5 * u_shifted),
            np.sqrt(radius - 1.0) * np.cos(0.5 * u_shifted),
        )
        phi = z / np.sqrt(radius**2 - 1.0)
        return np.column_stack(
            [
                (radius + np.cos(theta)) * np.cos(phi),
                np.sin(theta),
                (radius + np.cos(theta)) * np.sin(phi),
            ]
        )

    if R is not None:
        return bend(_validate_radius(R), phase=0.0)

    f, v = _validate_optimization_mesh(tube, f, v)
    height = float(np.ptp(z))
    R_min = float(np.sqrt(1.0 + (height / (2.0 * np.pi)) ** 2))
    # The finite upper bound avoids the unbent R -> infinity solution and is
    # consistent with the radius range used by the bending experiments.
    radius_bounds = _open_radius_bounds(R_min, max(10.0 * R_min, R_min + 1.0))

    def objective(params: tuple[float, float] | np.ndarray) -> float:
        radius, phase = params
        if not radius_bounds[0] <= radius <= radius_bounds[1]:
            return np.inf
        return _area_distortion_energy(v, f, bend(radius, phase))

    radius_init = minimize_scalar(
        lambda radius: objective((radius, 0.0)),
        method="bounded",
        bounds=radius_bounds,
        options={"xatol": 1e-4},
    ).x
    radius_opt, phase_opt = _optimize_radius_and_phase(
        objective,
        radius_init,
        radius_bounds,
        phase_bounds=(-np.pi, np.pi),
    )
    return bend(radius_opt, phase_opt)


def _optimize_radius_and_phase(
    objective,
    radius_init: float,
    radius_bounds: tuple[float, float],
    phase_bounds: tuple[float, float],
) -> tuple[float, float]:
    initial = np.array([radius_init, 0.0])
    initial_energy = objective(initial)
    opt = minimize(
        objective,
        x0=initial,
        method="Powell",
        bounds=(radius_bounds, phase_bounds),
        options={"xtol": 1e-4, "ftol": 1e-4, "maxiter": 120},
    )
    if np.isfinite(opt.fun) and opt.fun < initial_energy:
        return float(opt.x[0]), float(opt.x[1])
    return float(radius_init), 0.0


def _open_radius_bounds(lower: float, upper: float) -> tuple[float, float]:
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("No valid bending-radius interval exists for this tube")
    interval = upper - lower
    margin = min(max(1e-12, 1e-7 * interval), 0.25 * interval)
    return lower + margin, upper - margin


def _validate_radius(R: float) -> float:
    R = float(R)
    if not np.isfinite(R) or R <= 1.0:
        raise ValueError("R must be finite and greater than 1")
    return R


def _validate_tube(tube: np.ndarray) -> np.ndarray:
    tube = np.asarray(tube, dtype=float)
    if tube.ndim != 2 or tube.shape[1] != 3:
        raise ValueError("tube must have shape (n_vertices, 3)")
    if len(tube) == 0 or not np.all(np.isfinite(tube)):
        raise ValueError("tube must be non-empty and contain only finite values")
    return tube


def _validate_optimization_mesh(
    tube: np.ndarray, f: np.ndarray | None, v: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    if f is None or v is None:
        raise ValueError("f and v are required when R is optimized automatically")
    f = np.asarray(f, dtype=np.int64)
    v = np.asarray(v, dtype=float)
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
        raise ValueError("f must have shape (n_faces, 3) and be non-empty")
    if v.ndim != 2 or v.shape[1] not in (2, 3) or len(v) != len(tube):
        raise ValueError("v must have shape (n_vertices, 2 or 3) and match tube")
    if not np.all(np.isfinite(v)):
        raise ValueError("v must contain only finite values")
    if f.min() >= 1 and f.max() == len(tube):
        f = f - 1
    if f.min() < 0 or f.max() >= len(tube):
        raise ValueError("f contains an out-of-range vertex index")
    return f, v


def _area_distortion_energy(v: np.ndarray, f: np.ndarray, vmap: np.ndarray) -> float:
    distortion = area_distortion(v, f, vmap)
    if np.any(np.isinf(distortion)):
        return np.inf
    distortion = distortion[np.isfinite(distortion)]
    if len(distortion) == 0:
        return np.inf
    return float(np.mean(distortion**2))


def area_distortion(v: np.ndarray, f: np.ndarray, vmap: np.ndarray) -> np.ndarray:
    """Return normalized log-area distortion for every nondegenerate face."""
    area_v = face_area(v, f)
    area_map = face_area(vmap, f)
    scale = max(float(np.max(area_v)), 1.0)
    valid_reference = area_v > np.finfo(float).eps * scale
    distortion = np.full(len(f), np.nan, dtype=float)
    if not np.any(valid_reference):
        return distortion

    valid_map = valid_reference & (area_map > 0.0)
    distortion[valid_reference & ~valid_map] = np.inf
    total_v = np.sum(area_v[valid_reference])
    total_map = np.sum(area_map[valid_reference])
    if total_map <= 0.0:
        distortion[valid_reference] = np.inf
        return distortion

    distortion[valid_map] = np.log(
        (area_map[valid_map] / total_map) / (area_v[valid_map] / total_v)
    )
    return distortion


def face_area(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Compute the area of every triangular face."""
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=np.int64)
    if f.min() >= 1 and f.max() == len(v):
        f = f - 1
    edge1 = v[f[:, 1]] - v[f[:, 0]]
    edge2 = v[f[:, 2]] - v[f[:, 0]]
    if v.shape[1] == 2:
        return 0.5 * np.abs(edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])
    return 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)
