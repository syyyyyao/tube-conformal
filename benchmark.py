import csv
from pathlib import Path

import numpy as np
import trimesh

from src import (
    initial_tube,
    seam_correction,
    interior_refinement,
    tube_conformal_map,
    raw_extension,
    ring_smooth,
    conformal_bend_major,
    conformal_bend_minor,
    cut_path_finder,
)
from src.conformal_bend import area_distortion
from src.parallelogram_conformal_map import parallelogram_conformal_map
from src.slice_mesh import slice_mesh


def main():

    print("Running benchmark for correction width...")
    # run_benchmark_correction_width()
    print("Benchmark completed. Results saved to 'benchmark_results/parameter_results'.\n")

    print("Running benchmark for smoothed weight...")
    # run_benchmark_smoothed_weight()
    print("Benchmark completed. Results saved to 'benchmark_results/parameter_results'.\n")

    print("Running benchmark for extension layers...")
    # run_benchmark_extension_layers()
    print("Benchmark completed. Results saved to 'benchmark_results/parameter_results'.\n")

    print("Running benchmark for fixed boundary ablation...")
    # run_benchmark_ablation()
    print("Benchmark completed. Results saved to 'benchmark_results/ablation_results'.\n")

    print("Running benchmark for geometric fit...")
    run_benchmark_geometric_fit()
    print("Benchmark completed. Results saved to 'benchmark_results/geometric_fit_results'.\n")


def run_benchmark_correction_width():
    out_dir = Path('benchmark_results/parameter_results')
    out_dir.mkdir(exist_ok=True)
    seam_strip_widths = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    header = ['name']
    for w in seam_strip_widths:
        header.append(f'width={w}')

    for data_type in ['synthetic', 'real_single', 'real_multi']:
        files = sorted(Path(f'data/{data_type}').rglob("*.obj"))
        results = []
        n = 0

        for filepath in files:
            n = n + 1
            print(f"Processing {filepath} ({n}/{len(files)})...")
            mesh = trimesh.load(filepath)
            v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)

            tube0 = initial_tube(v, f)

            row = [filepath.name]
            for sw in seam_strip_widths:
                tube_seam = seam_correction(tube0, f, v, seam_strip_width=sw)
                tube_fixed = interior_refinement(tube_seam, f, v)
                row.append(_mean_abs_angular_distortion(v, f, tube_fixed))

            results.append(row)

        with open(out_dir / f'{data_type}_correction_width_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)

    return None


def run_benchmark_smoothed_weight():
    out_dir = Path('benchmark_results/parameter_results')
    out_dir.mkdir(exist_ok=True)
    smooth_weights = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    header = ['name', 'raw'] + [f'weight={w}' for w in smooth_weights]

    for data_type in ['synthetic', 'real_single', 'real_multi']:
        files = sorted(Path(f'data/{data_type}').rglob("*.obj"))
        results = []
        n = 0

        for filepath in files:
            n = n + 1
            print(f"Processing {filepath} ({n}/{len(files)})...")
            mesh = trimesh.load(filepath)
            v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)

            v_ext_raw, f_ext = raw_extension(v, f, normal_blend=0.15)
            tube_ext_raw = tube_conformal_map(v_ext_raw, f_ext, seam_strip_width=0.20)
            tube_raw = tube_ext_raw[:len(v)]
            dist_raw = _mean_abs_angular_distortion(v, f, tube_raw)

            row = [filepath.name, dist_raw]
            for w in smooth_weights:
                v_smooth = ring_smooth(v_ext_raw, f_ext, smooth_weight=w)
                tube_ext = tube_conformal_map(v_smooth, f_ext, seam_strip_width=0.20)
                tube_free = tube_ext[:len(v)]
                row.append(_mean_abs_angular_distortion(v, f, tube_free))

            results.append(row)

        with open(out_dir / f'{data_type}_smoothed_weight_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)

    return None


def run_benchmark_extension_layers():
    out_dir = Path('benchmark_results/parameter_results')
    out_dir.mkdir(exist_ok=True)
    max_layers = 3
    header = ['name'] + [f'layers={m}' for m in range(1, max_layers+1)]

    for data_type in ['synthetic', 'real_single', 'real_multi']:
        files = sorted(Path(f'data/{data_type}').rglob("*.obj"))
        results = []
        n = 0

        for filepath in files:
            n = n + 1
            print(f"Processing {filepath} ({n}/{len(files)})...")
            mesh = trimesh.load(filepath)
            v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)

            v_current, f_current = v.copy(), f.copy()
            row = [filepath.name]
            for m in range(max_layers):
                v_new_raw, f_new = raw_extension(v_current, f_current, normal_blend=0.15)
                v_new = ring_smooth(v_new_raw, f_new, smooth_weight=0.5)
                v_current, f_current = v_new, f_new
                tube_ext = tube_conformal_map(v_new, f_new, seam_strip_width=0.20)
                tube_free = tube_ext[:len(v)]
                row.append(_mean_abs_angular_distortion(v, f, tube_free))

            results.append(row)

        with open(out_dir / f'{data_type}_extension_layers_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)
    return None


def run_benchmark_ablation():
    out_dir = Path('benchmark_results/ablation_results')
    out_dir.mkdir(exist_ok=True)
    header = [
        'name',
        'fixed_initial_tube',
        'fixed_after_seam_correction',
        'fixed_after_interior_refinement',
        'free_tube',
    ]

    for data_type in ['synthetic', 'real_single', 'real_multi']:
        files = sorted(Path(f'data/{data_type}').rglob("*.obj"))
        results = []
        n = 0

        for filepath in files:
            n = n + 1
            print(f"Processing {filepath} ({n}/{len(files)})...")
            mesh = trimesh.load(filepath)
            v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)

            tube0 = initial_tube(v, f)
            tube_seam = seam_correction(tube0, f, v, seam_strip_width = 0.20)
            tube_full = interior_refinement(tube_seam, f, v)

            v_ext_raw, f_ext = raw_extension(v, f, normal_blend=0.15)
            v_ext = ring_smooth(v_ext_raw, f_ext, smooth_weight=0.50)
            tube_free_ext = tube_conformal_map(v_ext, f_ext, seam_strip_width=0.20)
            tube_free = tube_free_ext[:len(v)]

            row = [
                filepath.name,
                _mean_abs_angular_distortion(v, f, tube0),
                _mean_abs_angular_distortion(v, f, tube_seam),
                _mean_abs_angular_distortion(v, f, tube_full),
                _mean_abs_angular_distortion(v, f, tube_free),
            ]
            results.append(row)

        with open(out_dir / f'{data_type}_ablation_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)

    return None


def run_benchmark_geometric_fit():
    """Compare distortion across planar, tubular, and bent geometries."""
    out_dir = Path("benchmark_results/geometric_fit_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    geometry_names = ["parallelogram", "annulus", "tube", "major", "minor"]
    header = ["name"]
    for name in geometry_names:
        header.extend([f"{name}_angular", f"{name}_area"])

    for data_type in ["synthetic", "real_single", "real_multi"]:
        files = sorted(Path(f"data/{data_type}").rglob("*.obj"))
        results = []

        for index, filepath in enumerate(files, start=1):
            print(f"Processing {filepath} ({index}/{len(files)})...")
            mesh = trimesh.load(filepath)
            v = np.asarray(mesh.vertices)
            f = np.asarray(mesh.faces)

            cut_path = cut_path_finder(v, f)
            v_sliced, f_sliced = slice_mesh(v, f, cut_path)
            corner = np.array(
                [cut_path[0], cut_path[-1], len(v) + len(cut_path) - 1, len(v)],
                dtype=np.int64,
            )
            parallelogram = parallelogram_conformal_map(v_sliced, f_sliced, corner)

            para_original = parallelogram[: len(v)]
            initial = np.column_stack(
                [
                    np.cos(para_original[:, 1]),
                    np.sin(para_original[:, 1]),
                    para_original[:, 0],
                ]
            )
            corrected = seam_correction(initial, f, v, seam_strip_width=0.05)
            annulus = np.column_stack(
                [
                    np.exp(corrected[:, 2]) * corrected[:, 0],
                    np.exp(corrected[:, 2]) * corrected[:, 1],
                ]
            )
            tube = interior_refinement(corrected, f, v)
            major = conformal_bend_major(tube, f=f, v=v)
            minor = conformal_bend_minor(tube, f=f, v=v)

            geometries = [
                (v_sliced, f_sliced, parallelogram),
                (v, f, annulus),
                (v, f, tube),
                (v, f, major),
                (v, f, minor),
            ]
            row = [filepath.name]
            for reference, faces, mapped in geometries:
                row.extend(
                    [
                        _mean_abs_angular_distortion(reference, faces, mapped),
                        _mean_abs_area_distortion(reference, faces, mapped),
                    ]
                )
            results.append(row)

        output = out_dir / f"{data_type}_geometric_fit_results.csv"
        with open(output, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(results)


def _mean_abs_angular_distortion(
    v: np.ndarray,
    f: np.ndarray,
    vmap: np.ndarray,
    face_mask: np.ndarray | None = None,
) -> float:
    """Return the mean absolute corner-angle distortion in degrees."""
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=np.int64)
    vmap = np.asarray(vmap, dtype=float)

    def triangle_angles(vertices: np.ndarray) -> np.ndarray:
        triangles = vertices[f]
        edge_next = np.roll(triangles, -1, axis=1) - triangles
        edge_prev = np.roll(triangles, 1, axis=1) - triangles
        cosine = np.sum(edge_next * edge_prev, axis=2) / (
            np.linalg.norm(edge_next, axis=2)
            * np.linalg.norm(edge_prev, axis=2)
        )
        return np.arccos(np.clip(cosine, -1.0, 1.0))

    distortion = np.rad2deg(triangle_angles(vmap) - triangle_angles(v))
    if face_mask is not None:
        face_mask = np.asarray(face_mask, dtype=bool)
        if not np.any(face_mask):
            return float("nan")
        distortion = distortion[face_mask]

    return float(np.mean(np.abs(distortion)))


def _mean_abs_area_distortion(
    v: np.ndarray,
    f: np.ndarray,
    vmap: np.ndarray,
) -> float:
    """Return the mean absolute normalized log-area distortion."""
    distortion = area_distortion(v, f, vmap)
    distortion = distortion[np.isfinite(distortion)]
    if len(distortion) == 0:
        return float("nan")
    return float(np.mean(np.abs(distortion)))


if __name__ == "__main__":
    main()
