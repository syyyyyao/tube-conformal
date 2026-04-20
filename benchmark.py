import numpy as np
import trimesh
import time
import csv
from pathlib import Path
from src import initial_tube, tube_conformal_map, raw_extension, ring_smooth, conformal_bend_major, conformal_bend_minor


def main():
    print("Running benchmark for fixed boundary correction...")
    run_benchmark_fixed()
    print("Benchmark completed. Results saved to benchmark_results/ directory.\n")

    print("Running benchmark for smoothed weight...")
    run_benchmark_smoothed_weight()
    print("Benchmark completed. Results saved to benchmark_results/ directory.\n")

    print("Running benchmark for extension layers...")
    run_benchmark_extension_layers()
    print("Benchmark completed. Results saved to benchmark_results/ directory.\n")

    print("Running benchmark for major conformal bending...")
    run_benchmark_conformal_bend_major()
    print("Benchmark completed. Results saved to benchmark_results/ directory.\n")

    print("Running benchmark for minor conformal bending...")
    run_benchmark_conformal_bend_minor()
    print("Benchmark completed. Results saved to benchmark_results/ directory.\n")



def run_benchmark_fixed():
    out_dir = Path('benchmark_results')
    out_dir.mkdir(exist_ok=True)
    seam_strip_widths = [0.05, 0.25, 0.45, 0.65, 0.85, 1.0]
    header = ['name', 'init'] + [f'width={w}' for w in seam_strip_widths]

    for data_type in ['synthetic', 'real']:
        files = sorted(Path(f'data/{data_type}').rglob("*.obj"))
        results = []
        n = 0

        for filepath in files:
            n = n + 1
            print(f"Processing {filepath} ({n}/{len(files)})...")
            mesh = trimesh.load(filepath)
            v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)

            tube0 = initial_tube(v, f)
            dist_init = np.mean(np.abs(_angular_distortion(v, f, tube0)))

            row = [filepath.name, dist_init]
            for sw in seam_strip_widths:
                tube_fixed = tube_conformal_map(tube0, f, v, seam_strip_width=sw)
                row.append(np.mean(np.abs(_angular_distortion(v, f, tube_fixed))))

            results.append(row)

        with open(out_dir / f'{data_type}_fixed_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)


def run_benchmark_smoothed_weight():
    out_dir = Path('benchmark_results')
    out_dir.mkdir(exist_ok=True)
    smooth_weights = [0.05, 0.1, 0.25, 0.5]
    header = ['name', 'raw'] + [f'weight={w}' for w in smooth_weights]

    for data_type in ['synthetic', 'real']:
        files = sorted(Path(f'data/{data_type}').rglob("*.obj"))
        results = []
        n = 0

        for filepath in files:
            n = n + 1
            print(f"Processing {filepath} ({n}/{len(files)})...")
            mesh = trimesh.load(filepath)
            v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)

            v_ext_raw, f_ext = raw_extension(v, f, normal_blend=0.15)
            tube0_raw = initial_tube(v, f)
            tube_raw = tube_conformal_map(tube0_raw, f, v, seam_strip_width=0.05)
            dist_raw = np.mean(np.abs(_angular_distortion(v, f, tube_raw)))

            row = [filepath.name, dist_raw]
            for w in smooth_weights:
                v_smooth = ring_smooth(v_ext_raw, f_ext, smooth_weight=w)
                tube0_ext = initial_tube(v_smooth, f_ext)
                tube_ext = tube_conformal_map(tube0_ext, f_ext, v_smooth, seam_strip_width=0.05)
                tube_free = tube_ext[:len(v)]
                row.append(np.mean(np.abs(_angular_distortion(v, f, tube_free))))

            results.append(row)

        with open(out_dir / f'{data_type}_smoothed_weight_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)

    return None


def run_benchmark_extension_layers():
    out_dir = Path('benchmark_results')
    out_dir.mkdir(exist_ok=True)
    max_layers = 3
    header = ['name'] + [f'layers={m}' for m in range(1, max_layers+1)]

    for data_type in ['synthetic', 'real']:
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
                tube0_ext = initial_tube(v_new, f_new)
                tube_ext = tube_conformal_map(tube0_ext, f_new, v_new, seam_strip_width=0.05)
                tube_free = tube_ext[:len(v)]
                row.append(np.mean(np.abs(_angular_distortion(v, f, tube_free))))

            results.append(row)

        with open(out_dir / f'{data_type}_extension_layers_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)
    return None


def run_benchmark_conformal_bend_major():
    out_dir = Path('benchmark_results')
    out_dir.mkdir(exist_ok=True)
    major_ratios = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    header = ['name', 'tube'] + [f'ratio={a}' for a in major_ratios]

    for data_type in ['synthetic', 'real']:
        files = sorted(Path(f'data/{data_type}').rglob("*.obj"))
        results = []
        n = 0

        for filepath in files:
            n = n + 1
            print(f"Processing {filepath} ({n}/{len(files)})...")
            mesh = trimesh.load(filepath)
            v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)

            tube0_fixed = initial_tube(v, f)
            tube_fixed = tube_conformal_map(tube0_fixed, f, v, seam_strip_width=0.05)
            dist_tube = np.mean(np.abs(_angular_distortion(v, f, tube_fixed)))
            R_max = np.sqrt(1 + (2*np.pi / (np.max(tube_fixed[:,2]) - np.min(tube_fixed[:,2])))**2)

            row = [filepath.name, dist_tube]
            for a in major_ratios:
                R_major = a * R_max + 1-a
                bent_major = conformal_bend_major(tube_fixed, R_major)
                row.append(np.mean(np.abs(_angular_distortion(v, f, bent_major))))

            results.append(row)

        with open(out_dir / f'{data_type}_conformal_bend_major_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)

    return None


def run_benchmark_conformal_bend_minor():
    out_dir = Path('benchmark_results')
    out_dir.mkdir(exist_ok=True)
    minor_ratios = [1.1, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    header = ['name', 'tube'] + [f'ratio={a}' for a in minor_ratios]

    for data_type in ['synthetic', 'real']:
        files = sorted(Path(f'data/{data_type}').rglob("*.obj"))
        results = []
        n = 0

        for filepath in files:
            n = n + 1
            print(f"Processing {filepath} ({n}/{len(files)})...")
            mesh = trimesh.load(filepath)
            v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)

            tube0_fixed = initial_tube(v, f)
            tube_fixed = tube_conformal_map(tube0_fixed, f, v, seam_strip_width=0.05)
            dist_tube = np.mean(np.abs(_angular_distortion(v, f, tube_fixed)))
            R_min = np.sqrt(1 + ((np.max(tube_fixed[:,2]) - np.min(tube_fixed[:,2]))/(2*np.pi))**2)

            row = [filepath.name, dist_tube]
            for a in minor_ratios:
                R_minor = a * R_min
                bent_minor = conformal_bend_minor(tube_fixed, R_minor)
                row.append(np.mean(np.abs(_angular_distortion(v, f, bent_minor))))

            results.append(row)

        with open(out_dir / f'{data_type}_conformal_bend_minor_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)
            
    return None



def _angular_distortion(v: np.ndarray, f: np.ndarray, vmap: np.ndarray) -> np.ndarray:
    """
    Compute the angle distortion (in degree) of a mapping.
    
    Arguments:
    ----------
    v: nv x 3 or nv x 2 numpy array
       vertex coordinates
    f: nf x 3 numpy array
       triangular connectivity of mesh, 0-indexed
    vmap: nv x 3 or nv x 2 numpy array
          vertex coordinates of the mapping
    
    Returns:
    --------
    distortion: 3*nf x 1 numpy array
                angle distortion
    """
    
    f1 = f[:, 0]
    f2 = f[:, 1]
    f3 = f[:, 2]

    if np.size(v,1) == 2:
        v = np.hstack([v, np.zeros((len(v),1))])
        
    if np.size(vmap,1) == 2:
        vmap = np.hstack([vmap, np.zeros((len(vmap),1))])

    # calculate angles on v
    
    a3 = v[f1,:] - v[f3,:]
    b3 = v[f2,:] - v[f3,:]
    a1 = v[f2,:] - v[f1,:]
    b1 = v[f3,:] - v[f1,:]
    a2 = v[f3,:] - v[f2,:]
    b2 = v[f1,:] - v[f2,:]
    
    vcos1 = (a1[:,0]*b1[:,0]+a1[:,1]*b1[:,1]+a1[:,2]*b1[:,2])/np.sqrt(a1[:,0]**2+a1[:,1]**2+a1[:,2]**2)/np.sqrt(b1[:,0]**2+b1[:,1]**2+b1[:,2]**2)
    vcos2 = (a2[:,0]*b2[:,0]+a2[:,1]*b2[:,1]+a2[:,2]*b2[:,2])/np.sqrt(a2[:,0]**2+a2[:,1]**2+a2[:,2]**2)/np.sqrt(b2[:,0]**2+b2[:,1]**2+b2[:,2]**2)
    vcos3 = (a3[:,0]*b3[:,0]+a3[:,1]*b3[:,1]+a3[:,2]*b3[:,2])/np.sqrt(a3[:,0]**2+a3[:,1]**2+a3[:,2]**2)/np.sqrt(b3[:,0]**2+b3[:,1]**2+b3[:,2]**2)

    # calculate angles on vmap
    c3 = vmap[f1,:] - vmap[f3,:]
    d3 = vmap[f2,:] - vmap[f3,:]
    c1 = vmap[f2,:] - vmap[f1,:]
    d1 = vmap[f3,:] - vmap[f1,:]
    c2 = vmap[f3,:] - vmap[f2,:]
    d2 = vmap[f1,:] - vmap[f2,:]
    
    mapcos1 = (c1[:,0]*d1[:,0]+c1[:,1]*d1[:,1]+c1[:,2]*d1[:,2])/np.sqrt(c1[:,0]**2+c1[:,1]**2+c1[:,2]**2)/np.sqrt(d1[:,0]**2+d1[:,1]**2+d1[:,2]**2)
    mapcos2 = (c2[:,0]*d2[:,0]+c2[:,1]*d2[:,1]+c2[:,2]*d2[:,2])/np.sqrt(c2[:,0]**2+c2[:,1]**2+c2[:,2]**2)/np.sqrt(d2[:,0]**2+d2[:,1]**2+d2[:,2]**2)
    mapcos3 = (c3[:,0]*d3[:,0]+c3[:,1]*d3[:,1]+c3[:,2]*d3[:,2])/np.sqrt(c3[:,0]**2+c3[:,1]**2+c3[:,2]**2)/np.sqrt(d3[:,0]**2+d3[:,1]**2+d3[:,2]**2)

    # clamp to [-1, 1] to avoid arccos nan from floating point error
    vcos1 = np.clip(vcos1, -1, 1)
    vcos2 = np.clip(vcos2, -1, 1)
    vcos3 = np.clip(vcos3, -1, 1)
    mapcos1 = np.clip(mapcos1, -1, 1)
    mapcos2 = np.clip(mapcos2, -1, 1)
    mapcos3 = np.clip(mapcos3, -1, 1)

    # calculate the angle difference
    angular_distortion = np.hstack((np.arccos(mapcos1) - np.arccos(vcos1), np.arccos(mapcos2) - np.arccos(vcos2), np.arccos(mapcos3) - np.arccos(vcos3))) * 180 / np.pi

    return angular_distortion



if __name__ == "__main__":
    main()
