import numpy as np
from scipy import sparse


def beltrami_coefficient(v: np.ndarray, f: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=np.int64)
    target_coords = np.asarray(target_coords, dtype=float)

    nf = len(f)
    nv = len(v)

    mi = np.repeat(np.arange(nf), 3)
    mj = f.reshape(-1)

    e1 = v[f[:, 2], :2] - v[f[:, 1], :2]
    e2 = v[f[:, 0], :2] - v[f[:, 2], :2]
    e3 = v[f[:, 1], :2] - v[f[:, 0], :2]

    area = (-e2[:, 0] * e1[:, 1] + e1[:, 0] * e2[:, 1]) / 2.0
    area = np.where(area == 0, np.finfo(float).eps, area)

    mxt = np.vstack([e1[:, 1], e2[:, 1], e3[:, 1]]) / area
    myt = -np.vstack([e1[:, 0], e2[:, 0], e3[:, 0]]) / area
    mx = (mxt.T.reshape(-1)) / 2.0
    my = (myt.T.reshape(-1)) / 2.0

    dx = sparse.csr_matrix((mx, (mi, mj)), shape=(nf, nv))
    dy = sparse.csr_matrix((my, (mi, mj)), shape=(nf, nv))

    if target_coords.shape[1] == 3:
        d_x_du = dx @ target_coords[:, 0]
        d_x_dv = dy @ target_coords[:, 0]
        d_y_du = dx @ target_coords[:, 1]
        d_y_dv = dy @ target_coords[:, 1]
        d_z_du = dx @ target_coords[:, 2]
        d_z_dv = dy @ target_coords[:, 2]

        e = d_x_du**2 + d_y_du**2 + d_z_du**2
        g = d_x_dv**2 + d_y_dv**2 + d_z_dv**2
        f_term = d_x_du * d_x_dv + d_y_du * d_y_dv + d_z_du * d_z_dv
        den = e + g + 2 * np.sqrt(np.maximum(e * g - f_term**2, 0.0))
        den = np.where(den == 0, np.finfo(float).eps, den)
        return (e - g + 2j * f_term) / den

    z = target_coords[:, 0] + 1j * target_coords[:, 1]
    dz = (dx - 1j * dy) / 2.0
    dc = (dx + 1j * dy) / 2.0
    mu = (dc @ z) / (dz @ z)
    mu[~np.isfinite(mu)] = 1.0
    return mu
