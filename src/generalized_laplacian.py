import numpy as np
from scipy import sparse


def generalized_laplacian(v: np.ndarray, f: np.ndarray, mu: np.ndarray) -> sparse.csr_matrix:
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=np.int64)
    mu = np.asarray(mu, dtype=complex)

    af = (1 - 2 * np.real(mu) + np.abs(mu) ** 2) / (1.0 - np.abs(mu) ** 2)
    bf = -2 * np.imag(mu) / (1.0 - np.abs(mu) ** 2)
    gf = (1 + 2 * np.real(mu) + np.abs(mu) ** 2) / (1.0 - np.abs(mu) ** 2)

    f0, f1, f2 = f[:, 0], f[:, 1], f[:, 2]

    uxv0 = v[f1, 1] - v[f2, 1]
    uyv0 = v[f2, 0] - v[f1, 0]
    uxv1 = v[f2, 1] - v[f0, 1]
    uyv1 = v[f0, 0] - v[f2, 0]
    uxv2 = v[f0, 1] - v[f1, 1]
    uyv2 = v[f1, 0] - v[f0, 0]

    l = np.column_stack(
        [
            np.sqrt(uxv0**2 + uyv0**2),
            np.sqrt(uxv1**2 + uyv1**2),
            np.sqrt(uxv2**2 + uyv2**2),
        ]
    )
    s = 0.5 * np.sum(l, axis=1)
    area = np.sqrt(np.maximum(s * (s - l[:, 0]) * (s - l[:, 1]) * (s - l[:, 2]), 0.0))
    area = np.where(area == 0, np.finfo(float).eps, area)

    v00 = (af * uxv0 * uxv0 + 2 * bf * uxv0 * uyv0 + gf * uyv0 * uyv0) / area
    v11 = (af * uxv1 * uxv1 + 2 * bf * uxv1 * uyv1 + gf * uyv1 * uyv1) / area
    v22 = (af * uxv2 * uxv2 + 2 * bf * uxv2 * uyv2 + gf * uyv2 * uyv2) / area
    v01 = (af * uxv1 * uxv0 + bf * uxv1 * uyv0 + bf * uxv0 * uyv1 + gf * uyv1 * uyv0) / area
    v12 = (af * uxv2 * uxv1 + bf * uxv2 * uyv1 + bf * uxv1 * uyv2 + gf * uyv2 * uyv1) / area
    v20 = (af * uxv0 * uxv2 + bf * uxv0 * uyv2 + bf * uxv2 * uyv0 + gf * uyv0 * uyv2) / area

    i = np.concatenate([f0, f1, f2, f0, f1, f1, f2, f2, f0])
    j = np.concatenate([f0, f1, f2, f1, f0, f2, f1, f0, f2])
    vals = np.concatenate([v00, v11, v22, v01, v01, v12, v12, v20, v20]) / 2.0

    return sparse.csr_matrix((-vals, (i, j)), shape=(len(v), len(v)))
