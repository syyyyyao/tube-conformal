import numpy as np
from scipy import sparse


def cotangent_laplacian(v: np.ndarray, f: np.ndarray) -> sparse.csr_matrix:
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=np.int64)
    nv = len(v)

    f1, f2, f3 = f[:, 0], f[:, 1], f[:, 2]

    l1 = np.linalg.norm(v[f2] - v[f3], axis=1)
    l2 = np.linalg.norm(v[f3] - v[f1], axis=1)
    l3 = np.linalg.norm(v[f1] - v[f2], axis=1)

    s = 0.5 * (l1 + l2 + l3)
    area = np.sqrt(np.maximum(s * (s - l1) * (s - l2) * (s - l3), 0.0))
    area = np.where(area == 0, np.finfo(float).eps, area)

    cot12 = (l1**2 + l2**2 - l3**2) / area / 2.0
    cot23 = (l2**2 + l3**2 - l1**2) / area / 2.0
    cot31 = (l1**2 + l3**2 - l2**2) / area / 2.0

    diag1 = -cot12 - cot31
    diag2 = -cot12 - cot23
    diag3 = -cot31 - cot23

    ii = np.concatenate([f1, f2, f2, f3, f3, f1, f1, f2, f3])
    jj = np.concatenate([f2, f1, f3, f2, f1, f3, f1, f2, f3])
    vv = np.concatenate([cot12, cot12, cot23, cot23, cot31, cot31, diag1, diag2, diag3])

    return sparse.csr_matrix((vv, (ii, jj)), shape=(nv, nv))
