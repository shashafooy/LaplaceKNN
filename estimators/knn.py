import numpy as np
import scipy.special as spl
import subprocess

# If a GPU exists, cuml version of KNN should be used
try:
    subprocess.check_output("nvidia-smi")
    from cuml.neighbors import NearestNeighbors
except Exception:
    from sklearn.neighbors import NearestNeighbors


dtype = np.float32


# Lebesque Measure of a Ball of radius r, dimension d, p norm
# lambda(B) = 2^d *gamma(1+1/p)^d / gamma(1+d/p) * r^d


def knn_KL(data, k=1, n=None, bias_correction=True):
    # KNN
    p_norm = 2
    N, dim = data.shape
    correction = np.log(k) - spl.digamma(k)

    if n is None:
        n = N

    (dist, _) = neighbor_distances(data, k, n, p_norm=p_norm)
    r = dist[:, k]

    # phat = (k/dim)/lebesque(Ball(r))
    logp = np.log(k) - np.log(N) - log_lebesque_ball(dim, r, p_norm=p_norm)

    if bias_correction:
        H = -np.mean(logp) + correction
    else:
        H = -np.mean(logp)
    return H


def knn_laplace(data, k=1, n=None, p_norm=2):
    N, dim = data.shape
    k_list = k if isinstance(k, list) else [k]
    if n is None:
        n = N

    (dist, _) = neighbor_distances(data, max(k_list), n, p_norm=p_norm)
    H = np.empty(len(k_list))
    for i, k in enumerate(k_list):
        r = dist[:, k]

        phi = np.log(n) + log_lebesque_ball(dim, r, p_norm=p_norm) - spl.digamma(k)

        H[i] = np.mean(phi)

    return H


def tkl(y, k=1):
    """Evaluate truncated knn KL and KSG algorithms
    Ziqiao Ao and Jinglai Li. “Entropy estimation via uniformization”

    Args:
        y (_type_): data points to find the knn distances of
        n (_type_, optional): number of samples. Defaults to None.
        k (int, optional): number of neighbors to find. Defaults to 1.
        shuffle (bool, optional): True to shuffle the data samples. Defaults to True.
        rng (_type_, optional): type of rng generator. Defaults to np.random.

    Returns:
        _type_: entropy estimate
    """
    N, dim = y.shape

    dist, idx = neighbor_distances(y, k, p_norm=np.inf)

    zeros_mask = dist[:, k] != 0

    r = dist[:, k]
    r = np.tile(r[:, np.newaxis], (1, dim))
    lb = (y - r >= 0) * (y - r) + (y - r < 0) * 0
    ub = (y + r <= 1) * (y + r) + (y + r > 1) * 1

    zeta = (ub - lb)[zeros_mask]  # remove zeros, duplicate points result in 0 distance
    N = zeta.shape[0]
    hh = np.log(np.prod(zeta, axis=1))

    h = -spl.digamma(k) + spl.digamma(N) + np.mean(hh)

    return h


def lebesque_ball(dim, r, p_norm=2):
    if p_norm == 2:
        return np.pi ** (dim / 2) / spl.gamma(1 + dim / 2) * r**dim
    else:
        return (2 * spl.gamma(1 + 1 / p_norm)) ** dim / spl.gamma(1 + dim / p_norm) * r**dim


def log_lebesque_ball(dim, r, p_norm=2):
    if p_norm == 2:
        return dim / 2 * np.log(np.pi) - np.log(spl.gamma(1 + dim / 2)) + dim * np.log(r)
    else:
        return (
            dim * np.log(2 * spl.gamma(1 + 1 / p_norm))
            - np.log(spl.gamma(1 + dim / p_norm))
            + dim * np.log(r)
        )


def neighbor_distances(y, k=1, n=None, shuffle=True, p_norm=2):
    rng = np.random
    y = np.asarray(y, float)
    N, dim = y.shape

    if p_norm == np.inf:
        metric = "chebyshev"  # p=infinite norm
    elif p_norm == 2:
        metric = "euclidean"  # p=2 norm
    else:
        raise ValueError("Invalid p_norm input. Expected {2,np.inf}")

    if n is not None:
        n = min(n, N)
        y = y[:n, :]

    # permute y
    if shuffle is True:
        rng.shuffle(y)

    # knn search

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric=metric, n_jobs=-1).fit(y)
    dist, idx = nbrs.kneighbors(y)

    return dist, idx
