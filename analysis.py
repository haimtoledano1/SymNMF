#!/usr/bin/env python3
"""
Analysis CLI for SymNMF vs. K-Means.

Usage:
    python3 analysis.py <k> <file.txt>

Prints EXACTLY two lines on success:
    nmf: <score or NaN>
    kmeans: <score or NaN>

Error policy:
    On any error/invalid input, prints exactly "An Error Has Occurred" and exits
    with a non-zero code.

Notes:
    • Random seed is fixed (1234) per spec.
    • Silhouette is computed with scikit-learn if available; otherwise a
      NumPy fallback is used. In this version, undefined silhouettes raise
      an exception that is caught at top-level (printing the standard error).
"""

import sys, re, math
from typing import List
import numpy as np
import symnmfc
try:
    from sklearn.metrics import silhouette_score as _sk_silhouette
except Exception:
    _sk_silhouette = None

np.random.seed(1234)  # per spec
MAX_ITER = 300
EPSILON  = 1e-4
ERR = "An Error Has Occurred"

# =========================== I/O ===========================

def read_points_from_file(path: str) -> np.ndarray:
    """
    Read a text file of points into a dense numpy array.

    :param path: Path to the input file.
    :return: Array of shape (N, d) with dtype float64.
    :raises SystemExit: Prints "An Error Has Occurred" and exits if parsing
                        fails, file is empty, or rows are ragged.
    """
    try:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = [p for p in re.split(r"[,\s]+", s) if p]
                rows.append([float(x) for x in parts])
        if not rows:
            raise ValueError("input file has no data rows")
        m = len(rows[0])
        if any(len(r) != m for r in rows):
            raise ValueError("ragged rows")
        return np.asarray(rows, dtype=float)
    except Exception:
        print(ERR, flush=True)
        sys.exit(1)

# ====================== Silhouette (fallback) ======================

def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Euclidean distance matrix.

    Uses the identity ||x - y||^2 = ||x||^2 + ||y||^2 − 2·x·y and then
    applies sqrt, clamping small negatives to 0 for numerical stability.

    :param X: Data matrix of shape (N, d).
    :return: Distance matrix of shape (N, N).
    """
    # D^2 = ||x||^2 + ||y||^2 - 2 x·y ; take sqrt to get Euclidean
    XX = np.sum(X * X, axis=1, keepdims=True)
    D2 = XX + XX.T - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2)

def _silhouette_np(X: np.ndarray, labels: np.ndarray) -> float:

    n = X.shape[0]
    uniq, counts = np.unique(labels, return_counts=True)
    if uniq.size < 2 or n < 2 or np.any(counts == 1):
        raise ValueError("silhouette undefined")

    D = _pairwise_distances(X)
    a = np.zeros(n, dtype=float)
    b = np.full(n, np.inf, dtype=float)

    for c in uniq:
        idx_c = np.where(labels == c)[0]
        Dc = D[np.ix_(idx_c, idx_c)]
        if idx_c.size <= 1:
            raise ValueError("singleton cluster")
        # mean intra-cluster distance (diagonal is 0)
        a[idx_c] = (np.sum(Dc, axis=1)) / (idx_c.size - 1)

    for c in uniq:
        idx_c = np.where(labels == c)[0]
        others = uniq[uniq != c]
        best = np.full(idx_c.size, np.inf, dtype=float)
        for c2 in others:
            idx_o = np.where(labels == c2)[0]
            Do = D[np.ix_(idx_c, idx_o)]
            mean_to_other = np.mean(Do, axis=1)
            best = np.minimum(best, mean_to_other)
        b[idx_c] = best

    denom = np.maximum(a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (b - a) / denom
    if not np.isfinite(s).all():
        raise ValueError("silhouette is NaN/Inf")
    val = float(np.mean(s))
    if not np.isfinite(val):
        raise ValueError("silhouette is NaN/Inf")
    return val

def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Robust silhouette wrapper.

    :param X: Data matrix (N, d).
    :param labels: Cluster labels (length N).
    :return: Mean silhouette in [-1, 1].
    :raises ValueError: If fewer than two clusters, a singleton cluster exists,
                        or the score is NaN/Inf.
    """
    # stricter than a silent NaN: raise and let the CLI print ERR
    if X.shape[0] < 2 or np.unique(labels).size < 2:
        raise ValueError("silhouette undefined")
    if _sk_silhouette is not None:
        val = float(_sk_silhouette(X, labels, metric="euclidean"))
        if not np.isfinite(val):
            raise ValueError("silhouette NaN")
        return val
    return _silhouette_np(X, labels)

# ======================== K-MEANS (HW1) ========================

def _dist(a: List[float], b: List[float]) -> float:
    """
    Euclidean distance between two flat Python lists.

    :param a: First point.
    :param b: Second point.
    :return: sqrt(sum((a_i - b_i)^2)).
    """
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def _assign(centroids, clusters, data, k):
    """
    Assign each point to its nearest centroid (in-place).

    :param centroids: Current centroids, length k.
    :param clusters: Buckets to fill; list of k lists (modified in-place).
    :param data: Input points as list of lists.
    :param k: Number of clusters.
    """
    for p in data:
        j = min(range(k), key=lambda t: _dist(p, centroids[t]))
        clusters[j].append(p)

def _centroid(cluster, old):
    """
    Compute the centroid of a cluster and its shift from the previous centroid.

    :param cluster: Points assigned to the cluster.
    :param old: Previous centroid (used if cluster is empty).
    :return: (new_centroid, L2_shift_from_old).
    """
    if not cluster:
        return old, 0.0
    d = len(old)
    m = len(cluster)
    new = [sum(pt[i] for pt in cluster) / m for i in range(d)]
    return new, _dist(new, old)

def kmeans_hw1(X: np.ndarray, k: int) -> np.ndarray:
    """
    Basic K-Means (HW1 style): first-k initialization, Euclidean distance.

    Iterates until max centroid shift < EPSILON or ``MAX_ITER`` is reached.

    :param X: Data matrix (N, d).
    :param k: Number of clusters (must satisfy 2 ≤ k ≤ N-1).
    :return: Integer labels of shape (N,).
    :raises ValueError: If k is outside the allowed range.
    """
    data = [list(r) for r in np.asarray(X, float)]
    n = len(data)
    if not (2 <= k <= max(2, n - 1)):
        raise ValueError("k must be in [2, n-1]")
    centroids = [data[i] for i in range(k)]
    clusters = [[] for _ in range(k)]
    _assign(centroids, clusters, data, k)

    it, max_shift = 0, float("inf")
    while max_shift >= EPSILON and it < MAX_ITER:
        it += 1
        max_shift = 0.0
        newc = []
        for i in range(k):
            c, shift = _centroid(clusters[i], centroids[i])
            newc.append(c)
            if shift > max_shift:
                max_shift = shift
        centroids = newc
        clusters = [[] for _ in range(k)]
        _assign(centroids, clusters, data, k)

    labels = [min(range(k), key=lambda j: _dist(p, centroids[j])) for p in data]
    return np.asarray(labels, dtype=int)

# =========================== SymNMF ===========================

def run_symnmf(k: int, X: np.ndarray) -> np.ndarray:
    """
    Run Symmetric NMF on the normalized similarity matrix W.

    Steps:
      1) Build W = norm(X) via the C-extension (Gaussian similarity + normalization).
      2) Compute m = mean(W) and initialize H0 ~ U(0, 2*sqrt(m/k)).
      3) Call the C-extension ``symnmf(W, H0, MAX_ITER, EPSILON)``.
      4) Return the final H (N×k).

    :param k: Number of components/clusters.
    :param X: Data matrix (N, d).
    :return: Nonnegative factor H of shape (N, k).
    """
    W = np.asarray(symnmfc.norm(X.tolist()), dtype=float)
    m = float(W.mean())
    upper = 2.0 * math.sqrt(m / float(k)) if k > 0 else 0.0
    H0 = np.random.uniform(0.0, upper, size=(X.shape[0], k))
    H = np.asarray(symnmfc.symnmf(W.tolist(), H0.tolist(), MAX_ITER, EPSILON), dtype=float)
    return H

# ============================ CLI ============================

def main():
    """
    CLI entry point.

    Reads K and the data file, validates input, runs SymNMF and K-Means,
    computes silhouette scores, and prints two formatted lines.

    :raises SystemExit: Always exits (success or error).
    """
    if len(sys.argv) != 3:
        print(ERR, flush=True); sys.exit(1)
    try:
        k = int(float(sys.argv[1]))
        X = read_points_from_file(sys.argv[2])

        n = X.shape[0]
        if not (2 <= k <= max(2, n - 1)):
            raise ValueError("bad k")

        # SymNMF
        H = run_symnmf(k, X)
        nmf_labels = np.argmax(H, axis=1)
        nmf_score  = safe_silhouette(X, nmf_labels)

        # KMeans
        km_labels = kmeans_hw1(X, k)
        km_score  = safe_silhouette(X, km_labels)

        print(f"nmf: {nmf_score:.4f}")
        print(f"kmeans: {km_score:.4f}")

    except Exception:
        print(ERR, flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
