#!/usr/bin/env python3
"""
SymNMF command-line wrapper.

Accepted usage forms:
  (A) python3 symnmf.py K goal file [max_iter] [eps]   # for 'symnmf'
  (B) python3 symnmf.py goal file                      # for 'sym'/'ddg'/'norm'

Goals:
  - sym    : Gaussian similarity matrix A (diag = 0)
  - ddg    : Degree matrix D (returned as a full N×N diagonal matrix)
  - norm   : Normalized similarity W = D^{-1/2} A D^{-1/2}
  - symnmf : Symmetric NMF on W with random non-negative initialization H0

Exit behavior:
  - On any parse/validation/runtime error, prints exactly "An Error Has Occurred"
    and exits with non-zero code.
  - On success, prints the resulting matrix with 4 decimal digits per entry.

Notes:
  - Random seed is fixed (1234) to make H0 reproducible.
  - Input parsing accepts both commas and whitespace as separators.

Tester hooks:
  - set_seed(seed:int) — called by the tester before H initialization.
  - init_H(W, k)       — H0 initializer: U(0, 2*sqrt(mean(W)/k)).
"""

import sys, re
import numpy as np
import symnmfc

np.random.seed(1234)
ERR = "An Error Has Occurred"


def read_points_file(path: str):
    """
    Read a points file into a dense NumPy array.

    :param path: Path to a text file containing N lines of d numeric values.
    :return: Array of shape (N, d) with dtype float64.
    :raises SystemExit: On I/O/parse failure or ragged/empty input; prints
                        "An Error Has Occurred" and exits.
    """
    try:
        rows = []
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = [p for p in re.split(r"[,\s]+", s) if p != ""]
                rows.append([float(x) for x in parts])
        if not rows:
            raise ValueError("empty")
        d = len(rows[0])
        if any(len(r) != d for r in rows):
            raise ValueError("ragged")
        return np.asarray(rows, dtype=float)
    except Exception:
        print(ERR, flush=True); sys.exit(1)


def print_matrix(M: np.ndarray) -> None:
    """
    Print a matrix as CSV with 4 fractional digits per entry.

    :param M: Matrix to print, shape (rows, cols).
    :raises SystemExit: If formatting/printing fails; prints
                        "An Error Has Occurred" and exits.
    """
    try:
        for row in M:
            print(",".join(f"{float(v):.4f}" for v in row), flush=True)
    except Exception:
        print(ERR, flush=True); sys.exit(1)


def set_seed(seed: int) -> None:
    """
    Set NumPy's RNG seed. The tester calls this right before H initialization.

    :param seed: Integer seed value.
    """
    np.random.seed(int(seed))


def init_H(W, k: int) -> np.ndarray:
    """
    Initialize H0 ~ U(0, 2*sqrt(m/k)), where m = mean(W).

    :param W: Normalized similarity matrix (N×N), NumPy array-like.
    :param k: Number of components/clusters (>0).
    :return: Initial H0 of shape (N, k), dtype float64.
    :raises ValueError: If k <= 0 or W is empty.
    """
    W = np.asarray(W, dtype=float)
    if k <= 0 or W.size == 0:
        raise ValueError("bad k/W")
    m = float(W.mean())
    upper = 2.0 * (m / float(k)) ** 0.5
    n = W.shape[0]
    return np.random.uniform(0.0, upper, size=(n, k))


# ============================ CLI helpers ============================

def _err_exit() -> "NoReturn":
    """Print the standard error message and exit non-zero."""
    print(ERR, flush=True)
    sys.exit(1)

def _parse_form_A(argv: list[str]) -> tuple[int, str, str]:
    """Parse form (A): K goal path  →  (k, goal, path)."""
    try:
        k = int(float(argv[0]))
    except Exception:
        _err_exit()
    goal = argv[1].lower()
    path = argv[2]
    return k, goal, path

def _parse_form_B(argv: list[str]) -> tuple[int | None, str, str]:
    """Parse form (B): goal path  →  (None, goal, path)."""
    goal = argv[0].lower()
    path = argv[1]
    return None, goal, path

def _parse_cli(argv_full: list[str]) -> tuple[int | None, str, str, int, float]:
    """
    Parse argv into (k, goal, path, max_iter, eps).
    Accepts both forms; optional max_iter/eps רק ל־symnmf.
    """
    if len(argv_full) < 3:
        _err_exit()

    argv = argv_full[1:]
    if len(argv) >= 3:
        k, goal, path = _parse_form_A(argv)
    elif len(argv) == 2:
        k, goal, path = _parse_form_B(argv)
    else:
        _err_exit()

    if goal not in {"sym", "ddg", "norm", "symnmf"}:
        _err_exit()

    max_iter, eps = 300, 1e-4
    if goal == "symnmf":
        if len(argv_full) >= 5:
            try:
                max_iter = int(argv_full[4])
            except Exception:
                _err_exit()
        if len(argv_full) >= 6:
            try:
                eps = float(argv_full[5])
            except Exception:
                _err_exit()

    return k, goal, path, max_iter, eps

def _dispatch_goal(goal: str, X: np.ndarray, k: int | None, max_iter: int, eps: float) -> np.ndarray:
    """
    Run the requested goal via the C-extension and return the resulting matrix.
    Raises on any runtime error; caller ידפיס ERR.
    """
    if goal == "sym":
        return np.asarray(symnmfc.sym(X.tolist()), dtype=float)

    if goal == "ddg":
        return np.asarray(symnmfc.ddg(X.tolist()), dtype=float)

    if goal == "norm":
        return np.asarray(symnmfc.norm(X.tolist()), dtype=float)

    # symnmf
    if k is None:
        _err_exit()
    W = np.asarray(symnmfc.norm(X.tolist()), dtype=float)
    set_seed(1234)
    H0 = init_H(W, k)
    return np.asarray(symnmfc.symnmf(W.tolist(), H0.tolist(), max_iter, eps), dtype=float)

# ============================== main ==============================

def main():
    """
    CLI entry point:
      * Parse args (both forms)
      * Load data
      * Validate and run goal via C-extension
      * Print matrix with 4 digits after the decimal
    """
    k, goal, path, max_iter, eps = _parse_cli(sys.argv)

    X = read_points_file(path)
    n = X.shape[0]
    if goal == "symnmf":
        if k is None or k < 2 or k >= n:
            _err_exit()

    try:
        M = _dispatch_goal(goal, X, k, max_iter, eps)
        print_matrix(M)
    except Exception:
        _err_exit()

if __name__ == "__main__":
    main()
