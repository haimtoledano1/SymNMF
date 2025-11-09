#ifndef SYMNMF_H
#define SYMNMF_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 *  SymNMF — Public API
 * ========================================================================= */

/** @brief Dense dataset container (N points in R^dim). */
typedef struct {
    size_t n;      /** Number of points (rows). */
    size_t dim;    /** Dimensionality (columns). */
    double **X;    /** Row-major matrix of size n×dim. */
} Dataset;

/* ============================== Matrix utils ============================== */

/**
 * @brief Allocate a rows×cols matrix and zero-initialize it.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return Newly allocated matrix (double**), or NULL on allocation failure.
 * @note Free with free_matrix(A, rows).
 */
double **alloc_matrix(size_t rows, size_t cols);

/**
 * @brief Free a matrix previously returned by alloc_matrix().
 * @param A    Matrix pointer (may be NULL; then no-op).
 * @param rows Number of rows originally allocated.
 */
void free_matrix(double **A, size_t rows);

/**
 * @brief Set all entries of an existing matrix to 0.0.
 * @param A    Matrix of size rows×cols.
 * @param rows Number of rows.
 * @param cols Number of columns.
 */
void zero_matrix(double **A, size_t rows, size_t cols);

/**
 * @brief Copy the contents of src into dst (same shape).
 * @param dst  Destination matrix (rows×cols).
 * @param src  Source matrix (rows×cols).
 * @param rows Number of rows.
 * @param cols Number of columns.
 */
void copy_matrix(double **dst, double **src, size_t rows, size_t cols);

/* ============================== Dataset I/O =============================== */

/**
 * @brief Parse a text file of points into a Dataset.
 *
 * Each non-empty line represents one point. Both comma (',') and whitespace
 * (spaces/tabs) separators are accepted; empty lines are skipped.
 * All non-empty lines must have the exact same number of coordinates.
 *
 * @param path Path to the input file.
 * @return Newly allocated Dataset*, or NULL on I/O/parse/shape failure.
 * @note Free with free_dataset().
 */
Dataset *read_dataset_csv(const char *path);

/**
 * @brief Free a Dataset allocated by read_dataset_csv().
 * @param ds Dataset pointer (may be NULL; then no-op).
 */
void free_dataset(Dataset *ds);

/* ===================== Graph & derived matrices (N×N) ===================== */

/**
 * @brief Build the Gaussian similarity matrix A from raw points X.
 *
 * A[i][j] = exp(-||x_i - x_j||^2 / 2), and A[i][i] = 0.
 *
 * @param X   Input points matrix (n×dim).
 * @param n   Number of points.
 * @param dim Dimensionality.
 * @return Newly allocated N×N similarity matrix A, or NULL on failure.
 */
double **calc_sym(double **X, size_t n, size_t dim);

/**
 * @brief Build the (full) diagonal degree matrix D from similarity A.
 *
 * D[i][i] = sum_j A[i][j]; off-diagonal entries are 0.
 *
 * @param A Input similarity matrix (N×N).
 * @param n Number of points (matrix size).
 * @return Newly allocated N×N diagonal matrix D, or NULL on failure.
 */
double **calc_ddg(double **A, size_t n);

/**
 * @brief Compute the normalized similarity matrix W = D^{-1/2} A D^{-1/2}.
 *
 * @param A Similarity matrix (N×N).
 * @param D Diagonal degree matrix as a full N×N matrix.
 * @param n Number of points.
 * @return Newly allocated N×N matrix W, or NULL on failure.
 * @note If D[i][i] == 0, the corresponding factor 1/sqrt(D[i][i]) is treated as 0.
 */
double **calc_norm(double **A, double **D, size_t n);

/* ================================ SymNMF ================================== */

/**
 * @brief Perform a single multiplicative-update step on H for SymNMF.
 *
 * Computes:
 *   H_new = H ∘ ((1-β) + β · (W·H) / (H·H^T·H)),
 * with β = 0.5, elementwise non-negativity, and returns ||H_new - H||_F^2.
 *
 * @param W N×N normalized similarity matrix.
 * @param H N×k nonnegative factor (updated in-place to H_new).
 * @param n Number of rows (points).
 * @param k Number of components (clusters).
 * @return Squared Frobenius difference ||H_new - H||_F^2; negative value on error.
 */
double update_H_once(double **W, double **H, size_t n, size_t k);

/**
 * @brief Run Symmetric NMF iterations until convergence or max_iter.
 *
 * Repeats update_H_once() up to max_iter times and stops early if
 *   dist < eps.
 *
 * @param W        N×N normalized similarity matrix.
 * @param H        N×k nonnegative initial factor; updated in-place to the final H.
 * @param n        Number of points.
 * @param k        Number of components.
 * @param max_iter Maximum number of iterations.
 * @param eps      Convergence threshold on squared Frobenius change.
 * @return Number of iterations performed (>=0) on success;
 *         -1 on internal failure/allocation error.
 */
int calc_symnmf(double **W, double **H, size_t n, size_t k,
                size_t max_iter, double eps);

/* ============================== Helpers / Ops ============================= */

/**
 * @brief Frobenius norm of the difference: ||A - B||_F.
 * @param A First matrix (n×m).
 * @param B Second matrix (n×m).
 * @param n Rows.
 * @param m Columns.
 * @return Frobenius norm (non-negative scalar).
 */
double frobenius_norm_diff(double **A, double **B, size_t n, size_t m);

/**
 * @brief Print a matrix as CSV with 4 decimal digits per entry.
 * @param A    Matrix of size rows×cols.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @note Each row ends with '\n'; entries are comma-separated using "%.4f".
 */
void print_matrix(double **A, size_t rows, size_t cols);

/**
 * @brief Build D^{-1/2} as a full N×N diagonal matrix from a degree vector.
 * @param diag Vector of length N with strictly non-negative entries.
 * @param n    Length of diag / size of the output.
 * @return Newly allocated N×N diagonal matrix; entry i is 1/sqrt(diag[i]) (or 0 if diag[i]==0).
 */
double **diag_pow_minus_half(const double *diag, size_t n);

/**
 * @brief Multiply two dense matrices: C = A·B.
 * @param A   Left matrix of size ra×ca.
 * @param ra  Rows of A.
 * @param ca  Columns of A (must equal rb).
 * @param B   Right matrix of size rb×cb.
 * @param rb  Rows of B.
 * @param cb  Columns of B.
 * @return Newly allocated ra×cb matrix C, or NULL if ca!=rb or on allocation failure.
 */
double **mul_matrix(double **A, size_t ra, size_t ca,
                    double **B, size_t rb, size_t cb);

/**
 * @brief Squared Euclidean distance ||a - b||^2.
 * @param a   First vector (length dim).
 * @param b   Second vector (length dim).
 * @param dim Vector length.
 * @return Sum_t (a[t] - b[t])^2 (non-negative).
 */
double sq_euclidean(const double *a, const double *b, size_t dim);

/**
 * @brief Gram matrix G = H^T·H (k×k), symmetric.
 * @param H N×k matrix.
 * @param n Number of rows of H.
 * @param k Number of columns of H.
 * @return Newly allocated k×k Gram matrix, or NULL on failure.
 */
double **gram_ht_h(double **H, size_t n, size_t k);

#ifdef __cplusplus
}
#endif

#endif /* SYMNMF_H */
