#include "symnmf.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static double *parse_line_alloc(const char *line, size_t *out_m);
static double frobenius_sq_diff(double **A, double **B, size_t n, size_t m);

double **diag_pow_minus_half(const double *diag, size_t n);
double **mul_matrix(double **A, size_t ra, size_t ca, double **B, size_t rb, size_t cb);
double sq_euclidean(const double *a, const double *b, size_t dim);
double **gram_ht_h(double **H, size_t n, size_t k);

#define BETA 0.5

/* ============ Matrix utils ============ */
double **alloc_matrix(size_t rows, size_t cols) {
    size_t i, j;
    double **A = (double **)malloc(rows * sizeof(double *));
    if (A == NULL) return NULL;
    for (i = 0; i < rows; ++i) {
        A[i] = (double *)calloc(cols, sizeof(double));
        if (A[i] == NULL) {
            for (j = 0; j < i; ++j) free(A[j]);
            free(A);
            return NULL;
        }
    }
    return A;
}

void free_matrix(double **A, size_t rows) {
    size_t i;
    if (A == NULL) return;
    for (i = 0; i < rows; ++i) free(A[i]);
    free(A);
}

void zero_matrix(double **A, size_t rows, size_t cols) {
    size_t i, j;
    for (i = 0; i < rows; ++i)
        for (j = 0; j < cols; ++j)
            A[i][j] = 0.0;
}

void copy_matrix(double **dst, double **src, size_t rows, size_t cols) {
    size_t i, j;
    for (i = 0; i < rows; ++i)
        for (j = 0; j < cols; ++j)
            dst[i][j] = src[i][j];
}

/* ============ Dataset I/O (portable, no getline) ============ */

static double *parse_line_alloc(const char *line, size_t *out_m) {
    size_t cap = 8, m = 0;
    const char *p = line;
    char *endptr;
    double *row = (double *)malloc(cap * sizeof(double));
    if (row == NULL) return NULL;

    while (*p != '\0') {
        while (*p == ',' || *p == ' ' || *p == '\t' || *p == '\r') ++p;
        if (*p == '\0') break;
        endptr = NULL;
        {
            double val = strtod(p, &endptr);
            if (p == endptr) break; /* no conversion */
            if (m == cap) {
                size_t newcap = cap * 2;
                double *tmp = (double *)realloc(row, newcap * sizeof(double));
                if (tmp == NULL) { free(row); return NULL; }
                row = tmp; cap = newcap;
            }
            row[m++] = val;
            p = endptr;
        }
        while (*p == ',' || *p == ' ' || *p == '\t' || *p == '\r') ++p;
    }
    *out_m = m;
    return row;
}

static char* next_line_end(char *start){
    char *p = start;
    while (*p != '\n' && *p != '\0') ++p;
    return p;
}

static char* dup_line_segment(char *start, char *end){
    size_t len = (size_t)(end - start);
    char *copy = (char*)malloc(len + 1u);
    if (!copy) return NULL;
    memcpy(copy, start, len);
    copy[len] = '\0';
    return copy;
}

static void free_rows_upto(double **X, size_t row){
    size_t i;
    for (i = 0; i < row; ++i) free(X[i]);
    free(X);
}

static int ensure_cap(double ***pX, size_t *cap, size_t need){
    double **tmp;
    size_t newcap = *cap;
    if (need <= *cap) return 1;
    while (newcap < need) newcap <<= 1;
    tmp = (double**)realloc(*pX, newcap * sizeof(double*));
    if (!tmp) return 0;
    *pX = tmp; *cap = newcap; return 1;
}


static int build_dataset_from_buffer(Dataset *ds, char *buf) {
    size_t cap = 64u, row = 0u, dim = 0u, m;
    char *line_start = buf, *line_end, *line_copy;
    double *values;
    ds->X = (double**)malloc(cap * sizeof(double*));
    if (!ds->X) return 0;

    while (*line_start != '\0') {
        line_end = next_line_end(line_start);
        line_copy = dup_line_segment(line_start, line_end);
        if (!line_copy) { free_rows_upto(ds->X, row); return 0; }

        values = parse_line_alloc(line_copy, &m);
        free(line_copy);
        line_start = (*line_end == '\n') ? (line_end + 1) : line_end;

        if (!values || m == 0u) { if (values) free(values); continue; }
        if (row == 0u) dim = m;
        if (m != dim) { free(values); free_rows_upto(ds->X, row); return 0; }
        if (!ensure_cap(&ds->X, &cap, row + 1)) { free(values); free_rows_upto(ds->X, row); return 0; }

        ds->X[row++] = values;
    }
    ds->n = row; 
    ds->dim = dim;
    return 1;
}



Dataset *read_dataset_csv(const char *path) {
    FILE *fp;
    long sz;
    char *buf;
    Dataset *ds;
    size_t readn;

    fp = fopen(path, "r");
    if (fp == NULL) return NULL;

    if (fseek(fp, 0L, SEEK_END) != 0) { fclose(fp); return NULL; }
    sz = ftell(fp);
    if (sz < 0) { fclose(fp); return NULL; }
    if (fseek(fp, 0L, SEEK_SET) != 0) { fclose(fp); return NULL; }

    buf = (char *)malloc((size_t)sz + 1u);
    if (buf == NULL) { fclose(fp); return NULL; }
    if (sz > 0) {
        readn = fread(buf, 1u, (size_t)sz, fp);
        if (readn != (size_t)sz) { free(buf); fclose(fp); return NULL; }
    } else {
        readn = 0u;
    }
    buf[sz] = '\0';
    fclose(fp);

    ds = (Dataset *)calloc(1u, sizeof(Dataset));
    if (ds == NULL) { free(buf); return NULL; }

    if (!build_dataset_from_buffer(ds, buf)) {
        free(buf);
        free(ds);
        return NULL;
    }

    free(buf);
    return ds;
}


void free_dataset(Dataset *ds) {
    size_t i;
    if (ds == NULL) return;
    for (i = 0; i < ds->n; ++i) free(ds->X[i]);
    free(ds->X);
    free(ds);
}

/* ============ Graph & derived matrices ============ */

double **calc_sym(double **X, size_t n, size_t dim) {
    size_t i, j;
    double **A = alloc_matrix(n, n);
    if (A == NULL) return NULL;

    for (i = 0; i < n; ++i) {
        A[i][i] = 0.0;
        for (j = i + 1; j < n; ++j) {
            double d2  = sq_euclidean(X[i], X[j], dim);
            double val = exp(-0.5 * d2);
            A[i][j] = val;
            A[j][i] = val;
        }
    }
    return A;
}

double **calc_ddg(double **A, size_t n) {
    size_t i, j;
    double **D = alloc_matrix(n, n);
    if (D == NULL) return NULL;
    for (i = 0; i < n; ++i) {
        double s = 0.0;
        for (j = 0; j < n; ++j) s += A[i][j];
        D[i][i] = s;
    }
    return D;
}

double **calc_norm(double **A, double **D, size_t n) {
    size_t i;
    double **Dmh, **tmp, **W;
    double *deg = (double *)malloc(n * sizeof(double));
    if (deg == NULL) return NULL;
    for (i = 0; i < n; ++i) deg[i] = D[i][i];

    Dmh = diag_pow_minus_half(deg, n);
    free(deg);
    if (Dmh == NULL) return NULL;

    tmp = mul_matrix(Dmh, n, n, A, n, n);
    if (tmp == NULL) { free_matrix(Dmh, n); return NULL; }

    W = mul_matrix(tmp, n, n, Dmh, n, n);

    free_matrix(tmp, n);
    free_matrix(Dmh, n);

    return W;
}

/* ============ SymNMF ============ */
static void compute_updated_H(double **H, double **num, double **den,
                              double **Hnew, size_t n, size_t k) {
    size_t i, j;
    const double one_minus_beta = 1.0 - BETA;

    for (i = 0; i < n; ++i) {
        for (j = 0; j < k; ++j) {
            double hij   = H[i][j];
            double denij = den[i][j];
            double ratio = (denij > 0.0) ? (num[i][j] / denij) : 0.0;
            double mult  = one_minus_beta + BETA * ratio;
            double v     = hij * mult;

            if (v < 0.0 && v > -1e-16) v = 0.0;   
            Hnew[i][j] = (v >= 0.0) ? v : 0.0;    
        }
    }
}


double update_H_once(double **W, double **H, size_t n, size_t k) {
    double **num;
    double **G;
    double **den;
    double **Hnew;
    double delta_sq;

    num = mul_matrix(W, n, n, H, n, k);
    if (num == NULL) return -1.0;
    G = gram_ht_h(H, n, k);
    if (G == NULL) {
        free_matrix(num, n);
        return -1.0;
    }
    den = mul_matrix(H, n, k, G, k, k);
    if (den == NULL) {
        free_matrix(G, k);
        free_matrix(num, n);
        return -1.0;
    }

    Hnew = alloc_matrix(n, k);
    if (Hnew == NULL) {
        free_matrix(den, n);
        free_matrix(G, k);
        free_matrix(num, n);
        return -1.0;
    }

    compute_updated_H(H, num, den, Hnew, n, k);
    delta_sq = frobenius_sq_diff(Hnew, H, n, k);
    copy_matrix(H, Hnew, n, k);

    free_matrix(Hnew, n);
    free_matrix(den, n);
    free_matrix(G, k);
    free_matrix(num, n);
    return delta_sq;
}




int calc_symnmf(double **W, double **H, size_t n, size_t k,
                size_t max_iter, double eps) {
    size_t t;
    for (t = 0; t < max_iter; ++t) {
        double delta_sq = update_H_once(W, H, n, k);
        if (delta_sq < 0.0) return -1;            
        if (delta_sq < eps) return (int)(t + 1);  
    }
    return (int)max_iter;
}

/* ============ Helpers ============ */
double frobenius_norm_diff(double **A, double **B, size_t n, size_t m) {
    size_t i, j;
    double s = 0.0;
    for (i = 0; i < n; ++i)
        for (j = 0; j < m; ++j) {
            double d = A[i][j] - B[i][j];
            s += d * d;
        }
    return sqrt(s);
}

void print_matrix(double **A, size_t rows, size_t cols) {
    size_t i, j;
    for (i = 0; i < rows; ++i) {
        for (j = 0; j < cols; ++j) {
            if (j) printf(",");
            printf("%.4f", A[i][j]);
        }
        printf("\n");
    }
}

double **diag_pow_minus_half(const double *diag, size_t n) {
    size_t i;
    double **Dmh = alloc_matrix(n, n); /* zero-initialized */
    if (Dmh == NULL) return NULL;
    for (i = 0; i < n; ++i) {
        double di = diag[i];
        Dmh[i][i] = (di > 0.0) ? (1.0 / sqrt(di)) : 0.0;  /* if d_i=0 → 0 */
    }
    return Dmh;
}

double **mul_matrix(double **A, size_t ra, size_t ca, double **B, size_t rb, size_t cb) {
    size_t i, j, k;
    double **C;
    if (ca != rb) return NULL;
    C = alloc_matrix(ra, cb);
    if (C == NULL) return NULL;

    for (i = 0; i < ra; ++i) {
        for (k = 0; k < ca; ++k) {
            double aik = A[i][k];
            if (aik == 0.0) continue; 
            for (j = 0; j < cb; ++j) {
                C[i][j] += aik * B[k][j];
            }
        }
    }
    return C;
}

double sq_euclidean(const double *a, const double *b, size_t dim) {
    size_t t;
    double s = 0.0, d;
    for (t = 0; t < dim; ++t) {
        d = a[t] - b[t];
        s += d * d;
    }
    return s;
}

double **gram_ht_h(double **H, size_t n, size_t k) {
    size_t a, b, i;
    double **G = alloc_matrix(k, k);
    if (G == NULL) return NULL;
    for (a = 0; a < k; ++a) {
        for (b = a; b < k; ++b) {
            double s = 0.0;
            for (i = 0; i < n; ++i) s += H[i][a] * H[i][b];
            G[a][b] = s;
            G[b][a] = s;
        }
    }
    return G;
}

static double frobenius_sq_diff(double **A, double **B, size_t n, size_t m) {
    size_t i, j;
    double s = 0.0;
    for (i = 0; i < n; ++i)
        for (j = 0; j < m; ++j) {
            double d = A[i][j] - B[i][j];
            s += d * d;
        }
    return s;
}


#ifdef SYMNMF_CLI
#include <string.h> 

static void print_err(void) {
    printf("An Error Has Occurred\n");
}

static int run_sym(Dataset *ds){
    double **A = calc_sym(ds->X, ds->n, ds->dim);
    if (!A) return 0;
    print_matrix(A, ds->n, ds->n);
    free_matrix(A, ds->n);
    return 1;
}

static int run_ddg(Dataset *ds){
    double **A = calc_sym(ds->X, ds->n, ds->dim);
    double **D;
    if (!A) return 0;
    D = calc_ddg(A, ds->n);
    free_matrix(A, ds->n);
    if (!D) return 0;
    print_matrix(D, ds->n, ds->n);
    free_matrix(D, ds->n);
    return 1;
}

static int run_norm(Dataset *ds){
    double **A = calc_sym(ds->X, ds->n, ds->dim);
    double **D, **W;
    if (!A) return 0;
    D = calc_ddg(A, ds->n);
    if (!D){ free_matrix(A, ds->n); return 0; }
    W = calc_norm(A, D, ds->n);
    free_matrix(D, ds->n);
    free_matrix(A, ds->n);
    if (!W) return 0;
    print_matrix(W, ds->n, ds->n);
    free_matrix(W, ds->n);
    return 1;
}

static int dispatch_goal(const char *goal, Dataset *ds){
    if (strcmp(goal, "sym")  == 0) return run_sym(ds);
    if (strcmp(goal, "ddg")  == 0) return run_ddg(ds);
    if (strcmp(goal, "norm") == 0) return run_norm(ds);
    return 0; /* goal לא חוקי */
}


int main(int argc, char **argv) {
    const char *goal, *path;
    Dataset *ds;
    int ok;

    if (argc != 3) { print_err(); return 1; }
    goal = argv[1]; 
    path = argv[2];

    ds = read_dataset_csv(path);
    if (!ds) { print_err(); return 1; }

    ok = dispatch_goal(goal, ds);
    if (!ok) print_err();

    free_dataset(ds);
    return ok ? 0 : 1;
}

#endif /* SYMNMF_CLI */

