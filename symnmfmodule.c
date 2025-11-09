/* symnmfmodule.c — Python C-API bridge, module name MUST be "symnmfc" */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include "symnmf.h"

#define ERRMSG "An Error Has Occurred"

/* forward declarations so we can call these before their definitions */
static double **alloc_matrix_py(Py_ssize_t rows, Py_ssize_t cols);
static void     free_matrix_py(double **A, Py_ssize_t rows);


/* -------- helpers: Python <-> C matrices (size_t-safe) -------- */

/* Wrap PySequence_Fast with consistent error message. Returns 1 on success. */
static int seq_fast_or_err(PyObject *obj, PyObject **out) {
    *out = PySequence_Fast(obj, ERRMSG);
    return *out != NULL;
}

/* Infer number of columns (m) from the first row of an outer sequence. */
static int infer_cols_from_first_row(PyObject *outer, Py_ssize_t *m_out) {
    PyObject *row_seq = NULL;
    if (!outer) return 0;
    row_seq = PySequence_Fast(PySequence_Fast_GET_ITEM(outer, 0), ERRMSG);
    if (!row_seq) return 0;
    *m_out = PySequence_Fast_GET_SIZE(row_seq);
    Py_DECREF(row_seq);
    if (*m_out <= 0) { PyErr_SetString(PyExc_RuntimeError, ERRMSG); return 0; }
    return 1;
}

/* Fill a C row from a Python fast sequence of length m; returns 1 on success. */
static int fill_row_from_seq(PyObject *row_seq, double *row, Py_ssize_t m) {
    PyObject **items;
    Py_ssize_t j;
    if (PySequence_Fast_GET_SIZE(row_seq) != m) {
        PyErr_SetString(PyExc_RuntimeError, ERRMSG);
        return 0;
    }
    items = PySequence_Fast_ITEMS(row_seq);
    for (j = 0; j < m; ++j) {
        double v = PyFloat_AsDouble(items[j]); /* ints/floats/NumPy scalars */
        if (PyErr_Occurred()) { PyErr_SetString(PyExc_RuntimeError, ERRMSG); return 0; }
        row[j] = v;
    }
    return 1;
}

/* Allocate and copy list-of-lists into a newly allocated A (n×m). */
static int copy_outer_to_matrix(PyObject *outer, Py_ssize_t n, Py_ssize_t m, double ***A_out) {
    double **A = alloc_matrix_py(n, m);
    Py_ssize_t i;
    if (!A) { PyErr_SetString(PyExc_RuntimeError, ERRMSG); return 0; }
    for (i = 0; i < n; ++i) {
        PyObject *row_seq = PySequence_Fast(PySequence_Fast_GET_ITEM(outer, i), ERRMSG);
        if (!row_seq) { free_matrix_py(A, n); return 0; }
        if (!fill_row_from_seq(row_seq, A[i], m)) { Py_DECREF(row_seq); free_matrix_py(A, n); return 0; }
        Py_DECREF(row_seq);
    }
    *A_out = A;
    return 1;
}

/* Decide which matrix is W (n×n) and which is H (n×k). Returns 1 on success. */
static int choose_WH_from_two(double **M1, size_t r1, size_t c1,
                              double **M2, size_t r2, size_t c2,
                              double ***W, double ***H, size_t *n, size_t *k) {
    if (r1 == c1 && r1 == r2 && c2 > 0) {
        *W = M1; *H = M2; *n = r1; *k = c2; return 1;
    }
    if (r2 == c2 && r2 == r1 && c1 > 0) {
        *W = M2; *H = M1; *n = r2; *k = c1; return 1;
    }
    return 0;
}

static double **alloc_matrix_py(ssize_t rows, ssize_t cols) {
    ssize_t i, j;
    double **A = (double**)malloc((size_t)rows * sizeof(double*));
    if (!A) return NULL;
    for (i = 0; i < rows; ++i) {
        A[i] = (double*)calloc((size_t)cols, sizeof(double));
        if (!A[i]) {
            for (j = 0; j < i; ++j) free(A[j]);
            free(A);
            return NULL;
        }
    }
    return A;
}

static void free_matrix_py(double **A, ssize_t rows) {
    ssize_t i;
    if (!A) return;
    for (i = 0; i < rows; ++i) free(A[i]);
    free(A);
}

static int py_to_matrix(PyObject *obj, double ***A_out, size_t *rows_out, size_t *cols_out) {
    PyObject *outer = NULL;
    Py_ssize_t n, m;

    if (!seq_fast_or_err(obj, &outer)) return 0;

    n = PySequence_Fast_GET_SIZE(outer);
    if (n <= 0) { Py_DECREF(outer); PyErr_SetString(PyExc_RuntimeError, ERRMSG); return 0; }

    if (!infer_cols_from_first_row(outer, &m)) { Py_DECREF(outer); return 0; }

    if (!copy_outer_to_matrix(outer, n, m, A_out)) { Py_DECREF(outer); return 0; }

    Py_DECREF(outer);
    *rows_out = (size_t)n;
    *cols_out = (size_t)m;
    return 1;
}


static PyObject *matrix_to_py(double **A, size_t rows, size_t cols) {
    size_t i, j;
    PyObject *out = PyList_New((Py_ssize_t)rows);
    if (!out) return NULL;
    for (i = 0; i < rows; ++i) {
        PyObject *row = PyList_New((Py_ssize_t)cols);
        if (!row) { Py_DECREF(out); return NULL; }
        for (j = 0; j < cols; ++j) {
            PyObject *v = PyFloat_FromDouble(A[i][j]);
            if (!v) { Py_DECREF(row); Py_DECREF(out); return NULL; }
            PyList_SetItem(row, (Py_ssize_t)j, v); /* steals ref */
        }
        PyList_SetItem(out, (Py_ssize_t)i, row);   /* steals ref */
    }
    return out;
}

/* ================== exposed: all lowercase ================== */

/* sym(X) -> A (n×n) */
static PyObject *py_sym(PyObject *self, PyObject *args) {
    PyObject *x_obj = NULL;
    double **X = NULL, **A = NULL;
    size_t n = 0, d = 0;
    (void)self;

    if (!PyArg_ParseTuple(args, "O", &x_obj)) return NULL;
    if (!py_to_matrix(x_obj, &X, &n, &d)) { PyErr_SetString(PyExc_RuntimeError, ERRMSG); return NULL; }

    A = calc_sym(X, n, d);
    free_matrix_py(X, (ssize_t)n);
    if (!A) { PyErr_SetString(PyExc_RuntimeError, ERRMSG); return NULL; }

    PyObject *ret = matrix_to_py(A, n, n);
    free_matrix(A, n); 
    return ret;
}

/* ddg(X) -> D (n×n) */
static PyObject *py_ddg(PyObject *self, PyObject *args) {
    PyObject *x_obj = NULL;
    double **X = NULL, **A = NULL, **D = NULL;
    size_t n = 0, d = 0;
    (void)self;

    if (!PyArg_ParseTuple(args, "O", &x_obj)) return NULL;
    if (!py_to_matrix(x_obj, &X, &n, &d)) { PyErr_SetString(PyExc_RuntimeError, ERRMSG); return NULL; }

    A = calc_sym(X, n, d);
    free_matrix_py(X, (ssize_t)n);
    if (!A) { PyErr_SetString(PyExc_RuntimeError, ERRMSG); return NULL; }

    D = calc_ddg(A, n);
    free_matrix(A, n);
    if (!D) { PyErr_SetString(PyExc_RuntimeError, ERRMSG); return NULL; }

    PyObject *ret = matrix_to_py(D, n, n);
    free_matrix(D, n);
    return ret;
}

/* norm(X) -> W (n×n) */
static PyObject *py_norm(PyObject *self, PyObject *args) {
    PyObject *x_obj = NULL;
    double **X = NULL, **A = NULL, **D = NULL, **W = NULL;
    size_t n = 0, d = 0;
    (void)self;

    if (!PyArg_ParseTuple(args, "O", &x_obj)) return NULL;
    if (!py_to_matrix(x_obj, &X, &n, &d)) { PyErr_SetString(PyExc_RuntimeError, ERRMSG); return NULL; }

    A = calc_sym(X, n, d);
    free_matrix_py(X, (ssize_t)n);
    if (!A) { PyErr_SetString(PyExc_RuntimeError, ERRMSG); return NULL; }

    D = calc_ddg(A, n);
    if (!D) { free_matrix(A, n); PyErr_SetString(PyExc_RuntimeError, ERRMSG); return NULL; }

    W = calc_norm(A, D, n);
    free_matrix(D, n);
    free_matrix(A, n);
    if (!W) { PyErr_SetString(PyExc_RuntimeError, ERRMSG); return NULL; }

    PyObject *ret = matrix_to_py(W, n, n);
    free_matrix(W, n);
    return ret;
}

/* symnmf(A, B, [max_iter], [eps]) -> H (n×k)
   Accepts either order:
     - (W, H0, ...) where W is n×n and H0 is n×k
     - (H0, W, ...) where H0 is n×k and W is n×n
   max_iter defaults to 300, eps defaults to 1e-4.
*/
static PyObject *py_symnmf(PyObject *self, PyObject *args) {
    PyObject *obj1 = NULL, *obj2 = NULL;
    int max_iter = 300;
    double eps = 1e-4;
    double **M1 = NULL, **M2 = NULL, **W = NULL, **H = NULL;
    size_t r1=0, c1=0, r2=0, c2=0, n=0, k=0;
    PyObject *ret = NULL;
    (void)self;

    if (!PyArg_ParseTuple(args, "OO|id", &obj1, &obj2, &max_iter, &eps)) return NULL;

    if (!py_to_matrix(obj1, &M1, &r1, &c1)) { PyErr_SetString(PyExc_RuntimeError, ERRMSG); return NULL; }
    if (!py_to_matrix(obj2, &M2, &r2, &c2)) { free_matrix_py(M1, (Py_ssize_t)r1); PyErr_SetString(PyExc_RuntimeError, ERRMSG); return NULL; }

    if (!choose_WH_from_two(M1, r1, c1, M2, r2, c2, &W, &H, &n, &k)) {
        free_matrix_py(M1, (Py_ssize_t)r1); free_matrix_py(M2, (Py_ssize_t)r2);
        PyErr_SetString(PyExc_RuntimeError, ERRMSG); return NULL;
    }

    /* ownership moved; nothing left in M1/M2 to free */

    if (calc_symnmf(W, H, n, k, (size_t)max_iter, eps) < 0) {
        free_matrix(W, n); free_matrix(H, n);
        PyErr_SetString(PyExc_RuntimeError, ERRMSG); return NULL;
    }

    ret = matrix_to_py(H, n, k);
    free_matrix(W, n); free_matrix(H, n);
    return ret;
}



/* -------- method table & module init -------- */
static PyMethodDef Methods[] = {
    {"sym",    (PyCFunction)py_sym,   METH_VARARGS, "sym(X) -> A (n×n)"},
    {"ddg",    (PyCFunction)py_ddg,   METH_VARARGS, "ddg(X) -> D (n×n)"},
    {"norm",   (PyCFunction)py_norm,  METH_VARARGS, "norm(X) -> W (n×n)"},
    {"symnmf", (PyCFunction)py_symnmf,METH_VARARGS, "symnmf(W, H0, max_iter, eps) -> H (n×k)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfc_module = {
    PyModuleDef_HEAD_INIT,
    "symnmfc",
    NULL,               /* m_doc */
    -1,                 /* m_size */
    Methods,            /* m_methods */
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_symnmfc(void) {
    return PyModule_Create(&symnmfc_module);
}
