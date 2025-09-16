// hexfast.c
// A CPython extension that accelerates Hex evaluation and provides fast players.
//
// Public functions exposed to Python:
//   evaluate_hex(size: int, moves: list[tuple[int,int]]) -> (winner:str|None, idx:int|None)
//   random_choose_move(size: int, moves: list[tuple[int,int]], rng: object|None=None) -> (r,c)
//   one_ahead_choose_move(size: int, moves: list[tuple[int,int]], rng: object|None=None) -> (r,c)
//   two_ahead_choose_move(size: int, moves: list[tuple[int,int]], rng: object|None=None) -> (r,c)
//   monte_carlo_choose_move(size: int, moves: list[tuple[int,int]], sims: int=200, rng: object|None=None) -> (r,c)
//
// Build with setuptools (see pyproject.toml below).

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define BLACK 1
#define WHITE 2

static inline int idx_rc(int n, int r, int c) { return r * n + c; }

static const int drs[6] = {-1,-1, 0, 0, 1, 1};
static const int dcs[6] = { 0, 1,-1, 1,-1, 0};

typedef struct {
    int *parent;
    unsigned char *rank;
    int n;
} DSU;

static DSU dsu_new(int n) {
    DSU d;
    d.n = n;
    d.parent = (int*)malloc(sizeof(int)*n);
    d.rank = (unsigned char*)calloc(n, 1);
    for (int i=0;i<n;i++) d.parent[i]=i;
    return d;
}

static void dsu_free(DSU *d) {
    if (d->parent) free(d->parent);
    if (d->rank) free(d->rank);
    d->parent = NULL; d->rank = NULL; d->n = 0;
}

static int dsu_find(DSU *d, int x) {
    int p = d->parent[x];
    if (p != x) d->parent[x] = dsu_find(d, p);
    return d->parent[x];
}

static void dsu_union(DSU *d, int a, int b) {
    int ra = dsu_find(d, a);
    int rb = dsu_find(d, b);
    if (ra == rb) return;
    if (d->rank[ra] < d->rank[rb]) { int t=ra; ra=rb; rb=t; }
    d->parent[rb] = ra;
    if (d->rank[ra] == d->rank[rb]) d->rank[ra]++;
}

// simple RNG (xorshift64*)
static uint64_t RNG_STATE = 88172645463393265ULL;
static inline uint64_t xorshift64s(void) {
    uint64_t x = RNG_STATE;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    RNG_STATE = x;
    return x * 2685821657736338717ULL;
}
static inline int rand_index(int k) {
    if (k <= 0) return 0;
    // rejection to avoid modulo bias
    uint64_t limit = UINT64_MAX - (UINT64_MAX % (uint64_t)k);
    uint64_t r;
    do { r = xorshift64s(); } while (r > limit);
    return (int)(r % (uint64_t)k);
}
static void seed_rng(void) {
    RNG_STATE ^= (uint64_t)time(NULL) * 0x9E3779B97F4A7C15ULL;
}

// --- helpers to parse Python move list into a board ----

static int parse_moves_build_board(PyObject *moves, int n, unsigned char *board, char **err_msg) {
    // board: 0 empty, 1 black, 2 white
    Py_ssize_t m = PyList_Check(moves) ? PyList_Size(moves) :
                    (PyTuple_Check(moves) ? PyTuple_Size(moves) : -1);
    if (m < 0) { *err_msg = "moves must be a list or tuple"; return -1; }
    memset(board, 0, (size_t)(n*n));

    for (Py_ssize_t i=0;i<m;i++) {
        PyObject *item = PyList_Check(moves) ? PyList_GetItem(moves, i) : PyTuple_GetItem(moves, i);
        if (!PyTuple_Check(item) || PyTuple_Size(item) != 2) {
            *err_msg = "each move must be a (row, col) tuple";
            return -1;
        }
        PyObject *pr = PyTuple_GetItem(item, 0);
        PyObject *pc = PyTuple_GetItem(item, 1);
        long r = PyLong_AsLong(pr);
        long c = PyLong_AsLong(pc);
        if (PyErr_Occurred()) { *err_msg = "row/col must be integers"; return -1; }
        if (!(0 <= r && r < n && 0 <= c && c < n)) { *err_msg = "move out of bounds"; return -1; }
        int id = idx_rc(n, (int)r, (int)c);
        if (board[id] != 0) { *err_msg = "repeated move"; return -1; }
        board[id] = ((i % 2)==0) ? BLACK : WHITE;
    }
    return (int)m;
}

static inline int in_bounds(int n, int r, int c) {
    return (r>=0 && r<n && c>=0 && c<n);
}

// BFS test: does adding (r,c) for color create a connection NOW?
static int is_instant_win_after_place(unsigned char *board, int n, int color, int r, int c) {
    int cells = n*n;
    int id = idx_rc(n, r, c);
    if (board[id] != 0) return 0; // occupied can't place
    board[id] = (unsigned char)color;

    char *vis = (char*)calloc(cells, 1);
    int *q = (int*)malloc(sizeof(int)*cells);
    int qh=0, qt=0;

    int touchA = 0, touchB = 0; // edges
    // init from the placed stone
    q[qt++] = id; vis[id]=1;
    if (color==BLACK) {
        if (r==0) touchA=1;
        if (r==n-1) touchB=1;
    } else {
        if (c==0) touchA=1;
        if (c==n-1) touchB=1;
    }

    while (qh<qt && !(touchA && touchB)) {
        int cur = q[qh++];
        int rr = cur / n;
        int cc = cur % n;
        for (int k=0;k<6;k++) {
            int nr = rr + drs[k], nc = cc + dcs[k];
            if (!in_bounds(n, nr, nc)) continue;
            int nid = idx_rc(n, nr, nc);
            if (!vis[nid] && board[nid]==color) {
                vis[nid]=1; q[qt++]=nid;
                if (color==BLACK) {
                    if (nr==0) touchA=1;
                    if (nr==n-1) touchB=1;
                } else {
                    if (nc==0) touchA=1;
                    if (nc==n-1) touchB=1;
                }
            }
        }
    }

    board[id] = 0; // revert
    free(vis); free(q);
    return (touchA && touchB);
}

// Gather legal move indices into out[], returns count
static int gather_legal(const unsigned char *board, int n, int *out) {
    int cnt=0, cells=n*n;
    for (int i=0;i<cells;i++) if (board[i]==0) out[cnt++]=i;
    return cnt;
}

// Optionally use Python rng.randrange(k) if provided
static int choose_with_rng(PyObject *rng, int k, int *ok) {
    *ok = 0;
    if (!rng || rng == Py_None) return 0;
    if (!PyObject_HasAttrString(rng, "randrange")) return 0;
    PyObject *res = PyObject_CallMethod(rng, "randrange", "i", k);
    if (!res) { PyErr_Clear(); return 0; }
    long v = PyLong_AsLong(res);
    Py_DECREF(res);
    if (PyErr_Occurred()) { PyErr_Clear(); return 0; }
    if (v<0 || v>=k) return 0;
    *ok = 1; return (int)v;
}

// ------- evaluate_hex (Union–Find, virtual edges) ----------

static PyObject* py_evaluate_hex(PyObject *self, PyObject *args) {
    int n;
    PyObject *moves;
    if (!PyArg_ParseTuple(args, "iO", &n, &moves)) return NULL;
    if (n<=0) { PyErr_SetString(PyExc_ValueError, "size must be positive"); return NULL; }

    int cells = n*n;
    unsigned char *board = (unsigned char*)calloc(cells, 1);
    char *err = NULL;

    int m = parse_moves_build_board(moves, n, board, &err);
    if (m < 0) { free(board); PyErr_SetString(PyExc_ValueError, err); return NULL; }

    DSU dsuB = dsu_new(cells+2);
    DSU dsuW = dsu_new(cells+2);
    int TOP=cells, BOT=cells+1, LEFT=cells, RIGHT=cells+1;

    // incrementally rebuild DSU while scanning
    memset(dsuB.parent, 0, 0); // parent already init in dsu_new
    // (We’ll just union as we go)
    for (int i=0;i<m;i++) {
        // extract (r,c)
        PyObject *item = PyList_Check(moves) ? PyList_GetItem(moves, i) : PyTuple_GetItem(moves, i);
        long r = PyLong_AsLong(PyTuple_GetItem(item, 0));
        long c = PyLong_AsLong(PyTuple_GetItem(item, 1));
        int id = idx_rc(n, (int)r, (int)c);
        if ((i%2)==0) { // BLACK
            if (r==0) dsu_union(&dsuB, id, TOP);
            if (r==n-1) dsu_union(&dsuB, id, BOT);
            // union adjacent blacks
            for (int k=0;k<6;k++) {
                int nr=(int)r+drs[k], nc=(int)c+dcs[k];
                if (!in_bounds(n,nr,nc)) continue;
                int nid = idx_rc(n,nr,nc);
                if (board[nid]==BLACK) dsu_union(&dsuB, id, nid);
            }
            if (dsu_find(&dsuB, TOP) == dsu_find(&dsuB, BOT)) {
                dsu_free(&dsuB); dsu_free(&dsuW); free(board);
                PyObject *w = PyUnicode_FromString("Black");
                PyObject *movei = PyLong_FromLong(i+1);
                PyObject *t = PyTuple_Pack(2, w, movei);
                Py_DECREF(w); Py_DECREF(movei);
                return t;
            }
        } else { // WHITE
            if (c==0) dsu_union(&dsuW, id, LEFT);
            if (c==n-1) dsu_union(&dsuW, id, RIGHT);
            for (int k=0;k<6;k++) {
                int nr=(int)r+drs[k], nc=(int)c+dcs[k];
                if (!in_bounds(n,nr,nc)) continue;
                int nid = idx_rc(n,nr,nc);
                if (board[nid]==WHITE) dsu_union(&dsuW, id, nid);
            }
            if (dsu_find(&dsuW, LEFT) == dsu_find(&dsuW, RIGHT)) {
                dsu_free(&dsuB); dsu_free(&dsuW); free(board);
                PyObject *w = PyUnicode_FromString("White");
                PyObject *movei = PyLong_FromLong(i+1);
                PyObject *t = PyTuple_Pack(2, w, movei);
                Py_DECREF(w); Py_DECREF(movei);
                return t;
            }
        }
    }

    dsu_free(&dsuB); dsu_free(&dsuW); free(board);
    Py_INCREF(Py_None); Py_INCREF(Py_None);
    return PyTuple_Pack(2, Py_None, Py_None);
}

// ----- common: build board once, then implement players -----

static int get_legal_moves(unsigned char *board, int n, int *buf) {
    return gather_legal(board, n, buf);
}

static PyObject* choose_random_common(int n, PyObject *moves_obj, PyObject *rng_obj) {
    int cells = n*n;
    unsigned char *board = (unsigned char*)calloc(cells, 1);
    char *err=NULL;
    int m = parse_moves_build_board(moves_obj, n, board, &err);
    if (m<0) { free(board); PyErr_SetString(PyExc_ValueError, err); return NULL; }

    int *avail = (int*)malloc(sizeof(int)*cells);
    int k = get_legal_moves(board, n, avail);
    if (k<=0) { free(board); free(avail); PyErr_SetString(PyExc_ValueError, "no legal moves"); return NULL; }

    int ok=0, choice = choose_with_rng(rng_obj, k, &ok);
    if (!ok) choice = rand_index(k);

    int id = avail[choice]; int r=id/n, c=id % n;
    free(board); free(avail);
    return Py_BuildValue("(ii)", r, c);
}

static PyObject* py_random_choose_move(PyObject *self, PyObject *args) {
    int n; PyObject *moves_obj; PyObject *rng_obj = Py_None;
    if (!PyArg_ParseTuple(args, "iO|O", &n, &moves_obj, &rng_obj)) return NULL;
    if (n<=0) { PyErr_SetString(PyExc_ValueError, "size must be positive"); return NULL; }
    return choose_random_common(n, moves_obj, rng_obj);
}

// one_ahead: if any instant win exists, play the first found; else random
static PyObject* py_one_ahead_choose_move(PyObject *self, PyObject *args) {
    int n; PyObject *moves_obj; PyObject *rng_obj = Py_None;
    if (!PyArg_ParseTuple(args, "iO|O", &n, &moves_obj, &rng_obj)) return NULL;

    int cells = n*n;
    unsigned char *board = (unsigned char*)calloc(cells, 1);
    char *err=NULL;
    int m = parse_moves_build_board(moves_obj, n, board, &err);
    if (m<0) { free(board); PyErr_SetString(PyExc_ValueError, err); return NULL; }
    int my = (m % 2)==0 ? BLACK : WHITE;

    int *avail = (int*)malloc(sizeof(int)*cells);
    int k = get_legal_moves(board, n, avail);
    if (k<=0) { free(board); free(avail); PyErr_SetString(PyExc_ValueError, "no legal moves"); return NULL; }

    for (int i=0;i<k;i++) {
        int id = avail[i]; int r=id/n, c=id % n;
        if (is_instant_win_after_place(board, n, my, r, c)) {
            free(board); free(avail);
            return Py_BuildValue("(ii)", r, c);
        }
    }

    int ok=0, choice = choose_with_rng(rng_obj, k, &ok);
    if (!ok) choice = rand_index(k);
    int id = avail[choice]; int r=id/n, c=id % n;
    free(board); free(avail);
    return Py_BuildValue("(ii)", r, c);
}

// two_ahead: if you can win, do it; else if opponent has any instant-win RIGHT NOW, block one; else random
static PyObject* py_two_ahead_choose_move(PyObject *self, PyObject *args) {
    int n; PyObject *moves_obj; PyObject *rng_obj = Py_None;
    if (!PyArg_ParseTuple(args, "iO|O", &n, &moves_obj, &rng_obj)) return NULL;

    int cells = n*n;
    unsigned char *board = (unsigned char*)calloc(cells, 1);
    char *err=NULL;
    int m = parse_moves_build_board(moves_obj, n, board, &err);
    if (m<0) { free(board); PyErr_SetString(PyExc_ValueError, err); return NULL; }

    int my = (m % 2)==0 ? BLACK : WHITE;
    int opp = (my==BLACK) ? WHITE : BLACK;

    int *avail = (int*)malloc(sizeof(int)*cells);
    int k = get_legal_moves(board, n, avail);
    if (k<=0) { free(board); free(avail); PyErr_SetString(PyExc_ValueError, "no legal moves"); return NULL; }

    // 1) can we instantly win?
    for (int i=0;i<k;i++) {
        int id=avail[i], r=id/n, c=id % n;
        if (is_instant_win_after_place(board, n, my, r, c)) {
            free(board); free(avail);
            return Py_BuildValue("(ii)", r, c);
        }
    }

    // 2) does opponent have an instant-win now? If so, block one
    // i.e., find any empty cell that would be opp's instant win; play there.
    for (int i=0;i<k;i++) {
        int id=avail[i], r=id/n, c=id % n;
        if (is_instant_win_after_place(board, n, opp, r, c)) {
            free(board); free(avail);
            return Py_BuildValue("(ii)", r, c);
        }
    }

    // 3) random fallback
    int ok=0, choice = choose_with_rng(rng_obj, k, &ok);
    if (!ok) choice = rand_index(k);
    int id = avail[choice]; int r=id/n, c=id % n;
    free(board); free(avail);
    return Py_BuildValue("(ii)", r, c);
}

// BFS winner on full board (for playout scoring)
static int fullboard_winner(const unsigned char *board, int n) {
    int cells = n*n;
    // check BLACK via BFS from top edge
    char *vis = (char*)calloc(cells,1);
    int *q = (int*)malloc(sizeof(int)*cells);
    int qh=0, qt=0;
    for (int c=0;c<n;c++) {
        int id = idx_rc(n, 0, c);
        if (board[id]==BLACK) { vis[id]=1; q[qt++]=id; }
    }
    while (qh<qt) {
        int cur = q[qh++]; int r=cur/n, c=cur % n;
        if (r==n-1) { free(vis); free(q); return BLACK; }
        for (int k=0;k<6;k++) {
            int nr=r+drs[k], nc=c+dcs[k];
            if (!in_bounds(n,nr,nc)) continue;
            int nid = idx_rc(n,nr,nc);
            if (!vis[nid] && board[nid]==BLACK) { vis[nid]=1; q[qt++]=nid; }
        }
    }
    free(vis); free(q);
    return WHITE; // by Hex topology, exactly one winner exists
}

static void shuffle_ints(int *a, int n) {
    for (int i=n-1;i>0;i--) {
        int j = rand_index(i+1);
        int t=a[i]; a[i]=a[j]; a[j]=t;
    }
}

// uniform playout from (current moves + candidate) to end; return winner
static int playout_winner_after_move(unsigned char *board0, int n, int my_color, int move_id) {
    int cells = n*n;
    // copy board to tmp
    unsigned char *board = (unsigned char*)malloc(cells);
    memcpy(board, board0, cells);
    if (board[move_id]!=0) { free(board); return 0; }
    board[move_id] = (unsigned char)my_color;

    int *avail = (int*)malloc(sizeof(int)*cells);
    int k = 0;
    for (int i=0;i<cells;i++) if (board[i]==0) avail[k++]=i;
    shuffle_ints(avail, k);

    int cur_color = (my_color==BLACK) ? WHITE : BLACK;
    for (int i=0;i<k;i++) {
        board[avail[i]] = (unsigned char)cur_color;
        cur_color = (cur_color==BLACK) ? WHITE : BLACK;
    }
    int w = fullboard_winner(board, n);
    free(avail); free(board);
    return w;
}

static PyObject* py_monte_carlo_choose_move(PyObject *self, PyObject *args) {
    int n; PyObject *moves_obj; int sims=200; PyObject *rng_obj = Py_None;
    if (!PyArg_ParseTuple(args, "iO|iO", &n, &moves_obj, &sims, &rng_obj)) return NULL;
    if (sims<=0) sims=1;

    int cells = n*n;
    unsigned char *board = (unsigned char*)calloc(cells, 1);
    char *err=NULL;
    int m = parse_moves_build_board(moves_obj, n, board, &err);
    if (m<0) { free(board); PyErr_SetString(PyExc_ValueError, err); return NULL; }
    int my = (m % 2)==0 ? BLACK : WHITE;

    int *avail = (int*)malloc(sizeof(int)*cells);
    int k = get_legal_moves(board, n, avail);
    if (k<=0) { free(board); free(avail); PyErr_SetString(PyExc_ValueError, "no legal moves"); return NULL; }

    // quick win if possible
    for (int i=0;i<k;i++) {
        int id=avail[i], r=id/n, c=id % n;
        if (is_instant_win_after_place(board, n, my, r, c)) {
            free(board); free(avail);
            return Py_BuildValue("(ii)", r, c);
        }
    }

    // score each candidate by #wins in sims playouts
    int best_i = 0; int best_score = -1;
    for (int i=0;i<k;i++) {
        int id = avail[i];
        int wins = 0;
        for (int s=0;s<sims;s++) {
            int w = playout_winner_after_move(board, n, my, id);
            if (w == my) wins++;
        }
        if (wins > best_score) { best_score = wins; best_i = i; }
    }

    int id = avail[best_i]; int r=id/n, c=id % n;
    free(board); free(avail);
    return Py_BuildValue("(ii)", r, c);
}

// ---- module def ----

static PyMethodDef HexFastMethods[] = {
    {"evaluate_hex", (PyCFunction)py_evaluate_hex, METH_VARARGS, "Evaluate a Hex game; returns (winner, move_index)."},
    {"random_choose_move", (PyCFunction)py_random_choose_move, METH_VARARGS, "Random legal move."},
    {"one_ahead_choose_move", (PyCFunction)py_one_ahead_choose_move, METH_VARARGS, "Instant-win if any, else random."},
    {"two_ahead_choose_move", (PyCFunction)py_two_ahead_choose_move, METH_VARARGS, "Instant-win; else block opp instant-win; else random."},
    {"monte_carlo_choose_move", (PyCFunction)py_monte_carlo_choose_move, METH_VARARGS, "Monte Carlo playout chooser."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef hexfastmodule = {
    PyModuleDef_HEAD_INIT,
    "hexfast",
    "C accelerators for Hex (evaluator + simple players).",
    -1,
    HexFastMethods
};

PyMODINIT_FUNC PyInit_hexfast(void) {
    seed_rng();
    return PyModule_Create(&hexfastmodule);
}
