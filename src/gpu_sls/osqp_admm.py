import numpy as np
import scipy.sparse as sp
import osqp
import jax
import jax.numpy as jnp
import heapq

OSQP_total_iterations = 0
OSQP_total_admm_iterations = 0


class RunningMedian:
    def __init__(self):
        self.low = []   # max heap via negatives
        self.high = []  # min heap

    def add(self, x):
        x = float(x)

        if not self.low or x <= -self.low[0]:
            heapq.heappush(self.low, -x)
        else:
            heapq.heappush(self.high, x)

        if len(self.low) > len(self.high) + 1:
            heapq.heappush(self.high, -heapq.heappop(self.low))
        elif len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def median(self):
        if not self.low and not self.high:
            raise ValueError("No elements yet")
        if len(self.low) > len(self.high):
            return -self.low[0]
        return 0.5 * (-self.low[0] + self.high[0])


OSQP_running_admm_median = RunningMedian()


def _np_list(x, name="array"):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [np.asarray(v, dtype=np.float64) for v in x]
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"{name} must be stage-stacked or a list.")
    return [np.asarray(arr[i], dtype=np.float64) for i in range(arr.shape[0])]


def _normalize_constraint_data(C, D, f, T, nx, nu):
    """
    Normalize constraint data to stage lists.

    Supported forms:
      - path only:
          C: length T
          D: length T
          f: length T
      - path + terminal:
          C: length T+1
          D: length T     or length T+1 with final block ignored/zero
          f: length T+1

    Returns:
      C_path: list length T, each (m_t, nx)
      D_path: list length T, each (m_t, nu)
      f_path: list length T, each (m_t,)
      C_term: None or array (m_T, nx)
      f_term: None or array (m_T,)
    """
    if C is None or D is None or f is None:
        return None, None, None, None, None

    C_list = _np_list(C, "C")
    D_list = _np_list(D, "D")
    f_list = _np_list(f, "f")

    if len(C_list) not in (T, T + 1):
        raise ValueError(f"C has length {len(C_list)}, expected T or T+1 with T={T}.")
    if len(f_list) not in (T, T + 1):
        raise ValueError(f"f has length {len(f_list)}, expected T or T+1 with T={T}.")
    if len(D_list) not in (T, T + 1):
        raise ValueError(f"D has length {len(D_list)}, expected T or T+1 with T={T}.")

    has_terminal = (len(C_list) == T + 1) and (len(f_list) == T + 1)

    C_path = []
    D_path = []
    f_path = []

    for t in range(T):
        Ct = np.asarray(C_list[t], dtype=np.float64)
        Dt = np.asarray(D_list[t], dtype=np.float64)
        ft = np.asarray(f_list[t], dtype=np.float64).reshape(-1)

        if Ct.ndim != 2 or Ct.shape[1] != nx:
            raise ValueError(f"C[{t}] has shape {Ct.shape}, expected (m, {nx}).")
        if Dt.ndim != 2 or Dt.shape[1] != nu:
            raise ValueError(f"D[{t}] has shape {Dt.shape}, expected (m, {nu}).")
        if Ct.shape[0] != Dt.shape[0] or Ct.shape[0] != ft.shape[0]:
            raise ValueError(
                f"Inconsistent constraint sizes at stage {t}: "
                f"C[{t}].shape={Ct.shape}, D[{t}].shape={Dt.shape}, f[{t}].shape={ft.shape}"
            )

        C_path.append(Ct)
        D_path.append(Dt)
        f_path.append(ft)

    C_term = None
    f_term = None
    if has_terminal:
        C_term = np.asarray(C_list[T], dtype=np.float64)
        f_term = np.asarray(f_list[T], dtype=np.float64).reshape(-1)
        if C_term.ndim != 2 or C_term.shape[1] != nx:
            raise ValueError(f"C[T] has shape {C_term.shape}, expected (m, {nx}).")
        if C_term.shape[0] != f_term.shape[0]:
            raise ValueError(
                f"Inconsistent terminal constraint sizes: "
                f"C[T].shape={C_term.shape}, f[T].shape={f_term.shape}"
            )

        if len(D_list) == T + 1:
            DT = np.asarray(D_list[T], dtype=np.float64)
            if DT.ndim != 2:
                raise ValueError(f"D[T] has shape {DT.shape}, expected 2D.")
            if DT.shape[0] != C_term.shape[0] or DT.shape[1] != nu:
                raise ValueError(
                    f"D[T] has shape {DT.shape}, expected ({C_term.shape[0]}, {nu})."
                )
            if np.linalg.norm(DT) > 1e-12:
                raise ValueError(
                    "Terminal D[T] is nonzero, but this formulation only supports "
                    "terminal constraints of the form C[T] x_T <= f[T]."
                )

    return C_path, D_path, f_path, C_term, f_term


def _build_qp(Q, q, R, r, M, A, B, c, C, D, f):
    """
    QP with:
      x0 = c[0]
      x_{t+1} = A[t] x_t + B[t] u_t + c[t+1]
      C[t] x_t + D[t] u_t <= f[t]
      optional terminal constraint C[T] x_T <= f[T]

    Cost:
      sum_{t=0}^{T-1} 0.5 x_t^T Q[t] x_t + q[t]^T x_t
                    + 0.5 u_t^T R[t] u_t + r[t]^T u_t
                    + x_t^T M[t] u_t
      + terminal 0.5 x_T^T Q[T] x_T + q[T]^T x_T

    Here M[t] is assumed to have shape (nx, nu).
    """
    T = len(R)
    nx = Q[0].shape[0]
    nu = R[0].shape[0]

    C_path, D_path, f_path, C_term, f_term = _normalize_constraint_data(C, D, f, T, nx, nu)

    nX = (T + 1) * nx
    nU = T * nu
    nz = nX + nU

    def xs(t):
        return slice(t * nx, (t + 1) * nx)

    def us(t):
        return slice(nX + t * nu, nX + (t + 1) * nu)

    P = sp.lil_matrix((nz, nz), dtype=np.float64)
    lin = np.zeros((nz,), dtype=np.float64)

    for t in range(T):
        Qt = np.asarray(Q[t], dtype=np.float64)
        qt = np.asarray(q[t], dtype=np.float64).reshape(nx)
        Rt = np.asarray(R[t], dtype=np.float64)
        rt = np.asarray(r[t], dtype=np.float64).reshape(nu)

        if M is None:
            Mt = np.zeros((nx, nu), dtype=np.float64)
        else:
            Mt = np.asarray(M[t], dtype=np.float64)
            if Mt.shape != (nx, nu):
                raise ValueError(f"M[{t}] has shape {Mt.shape}, expected ({nx}, {nu}).")

        P[xs(t), xs(t)] += Qt
        P[us(t), us(t)] += Rt
        P[xs(t), us(t)] += Mt
        P[us(t), xs(t)] += Mt.T

        lin[xs(t)] += qt
        lin[us(t)] += rt

    QT = np.asarray(Q[T], dtype=np.float64)
    qT = np.asarray(q[T], dtype=np.float64).reshape(nx)
    P[xs(T), xs(T)] += QT
    lin[xs(T)] += qT

    P = P.tocsc()

    meq = (T + 1) * nx

    mineq = 0
    if C_path is not None:
        mineq += sum(Ct.shape[0] for Ct in C_path)
        if C_term is not None:
            mineq += C_term.shape[0]

    m = meq + mineq
    Aall = sp.lil_matrix((m, nz), dtype=np.float64)
    l = np.full((m,), -np.inf, dtype=np.float64)
    u = np.full((m,), np.inf, dtype=np.float64)

    row = 0

    x_init = np.asarray(c[0], dtype=np.float64).reshape(nx)
    Aall[row:row + nx, xs(0)] = sp.eye(nx, format="lil")
    l[row:row + nx] = x_init
    u[row:row + nx] = x_init
    row += nx

    for t in range(T):
        At = np.asarray(A[t], dtype=np.float64)
        Bt = np.asarray(B[t], dtype=np.float64)
        ct = np.asarray(c[t + 1], dtype=np.float64).reshape(nx)

        Aall[row:row + nx, xs(t + 1)] = sp.eye(nx, format="lil")
        Aall[row:row + nx, xs(t)] = -At
        Aall[row:row + nx, us(t)] = -Bt
        l[row:row + nx] = ct
        u[row:row + nx] = ct
        row += nx

    if C_path is not None:
        for t in range(T):
            Ct = C_path[t]
            Dt = D_path[t]
            ft = f_path[t]
            mt = Ct.shape[0]

            Aall[row:row + mt, xs(t)] = Ct
            Aall[row:row + mt, us(t)] = Dt
            l[row:row + mt] = -np.inf
            u[row:row + mt] = ft
            row += mt

        if C_term is not None:
            mt = C_term.shape[0]
            Aall[row:row + mt, xs(T)] = C_term
            l[row:row + mt] = -np.inf
            u[row:row + mt] = f_term
            row += mt

    assert row == m

    return P, lin, Aall.tocsc(), l, u, meq, C_path, D_path, C_term


def _recover_costates(Q, q, M, A, X, U, y, meq, C_path, D_path, C_term):
    T = U.shape[0]
    nx = X.shape[1]
    V = np.zeros((T + 1, nx), dtype=np.float64)

    lam_ineq = y[meq:]
    offset = 0

    lam_path = []
    if C_path is not None:
        for Ct in C_path:
            mt = Ct.shape[0]
            lam_path.append(lam_ineq[offset:offset + mt])
            offset += mt
    else:
        lam_path = [None] * T

    lam_term = None
    if C_term is not None:
        mt = C_term.shape[0]
        lam_term = lam_ineq[offset:offset + mt]
        offset += mt

    QT = np.asarray(Q[T], dtype=np.float64)
    qT = np.asarray(q[T], dtype=np.float64).reshape(nx)
    V[T] = QT @ X[T] + qT
    if C_term is not None:
        V[T] += C_term.T @ lam_term

    for t in range(T - 1, -1, -1):
        Qt = np.asarray(Q[t], dtype=np.float64)
        qt = np.asarray(q[t], dtype=np.float64).reshape(nx)
        At = np.asarray(A[t], dtype=np.float64)

        if M is None:
            Mu = np.zeros((nx,), dtype=np.float64)
        else:
            Mt = np.asarray(M[t], dtype=np.float64)
            Mu = Mt @ U[t]

        V[t] = Qt @ X[t] + qt + Mu + At.T @ V[t + 1]

        if C_path is not None:
            V[t] += C_path[t].T @ lam_path[t]

    return V


def _solve_osqp_host(Q, q, R, r, M, A, B, c, C, D, f):
    global OSQP_total_iterations, OSQP_total_admm_iterations, OSQP_running_admm_median

    Q = _np_list(Q, "Q")
    q = _np_list(q, "q")
    R = _np_list(R, "R")
    r = _np_list(r, "r")
    M = None if M is None else _np_list(M, "M")
    A = _np_list(A, "A")
    B = _np_list(B, "B")
    c = _np_list(c, "c")

    T = len(R)
    nx = Q[0].shape[0]
    nu = R[0].shape[0]

    P, lin, Aall, l, u, meq, C_path, D_path, C_term = _build_qp(
        Q, q, R, r, M, A, B, c, C, D, f
    )

    solver = osqp.OSQP()
    solver.setup(
        P=P,
        q=lin,
        A=Aall,
        l=l,
        u=u,
        verbose=False,
        warm_start=True,
        eps_abs=1e-2,
        eps_rel=1e-2,
        max_iter=10000,
        polish=False,
        adaptive_rho=True,
    )

    res = solver.solve()

    OSQP_total_iterations += 1
    OSQP_total_admm_iterations += res.info.iter
    OSQP_running_admm_median.add(res.info.iter)

    current_avg = OSQP_total_admm_iterations / OSQP_total_iterations
    current_median = OSQP_running_admm_median.median()

    print(f"Current ADMM iteration count: {res.info.iter}")
    print(f"Running ADMM iteration count average: {current_avg}")
    print(f"Running ADMM iteration count median: {current_median}")

    z = np.asarray(res.x, dtype=np.float64)
    y = np.asarray(res.y, dtype=np.float64)

    nX = (T + 1) * nx

    def xs(t):
        return slice(t * nx, (t + 1) * nx)

    def us(t):
        return slice(nX + t * nu, nX + (t + 1) * nu)

    X = np.vstack([z[xs(t)] for t in range(T + 1)])
    U = np.vstack([z[us(t)] for t in range(T)]) if T > 0 else np.zeros((0, nu), dtype=np.float64)

    V = _recover_costates(Q, q, M, A, X, U, y, meq, C_path, D_path, C_term)

    return (
        X.astype(np.float32),
        U.astype(np.float32),
        V.astype(np.float32),
    )


def _constrained_solve_osqp(Q, q, R, r, M, A, B, c, C, D, f):
    T = R.shape[0]
    nx = Q.shape[-1]
    nu = R.shape[-1]

    result_shape = (
        jax.ShapeDtypeStruct((T + 1, nx), jnp.float32),
        jax.ShapeDtypeStruct((T, nu), jnp.float32),
        jax.ShapeDtypeStruct((T + 1, nx), jnp.float32),
    )

    def _callback(Q, q, R, r, M, A, B, c, C, D, f):
        return _solve_osqp_host(Q, q, R, r, M, A, B, c, C, D, f)

    return jax.pure_callback(
        _callback,
        result_shape,
        Q, q, R, r, M, A, B, c, C, D, f,
        vmap_method="sequential",
    )


def constrained_solve_osqp(Q, q, R, r, M, A, B, c, C, D, f):
    return _constrained_solve_osqp(Q, q, R, r, M, A, B, c, C, D, f)