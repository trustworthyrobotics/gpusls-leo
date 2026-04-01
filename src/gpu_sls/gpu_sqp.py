from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.tree_util import register_pytree_node_class
from trajax.optimizers import linearize, quadratize, vectorize

from gpu_sls.external.primal_dual_ilqr.primal_dual_ilqr.optimizers import (
    merit_rho,
    model_evaluator_helper,
    slope,
)
from gpu_sls.gpu_admm import ADMMConfig, constrained_solve
from gpu_sls.gpu_sls import SLSConfig, sls_solve_gpu, get_tube_width


@partial(jit, static_argnums=(0, 1))
def line_search(
    merit_function,
    model_evaluator,
    X_in,
    U_in,
    V_in,
    dX,
    dU,
    dV,
    current_merit,
    current_g,
    current_c,
    merit_slope,
    armijo_factor,
    alpha_0,
    alpha_mult,
    alpha_min,
    Phi_x_prev, Phi_u_prev, r_center_prev,
):
    """Performs a primal-dual line search on an augmented Lagrangian merit function.

    Args:
      merit_function:  merit function mapping V, g, c, X, U to the merit scalar.
      X_in:            [T+1, n]      numpy array.
      U_in:            [T, m]        numpy array.
      V_in:            [T+1, n]      numpy array.
      dX:              [T+1, n]      numpy array.
      dU:              [T, m]        numpy array.
      dV:              [T+1, n]      numpy array.
      current_merit:   the merit function value at X, U, V.
      current_g:       the cost value at X, U, V.
      current_c:       the constraint values at X, U, V.
      merit_slope:     the directional derivative of the merit function.
      armijo_factor:   the Armijo parameter to be used in the line search.
      alpha_0:         initial line search value.
      alpha_mult:      a constant in (0, 1) that gets multiplied to alpha to update it.
      alpha_min:       minimum line search value.

    Returns:
      X: [T+1, n]     numpy array, representing the optimal state trajectory.
      U: [T, m]       numpy array, representing the optimal control trajectory.
      V: [T+1, n]     numpy array, representing the optimal multiplier trajectory.
      new_g:          the cost value at the new X, U, V.
      new_c:          the constraint values at the new X, U, V.
      no_errors:      whether no error occurred during the line search.
    """

    def continuation_criterion(inputs):
        _, _, _, _, _, new_merit, alpha = inputs
        return jnp.logical_and(
            new_merit > current_merit + alpha * armijo_factor * merit_slope,
            alpha > alpha_min,
        )

    def body(inputs):
        _, _, _, _, _, _, alpha = inputs
        alpha *= alpha_mult
        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in + alpha * dV
        new_g, new_c = model_evaluator(X_new, U_new)
        new_merit = merit_function(V_new, new_g, new_c, X_new, U_new, Phi_x_prev, Phi_u_prev, r_center_prev)
        new_merit = jnp.where(jnp.isnan(new_merit), current_merit, new_merit)
        return X_new, U_new, V_new, new_g, new_c, new_merit, alpha

    X, U, V, new_g, new_c, new_merit, alpha = lax.while_loop(
        continuation_criterion,
        body,
        (X_in, U_in, V_in, current_g, current_c, jnp.inf, alpha_0 / alpha_mult),
    )
    no_errors = alpha > alpha_min

    return X, U, V, new_g, new_c, no_errors


@register_pytree_node_class
@dataclass(frozen=True)
class SQPConfig:
    max_sqp_iterations: int = 1
    feas_tol: float = 1e-2
    step_tol: float = 1e-4
    warm_start: bool = True
    line_search: bool = True

    def tree_flatten(self):
        children = (self.max_sqp_iterations, self.feas_tol, self.step_tol, self.warm_start, self.line_search)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

# TODO: Add constraints to this?
def lagrangian(cost, dynamics, x0):
    def fun(x, u, t, v, v_prev):
        c1 = cost(x, u, t)
        c2 = jnp.dot(v, dynamics(x, u, t))
        c3 = jnp.dot(v_prev, lax.select(t == 0, x0 - x, -x))
        return c1 + c2 + c3

    return fun

@jax.jit
def add_obstacle_constraints(C: jnp.ndarray, D: jnp.ndarray, f: jnp.ndarray,
                             obstacles: jnp.ndarray, x_curr: jnp.ndarray, eps=1e-5):
    if obstacles.shape[0] == 0:
        return C, D, f

    Tp1, _, nx = C.shape
    _,  _, nu = D.shape

    centers = obstacles[:, :2]
    radii   = obstacles[:, 2]
    pos = x_curr[:, :2]
    diff = pos[:, None, :] - centers[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1) + eps
    n = diff / dist[..., None]
    coeffs = -n

    C_obstacle = jnp.zeros((Tp1, centers.shape[0], nx), dtype=C.dtype)
    D_obstacle = jnp.zeros((Tp1, centers.shape[0], nu), dtype=D.dtype)

    C_obstacle = C_obstacle.at[..., 0:2].set(coeffs)

    f_obstacle = (dist - radii[None, :]).astype(f.dtype)

    C_all = jnp.concatenate([C, C_obstacle], axis=1)
    D_all = jnp.concatenate([D, D_obstacle], axis=1)
    f_all = jnp.concatenate([f, f_obstacle], axis=1)
    
    return C_all, D_all, f_all

def merit_function_factory(rho_merit, lambda_rem, remainder_func, eps_rem, disturbance):
    @partial(jax.jit, static_argnames=("remainder_func", "disturbance"))
    def merit_fn(V, g, c, X, U, Phi_x_prev, Phi_u_prev, r_center_prev,
                 remainder_func=remainder_func, disturbance=disturbance):
        T = U.shape[0]
        nu = U.shape[1]
        t = jnp.arange(T + 1, dtype=X.dtype)[:, None]
        t_width = jnp.zeros_like(t)

        tube_center_shift_x = jnp.einsum("kjxn,jn->kx", Phi_x_prev, r_center_prev)
        tube_center_shift_u = jnp.einsum("kjun,jn->ku", Phi_u_prev, r_center_prev)
        tube_center_shift_u_pad = jnp.concatenate(
            [tube_center_shift_u, jnp.zeros((1, nu), dtype=tube_center_shift_u.dtype)],
            axis=0
        )
        tube_center_shift = jnp.concatenate(
            [tube_center_shift_x, tube_center_shift_u_pad, jnp.zeros_like(t_width)],
            axis=-1,
        )

        r = remainder_penalty(
            X, U, eps_rem, tube_center_shift,
            remainder_func, disturbance, Phi_x_prev, Phi_u_prev
        )
        return g + jnp.sum(V * c) + 0.5 * rho_merit * jnp.sum(c * c) + lambda_rem * r
        # return g + jnp.sum(V * c) + 0.5 * rho_merit * jnp.sum(c * c)
    return merit_fn

def get_tube_width(Phi_x, Phi_u, E):
    # Phi_x: [T+1, T+1, nx, nw]
    # Phi_u: [T,   T+1, nu, nw]
    # E:     [T+1, nw, ne]

    Phi_x_E = jnp.einsum("kjxn,jne->kjxe", Phi_x, E)   # [T+1, T+1, nx, ne]
    Phi_u_E = jnp.einsum("kjun,jne->kjue", Phi_u, E)   # [T,   T+1, nu, ne]

    x_width = jnp.linalg.norm(Phi_x_E, ord=1, axis=-1).sum(axis=1)
    u_width = jnp.linalg.norm(Phi_u_E, ord=1, axis=-1).sum(axis=1)

    return x_width, u_width

def get_tube_width_surrogate(Phi_x, Phi_u, E):
    # Phi_x: [T+1, T+1, nx, nw]
    # Phi_u: [T,   T+1, nu, nw]
    # E:     [T+1, nw, ne]

    Phi_x_E = jnp.einsum("kjxn,jne->kjxe", Phi_x, E)   # [T+1, T+1, nx, ne]
    Phi_u_E = jnp.einsum("kjun,jne->kjue", Phi_u, E)   # [T,   T+1, nu, ne]
    eps = 1e-6
    x_width = jnp.sqrt(jnp.sum(Phi_x_E**2, axis=-1) + eps).sum(axis=1)
    u_width = jnp.sqrt(jnp.sum(Phi_u_E**2, axis=-1) + eps).sum(axis=1)

    return x_width, u_width

@partial(jax.jit, static_argnames=("remainder_func", "disturbance"))
def remainder_penalty(X_, U_, eps_, tube_center_shift, remainder_func, disturbance, Phi_x, Phi_u):
    T = U_.shape[0]
    t = jnp.arange(T + 1, dtype=X_.dtype)[:, None]
    U_pad_ = jnp.pad(U_, ((0, 1), (0, 0)))
    z_nom_ = jnp.concatenate([X_, U_pad_, t], axis=1)
    z_lo_ = z_nom_ - eps_ + tube_center_shift
    z_up_ = z_nom_ + eps_ + tube_center_shift

    r_lower, r_upper = jax.vmap(remainder_func, in_axes=(0, 0))(z_lo_, z_up_)
    r_radius = 0.5 * (r_upper - r_lower)
    diag_r = jax.vmap(jnp.diag)(r_radius)
    E = disturbance(X_)
    E_combined = jnp.concatenate([E, diag_r], axis=2)
    x_width, u_width = get_tube_width_surrogate(Phi_x, Phi_u, E_combined)
    # return jnp.sum(E_combined)
    return jnp.sum(x_width) + jnp.sum(u_width)


@partial(jax.jit, static_argnames=("remainder_func", "disturbance"))
def get_remainder_gradients(X, U, Phi_x_prev, Phi_u_prev, E_prev, r_center_prev, remainder_func, disturbance):
    T = U.shape[0]
    nu = U.shape[1]
    t = jnp.arange(T + 1, dtype=X.dtype)[:, None]
    t_width = jnp.zeros_like(t)
    x_tube_widths, u_tube_widths = get_tube_width(Phi_x_prev, Phi_u_prev, E_prev)
    tube_center_shift_x = jnp.einsum("kjxn,jn->kx", Phi_x_prev, r_center_prev)
    tube_center_shift_u = jnp.einsum("kjun,jn->ku", Phi_u_prev, r_center_prev)

    u_width_pad = jnp.concatenate(
        [u_tube_widths, jnp.zeros((1, nu), dtype=u_tube_widths.dtype)],
        axis=0
    )
    tube_center_shift_u_pad = jnp.concatenate(
        [tube_center_shift_u, jnp.zeros((1, nu), dtype=tube_center_shift_u.dtype)],
        axis=0
    )
    z_width  = jnp.concatenate([x_tube_widths, u_width_pad, t_width], axis=-1)
    tube_center_shift = jnp.concatenate([tube_center_shift_x, tube_center_shift_u_pad, jnp.zeros_like(t_width)], axis=-1)
    _, (grad_X_rem, grad_U_rem) = jax.value_and_grad(
        remainder_penalty,
        argnums=(0, 1)
    )(X, U, z_width, tube_center_shift, remainder_func, disturbance, Phi_x_prev, Phi_u_prev)
    grad_X_rem = grad_X_rem.at[0].set(0.0)
    return grad_X_rem, grad_U_rem


@partial(jit, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
def compute_search_direction(
    sqp_config: SQPConfig, sls_config: SLSConfig, admm_config: ADMMConfig,
    cost, dynamics, hessian_approx,
    constraints, disturbance, remainder_func, splits_cfg,
    sqp_iteration,
    obstacles, disturbance_center,
    x0, X, U, V, c,
    w, y, rho,
    Q_bar, R_bar,
    h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws, E_prev, r_center_prev, Phi_x_prev, Phi_u_prev,
):
    T = U.shape[0]
    nx = X.shape[1]
    nu = U.shape[1]
    nc = w.shape[1]
    pad = lambda A: jnp.pad(A, [[0, 1], [0, 0]])

    if hessian_approx is None:
        quadratizer = quadratize(cost)
        Q, R_pad, M_pad = quadratizer(X, pad(U), jnp.arange(T + 1))
    else:
        Q, R_pad, M_pad = jax.vmap(hessian_approx)(X, pad(U), jnp.arange(T + 1))

    R = R_pad[:-1]
    M = M_pad[:-1]

    linearizer = linearize(lagrangian(cost, dynamics, x0), argnums=5)
    dynamics_linearizer = linearize(dynamics)
    q, r_pad = linearizer(X, pad(U), jnp.arange(T + 1), pad(V[1:]), V)
    r = r_pad[:-1]
    if sls_config.enable_linearization_gradients:
        grad_X_rem, grad_U_rem = get_remainder_gradients(X, U, Phi_x_prev, Phi_u_prev, E_prev, r_center_prev, remainder_func, disturbance)
        t = sqp_iteration / jnp.maximum(
            sls_config.max_initial_sqp_iterations + sqp_config.max_sqp_iterations - 1,
            1,
        )
        decay = jnp.maximum(0.01, (1.0 - t) ** 2)
        weight = sls_config.lambda_rem * decay
        # jax.debug.print("{}", grad_X_rem)
        q = q + weight * grad_X_rem
        r = r + weight * grad_U_rem

    # jax.debug.print("q: {}", q)


    A_pad, B_pad = dynamics_linearizer(X, pad(U), jnp.arange(T + 1))
    A = A_pad[:-1]
    B = B_pad[:-1]

    pad = lambda A: jnp.pad(A, ((0, 1), (0, 0)))
    U_pad = pad(U)
    t = jnp.arange(X.shape[0])
    g = vectorize(constraints)(X, U_pad, t)
    f = -g
    C, D = linearize(constraints)(X, U_pad, t)
    C_all, D_all, f_all = add_obstacle_constraints(C, D, f, obstacles, X)
    E = disturbance(X)

    n_obs = obstacles.shape[0]

    def run_nominal(_):
        dX, dU, dV, w1, y1, rho1, _, converged_admm = constrained_solve(
            admm_config, Q, q, R, r, M, A, B, c, C_all, D_all, f_all, w, y, rho
        )
        backoffs = jnp.zeros((T + 1, nc - n_obs))
        Phi_x   = jnp.zeros((T + 1, T + 1, nx, nx))
        Phi_u   = jnp.zeros((T, T + 1, nu, nx))
        betaN   = jnp.ones((T + 1, T + 1, nc - n_obs)) * 1e-10
        muN     = jnp.zeros((T + 1, nc))
        EN = jnp.zeros((T + 1, nx, nx + nx))
        r_centerN = jnp.zeros((T + 1, nx))
        return dX, dU, dV, w1, y1, rho1, backoffs, Phi_x, Phi_u, betaN, muN, EN, r_centerN

    def run_sls(_):
        dX, dU, dV, w1, y1, rho1, converged, converged_admm, backoffs, Phi_x, Phi_u, betaN, muN, EN, r_centerN = sls_solve_gpu(
            admm_config, remainder_func,
            Q, q, R, r, M, A, B, c,
            C_all, D_all, f_all, w, y, rho, sls_config, splits_cfg,
            E, E_prev, r_center_prev, Q_bar, R_bar, obstacles, X, h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws, X, U, disturbance_center
        )
        return dX, dU, dV, w1, y1, rho1, backoffs, Phi_x, Phi_u, betaN, muN, EN, r_centerN
    
    use_nominal = jnp.logical_or(
        jnp.logical_not(sls_config.enable_fastsls),
        jnp.logical_and(sls_config.initialize_nominal, sqp_iteration < sls_config.max_initial_sqp_iterations)
    )
    dX, dU, dV, w1, y1, rho1, backoffs, Phi_x, Phi_u, betaN, muN, EN, r_centerN = lax.cond(
        use_nominal, run_nominal, run_sls, operand=None
    )
    # jax.debug.print("{}", use_nominal)
    return dX, dU, dV, q, r, w1, y1, rho1, backoffs, Phi_x, Phi_u, betaN, muN, EN, r_centerN


@partial(jit, static_argnums=(0,1,2,3,4,5,6,7,8,9))
def sqp(
    sls_config: SLSConfig, sqp_config: SQPConfig, admm_config: ADMMConfig,
    cost, dynamics, hessian_approx,
    constraints, disturbance,
    remainder_func, splits_cfg,
    Q_bar, R_bar,
    reference, parameter,
    W, disturbance_center,
    x0, X_in, U_in, V_in,
    w, y, rho,
    obstacles,
    h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws, E_prev, r_center_prev
):
    _cost = partial(cost, W, reference)
    if hessian_approx is not None:
        _hessian_approx = partial(hessian_approx, W, reference)
    else:
        _hessian_approx = None

    _dynamics = partial(dynamics, parameter=parameter)
    model_evaluator = partial(model_evaluator_helper, _cost, _dynamics, x0)

    def body(i, carry):
        i, X_curr, U_curr, V_curr, w, y, rho, converged, backoffs, Phi_x, Phi_u, beta_ws, mu_w, E_prev, r_center_prev = carry

        def do_nothing(_):
            return carry

        def do_iter(_):
            g, c = model_evaluator(X_curr, U_curr)
            feas = jnp.max(jnp.abs(c))
            warm_flag = jnp.array(bool(sqp_config.warm_start))

            w0   = lax.select(warm_flag, w, jnp.zeros_like(w))
            y0   = lax.select(warm_flag, y, jnp.zeros_like(y))
            rho0 = lax.select(warm_flag, rho, jnp.asarray(admm_config.initial_rho, dtype=rho.dtype))
            h_ct_ws = backoffs
            dX, dU, dV, q, r, w1, y1, rho1, backoffs1, Phi_x1, Phi_u1, betaN, muN, EN, r_centerN = compute_search_direction(
                sqp_config, sls_config, admm_config,
                _cost, _dynamics, _hessian_approx,
                constraints, disturbance,
                remainder_func, splits_cfg,
                i,
                obstacles, disturbance_center,
                x0, X_curr, U_curr, V_curr, c,
                w0, y0, rho0,
                Q_bar, R_bar,
                h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws, E_prev, r_center_prev, Phi_x, Phi_u
            )

            step = jnp.maximum(
                jnp.max(jnp.abs(dX)),
                jnp.max(jnp.abs(dU))
            )
            z_norm = jnp.maximum(
                jnp.max(jnp.abs(X_curr)),
                jnp.max(jnp.abs(U_curr))
            )

            feas_ok = feas <= sqp_config.feas_tol
            step_ok = step <= sqp_config.step_tol * (1.0 + z_norm)
            jax.debug.print("SQP Iteration {} Feas {} (<= {}) Step {} (<= {})", i, feas, sqp_config.feas_tol, step, sqp_config.step_tol)
            converged1 = jnp.logical_and(feas_ok, step_ok)

            g, c = model_evaluator(X_curr, U_curr)

            rho_merit = merit_rho(c, dV)

            t_width = jnp.zeros((X_curr.shape[0], 1), dtype=X_curr.dtype)
            x_tube_widths, u_tube_widths = get_tube_width(Phi_x, Phi_u, E_prev)
            u_width_pad = jnp.concatenate(
                [u_tube_widths, jnp.zeros((1, U_curr.shape[1]), dtype=u_tube_widths.dtype)],
                axis=0,
            )
            z_width = jnp.concatenate([x_tube_widths, u_width_pad, t_width], axis=-1)

            merit_fn = merit_function_factory(
                rho_merit,
                sls_config.lambda_rem,
                remainder_func,
                z_width,
                disturbance
            )
            current_merit = merit_fn(V_curr, g, c, X_curr, U_curr, Phi_x1, Phi_u1, r_center_prev)
            merit_slope = slope(dX, dU, dV, c, q, r, rho_merit)
            last_iter = (i >= (sqp_config.max_sqp_iterations + sls_config.max_initial_sqp_iterations + 3))
            use_line_search = jnp.logical_and(
                jnp.logical_and(jnp.array(bool(sqp_config.line_search)), jnp.logical_not(last_iter)),
                jnp.logical_not(converged1),
            )

            def ls_branch(_):
                Xn, Un, Vn, g_new, c_new, ok = line_search(
                    merit_fn, model_evaluator,
                    X_curr, U_curr, V_curr,
                    dX, dU, dV,
                    current_merit, g, c,
                    merit_slope,
                    armijo_factor=1e-4,
                    alpha_0=1.0,
                    alpha_mult=0.5,
                    alpha_min=1e-6,
                    Phi_x_prev=Phi_x1, Phi_u_prev=Phi_u1, r_center_prev=r_center_prev,
                )
                return Xn, Un, Vn

            def fullstep_branch(_):
                return X_curr + dX, U_curr + dU, V_curr + dV

            # If converged1 is true, force alpha = 1 by skipping line search.
            X_next, U_next, V_next = lax.cond(
                converged1,
                fullstep_branch,
                lambda _: lax.cond(use_line_search, ls_branch, fullstep_branch, operand=None),
                operand=None,
            )

            w_next = w1
            y_next = y1
            # rho_next = jnp.where(
            #     rho1 > admm_config.initial_rho,
            #     jnp.maximum(rho1 * 0.9, admm_config.initial_rho),
            #     rho1
            # )
            rho_next = rho1
            y_next = rho1 / rho_next * y_next
            backoffs_next = backoffs1
            Phi_x_next = Phi_x1
            Phi_u_next = Phi_u1

            def fastsls_branch(_):
                return (
                    sls_config.max_initial_sqp_iterations,
                    jnp.array(False)
                )

            def no_fastsls_branch(_):
                return (i, converged1)

            i_new, converged1_new = lax.cond(
                jnp.logical_and(
                    jnp.logical_and(jnp.array(sls_config.enable_fastsls), converged1),
                    i < sls_config.max_initial_sqp_iterations
                ),
                fastsls_branch,
                no_fastsls_branch,
                operand=None
            )
            # w_next = lax.select(converged1, w, w1)
            # y_next = lax.select(converged1, y, y1)
            # rho_next = lax.select(converged1, rho, rho1)
            # rho_next = jnp.minimum(rho, admm_config.initial_rho)
            # rho_next = jnp.where(
            #     rho > admm_config.initial_rho,
            #     jnp.maximum(rho * 0.9, admm_config.initial_rho),
            #     rho
            # )
            # y_next = rho / rho_next * y_next
            # backoffs_next = lax.select(converged1, backoffs, backoffs1)
            # Phi_x_next = lax.select(converged1, Phi_x, Phi_x1)
            # Phi_u_next = lax.select(converged1, Phi_u, Phi_u1)

            return (i_new + 1, X_next, U_next, V_next, w_next, y_next, rho_next,
                    converged1_new,
                    backoffs_next, Phi_x_next, Phi_u_next, betaN, muN, EN, r_centerN)

        return lax.cond(converged, do_nothing, do_iter, operand=None)

    backoffs0 = h_ct_ws
    carry0 = (0, X_in, U_in, V_in, w, y, rho, jnp.array(False), backoffs0, Phi_x_ws, Phi_u_ws, beta_ws, mu_ws, E_prev, r_center_prev)
    total_iterations, X_out, U_out, V_out, w_out, y_out, rho_out, converged, backoffs, Phi_x, Phi_u, betaN, muN, EN, r_centerN = lax.fori_loop(
        0, sqp_config.max_sqp_iterations + sls_config.max_initial_sqp_iterations, body, carry0
    )
    return X_out, U_out, V_out, w_out, y_out, rho_out, backoffs, Phi_x, Phi_u, betaN, muN, EN, r_centerN
