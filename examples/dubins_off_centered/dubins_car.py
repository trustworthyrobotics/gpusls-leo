from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable


import jax
import jax.numpy as jnp
from jax import config

import numpy as np

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.utils.constraint_utils import combine_constraints
from gpu_sls.utils.sls_visual import get_trajectory_tubes
from visualize_experiment import plot_rollouts_tubes_centers, plot_tube_graph

config.update("jax_enable_x64", False)
config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
config.update("jax_persistent_cache_min_compile_time_secs", 0)
config.update("jax_persistent_cache_min_entry_size_bytes", -1)
config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)

# -----------------------------
# Goal stopping config
# -----------------------------
GOAL_TOL = 0.2  # meters (XY distance)
DISTURBANCE_OFFSET = 0.0005

def reached_goal_xy(x: jnp.ndarray, x_goal: jnp.ndarray, tol: float = GOAL_TOL) -> jnp.bool_:
    dxy = x[:2] - x_goal[:2]
    return (dxy @ dxy) <= (tol * tol)


# -----------------------------
# Dubins car dynamics
# x = [px, py, theta], u = [omega]
# -----------------------------
V_CONST = 0.2
NUM_RANDOM = 5
NUM_ADV = 26

def dubins_step_with_disturbance(
    key: jax.Array,          # PRNGKey
    x: jnp.ndarray,          # (3,)
    u: jnp.ndarray,          # (1,)
    E: jnp.ndarray,          # (3,3)
    dt: float,
    i: int
) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    """
    Simulates: x_{k+1} = f(x_k,u_k) + E w,   with ||w||_2 <= 1
    where w is sampled from a unit-ball-ish distribution (plus some deterministic cases).

    Returns (key_next, x_next, w).
    """
    px, py, th = x
    om = u[0]

    # Nominal Dubins step
    px_next = px + dt * V_CONST * jnp.cos(th)
    py_next = py + dt * V_CONST * jnp.sin(th)
    th_next = th + dt * om
    x_nom = jnp.array([px_next, py_next, th_next], dtype=x.dtype)

    # Stronger disturbance sampling
    key, key_dir, key_rad = jax.random.split(key, 3)

    z = jax.random.normal(key_dir, (x.shape[0],), dtype=x.dtype)
    z = z / (jnp.linalg.norm(z) + jnp.asarray(1e-12, dtype=x.dtype))

    n = jnp.asarray(x.shape[0], dtype=x.dtype)
    a = jnp.asarray(1.0, dtype=x.dtype)
    b = jnp.asarray(1.0, dtype=x.dtype)

    uu = jax.random.uniform(key_rad, (), dtype=x.dtype)
    r = (a**n + (b**n - a**n) * uu) ** (1.0 / n)
    w = r * z

    # Optional deterministic set of w's for "adversarial" rollouts
    # jax.debug.print("{}", w)
    start = i - NUM_RANDOM + 5
    if start == 5:
        w = jnp.array([1.0, 1.0, 1.0], dtype=x.dtype)
    if start == 6:
        w = jnp.array([1.0, -1.0, 1.0], dtype=x.dtype)
    if start == 7:
        w = jnp.array([1.0, 1.0, -1.0], dtype=x.dtype)
    if start == 8:
        w = jnp.array([-1.0, -1.0, 1.0], dtype=x.dtype)
    if start == 9:
        w = jnp.array([-1.0, 1.0, -1.0], dtype=x.dtype)
    if start == 10:
        w = jnp.array([1.0, -1.0, -1.0], dtype=x.dtype)
    if start == 11:
        w = jnp.array([-1.0, -1.0, -1.0], dtype=x.dtype)
    if start == 12:
        w = jnp.array([-0.707, 0.707, 0.0], dtype=x.dtype)
    if start == 13:
        w = jnp.array([0.707, -0.707, 0.0], dtype=x.dtype)
    if start == 14:
        w = jnp.array([-0.707, -0.707, 0.0], dtype=x.dtype)
    if start == 15:
        w = jnp.array([0.707, 0.0, 0.707], dtype=x.dtype)
    if start == 16:
        w = jnp.array([-0.707, 0.0, 0.707], dtype=x.dtype)
    if start == 17:
        w = jnp.array([0.707, 0.0, -0.707], dtype=x.dtype)
    if start == 18:
        w = jnp.array([-0.707, 0.0, -0.707], dtype=x.dtype)
    if start == 19:
        w = jnp.array([0.0, 0.707, 0.707], dtype=x.dtype)
    if start == 20:
        w = jnp.array([0.0, -0.707, 0.707], dtype=x.dtype)
    if start == 21:
        w = jnp.array([0.0, 0.707, -0.707], dtype=x.dtype)
    if start == 22:
        w = jnp.array([0.0, -0.707, -0.707], dtype=x.dtype)
    if start == 23:
        w = jnp.array([0.577, 0.577, 0.577], dtype=x.dtype)
    if start == 24:
        w = jnp.array([-0.577, 0.577, 0.577], dtype=x.dtype)
    if start == 25:
        w = jnp.array([0.577, -0.577, 0.577], dtype=x.dtype)
    if start == 26:
        w = jnp.array([0.577, 0.577, -0.577], dtype=x.dtype)
    if start == 27:
        w = jnp.array([-0.577, -0.577, 0.577], dtype=x.dtype)
    if start == 28:
        w = jnp.array([0.577, -0.577, -0.577], dtype=x.dtype)
    if start == 29:
        w = jnp.array([-0.577, 0.577, -0.577], dtype=x.dtype)
    if start == 30:
        w = jnp.array([-0.577, -0.577, -0.577], dtype=x.dtype)

    # Additive disturbance
    injected_disturbance = (E @ w + DISTURBANCE_OFFSET)
    x_next = x_nom + injected_disturbance
    return key, x_next, injected_disturbance

def dynamics(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray, *, parameter: Any) -> jnp.ndarray:
    """Discrete-time dynamics required by your model evaluator."""
    dt = parameter
    px, py, th = x[0], x[1], x[2]
    om = u[0]
    px_next = px + dt * V_CONST * jnp.cos(th)
    py_next = py + dt * V_CONST * jnp.sin(th)
    th_next = th + dt * om
    return jnp.array([px_next, py_next, th_next], dtype=x.dtype)

def cost(W, reference, x, u, t):
    """
    W = [wx, wy, wtheta, womega]
    """
    wx, wy, wtheta, womega = W
    xref = reference[t]

    dx = x[0] - xref[0]
    dy = x[1] - xref[1]
    dth = x[2] - xref[2]
    theta_cost = 1 - jnp.cos(dth)

    om = u[0]

    return (
        wx * (dx * dx)
        + wy * (dy * dy)
        + wtheta * theta_cost
        + womega * (om * om)
    )

def make_control_box_constraints(
    u_min: jnp.ndarray,
    u_max: jnp.ndarray
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Inequality constraints g(x,u,t) <= 0 for control bounds:
      u - u_max <= 0
      u_min - u <= 0
    """
    u_min = jnp.asarray(u_min)
    u_max = jnp.asarray(u_max)

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([u - u_max, u_min - u], axis=0)

    return constraints

def make_state_box_constraints(
    x_min: jnp.ndarray,
    x_max: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Inequality constraints g(x,u,t) <= 0 for state bounds:
      x - x_max <= 0
      x_min - x <= 0
    """
    x_min = jnp.asarray(x_min)
    x_max = jnp.asarray(x_max)

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([x - x_max, x_min - x], axis=0)

    return constraints

def make_constant_disturbance(
    n: int,
    alpha: float,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Returns a constant disturbance E with shape (T, n, n),
    where E[t] = alpha * I for all t.
    """
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        T = X_prefix.shape[0]
        E0 = alpha * jnp.eye(n, n, dtype=X_prefix.dtype)  # (n, n)
        return jnp.broadcast_to(E0, (T, n, n))

    return disturbance

def make_straight_line_reference(x0, x_goal, N):
    t = jnp.linspace(0.0, 1.0, N + 1)[:, None]   # (N+1, 1)
    return x0[None, :] + t * (x_goal - x0)[None, :]

# -----------------------------
# Main experiment
# -----------------------------
def main():
    # Dimensions
    n = 3      # [px, py, theta]
    nu = 1     # [omega]

    # Horizon and dt
    N = 110
    dt = 0.1

    # Weights: (x, y, theta, omega)
    W = jnp.array([25.0, 10.0, 0.01, 0.01], dtype=jnp.float64)

    cfg = MPCConfig(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.zeros((nu,), dtype=jnp.float64),
        dt=dt,
    )

    parameter = dt

    om_max = 100.0
    u_min = jnp.array([-om_max], dtype=jnp.float64)
    u_max = jnp.array([om_max], dtype=jnp.float64)

    constraints_u = make_control_box_constraints(u_min, u_max)

    x_max = jnp.array([15.0, 15.0, jnp.inf], dtype=jnp.float64)
    x_min = -x_max
    constraints_x = make_state_box_constraints(x_min, x_max)

    constraints_all = combine_constraints(constraints_x, constraints_u)

    obstacles = jnp.array([[0.0, 0.0, 0.3]], dtype=jnp.float64)
    n_obs = obstacles.shape[0]
    nc = 2 * nu + 2 * n + n_obs

    E_mag = 0.01
    alpha_sim = E_mag * dt
    disturbance = make_constant_disturbance(n=n, alpha=alpha_sim)

    x0 = jnp.array([-0.75, -0.75, 0.0], dtype=jnp.float64)
    x_goal = jnp.array([1.0, 0.6, 0.0], dtype=jnp.float64)

    X_ref = make_straight_line_reference(x0, x_goal, N)
    reference = X_ref
    T_steps = N

    key = jax.random.PRNGKey(0)
    E_sim = alpha_sim * jnp.eye(3, dtype=jnp.float64)

    # -----------------------------
    # Update configs for robust run
    # -----------------------------
    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=0,
        rho_max=1e5,
        max_iterations=1000,
        rho_update_frequency=10,
        initial_rho=0.1
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=2,
        sls_primal_tol=1e-2,
        enable_fastsls=True,
        initialize_nominal=True,
        max_initial_sqp_iterations=100,
        warm_start=False,
        rti=False,
        enable_linearization_bounds=True,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=100,
        warm_start=False,
        feas_tol=1e-2,
        step_tol=1e-8,
        line_search=False
    )

    Q_bar = jnp.broadcast_to(jnp.eye(n), (N + 1, n, n))
    R_bar = jnp.broadcast_to(jnp.eye(nu), (N, nu, nu))
    # Q = jnp.diag(jnp.array([25.0, 10.0, 0.01]))
    # R = jnp.diag(jnp.array([0.01]))
    # Q_bar = jnp.broadcast_to(Q, (N + 1, n, n))
    # R_bar = jnp.broadcast_to(R, (N, nu, nu))

    # disturbance_center = jnp.zeros((N + 1, n))
    disturbance_center = jnp.full((N + 1, n), DISTURBANCE_OFFSET)

    controller = GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        disturbance_center=disturbance_center,
        constraints=constraints_all,
        obstacles=obstacles,
        cost=cost,
        Q_bar=Q_bar,
        R_bar=R_bar,
        num_constraints=nc,
        disturbance=disturbance,
        shift=1,
        # X_in=jnp.zeros((cfg.N + 1, cfg.n), dtype=jnp.float64),
        X_in=X_ref,
        U_in=jnp.zeros((cfg.N, cfg.nu), dtype=jnp.float64),
    )

    # robust plan (single call in your script)
    N_ROLLOUTS = NUM_RANDOM + NUM_ADV
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u, EN, r_centerN = controller.run(
        x0=x0, reference=reference, parameter=parameter
    )
    jax.debug.print("{}", r_centerN)
    # -----------------------------
    # Rollout simulations with early stopping
    # -----------------------------
    xs = np.full((N_ROLLOUTS, T_steps, 3), np.nan, dtype=np.float64)
    disturbed = np.full((N_ROLLOUTS, T_steps, 3), np.nan, dtype=np.float64)
    stop_steps = np.full((N_ROLLOUTS,), T_steps, dtype=np.int32)

    for i in range(N_ROLLOUTS):
        disturbance_history = [jnp.zeros((n,), dtype=jnp.float64)]
        x = x0
        jax.debug.print("Rolling out iteration {}", i)

        for k in range(T_steps):
            if bool(reached_goal_xy(x, x_goal, GOAL_TOL)):
                stop_steps[i] = k
                break

            disturbance_feedback = jnp.zeros((nu,), dtype=jnp.float64)
            for j in range(k + 1):
                disturbance_feedback = disturbance_feedback + Phi_u[k, j] @ disturbance_history[j]

            u = U_pred[k] + disturbance_feedback

            key, x, injected_disturbance = dubins_step_with_disturbance(key, x, u, E_sim, dt, i)

            # disturbed[i, k] = np.asarray(x - X_pred[k + 1])
            disturbed[i, k] = np.asarray(x)

            disturbance_history.append(injected_disturbance)

            xs[i, k] = np.asarray(x)


    plans_xy = []
    lowers_xy = []
    uppers_xy = []

    # symmetric tube radius
    tube = np.asarray(get_trajectory_tubes(Phi_x, EN))          # (N+1, n)

    # nominal plan
    plan = np.asarray(X_pred)                                   # (N+1, n)

    # propagated center shift
    tube_center_shift = jnp.einsum("kjxn,jn->kx", Phi_x, r_centerN)
    shift = np.asarray(tube_center_shift)                       # (N+1, n)

    # off-centered reachable tube
    lower = plan + shift - tube                                 # (N+1, n)
    upper = plan + shift + tube                                 # (N+1, n)

    # lower_real = shift - tube
    # upper_real = shift + tube

    # tube center for plotting
    plan_center = plan                                   # (N+1, n)

    # XY slices for plotting
    plans_xy.append(plan_center[:, :2])
    lowers_xy.append(lower[:, :2])
    uppers_xy.append(upper[:, :2])

    plot_rollouts_tubes_centers(
        xs=xs,
        centers=np.asarray(obstacles[:, :2]),
        radii=np.asarray(obstacles[:, 2]),
        plans_xy=np.asarray(plans_xy),
        lowers_xy=np.asarray(lowers_xy),
        uppers_xy=np.asarray(uppers_xy),
        step_idx=0,
        tube_stride=1,
        filename="rollouts_tubes_centers.png",
        show_plan=False,
        tube_alpha=0.1,
        margin=0.2,
        rollout_alpha=0.5,
    )

    plot_tube_graph(
        disturbed=disturbed,
        lower=lower,
        upper=upper,
        dt=dt,
    )


if __name__ == "__main__":
    main()