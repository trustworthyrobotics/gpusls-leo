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
from gpu_sls.utils.constraint_utils import combine_constraints, make_control_box_constraints, make_state_box_constraints, make_constant_disturbance
from gpu_sls.utils.sls_visual import get_trajectory_tubes
from visualize_experiment import plot_rollouts_tubes_centers, plot_tube_graph_quadrotor

config.update("jax_enable_x64", True)
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
GOAL_TOL = 0.25  # meters in xyz


def reached_goal_xyz(x: jnp.ndarray, x_goal: jnp.ndarray, tol: float = GOAL_TOL) -> jnp.bool_:
    dpos = x[:3] - x_goal[:3]
    return (dpos @ dpos) <= (tol * tol)

# -----------------------------
# 3D quadrotor parameters
# x = [px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r]
# u = [T, tau_phi, tau_theta, tau_psi]
# -----------------------------
MASS = 1.0
GRAVITY = 9.81

JX = 0.02
JY = 0.02
JZ = 0.04

NUM_RANDOM = 5
NUM_ADV = 12 + 12 + 2**12  # +e_i, -e_i, all ±1 combos = 4120


def dynamics(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray, *, parameter) -> jnp.ndarray:
    dt = parameter

    px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r = x
    T, tau_phi, tau_theta, tau_psi = u

    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    cth, sth = jnp.cos(theta), jnp.sin(theta)
    cpsi, spsi = jnp.cos(psi), jnp.sin(psi)
    tth = jnp.tan(theta)

    dpx = vx
    dpy = vy
    dpz = vz

    dphi = p + q * sphi * tth + r * cphi * tth
    dtheta = q * cphi - r * sphi
    dpsi = q * sphi / cth + r * cphi / cth

    dvx = (T / MASS) * (cpsi * sth * cphi + spsi * sphi)
    dvy = (T / MASS) * (spsi * sth * cphi - cpsi * sphi)
    dvz = (T / MASS) * (cth * cphi) - GRAVITY

    dp = ((JY - JZ) / JX) * q * r + tau_phi / JX
    dq = ((JZ - JX) / JY) * p * r + tau_theta / JY
    dr = ((JX - JY) / JZ) * p * q + tau_psi / JZ

    return x + dt * jnp.stack((
        dpx, dpy, dpz,
        dphi, dtheta, dpsi,
        dvx, dvy, dvz,
        dp, dq, dr,
    ))


def cost(W, reference, x, u, t):
    """
    W =
    [wpx, wpy, wpz,
     wphi, wtheta, wpsi,
     wvx, wvy, wvz,
     wp, wq, wr,
     wT, wtau_phi, wtau_theta, wtau_psi]
    """
    (
        wpx, wpy, wpz,
        wphi, wtheta, wpsi,
        wvx, wvy, wvz,
        wp, wq, wr,
        wT, wtau_phi, wtau_theta, wtau_psi
    ) = W

    xref = reference[t]

    dpos = x[:3] - xref[:3]
    dang = x[3:6] - xref[3:6]
    dvel = x[6:9] - xref[6:9]
    drates = x[9:12] - xref[9:12]

    T_hover = MASS * GRAVITY
    du = jnp.array([
        u[0] - T_hover,
        u[1],
        u[2],
        u[3],
    ], dtype=x.dtype)

    angle_cost = (
        wphi * (1.0 - jnp.cos(dang[0]))
        + wtheta * (1.0 - jnp.cos(dang[1]))
        + wpsi * (1.0 - jnp.cos(dang[2]))
    )

    return (
        wpx * dpos[0] ** 2
        + wpy * dpos[1] ** 2
        + wpz * dpos[2] ** 2
        + angle_cost
        + wvx * dvel[0] ** 2
        + wvy * dvel[1] ** 2
        + wvz * dvel[2] ** 2
        + wp * drates[0] ** 2
        + wq * drates[1] ** 2
        + wr * drates[2] ** 2
        + wT * du[0] ** 2
        + wtau_phi * du[1] ** 2
        + wtau_theta * du[2] ** 2
        + wtau_psi * du[3] ** 2
    )

def build_piecewise_reference(x0: jnp.ndarray, x_goal: jnp.ndarray, N: int, dt: float) -> jnp.ndarray:
    """
    Build a straight-line reference trajectory from x0 to x_goal.

    Only position (px,py,pz) and yaw are interpolated.
    All other states are set to zero reference.
    """

    t = jnp.linspace(0.0, 1.0, N + 1)

    # Linear interpolation for position
    pos = (1.0 - t[:, None]) * x0[:3] + t[:, None] * x_goal[:3]

    # Shortest-path yaw interpolation
    dpsi = x_goal[5] - x0[5]
    psi = x0[5] + t * dpsi

    X_ref = jnp.zeros((N + 1, 12), dtype=jnp.float64)

    X_ref = X_ref.at[:, :3].set(pos)
    X_ref = X_ref.at[:, 5].set(psi)

    # Optionally compute velocity reference from the line
    vel = (x_goal[:3] - x0[:3]) / (N * dt)
    X_ref = X_ref.at[:, 6:9].set(vel)

    return X_ref


# -----------------------------
# Parallel rollout helpers
# -----------------------------

def make_w(
    rollout_idx: int,
    x: jnp.ndarray,
    key: jax.Array,
) -> tuple[jax.Array, jnp.ndarray]:
    n = x.shape[0]
    dtype = x.dtype

    # --- Random: uniform in the inf-norm unit ball ---
    key, subkey = jax.random.split(key)
    w_random = jax.random.uniform(subkey, (n,), dtype=dtype, minval=-1.0, maxval=1.0)

    # --- Adversarial corners ---
    # rows   0-11:      +e_i
    # rows  12-23:      -e_i
    # rows  24-4119:    all 2^12 combos of ±1
    eye12 = jnp.eye(12, dtype=dtype)

    # Build all 2^12 ±1 vectors: decode row index as 12-bit integer
    combo_indices = jnp.arange(2**12)                          # (4096,)
    bits = ((combo_indices[:, None] >> jnp.arange(12)) & 1)    # (4096, 12)
    pm_combos = (2.0 * bits - 1.0).astype(dtype)               # (4096, 12)  values in {-1, +1}

    adv_table = jnp.concatenate([
        eye12,       # rows   0-11
        -eye12,      # rows  12-23
        pm_combos,   # rows  24-4119
    ], axis=0)  # (NUM_ADV, 12)

    adv_idx = jnp.clip(rollout_idx - NUM_RANDOM, 0, NUM_ADV - 1)
    w_adv = adv_table[adv_idx]

    w = jnp.where(rollout_idx >= NUM_RANDOM, w_adv, w_random)
    return key, w


def run_single_rollout(
    rollout_idx: int,
    x0: jnp.ndarray,
    X_pred: jnp.ndarray,    # (T+1, n)  nominal state plan
    U_pred: jnp.ndarray,    # (T, nu)   nominal control plan
    Phi_u: jnp.ndarray,     # (T, T+1, nu, n)  disturbance-feedback gains
    E_sim: jnp.ndarray,     # (n, n)    disturbance matrix
    dt: float,
    u_min: jnp.ndarray,
    u_max: jnp.ndarray,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate one closed-loop rollout under disturbance feedback.

    Returns:
        xs       (T, n)  - simulated states at each step
        disturbed (T, n) - abs error vs nominal plan at each step
    """
    T_steps = U_pred.shape[0]
    n = x0.shape[0]

    def step_fn(carry, k):
        x, key, disturbance_history = carry
        # disturbance_history: (T+1, n)
        #   slot 0   = zeros (initial)
        #   slot j+1 = E_sim @ w_j  after step j

        # Disturbance feedback: sum_{j=0}^{k} Phi_u[k, j] @ disturbance_history[j]
        # Phi_u[k]: (T+1, nu, n),  disturbance_history: (T+1, n)
        feedback_all = jnp.einsum('jab,jb->ja', Phi_u[k], disturbance_history)  # (T+1, nu)
        mask = (jnp.arange(T_steps + 1) <= (k + 1))[:, None]                           # (T+1, 1)
        disturbance_feedback = jnp.sum(feedback_all * mask, axis=0)              # (nu,)

        u = U_pred[k] + disturbance_feedback
        u = jnp.clip(u, u_min, u_max)

        # Step dynamics + disturbance
        x_nom = dynamics(x, u, 0, parameter=dt)
        key, w = make_w(rollout_idx, x, key)
        x_next = x_nom + E_sim @ w

        # Tracking error vs nominal plan
        err = jnp.abs(X_pred[k + 1] - x_next)

        # Append new disturbance to history
        disturbance_history = disturbance_history.at[k + 1].set(E_sim @ w)

        carry = (x_next, key, disturbance_history)
        return carry, (x_next, err)

    init_disturbance_history = jnp.zeros((T_steps + 1, n), dtype=x0.dtype)
    init_carry = (x0, key, init_disturbance_history)

    ks = jnp.arange(T_steps)
    _final_carry, (xs, disturbed) = jax.lax.scan(step_fn, init_carry, ks)

    return xs, disturbed


def main():
    # -----------------------------
    # Dimensions
    # -----------------------------
    n = 12
    nu = 4

    # -----------------------------
    # Horizon and dt
    # -----------------------------
    N = 110
    dt = 0.01

    # -----------------------------
    # Cost weights
    # -----------------------------
    W = jnp.array([
        50.0, 50.0, 50.0,     # position
        2.0, 2.0, 0.5,        # roll, pitch, yaw
        0.5, 0.5, 0.5,        # velocities
        0.05, 0.05, 0.05,     # body rates
        0.01, 0.01, 0.01, 0.01  # control
    ], dtype=jnp.float64)

    cfg = MPCConfig(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=jnp.float64),
        dt=dt,
    )

    parameter = dt

    # -----------------------------
    # Control limits
    # -----------------------------
    T_hover = MASS * GRAVITY
    T_max = 2.0 * T_hover
    tau_max = 10.0

    u_min = jnp.array([0.0, -tau_max, -tau_max, -tau_max], dtype=jnp.float64)
    u_max = jnp.array([T_max, tau_max, tau_max, tau_max], dtype=jnp.float64)
    constraints_u = make_control_box_constraints(u_min, u_max)

    # -----------------------------
    # State limits
    # -----------------------------
    x_max = jnp.array([
        15.0, 15.0, 15.0,       # px, py, pz
        jnp.pi / 2.0,           # phi
        jnp.pi / 2.0,           # theta
        10.0 * jnp.pi,          # psi
        5.0, 5.0, 5.0,          # vx, vy, vz
        8.0, 8.0, 8.0           # p, q, r
    ], dtype=jnp.float64)
    x_min = -x_max
    x_min = x_min.at[2].set(-1.0)

    constraints_x = make_state_box_constraints(x_min, x_max)
    constraints_all = combine_constraints(constraints_x, constraints_u)

    obstacles = jnp.array([
        [0.0, 0.0, 0.35],
    ], dtype=jnp.float64)

    n_obs = obstacles.shape[0]
    nc = 2 * nu + 2 * n + n_obs

    E_mag = 0.1
    alpha_sim = E_mag * dt
    disturbance = make_constant_disturbance(n=n, alpha=alpha_sim)

    # -----------------------------
    # Initial / goal
    # -----------------------------
    x0 = jnp.array([
        -0.75, -0.75, 0.25,    # px, py, pz
        0.0, 0.0, 0.0,          # phi, theta, psi
        0.0, 0.0, 0.0,          # vx, vy, vz
        0.0, 0.0, 0.0           # p, q, r
    ], dtype=jnp.float64)

    x_goal = jnp.array([
        1.0, 0.8, 0.5,          # px, py, pz
        0.0, 0.0, 0.0,          # phi, theta, psi
        0.0, 0.0, 0.0,          # vx, vy, vz
        0.0, 0.0, 0.0           # p, q, r
    ], dtype=jnp.float64)

    X_ref = build_piecewise_reference(x0, x_goal, N, dt)
    reference = X_ref

    key = jax.random.PRNGKey(0)
    E_sim = alpha_sim * jnp.eye(n, dtype=jnp.float64)

    # -----------------------------
    # Solver configs
    # -----------------------------
    admm_cfg = ADMMConfig(
        eps_abs=1e-1,
        eps_rel=1e-2,
        rho_max=1e6,
        max_iterations=400,
        rho_update_frequency=20,
        initial_rho=30.0,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=3,
        sls_primal_tol=1e-2,
        enable_fastsls=True,
        initialize_nominal=False,
        max_initial_sqp_iterations=30,
        warm_start=False,
        rti=False,
        enable_linearization_bounds=True,
        enable_linearization_gradients=True,
        lambda_rem=2.0,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=30,
        warm_start=False,
        feas_tol=0.01,
        step_tol=0.01,
        line_search=False,
    )
    Q = jnp.diag(jnp.array(W[:-4]))
    R = jnp.diag(jnp.array(W[-4:]))
    Q_bar = jnp.broadcast_to(Q, (N + 1, n, n))
    R_bar = jnp.broadcast_to(R, (N, nu, nu))
    # Q_bar = jnp.broadcast_to(jnp.eye(n) * 1, (N + 1, n, n))
    # R_bar = jnp.broadcast_to(jnp.eye(nu), (N, nu, nu))

    controller = GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        constraints=constraints_all,
        obstacles=obstacles,
        cost=cost,
        Q_bar=Q_bar,
        R_bar=R_bar,
        num_constraints=nc,
        disturbance=disturbance,
        shift=1,
        X_in=jnp.zeros((cfg.N + 1, cfg.n), dtype=jnp.float64),
        U_in=jnp.zeros((cfg.N, cfg.nu), dtype=jnp.float64).at[:, 0].set(T_hover),
    )

    # -----------------------------
    # Robust plan
    # -----------------------------
    N_ROLLOUTS = NUM_RANDOM + NUM_ADV
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u, E_prev = controller.run(
        x0=x0, reference=reference, parameter=parameter
    )

    # -----------------------------
    # Parallel rollouts via vmap + lax.scan
    # -----------------------------

    # Give each rollout a unique PRNG key
    keys = jax.random.split(key, N_ROLLOUTS)
    rollout_indices = jnp.arange(N_ROLLOUTS)

    run_vmapped = jax.vmap(
        run_single_rollout,
        in_axes=(0, None, None, None, None, None, None, None, None, 0),
    )

    xs_jax, disturbed_jax = run_vmapped(
        rollout_indices,   # (N_ROLLOUTS,)   -- mapped
        x0,                # (n,)            -- broadcast
        X_pred,            # (T+1, n)        -- broadcast
        U_pred,            # (T, nu)         -- broadcast
        Phi_u,             # (T, T+1, nu, n) -- broadcast
        E_sim,             # (n, n)          -- broadcast
        dt,                # scalar          -- broadcast
        u_min,             # (nu,)           -- broadcast
        u_max,             # (nu,)           -- broadcast
        keys,              # (N_ROLLOUTS, 2) -- mapped
    )

    xs       = np.asarray(xs_jax)        # (N_ROLLOUTS, T_steps, n)
    disturbed = np.asarray(disturbed_jax) # (N_ROLLOUTS, T_steps, n)

    # -----------------------------
    # Tube visualization
    # NOTE: plotting is still 2D, using xy projection
    # -----------------------------
    plans_xy = []
    lowers_xy = []
    uppers_xy = []

    tube = get_trajectory_tubes(Phi_x, E_prev)

    plan_xy = X_pred[:, :2]       # (px, py)
    lower = plan_xy - tube[:, :2]
    upper = plan_xy + tube[:, :2]

    plans_xy.append(plan_xy)
    lowers_xy.append(lower)
    uppers_xy.append(upper)

    plot_rollouts_tubes_centers(
        xs=xs,
        centers=np.asarray(obstacles[:, :2]),
        radii=np.asarray(obstacles[:, 2]),
        plans_xy=np.asarray(plans_xy),
        lowers_xy=np.asarray(lowers_xy),
        uppers_xy=np.asarray(uppers_xy),
        step_idx=0,
        tube_stride=1,
        filename="quadrotor_3d_rollouts_xy_projection.png",
        show_plan=False,
        tube_alpha=0.1,
        margin=0.2,
        rollout_alpha=0.5,
        x_label="x",
        y_label="y",
        title="3D Quadrotor: XY Projection of Rollouts + Tube",
    )

    plot_tube_graph_quadrotor(
        disturbed=disturbed[:, :, :12],   # position + Euler angles only
        tube=tube[:, :12],
        dt=dt,
        filename="quadrotor_3d_disturbance_vs_tube_size_pose.png",
    )


if __name__ == "__main__":
    main()