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
from gpu_sls.utils.constraint_utils import (
    combine_constraints,
    make_control_box_constraints,
    make_state_box_constraints,
    make_constant_disturbance,
)
from gpu_sls.utils.sls_visual import get_trajectory_tubes
from visualize_experiment import plot_plan_xy_with_tubes
# from visualize_experiment import plot_rollouts_tubes_centers, plot_tube_graph_quadrotor

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
GOAL_TOL = 0.25  # meters in xyz

# -----------------------------
# Single-quad parameters
# State per quad:
# x_i = [px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r]
# Input per quad:
# u_i = [T, tau_phi, tau_theta, tau_psi]
# -----------------------------
MASS = 1.0
GRAVITY = 9.81

JX = 0.02
JY = 0.02
JZ = 0.04

# -----------------------------
# Multi-quad parameters
# -----------------------------
N_QUADS = 4
SINGLE_N = 12
SINGLE_NU = 4
N = N_QUADS * SINGLE_N
NU = N_QUADS * SINGLE_NU

COUPLING_ALPHA = 0.75   # spring-to-centroid strength
COUPLING_BETA = 0.25    # velocity damping to mean velocity

NUM_RANDOM = 5
NUM_ADV = 2 * N + 2**N   # this explodes for large N; keep only for small N if used


def reshape_state(x: jnp.ndarray) -> jnp.ndarray:
    return x.reshape(N_QUADS, SINGLE_N)


def reshape_control(u: jnp.ndarray) -> jnp.ndarray:
    return u.reshape(N_QUADS, SINGLE_NU)


def single_quad_derivative(xi: jnp.ndarray, ui: jnp.ndarray) -> jnp.ndarray:
    px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r = xi
    T, tau_phi, tau_theta, tau_psi = ui

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

    return jnp.array([
        dpx, dpy, dpz,
        dphi, dtheta, dpsi,
        dvx, dvy, dvz,
        dp, dq, dr,
    ], dtype=xi.dtype)


def multi_quad_dynamics(
    x: jnp.ndarray,
    u: jnp.ndarray,
    t: jnp.ndarray,
    *,
    parameter,
) -> jnp.ndarray:
    """
    Discrete Euler dynamics:
        x_{k+1} = x_k + dt * f(x_k, u_k)

    with optional centroid spring-damper coupling in translational acceleration.
    """
    dt = parameter

    xq = reshape_state(x)    # (N_QUADS, 12)
    uq = reshape_control(u)  # (N_QUADS, 4)

    dxq = jax.vmap(single_quad_derivative)(xq, uq)  # (N_QUADS, 12)

    # centroid coupling on translational acceleration only
    pos = xq[:, 0:3]
    vel = xq[:, 6:9]

    pos_mean = jnp.mean(pos, axis=0, keepdims=True)
    vel_mean = jnp.mean(vel, axis=0, keepdims=True)

    coupling_pos = -COUPLING_ALPHA * N_QUADS * (pos - pos_mean)
    coupling_vel = -COUPLING_BETA * (vel - vel_mean)

    accel_coupling = coupling_pos + coupling_vel

    dxq = dxq.at[:, 6:9].add(accel_coupling)

    return (xq + dt * dxq).reshape(-1)


def cost(W, reference, x, u, t):
    """
    Multi-quad separable tracking cost.

    W per quad:
    [wpx, wpy, wpz,
     wphi, wtheta, wpsi,
     wvx, wvy, wvz,
     wp, wq, wr,
     wT, wtau_phi, wtau_theta, wtau_psi]
    """
    xref = reference[t]

    xq = reshape_state(x)
    uq = reshape_control(u)
    xrq = reshape_state(xref)

    Wq = W.reshape(N_QUADS, 16)

    def quad_cost(xi, ui, xri, wi):
        (
            wpx, wpy, wpz,
            wphi, wtheta, wpsi,
            wvx, wvy, wvz,
            wp, wq, wr,
            wT, wtau_phi, wtau_theta, wtau_psi
        ) = wi

        dpos = xi[:3] - xri[:3]
        dang = xi[3:6] - xri[3:6]
        dvel = xi[6:9] - xri[6:9]
        drates = xi[9:12] - xri[9:12]

        T_hover = MASS * GRAVITY
        du = jnp.array([
            ui[0] - T_hover,
            ui[1],
            ui[2],
            ui[3],
        ], dtype=xi.dtype)

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

    return jnp.sum(jax.vmap(quad_cost)(xq, uq, xrq, Wq))


def build_piecewise_reference_multi(
    x0: jnp.ndarray,
    x_goal: jnp.ndarray,
    horizon: int,
    dt: float,
) -> jnp.ndarray:
    t = jnp.linspace(0.0, 1.0, horizon + 1)
    x0q = reshape_state(x0)
    xgq = reshape_state(x_goal)

    def one_quad_ref(x0i, xgi):
        pos = (1.0 - t[:, None]) * x0i[:3] + t[:, None] * xgi[:3]
        dpsi = xgi[5] - x0i[5]
        psi = x0i[5] + t * dpsi

        Xi = jnp.zeros((horizon + 1, SINGLE_N), dtype=jnp.float64)
        Xi = Xi.at[:, :3].set(pos)
        Xi = Xi.at[:, 5].set(psi)

        vel = (xgi[:3] - x0i[:3]) / (horizon * dt)
        Xi = Xi.at[:, 6:9].set(vel)
        return Xi

    X_ref_quads = jax.vmap(one_quad_ref)(x0q, xgq)          # (N_QUADS, H+1, 12)
    X_ref = X_ref_quads.transpose(1, 0, 2).reshape(horizon + 1, N)
    return X_ref


def make_w(
    rollout_idx: int,
    x: jnp.ndarray,
    key: jax.Array,
) -> tuple[jax.Array, jnp.ndarray]:
    n = x.shape[0]
    dtype = x.dtype

    key, subkey = jax.random.split(key)
    w_random = jax.random.uniform(subkey, (n,), dtype=dtype, minval=-1.0, maxval=1.0)

    # For large N, do not build 2^N adversarial corners.
    # Use +/- basis only.
    eye_n = jnp.eye(n, dtype=dtype)
    adv_table = jnp.concatenate([eye_n, -eye_n], axis=0)

    adv_idx = jnp.clip(rollout_idx - NUM_RANDOM, 0, adv_table.shape[0] - 1)
    w_adv = adv_table[adv_idx]

    w = jnp.where(rollout_idx >= NUM_RANDOM, w_adv, w_random)
    return key, w


def run_single_rollout(
    rollout_idx: int,
    x0: jnp.ndarray,
    X_pred: jnp.ndarray,
    U_pred: jnp.ndarray,
    Phi_u: jnp.ndarray,
    E_sim: jnp.ndarray,
    dt: float,
    u_min: jnp.ndarray,
    u_max: jnp.ndarray,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    T_steps = U_pred.shape[0]
    n = x0.shape[0]

    def step_fn(carry, k):
        x, key, disturbance_history = carry

        feedback_all = jnp.einsum("jab,jb->ja", Phi_u[k], disturbance_history)
        mask = (jnp.arange(T_steps + 1) <= (k + 1))[:, None]
        disturbance_feedback = jnp.sum(feedback_all * mask, axis=0)

        u = U_pred[k] + disturbance_feedback
        u = jnp.clip(u, u_min, u_max)

        x_nom = multi_quad_dynamics(x, u, 0, parameter=dt)
        key, w = make_w(rollout_idx, x, key)
        x_next = x_nom + E_sim @ w

        err = jnp.abs(X_pred[k + 1] - x_next)

        disturbance_history = disturbance_history.at[k + 1].set(E_sim @ w)

        carry = (x_next, key, disturbance_history)
        return carry, (x_next, err)

    init_disturbance_history = jnp.zeros((T_steps + 1, n), dtype=x0.dtype)
    init_carry = (x0, key, init_disturbance_history)

    ks = jnp.arange(T_steps)
    _final_carry, (xs, disturbed) = jax.lax.scan(step_fn, init_carry, ks)

    return xs, disturbed


def make_multi_quad_circle_obstacle_constraints(
    obstacles: jnp.ndarray,
    n_quads: int,
    single_state_dim: int = 12,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Creates circle obstacle constraints for a multi-quad state.

    Each quad uses its (px, py) = x_i[:2].
    Each obstacle row is [cx, cy, radius].

    Returns constraints g(x,u,t) such that:
        g <= 0  <=>  every quad is outside every circle

    Output shape:
        (n_quads * n_obs,)
    """
    obstacles = jnp.asarray(obstacles)
    centers = obstacles[:, :2]   # (n_obs, 2)
    radii = obstacles[:, 2]      # (n_obs,)

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        xq = x.reshape(n_quads, single_state_dim)   # (n_quads, 12)
        pos_xy = xq[:, :2]                          # (n_quads, 2)

        # diff[i, j] = quad i position - obstacle j center
        diff = pos_xy[:, None, :] - centers[None, :, :]   # (n_quads, n_obs, 2)
        dist_sq = jnp.sum(diff ** 2, axis=-1)             # (n_quads, n_obs)

        # outside circle means dist_sq >= r^2
        # so constraint is satisfied when r^2 - dist_sq <= 0
        g_obs = radii[None, :] ** 2 - dist_sq             # (n_quads, n_obs)

        return g_obs.reshape(-1)

    return constraints


def main():
    # -----------------------------
    # Horizon and dt
    # -----------------------------
    H = 25
    dt = 0.04

    # -----------------------------
    # Cost weights: one 16-vector per quad
    # -----------------------------
    W_single = jnp.array([
        50.0, 50.0, 50.0,     # position
        2.0, 2.0, 0.5,        # roll, pitch, yaw
        0.5, 0.5, 0.5,        # velocities
        0.05, 0.05, 0.05,     # body rates
        0.01, 0.01, 0.01, 0.01
    ], dtype=jnp.float64)
    W = jnp.tile(W_single, N_QUADS)

    u_ref_single = jnp.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=jnp.float64)
    u_ref = jnp.tile(u_ref_single, N_QUADS)

    cfg = MPCConfig(
        n=N,
        nu=NU,
        N=H,
        W=W,
        u_ref=u_ref,
        dt=dt,
    )

    parameter = dt

    # -----------------------------
    # Control limits
    # -----------------------------
    T_hover = MASS * GRAVITY
    T_max = 3.0 * T_hover
    tau_max = 10.0

    u_min_single = jnp.array([0.0, -tau_max, -tau_max, -tau_max], dtype=jnp.float64)
    u_max_single = jnp.array([T_max, tau_max, tau_max, tau_max], dtype=jnp.float64)

    u_min = jnp.tile(u_min_single, N_QUADS)
    u_max = jnp.tile(u_max_single, N_QUADS)

    constraints_u = make_control_box_constraints(u_min, u_max)

    # -----------------------------
    # State limits
    # -----------------------------
    x_max_single = jnp.array([
        5.0, 5.0, 5.0,
        jnp.pi / 2.0,
        jnp.pi / 2.0,
        10.0 * jnp.pi,
        5.0, 5.0, 5.0,
        8.0, 8.0, 8.0
    ], dtype=jnp.float64)
    x_min_single = -x_max_single
    x_min_single = x_min_single.at[2].set(-1.0)

    x_max = jnp.tile(x_max_single, N_QUADS)
    x_min = jnp.tile(x_min_single, N_QUADS)

    # -----------------------------
    # Obstacles
    # One circular obstacle per quad in XY projection, repeated in constraints interface.
    # -----------------------------
    obstacles = jnp.array([
        [0.0, 0.3, 0.35],
        [0.4, -0.5, 0.3],
    ], dtype=jnp.float64)


    constraints_x = make_state_box_constraints(x_min, x_max)
    obstacle_constraints = make_multi_quad_circle_obstacle_constraints(obstacles, N_QUADS)
    constraints_all = combine_constraints(constraints_x, constraints_u)
    constraints_all = combine_constraints(constraints_all, obstacle_constraints)

    n_obs = obstacles.shape[0]
    nc = 2 * NU + 2 * N + n_obs * N_QUADS
    # nc = 2 * NU + 2 * N

    E_mag = 0.06
    alpha_sim = E_mag * dt
    disturbance = make_constant_disturbance(n=N, alpha=alpha_sim)

    # -----------------------------
    # Initial / goal
    # -----------------------------
    x0_quads = jnp.array([
        [-0.75, -0.75, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.75,  -0.6, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.75,  -0.4, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.5, -0.6, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # [-0.6,  -0.9, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=jnp.float64)

    xg_quads = jnp.array([
        [1.00, -0.75, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.00,  0.00, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.00,  0.40, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.00, -0.90, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # [1.00,  -0.75, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=jnp.float64)

    x0 = x0_quads.reshape(-1)
    x_goal = xg_quads.reshape(-1)

    X_ref = build_piecewise_reference_multi(x0, x_goal, H, dt)
    reference = X_ref

    key = jax.random.PRNGKey(0)
    E_sim = alpha_sim * jnp.eye(N, dtype=jnp.float64)

    # -----------------------------
    # Solver configs
    # -----------------------------
    admm_cfg = ADMMConfig(
        eps_abs=1e-1,
        eps_rel=1e-4,
        rho_max=5e1,
        max_iterations=400,
        rho_update_frequency=2,
        initial_rho=1.0,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=3,
        sls_primal_tol=1e-2,
        enable_fastsls=True,
        initialize_nominal=True,
        max_initial_sqp_iterations=30,
        warm_start=False,
        rti=False,
        enable_linearization_bounds=True,
        enable_linearization_gradients=False,
        lambda_rem=2.0,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=30,
        warm_start=False,
        feas_tol=1e-2,
        step_tol=1e-4,
        line_search=False,
    )

    # Q_single = jnp.diag(jnp.array(W_single[:-4]))
    # R_single = jnp.diag(jnp.array(W_single[-4:]))
    Q_single = jnp.eye(12)
    R_single = jnp.eye(4)

    Q = jax.scipy.linalg.block_diag(*([Q_single] * N_QUADS))
    R = jax.scipy.linalg.block_diag(*([R_single] * N_QUADS))

    Q_bar = jnp.broadcast_to(Q, (H + 1, N, N))
    R_bar = jnp.broadcast_to(R, (H, NU, NU))
    no_obstacles = jnp.zeros((0, 3), dtype=jnp.float64)
    controller = GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=multi_quad_dynamics,
        constraints=constraints_all,
        obstacles=no_obstacles,
        cost=cost,
        Q_bar=Q_bar,
        R_bar=R_bar,
        num_constraints=nc,
        disturbance=disturbance,
        shift=1,
        X_in=jnp.zeros((cfg.N + 1, cfg.n), dtype=jnp.float64),
        U_in=jnp.zeros((cfg.N, cfg.nu), dtype=jnp.float64).at[:, 0::4].set(T_hover),
    )

    # -----------------------------
    # Robust plan
    # -----------------------------
    N_ROLLOUTS = NUM_RANDOM + 2 * N
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u, E_prev = controller.run(
        x0=x0, reference=reference, parameter=parameter
    )
    plot_plan_xy_with_tubes(
        X_pred=X_pred,
        x0_quads=x0_quads,
        xg_quads=xg_quads,
        obstacles=obstacles,
        Phi_x=Phi_x,
        E_prev=E_prev,
        N_QUADS=N_QUADS,
        SINGLE_N=SINGLE_N,
        filename="multi_quadrotor_plan_tubes.png",
        stride=1,        # plot every 5 timesteps (recommended)
        box_alpha=0.15,  # transparency of squares
    )
    # -----------------------------
    # Parallel rollouts
    # -----------------------------
    # keys = jax.random.split(key, N_ROLLOUTS)
    # rollout_indices = jnp.arange(N_ROLLOUTS)

    # run_vmapped = jax.vmap(
    #     run_single_rollout,
    #     in_axes=(0, None, None, None, None, None, None, None, None, 0),
    # )

    # xs_jax, disturbed_jax = run_vmapped(
    #     rollout_indices,
    #     x0,
    #     X_pred,
    #     U_pred,
    #     Phi_u,
    #     E_sim,
    #     dt,
    #     u_min,
    #     u_max,
    #     keys,
    # )

    # xs = np.asarray(xs_jax)
    # disturbed = np.asarray(disturbed_jax)

    # # -----------------------------
    # # Visualize first quad only in XY
    # # -----------------------------
    # tube = get_trajectory_tubes(Phi_x, E_prev)

    # quad0_slice = slice(0, SINGLE_N)
    # plan_xy = np.asarray(X_pred[:, quad0_slice])[:, :2]
    # tube0 = np.asarray(tube[:, quad0_slice])

    # lower = plan_xy - tube0[:, :2]
    # upper = plan_xy + tube0[:, :2]

    # xs_quad0 = xs[:, :, quad0_slice]

    # plot_rollouts_tubes_centers(
    #     xs=xs_quad0,
    #     centers=np.asarray(obstacles[:, :2]),
    #     radii=np.asarray(obstacles[:, 2]),
    #     plans_xy=np.asarray([plan_xy]),
    #     lowers_xy=np.asarray([lower]),
    #     uppers_xy=np.asarray([upper]),
    #     step_idx=0,
    #     tube_stride=1,
    #     filename="multi_quadrotor_rollouts_xy_projection_quad0.png",
    #     show_plan=False,
    #     tube_alpha=0.1,
    #     margin=0.2,
    #     rollout_alpha=0.5,
    #     x_label="x",
    #     y_label="y",
    #     title="Multi-Quadrotor: Quad 0 XY Projection",
    # )

    # plot_tube_graph_quadrotor(
    #     disturbed=disturbed[:, :, quad0_slice],
    #     tube=tube[:, quad0_slice],
    #     dt=dt,
    #     filename="multi_quadrotor_disturbance_vs_tube_size_quad0.png",
    # )


if __name__ == "__main__":
    main()