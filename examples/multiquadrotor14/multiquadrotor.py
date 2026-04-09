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
from visualize_experiment import plot_plan_xy_with_tubes, plot_tube_graph_multiquadrotor
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
N_QUADS = 14
SINGLE_N = 12
SINGLE_NU = 4
N = N_QUADS * SINGLE_N
NU = N_QUADS * SINGLE_NU

COUPLING_ALPHA = 0.75   # spring-to-centroid strength
COUPLING_BETA = 0.25    # velocity damping to mean velocity

NUM_RANDOM = 5
NUM_ADV = 2 * N + 2**N   # this explodes for large N; keep only for small N if used


import numpy as np
from pathlib import Path
from typing import Any


def save_multiquad_experiment_npz(
    filename: str,
    *,
    # core plan / controller outputs
    x0: Any,
    x_goal: Any,
    X_ref: Any,
    X_pred: Any,
    U_pred: Any,
    V_pred: Any | None = None,
    backoffs: Any | None = None,
    Phi_x: Any | None = None,
    Phi_u: Any | None = None,
    E_prev: Any | None = None,
    r_centerN: Any | None = None,
    E_sim: Any | None = None,
    # rollout outputs
    xs: Any | None = None,
    disturbed: Any | None = None,
    # visualization inputs
    x0_quads: Any | None = None,
    xg_quads: Any | None = None,
    obstacles: Any | None = None,
    # scalar metadata
    dt: float | None = None,
    H: int | None = None,
    N_QUADS: int | None = None,
    SINGLE_N: int | None = None,
    SINGLE_NU: int | None = None,
    u_min: Any | None = None,
    u_max: Any | None = None,
    notes: str = "",
) -> None:
    """
    Save everything needed to reconstruct rollouts and visualizations.

    The file is written as a compressed .npz archive.

    Parameters
    ----------
    filename :
        Output path, e.g. "multi_quad_experiment.npz"
    All other arguments :
        JAX arrays, NumPy arrays, Python scalars, or None.
    """

    def _to_numpy(x):
        if x is None:
            return None
        return np.asarray(x)

    payload = {
        # plan / controller state
        "x0": _to_numpy(x0),
        "x_goal": _to_numpy(x_goal),
        "X_ref": _to_numpy(X_ref),
        "X_pred": _to_numpy(X_pred),
        "U_pred": _to_numpy(U_pred),
        "V_pred": _to_numpy(V_pred),
        "backoffs": _to_numpy(backoffs),
        "Phi_x": _to_numpy(Phi_x),
        "Phi_u": _to_numpy(Phi_u),
        "E_prev": _to_numpy(E_prev),
        "r_centerN": _to_numpy(r_centerN),
        "E_sim": _to_numpy(E_sim),
        # rollout data
        "xs": _to_numpy(xs),
        "disturbed": _to_numpy(disturbed),
        # plotting / reconstruction inputs
        "x0_quads": _to_numpy(x0_quads),
        "xg_quads": _to_numpy(xg_quads),
        "obstacles": _to_numpy(obstacles),
        "u_min": _to_numpy(u_min),
        "u_max": _to_numpy(u_max),
        # metadata
        "dt": np.array(-1.0 if dt is None else dt, dtype=np.float64),
        "H": np.array(-1 if H is None else H, dtype=np.int32),
        "N_QUADS": np.array(-1 if N_QUADS is None else N_QUADS, dtype=np.int32),
        "SINGLE_N": np.array(-1 if SINGLE_N is None else SINGLE_N, dtype=np.int32),
        "SINGLE_NU": np.array(-1 if SINGLE_NU is None else SINGLE_NU, dtype=np.int32),
        "notes": np.array(notes),
    }

    # remove keys that are truly None so the archive stays clean
    payload = {k: v for k, v in payload.items() if v is not None}

    filename = str(Path(filename))
    np.savez_compressed(filename, **payload)
    print(f"Saved experiment archive to {filename}")

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

    inv_cth = 1.0 / cth
    tth = sth * inv_cth
    # tth = jnp.tan(theta)

    dpx = vx
    dpy = vy
    dpz = vz

    dphi = p + (q * sphi + r * cphi) * tth
    dtheta = q * cphi - r * sphi
    dpsi = (q * sphi + r * cphi) * inv_cth

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

    # dxq = dxq.at[:, 6:9].add(accel_coupling)
    dxq = jnp.concatenate(
        [
            dxq[:, :6],
            dxq[:, 6:9] + accel_coupling,
            dxq[:, 9:],
        ],
        axis=1,
    )

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

    # Random corner of {-1, +1}^n
    w = jax.random.rademacher(subkey, (n,), dtype=dtype)

    return key, w


# def run_single_rollout(
#     rollout_idx: int,
#     x0: jnp.ndarray,
#     X_pred: jnp.ndarray,
#     U_pred: jnp.ndarray,
#     Phi_u: jnp.ndarray,
#     E_sim: jnp.ndarray,
#     dt: float,
#     u_min: jnp.ndarray,
#     u_max: jnp.ndarray,
#     key: jax.Array,
# ) -> tuple[jnp.ndarray, jnp.ndarray]:
#     T_steps = U_pred.shape[0]
#     n = x0.shape[0]

#     def step_fn(carry, k):
#         x, key, disturbance_history = carry

#         feedback_all = jnp.einsum("jab,jb->ja", Phi_u[k], disturbance_history)
#         mask = (jnp.arange(T_steps + 1) <= (k + 1))[:, None]
#         disturbance_feedback = jnp.sum(feedback_all * mask, axis=0)

#         u = U_pred[k] + disturbance_feedback
#         u = jnp.clip(u, u_min, u_max)

#         x_nom = multi_quad_dynamics(x, u, 0, parameter=dt)
#         key, w = make_w(rollout_idx, x, key)
#         x_next = x_nom + E_sim @ w

#         err = jnp.abs(X_pred[k + 1] - x_next)

#         disturbance_history = disturbance_history.at[k + 1].set(E_sim @ w)

#         carry = (x_next, key, disturbance_history)
#         return carry, (x_next, err)

#     init_disturbance_history = jnp.zeros((T_steps + 1, n), dtype=x0.dtype)
#     init_carry = (x0, key, init_disturbance_history)

#     ks = jnp.arange(T_steps)
#     _final_carry, (xs, disturbed) = jax.lax.scan(step_fn, init_carry, ks)

#     return xs, disturbed

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

    # Sample one disturbance direction for the entire rollout
    key, subkey = jax.random.split(key)
    w_fixed = jax.random.rademacher(subkey, (n,), dtype=x0.dtype)
    d_fixed = E_sim @ w_fixed

    def step_fn(carry, k):
        x, disturbance_history = carry

        feedback_all = jnp.einsum("jab,jb->ja", Phi_u[k], disturbance_history)
        mask = (jnp.arange(T_steps + 1) <= (k + 1))[:, None]
        disturbance_feedback = jnp.sum(feedback_all * mask, axis=0)

        u = U_pred[k] + disturbance_feedback
        u = jnp.clip(u, u_min, u_max)

        x_nom = multi_quad_dynamics(x, u, 0, parameter=dt)
        x_next = x_nom + d_fixed

        err = x_next

        disturbance_history = disturbance_history.at[k + 1].set(d_fixed)

        carry = (x_next, disturbance_history)
        return carry, (x_next, err)

    init_disturbance_history = jnp.zeros((T_steps + 1, n), dtype=x0.dtype)
    init_carry = (x0, init_disturbance_history)

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
    dt = 0.1

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
    # T_max = 3.0 * T_hover
    # tau_max = 20.0
    tau_max = 80.0

    # u_min_single = jnp.array([0.0, -tau_max, -tau_max, -tau_max], dtype=jnp.float64)
    u_min_single = jnp.array([-tau_max, -tau_max, -tau_max, -tau_max], dtype=jnp.float64)
    u_max_single = jnp.array([tau_max, tau_max, tau_max, tau_max], dtype=jnp.float64)

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
    # x_min_single = x_min_single.at[2].set(-1.0)

    x_max = jnp.tile(x_max_single, N_QUADS)
    x_min = jnp.tile(x_min_single, N_QUADS)

    # -----------------------------
    # Obstacles
    # One circular obstacle per quad in XY projection, repeated in constraints interface.
    # -----------------------------
    # obstacles = jnp.array([
    #     # [0.1, 0.35, 0.27],
    #     [0.1, 0.345, 0.27],
    #     [0.45, -0.32, 0.35],
    #     [0.0, -0.5, 0.2],
    #     [-0.1, -1.1, 0.3],
    # ], dtype=jnp.float64)
    scale = 1
    obstacles = jnp.array([
        # [0.1, 0.35, 0.27],
        [0.1 * scale, 0.3 * scale, 0.26 * scale],
        [0.45 * scale, -0.32 * scale, 0.3 * scale],
        [0.0 * scale, -0.5 * scale, 0.18 * scale],
        [-0.1 * scale, -1.1 * scale, 0.26 * scale],
    ], dtype=jnp.float64)


    # constraints_x = make_state_box_constraints(x_min, x_max)
    obstacle_constraints = make_multi_quad_circle_obstacle_constraints(obstacles, N_QUADS)
    # constraints_all = make_multi_quad_circle_obstacle_constraints(obstacles, N_QUADS)
    # constraints_all = combine_constraints(constraints_x, constraints_u)
    constraints_all = combine_constraints(constraints_u, obstacle_constraints)
    # constraints_all = combine_constraints(constraints_u, obstacle_constraints)

    n_obs = obstacles.shape[0]
    # nc = 2 * NU + 2 * N + n_obs * N_QUADS
    nc = 2 * NU + n_obs * N_QUADS
    # nc = n_obs * N_QUADS

    # E_mag = 0.01
    E_mag = 0.0175
    alpha_sim = E_mag * dt
    disturbance = make_constant_disturbance(n=N, alpha=alpha_sim)

    # -----------------------------
    # Initial / goal
    # -----------------------------
    x0_quads = jnp.array([
        [-0.75 * scale, -0.75 * scale, 0.25 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.6 * scale,  -0.9 * scale, 0.1 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.5 * scale, -0.6 * scale, 0.5 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.75 * scale,  -0.4 * scale, 0.25 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.75 * scale,  -0.2 * scale, 0.4 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.4 * scale,  -0.3 * scale, 0.1 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.75 * scale,  0.4 * scale, 0.25 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.6 * scale,  0.5 * scale, 0.35 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.5 * scale,  0.45 * scale, 0.25 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.5 * scale,  -0.2 * scale, 0.3 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.4 * scale,  0.5 * scale, 0.4 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.4 * scale,  0.6 * scale, 0.3 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.5 * scale,  -0.7 * scale, 0.4 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.6 * scale,  -1.5 * scale, 0.3 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=jnp.float64)

    xg_quads = jnp.array([
        [1.00 * scale, -0.75 * scale, 0.50 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.00 * scale,  -0.6 * scale, 0.2 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8 * scale,  -1.1 * scale, 0.25 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.00 * scale,  0.40 * scale, 0.50 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.00 * scale, 0.20 * scale, 0.25 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8 * scale,  0.3 * scale, 0.35 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.7 * scale,  1.1 * scale, 0.5 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8 * scale, 1.0 * scale, 0.10 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8 * scale, 1.15 * scale, 0.50 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.9 * scale, 0.2 * scale, 0.50 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8 * scale, 0.8 * scale, 0.1 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8 * scale, 0.9 * scale, 0.30 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.9 * scale, -0.7 * scale, 0.50 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.75 * scale, -1.5 * scale, 0.30 * scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=jnp.float64)

    x0 = x0_quads[:N_QUADS].reshape(-1)
    x_goal = xg_quads[:N_QUADS].reshape(-1)

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
        rho_max=1e1,
        max_iterations=20,
        rho_update_frequency=2,
        initial_rho=5e-3,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=2,
        sls_primal_tol=1e-10,
        enable_fastsls=True,
        initialize_nominal=True,
        max_initial_sqp_iterations=50,
        warm_start=False,
        rti=False,
        enable_linearization_bounds=False,
        enable_linearization_gradients=False,
        lambda_rem=0.0,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=50,
        warm_start=False,
        feas_tol=1e-2,
        step_tol=1e-10,
        line_search=False,
    )

    Q_term_single = jnp.diag(jnp.array([
        30.0, 30.0, 1.0,   # position
        1.0, 1.0, 1.0,     # roll, pitch, yaw
        1.0, 1.0, 1.0,     # velocities
        1.0, 1.0, 1.0
    ]))

    Q_single = jnp.eye(12)
    R_single = jnp.eye(4)

    Q = jax.scipy.linalg.block_diag(*([Q_single] * N_QUADS))
    Q_term = jax.scipy.linalg.block_diag(*([Q_term_single] * N_QUADS))
    R = jax.scipy.linalg.block_diag(*([R_single] * N_QUADS))

    Q_bar = jnp.broadcast_to(Q, (H + 1, Q.shape[0], Q.shape[1]))
    Q_bar = Q_bar.at[-5:].set(Q_term)

    R_bar = jnp.broadcast_to(R, (H, R.shape[0], R.shape[1]))
    no_obstacles = jnp.zeros((0, 3), dtype=jnp.float64)
    disturbance_center = jnp.full((H + 1, N), 0.0)
    controller = GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=multi_quad_dynamics,
        constraints=constraints_all,
        obstacles=no_obstacles,
        cost=cost,
        disturbance_center=disturbance_center,
        Q_bar=Q_bar,
        R_bar=R_bar,
        num_constraints=nc,
        disturbance=disturbance,
        shift=1,
        X_in=X_ref,
        use_taylor_model=False,
        U_in=jnp.zeros((cfg.N, cfg.nu), dtype=jnp.float64).at[:, 0::4].set(T_hover),
    )

    # -----------------------------
    # Robust plan
    # -----------------------------
    N_ROLLOUTS = 1000
    jax.debug.print("START SCRIPT")
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u, E_prev, r_centerN = controller.run(
        x0=x0, reference=reference, parameter=parameter
    )
    import time
    jax.debug.print("BEGINNING BENCHMARKING")
    controller.reset(X0=jnp.zeros((cfg.N + 1, cfg.n), dtype=jnp.float64), U0=jnp.zeros((cfg.N, cfg.nu), dtype=jnp.float64).at[:, 0::4].set(T_hover))
    start = time.perf_counter()
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u, E_prev, r_centerN = controller.run(
        x0=x0, reference=reference, parameter=parameter
    )
    end = time.perf_counter()
    jax.debug.print("TOTAL_TIME: {}", (end - start))

    # jax.debug.print("{}", r_centerN)
    jax.debug.print(
        "nonzero values: {}",
        r_centerN[r_centerN != 0]
    )

    # plot_plan_xy_with_tubes(
    #     X_pred=X_pred,
    #     x0_quads=x0_quads,
    #     xg_quads=xg_quads,
    #     obstacles=obstacles,
    #     Phi_x=Phi_x,
    #     E_prev=E_prev,
    #     N_QUADS=N_QUADS,
    #     SINGLE_N=SINGLE_N,
    #     filename="multi_quadrotor_plan_tubes.png",
    #     stride=1,        # plot every 5 timesteps (recommended)
    #     box_alpha=0.15,  # transparency of squares
    # )
    # # -----------------------------
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

    # # xs = np.asarray(xs_jax)
    # xs = None
    # disturbed = np.asarray(disturbed_jax)
    # plot_plan_xy_with_tubes(
    #     X_pred=X_pred,
    #     x0_quads=x0_quads,
    #     xg_quads=xg_quads,
    #     obstacles=obstacles,
    #     Phi_x=Phi_x,
    #     r_centerN=r_centerN,
    #     E_prev=E_prev,
    #     N_QUADS=N_QUADS,
    #     SINGLE_N=SINGLE_N,
    #     xs=xs,
    #     filename="multi_quadrotor_plan_tubes_rollouts.png",
    #     stride=1,
    #     box_alpha=0.15,
    #     rollout_alpha=0.6,
    #     rollout_linewidth=1.0,
    # )
    # # disturbed = None
    # if xs is not None:
    #     xs_np = np.asarray(xs)
    # else:
    #     xs_np = None


    # save_multiquad_experiment_npz(
    #     "multi_quad_experiment.npz",
    #     x0=x0,
    #     x_goal=x_goal,
    #     X_ref=X_ref,
    #     X_pred=X_pred,
    #     U_pred=U_pred,
    #     V_pred=V_pred,
    #     backoffs=backoffs,
    #     Phi_x=Phi_x,
    #     Phi_u=Phi_u,
    #     E_prev=E_prev,
    #     r_centerN=r_centerN,
    #     E_sim=E_sim,
    #     xs=xs_np,
    #     disturbed=disturbed,
    #     x0_quads=x0_quads,
    #     xg_quads=xg_quads,
    #     obstacles=obstacles,
    #     dt=dt,
    #     H=H,
    #     N_QUADS=N_QUADS,
    #     SINGLE_N=SINGLE_N,
    #     SINGLE_NU=SINGLE_NU,
    #     u_min=u_min,
    #     u_max=u_max,
    #     notes="8-quad robust MPC experiment with centroid coupling",
    # )

    # tube = get_trajectory_tubes(Phi_x, E_prev)

    # tube_center_shift = jnp.einsum("kjxn,jn->kx", Phi_x, r_centerN)
    # shift = np.asarray(tube_center_shift)                       # (N+1, n)

    # lower_real = X_pred + shift - tube                                 # (N+1, n)
    # upper_real = X_pred + shift + tube  

    # plot_tube_graph_multiquadrotor(
    #     disturbed=disturbed,
    #     X_pred=X_pred,
    #     Phi_x=Phi_x,
    #     r_centerN=r_centerN,
    #     tube=tube,
    #     dt=dt,
    #     N_QUADS=N_QUADS,
    #     SINGLE_N=SINGLE_N,
    #     filename="multi_quadrotor_disturbance_vs_tube_size.png",
    # )
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