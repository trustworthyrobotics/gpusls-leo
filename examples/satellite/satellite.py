from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
LINEARIZATION_ERROR = os.path.join(
    ROOT, "src", "gpu_sls", "external", "linearization_error"
)

sys.path.insert(0, LINEARIZATION_ERROR)

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", False)
config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
config.update("jax_persistent_cache_min_compile_time_secs", 0)
config.update("jax_persistent_cache_min_entry_size_bytes", -1)
config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)
import numpy as np

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.utils.constraint_utils import combine_constraints
from gpu_sls.utils.sls_visual import get_trajectory_tubes, plot_tube_graph

NUM_RANDOM = 5
NUM_ADV = 26


# -----------------------------
# Quaternion helpers
# q = [qw, qx, qy, qz]
# -----------------------------
def quat_mul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return jnp.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=a.dtype)


def quat_normalize(q: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    return q / (jnp.linalg.norm(q) + jnp.asarray(eps, dtype=q.dtype))


def euler_to_quat(roll: float, pitch: float, yaw: float, dtype=jnp.float64) -> jnp.ndarray:
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return jnp.array([qw, qx, qy, qz], dtype=dtype)


# -----------------------------
# Rigid-body rotation dynamics
# x = [q(4), w(3)]
# u = [tau_x, tau_y, tau_z]
# -----------------------------
I_BODY = jnp.diag(jnp.array([5.0, 2.0, 1.0], dtype=jnp.float64))
I_BODY_INV = jnp.linalg.inv(I_BODY)


def ode_rigid_body_rotation(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    q = x[:4]
    w = x[4:7]

    qdot = 0.5 * quat_mul(jnp.array([0.0, w[0], w[1], w[2]], dtype=x.dtype), q)
    wdot = I_BODY_INV @ (u - jnp.cross(w, I_BODY @ w))

    return jnp.concatenate([qdot, wdot], axis=0)


def dynamics(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray, *, parameter: Any) -> jnp.ndarray:
    """
    Forward-Euler discretization of the same continuous-time dynamics
    used in RigidBodyRotation.m, instead of RK4.
    """
    dt = parameter
    x_next = x + dt * ode_rigid_body_rotation(x, u)

    # keep quaternion normalized
    q_next = quat_normalize(x_next[:4])
    w_next = x_next[4:7]
    return jnp.concatenate([q_next, w_next], axis=0)


def rigid_body_step_with_disturbance(
    key: jax.Array,
    x: jnp.ndarray,      # (7,)
    u: jnp.ndarray,      # (3,)
    E: jnp.ndarray,      # (7,3)
    dt: float,
    i: int,
) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    """
    x_{k+1} = x_k + dt * f(x_k,u_k) + E w
    with ||w||_2 <= 1.
    Disturbance enters only through angular-velocity states, matching
    the MATLAB model structure.
    """
    x_nom = dynamics(x, u, jnp.asarray(0), parameter=dt)

    key, key_dir, key_rad = jax.random.split(key, 3)
    z = jax.random.normal(key_dir, (3,), dtype=x.dtype)
    z = z / (jnp.linalg.norm(z) + jnp.asarray(1e-12, dtype=x.dtype))

    uu = jax.random.uniform(key_rad, (), dtype=x.dtype)
    r = uu ** (1.0 / 3.0)
    w = r * z

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
    w_real = jnp.array([0.0, 0.0, 0.0, 0.0, w[0], w[1], w[2]])
    x_next = x_nom + E @ w_real
    q_next = quat_normalize(x_next[:4])
    x_next = jnp.concatenate([q_next, x_next[4:7]], axis=0)
    return key, x_next, w_real


def cost(W, reference, x, u, t):
    """
    W = [wq, ww, wu]
    """
    wq, ww, wu = W
    xref = reference[t]

    q = quat_normalize(x[:4])
    qref = quat_normalize(xref[:4])
    w = x[4:7]
    wref = xref[4:7]

    # quaternion alignment cost; sign ambiguity handled with abs(dot)
    d = jnp.dot(q, qref)
    q_align = 1.0 - d * d
    w_err = w - wref

    return wq * q_align + ww * (w_err @ w_err) + wu * (u @ u)


def make_control_box_constraints(
    u_min: jnp.ndarray,
    u_max: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    u_min = jnp.asarray(u_min)
    u_max = jnp.asarray(u_max)

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([u - u_max, u_min - u], axis=0)

    return constraints


def make_state_box_constraints(
    x_min: jnp.ndarray,
    x_max: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    x_min = jnp.asarray(x_min)
    x_max = jnp.asarray(x_max)

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([x - x_max, x_min - x], axis=0)

    return constraints


def slerp(q0, q1, t):
    dot = jnp.dot(q0, q1)

    # handle antipodal case
    q1 = jnp.where(dot < 0, -q1, q1)
    dot = jnp.abs(dot)

    dot = jnp.clip(dot, -1.0, 1.0)
    theta = jnp.arccos(dot)

    def small_angle():
        return (1 - t) * q0 + t * q1

    def normal():
        sin_theta = jnp.sin(theta)
        w0 = jnp.sin((1 - t) * theta) / sin_theta
        w1 = jnp.sin(t * theta) / sin_theta
        return w0 * q0 + w1 * q1

    return jax.lax.cond(theta < 1e-6, small_angle, normal)


def create_reference(x0, x_goal, N, dt):
    q0 = x0[:4]
    qg = x_goal[:4]

    ts = jnp.linspace(0.0, 1.0, N + 1)

    # smooth time scaling (reduces aggressive early motion)
    ts = ts * ts * (3 - 2 * ts)   # cubic smoothstep

    q_ref = jax.vmap(lambda t: quat_normalize(slerp(q0, qg, t)))(ts)

    # finite-difference angular velocity
    w_ref = jnp.zeros((N + 1, 3))
    for k in range(N):
        dq = quat_mul(q_ref[k + 1], jnp.array([q_ref[k][0], -q_ref[k][1], -q_ref[k][2], -q_ref[k][3]]))
        w = 2.0 * dq[1:4] / dt
        w_ref = w_ref.at[k].set(w)

    w_ref = w_ref.at[-1].set(jnp.zeros(3))

    return jnp.concatenate([q_ref, w_ref], axis=1)

def make_constant_disturbance_omega(alpha: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Return E[t] with shape (7,7), diagonal matrix.
    Only angular velocity states (indices 4–6) are disturbed.
    """
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        T = X_prefix.shape[0]

        E0 = jnp.zeros((7, 7), dtype=X_prefix.dtype)
        E0 = E0.at[4, 4].set(alpha)
        E0 = E0.at[5, 5].set(alpha)
        E0 = E0.at[6, 6].set(alpha)

        return jnp.broadcast_to(E0, (T, 7, 7))

    return disturbance


def main():
    n = 7
    nu = 3

    N = 10
    dt = 1.0

    # [quaternion, angular velocity, input]
    W = jnp.array([5.0, 10.0, 0.1], dtype=jnp.float64)

    cfg = MPCConfig(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.zeros((nu,), dtype=jnp.float64),
        dt=dt,
    )

    parameter = dt

    T_max = 0.1
    w_max = 0.1

    u_min = jnp.array([-T_max, -T_max, -T_max], dtype=jnp.float64)
    u_max = jnp.array([ T_max,  T_max,  T_max], dtype=jnp.float64)
    constraints_u = make_control_box_constraints(u_min, u_max)

    # bound only angular velocity like the MATLAB file
    big = 1e6
    x_max = jnp.array([big, big, big, big, w_max, w_max, w_max], dtype=jnp.float64)
    x_min = jnp.array([-big, -big, -big, -big, -w_max, -w_max, -w_max], dtype=jnp.float64)
    constraints_x = make_state_box_constraints(x_min, x_max)

    constraints_all = combine_constraints(constraints_x, constraints_u)

    # no XY obstacles for this system
    obstacles = jnp.zeros((0, 3), dtype=jnp.float64)
    nc = 2 * nu + 2 * n

    # same structure as MATLAB E, but already scaled for discrete Euler step
    E_mag = 0.005
    alpha_sim = E_mag * dt
    disturbance = make_constant_disturbance_omega(alpha=alpha_sim)

    q0 = euler_to_quat(jnp.pi, jnp.pi / 4.0, jnp.pi / 4.0, dtype=jnp.float64)
    w0 = jnp.deg2rad(jnp.array([-1.0, -4.5, 4.5], dtype=jnp.float64))
    x0 = jnp.concatenate([q0, w0], axis=0)

    q_goal = euler_to_quat(0.0, 0.0, 0.0, dtype=jnp.float64)
    w_goal = jnp.zeros((3,), dtype=jnp.float64)
    x_goal = jnp.concatenate([q_goal, w_goal], axis=0)

    reference = jnp.tile(x_goal[None, :], (N + 1, 1))
    # reference = create_reference(x0, x_goal, N, dt)

    key = jax.random.PRNGKey(0)
    E_sim = jnp.zeros((7, 7), dtype=jnp.float64)
    E_sim = E_sim.at[4, 4].set(alpha_sim)
    E_sim = E_sim.at[5, 5].set(alpha_sim)
    E_sim = E_sim.at[6, 6].set(alpha_sim)

    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=1e-1,
        rho_max=2e2,
        max_iterations=30,
        rho_update_frequency=2,
        initial_rho=1e-1,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=2,
        sls_primal_tol=1e-5,
        enable_fastsls=True,
        initialize_nominal=True,
        max_initial_sqp_iterations=100,
        warm_start=False,
        rti=False,
        enable_linearization_bounds=True,
        enable_linearization_gradients=False,
        lambda_rem=0.0
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=50,
        warm_start=False,
        feas_tol=0.01,
        step_tol=0.0001,
        line_search=True,
    )

    Q_bar = jnp.broadcast_to(jnp.eye(n), (N + 1, n, n))
    R_bar = jnp.broadcast_to(jnp.eye(nu), (N, nu, nu))
    disturbance_center = jnp.zeros((N + 1, n))
    X_init = jnp.tile(x0[None, :], (cfg.N + 1, 1))
    U_init = jnp.zeros((cfg.N, cfg.nu), dtype=jnp.float64)
    controller = GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        constraints=constraints_all,
        obstacles=obstacles,
        disturbance_center=disturbance_center,
        cost=cost,
        Q_bar=Q_bar,
        R_bar=R_bar,
        num_constraints=nc,
        disturbance=disturbance,
        shift=1,
        X_in=X_init,
        U_in=U_init,
    )
    
    N_ROLLOUTS = NUM_RANDOM + NUM_ADV
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u, EN, r_centerN = controller.run(
        x0=x0, reference=reference, parameter=parameter
    )
    jax.debug.print("{}", r_centerN)
    xs = np.full((N_ROLLOUTS, N, n), np.nan, dtype=np.float64)
    disturbed = np.full((N_ROLLOUTS, N, n), np.nan, dtype=np.float64)

    for i in range(N_ROLLOUTS):
        print("Rolling out iteration:", i)
        disturbance_history = [jnp.zeros((n,), dtype=jnp.float64)]
        x = x0

        for k in range(N):
            disturbance_feedback = jnp.zeros((nu,), dtype=jnp.float64)
            for j in range(k + 1):
                disturbance_feedback = disturbance_feedback + Phi_u[k, j] @ disturbance_history[j]

            u = U_pred[k] + disturbance_feedback
            key, x, w = rigid_body_step_with_disturbance(key, x, u, E_sim, dt, i)

            disturbed[i, k, :] = np.asarray(x)
            disturbance_history.append(E_sim @ w)
            xs[i, k] = np.asarray(x)

    tube = get_trajectory_tubes(Phi_x, EN)
    tube_center_shift = jnp.einsum("kjxn,jn->kx", Phi_x, r_centerN)
    shift = np.asarray(tube_center_shift)                       # (N+1, n)
    jax.debug.print("{}", tube_center_shift)

    # off-centered reachable tube
    lower = X_pred + shift - tube                                 # (N+1, n)
    upper = X_pred + shift + tube  
    plot_tube_graph(disturbed, lower, upper, dt, output_folder=".")
    print("X_pred shape:", X_pred.shape)
    print("U_pred shape:", U_pred.shape)
    print("tube shape:", tube.shape)
    print("final nominal state:", np.asarray(X_pred[-1]))
    print("final rollout state (first rollout):", xs[0, np.where(~np.isnan(xs[0,:,0]))[0][-1]])
    rollout_save_path = "satellite_rollout_data.npz"
    np.savez_compressed(
        rollout_save_path,
        disturbed=disturbed,
        x0=np.asarray(x0),
        reference=np.asarray(reference),
        U_pred=np.asarray(U_pred),
        X_pred=np.asarray(X_pred),
        Phi_u=np.asarray(Phi_u),
        Phi_x=np.asarray(Phi_x),
        EN=np.asarray(EN),
        r_centerN=np.asarray(r_centerN),
        E_sim=np.asarray(E_sim),
        dt=np.asarray(dt),
        N=np.asarray(N),
        n=np.asarray(n),
        nu=np.asarray(nu),
        NUM_RANDOM=np.asarray(NUM_RANDOM),
        NUM_ADV=np.asarray(NUM_ADV),
        N_ROLLOUTS=np.asarray(N_ROLLOUTS),
        seed=np.asarray(0, dtype=np.int32),
        tube=tube,
    )
    print(f"Saved rollout data to {rollout_save_path}")

if __name__ == "__main__":
    main()