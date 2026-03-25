"""
T-Pushing Rollout Experiment  (GenericMPC / SLS+SQP+ADMM solver)
=================================================================
Mirrors the quadrotor experiment exactly:
  - GenericMPC with SLSConfig / SQPConfig / ADMMConfig
  - Disturbance feedback  u_k = U_pred[k] + sum_j Phi_u[k,j] @ w_j
  - Robust tube from  get_trajectory_tubes(Phi_x, E_prev)

Visualization is handled by visualize_experiment.py.

Edit the CONFIG block below, then run:
    python visualize_t_pushing.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# =============================================================================
# JAX config
# =============================================================================
from jax import config
config.update("jax_enable_x64", False)
config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
config.update("jax_persistent_cache_min_compile_time_secs", 0)
config.update("jax_persistent_cache_min_entry_size_bytes", -1)
config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)

# =============================================================================
# CONFIG
# =============================================================================

MODEL_DIR = (
    "/home/jeff/trustworthroboticsgroup/gpu_sls/src/gpu_sls/external/"
    "linearization_sls/models/certified"
)

# [obj_x, obj_y, obj_theta, pusher_x, pusher_y]
X0     = [1.8707293, 2.482089, 1.7709253, 2.12, 2.33]
X_GOAL = [
    2.5, 3.0, 1.7709253,
    2.12 + (2.5  - 1.8707293),
    2.33 + (3.0  - 2.482089),
]

N            = 60    # planning horizon (steps)
DT           = 0.1   # seconds per step
N_ROLLOUTS   = 30    # number of noisy rollouts  (rollout 0 = noise-free)
# NOISE_STD = 1.0 means ||w||~1, so per-step disturbance = E_MAG*DT,
# matching what the SLS tube is sized for.
NOISE_STD    = 1.0
SEED         = 0
U_MAX        = 0.30  # max pusher displacement  [m/step]
CONTACT_DIST = 0.05
OUT_DIR      = "figures"
E_MAG        = 0.01  # disturbance magnitude for SLS tube

# T-shape geometry — in the same units as X0/X_GOAL.
STEM_W      = 0.06
STEM_H      = 0.18
BAR_W       = 0.24
BAR_H       = 0.06
PUSHER_R    = 0.012
WINDOW_HALF = 1.0
FPS         = 8

# =============================================================================

# -- solver imports -----------------------------------------------------------
from gpu_sls.external.linearization_sls.neural.load import load_model
from gpu_sls.external.linearization_sls.neural.dt_dyn import T_Dynamics

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls  import SLSConfig
from gpu_sls.gpu_sqp  import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.utils.constraint_utils import (
    make_state_box_constraints,
    make_constant_disturbance,
)
from gpu_sls.utils.sls_visual import get_trajectory_tubes

# -- visualization ------------------------------------------------------------
from visualize_experiment import (
    animate_rollouts_t_shape,
    animate_tube_t_shape,
    plot_disturbance_vs_tube,
)

# -- fixed dims ---------------------------------------------------------------
Ds = 3        # object pose  (obj_x, obj_y, obj_theta)
Du = 2        # action       (d_pusher_x, d_pusher_y)
Dx = Ds + Du  # 5

STATE_LABELS = ("obj x", "obj y", "obj theta", "pusher x", "pusher y")


# =============================================================================
# Neural dynamics wrapper
# =============================================================================

def make_neural_dynamics(model: T_Dynamics):
    def dynamics(x: jnp.ndarray, u: jnp.ndarray, t, *, parameter=None) -> jnp.ndarray:
        dtype  = x.dtype
        x_next = model.forward_batchless(
            x.astype(jnp.float32), u.astype(jnp.float32)
        )
        return x_next.astype(dtype)
    return dynamics


# =============================================================================
# Cost function
# =============================================================================

def pushing_cost(W, reference, x, u, t):
    w_state = W[:Dx]
    w_ctrl  = W[Dx:]
    xref    = reference[t]
    dx      = x - xref
    return jnp.dot(w_state * dx, dx) + jnp.dot(w_ctrl * u, u)


# =============================================================================
# Reference trajectory
# =============================================================================

def build_pushing_reference(x0, x_goal, n, contact_dist=0.05):
    t         = jnp.linspace(0.0, 1.0, n + 1)
    obj_pos   = (1.0 - t[:, None]) * x0[:2] + t[:, None] * x_goal[:2]
    obj_theta = x0[2] + t * (x_goal[2] - x0[2])
    push_dir  = x_goal[:2] - x0[:2]
    push_unit = push_dir / (jnp.linalg.norm(push_dir) + 1e-8)
    pusher    = obj_pos - contact_dist * push_unit[None, :]
    return jnp.concatenate([obj_pos, obj_theta[:, None], pusher], axis=1)


# =============================================================================
# Rollout with disturbance feedback
# =============================================================================

def run_rollouts(x0, U_pred, Phi_u, E_sim, dynamics_fn,
                 n_rollouts, noise_std, u_min, u_max, key):
    n_steps = U_pred.shape[0]
    xs = np.full((n_rollouts, n_steps, Dx), np.nan, dtype=np.float64)

    for i in range(n_rollouts):
        disturbance_history = [jnp.zeros((Dx,), dtype=jnp.float64)]
        x = x0
        jax.debug.print("Rolling out iteration {}", i)

        for k in range(n_steps):
            # dist_fb = jnp.zeros((Du,), dtype=jnp.float64)
            # for j in range(k + 1):
            #     dist_fb = dist_fb + Phi_u[k, j] @ disturbance_history[j]

            u = U_pred[k]

            # if i == 0:
            #     w = jnp.zeros((Dx,), dtype=jnp.float64)
            # else:
            #     key, subkey = jax.random.split(key)
            #     w = jax.random.normal(subkey, (Dx,), dtype=jnp.float64) * noise_std
            x_next = dynamics_fn(x, u, k)
            # x_next = dynamics_fn(x, u, k) + E_sim @ w
            # disturbance_history.append(E_sim @ w)
            xs[i, k] = np.asarray(x_next)
            x = x_next

    return xs



# =============================================================================
# Dynamic feasibility check
# =============================================================================

def check_dynamic_feasibility(X_pred, U_pred, dynamics_fn, dt):
    """
    Verify that the planned trajectory is dynamically consistent:
        f(X_pred[k], U_pred[k]) ≈ X_pred[k+1]  for all k in [0, N-1]

    Parameters
    ----------
    X_pred      : (N+1, Dx)  planned state trajectory
    U_pred      : (N,   Du)  planned control sequence
    dynamics_fn : callable   dynamics(x, u, t, *, parameter=None) -> x_next
    dt          : float      timestep passed as `parameter` to dynamics_fn

    Returns
    -------
    residuals : (N, Dx)  signed one-step prediction errors
    norms     : (N,)     per-step residual 2-norm
    max_norm  : float    worst-case residual across the horizon
    feasible  : bool     True if max_norm < 1e-3
    """
    N   = U_pred.shape[0]
    tol = 1e-3

    residuals = np.zeros((N, Dx), dtype=np.float64)
    norms     = np.zeros(N,       dtype=np.float64)

    for k in range(N):
        x_k    = jnp.array(X_pred[k], dtype=jnp.float64)
        u_k    = jnp.array(U_pred[k], dtype=jnp.float64)
        x_next = np.asarray(dynamics_fn(x_k, u_k, k, parameter=dt))
        residuals[k] = np.asarray(X_pred[k + 1]) - x_next
        norms[k]     = float(np.linalg.norm(residuals[k]))

    max_norm  = float(norms.max())
    mean_norm = float(norms.mean())
    feasible  = max_norm < tol

    print("\n=== Dynamic Feasibility Check ===")
    print(f"  Horizon N      : {N}")
    print(f"  Tolerance      : {tol:.1e}")
    print(f"  Max  residual  : {max_norm:.4e}")
    print(f"  Mean residual  : {mean_norm:.4e}")
    print(f"  Worst step     : k={int(norms.argmax())}  ||r||={max_norm:.4e}")
    print(f"  Per-state max  : " +
          "  ".join(f"{STATE_LABELS[i]}={np.abs(residuals[:, i]).max():.3e}"
                    for i in range(Dx)))
    print("=================================\n")

    return residuals, norms, max_norm, feasible


# =============================================================================
# Main
# =============================================================================

def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {MODEL_DIR}")
    model       = load_model(MODEL_DIR)
    dynamics_fn = make_neural_dynamics(model)

    x0     = jnp.array(X0,     dtype=jnp.float64)
    x_goal = jnp.array(X_GOAL, dtype=jnp.float64)
    print(f"x0:     {np.array(x0)}")
    print(f"x_goal: {np.array(x_goal)}")

    X_ref = build_pushing_reference(x0, x_goal, N, contact_dist=CONTACT_DIST)

    # -------------------------------------------------------------------------
    # MPC setup
    # -------------------------------------------------------------------------
    W = jnp.array([
        20.0, 20.0, 5.0,  # obj position + orientation
        5.0,  5.0,        # pusher position
        0.1,  0.1,        # control effort
    ], dtype=jnp.float64)

    cfg = MPCConfig(
        n=Dx, nu=Du, N=N, W=W,
        u_ref=jnp.zeros(Du, dtype=jnp.float64),
        dt=DT,
    )

    u_min = jnp.full((Du,), -U_MAX, dtype=jnp.float64)
    u_max = jnp.full((Du,),  U_MAX, dtype=jnp.float64)

    x_max = jnp.array([100.0, 100.0, 4*jnp.pi, 100.0, 100.0], dtype=jnp.float64)
    constraints_all = make_state_box_constraints(-x_max, x_max)
    nc = 2 * Dx

    alpha_sim   = E_MAG * DT
    disturbance = make_constant_disturbance(n=Dx, alpha=alpha_sim)
    E_sim       = alpha_sim * jnp.eye(Dx, dtype=jnp.float64)

    admm_cfg = ADMMConfig(
        eps_abs=1e-2, eps_rel=1e-2, rho_max=1e6,
        max_iterations=400, rho_update_frequency=20, initial_rho=1.0,
    )
    sls_cfg = SLSConfig(
        max_sls_iterations=100, sls_primal_tol=1e-2,
        enable_fastsls=False, initialize_nominal=False,
        max_initial_sqp_iterations=30, warm_start=False, rti=False,
        # Must be False: immrax natif cannot handle custom_jvp in neural MLP
        enable_linearization_bounds=False,
    )
    sqp_cfg = SQPConfig(
        max_sqp_iterations=500, warm_start=False,
        feas_tol=0.01, step_tol=0.01, line_search=True,
    )

    Q_bar = jnp.broadcast_to(jnp.eye(Dx), (N + 1, Dx, Dx))
    R_bar = jnp.broadcast_to(jnp.eye(Du), (N,     Du, Du))

    controller = GenericMPC(
        sls_cfg, sqp_cfg, admm_cfg,
        config=cfg,
        dynamics=dynamics_fn,
        constraints=constraints_all,
        obstacles=jnp.zeros((0, 3), dtype=jnp.float64),
        cost=pushing_cost,
        Q_bar=Q_bar, R_bar=R_bar,
        num_constraints=nc,
        disturbance=disturbance,
        shift=1,
        X_in=jnp.zeros((N + 1, Dx), dtype=jnp.float64),
        U_in=jnp.zeros((N, Du),     dtype=jnp.float64),
    )

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    print("Running GenericMPC solver ...")
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u, E_prev = controller.run(
        x0=x0, reference=X_ref, parameter=DT
    )
    print("Solver done.")

    # -------------------------------------------------------------------------
    # Dynamic feasibility check
    # -------------------------------------------------------------------------
    check_dynamic_feasibility(
        X_pred=np.asarray(X_pred),
        U_pred=np.asarray(U_pred),
        dynamics_fn=dynamics_fn,
        dt=DT,
    )

    tube = get_trajectory_tubes(Phi_x, E_prev)   # (N+1, Dx)

    # -------------------------------------------------------------------------
    # Rollouts
    # -------------------------------------------------------------------------
    key = jax.random.PRNGKey(SEED)
    print(f"Running {N_ROLLOUTS} rollouts ...")
    xs = run_rollouts(
        x0=x0, U_pred=U_pred, Phi_u=Phi_u, E_sim=E_sim,
        dynamics_fn=dynamics_fn, n_rollouts=N_ROLLOUTS,
        noise_std=NOISE_STD, u_min=u_min, u_max=u_max, key=key,
    )
    print(f"xs shape: {xs.shape}")

    # Tracking error vs tube
    disturbed = np.zeros_like(xs)
    for i in range(N_ROLLOUTS):
        for k in range(N):
            disturbed[i, k] = np.abs(np.asarray(X_pred[k + 1]) - xs[i, k])

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    vis_kwargs = dict(
        dt          = DT,
        window_half = WINDOW_HALF,
        pusher_r    = PUSHER_R,
        stem_w      = STEM_W,
        stem_h      = STEM_H,
        bar_w       = BAR_W,
        bar_h       = BAR_H,
    )

    animate_rollouts_t_shape(
        xs=xs,
        x0=np.asarray(x0),
        x_goal=np.asarray(x_goal),
        fps=FPS,
        filename=str(out_dir / "rollouts_vis.gif"),
    )

    animate_tube_t_shape(
        X_pred=np.asarray(X_pred),
        tube=np.asarray(tube),
        x_goal=np.asarray(x_goal),
        fps=FPS,
        filename=str(out_dir / "tube_vis.gif"),
        **vis_kwargs,
    )

    plot_disturbance_vs_tube(
        disturbed=disturbed,
        tube=np.asarray(tube),
        dt=DT,
        filename=str(out_dir / "t_pushing_disturbance_vs_tube.png"),
        state_labels=STATE_LABELS,
    )

    print("\nAll done!")


if __name__ == "__main__":
    main()