#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulator-in-the-loop MPC for T-pushing.

This file keeps the learned/neural dynamics model inside GenericMPC for prediction,
but uses the external PyMunk simulator package as the true plant in the receding-
horizon loop.

You MUST edit the simulator import section to match your repo.

State convention used here:
    x = [obj_x, obj_y, obj_theta, pusher_x, pusher_y]

Control convention used here:
    u = [d_pusher_x, d_pusher_y]

Simulator convention assumed from your package:
    env.update((uxf, uyf), rel=False, n_sim_time=...) expects an ABSOLUTE target
    pusher position, not a delta.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# =============================================================================
# EDIT THESE IMPORTS TO MATCH YOUR REPO
# =============================================================================
# These are placeholders based on the code structure you showed.
# Replace them with the actual paths/class names from your package.

# --- neural dynamics model ---
from models.dt import T_Dynamics                    # <-- edit if needed
from utils.model_utils import load_model            # <-- edit if needed

# --- MPC / SLS stack ---
from mpx import GenericMPC                          # <-- edit if needed
from mpx.config import ADMMConfig, SLSConfig, SQPConfig  # <-- edit if needed
from mpx.utils.fast_sls_visual import get_trajectory_tubes  # <-- edit if needed

# --- visualization helpers ---
from utils.visualize import (                       # <-- edit if needed
    animate_rollouts_t_shape,
    animate_tube_t_shape,
    plot_disturbance_vs_tube,
)

# --- simulator package ---
# Replace this with your actual simulator class for the T-shape pushing task.
# Example possibilities:
# from tasks.sim.t_sim import TSim
# from sim.t_pushing import T_Sim
# from simulator.t_sim import T_Sim
from simulator.t_sim import T_Sim                  # <-- THIS IS THE MAIN ONE TO EDIT


# =============================================================================
# User / experiment config
# =============================================================================

MODEL_DIR = "/path/to/your/model_dir"   # <-- edit
OUT_DIR = "./sim_mpc_out"

# horizon / timing
N = 25
SIM_STEPS = 80
DT = 1.0
FPS = 10

# state and control sizes
Ds = 3   # object pose: x, y, theta
Du = 2   # pusher delta: dx_pusher, dy_pusher
Dx = 5   # [obj_x, obj_y, obj_theta, pusher_x, pusher_y]

STATE_LABELS = ["obj_x", "obj_y", "obj_theta", "pusher_x", "pusher_y"]

# initial / goal states
X0 = np.array([110.0, 110.0, 0.0, 75.0, 110.0], dtype=np.float64)
X_GOAL = np.array([200.0, 180.0, 0.0, 165.0, 180.0], dtype=np.float64)

CONTACT_DIST = 20.0

# simple bounds (edit to match your setup)
OBJ_X_MIN, OBJ_X_MAX = 0.0, 300.0
OBJ_Y_MIN, OBJ_Y_MAX = 0.0, 300.0
THETA_MIN, THETA_MAX = -np.pi, np.pi
PUSHER_X_MIN, PUSHER_X_MAX = 0.0, 300.0
PUSHER_Y_MIN, PUSHER_Y_MAX = 0.0, 300.0
DU_MIN = np.array([-20.0, -20.0], dtype=np.float64)
DU_MAX = np.array([20.0, 20.0], dtype=np.float64)

# simulator config
WINDOW_SIZE = 300
PUSHER_SIZE = 5
SAVE_IMG = False
ENABLE_VIS = False

# disturbance injection in simulator coordinates (optional)
USE_SIM_DISTURBANCE = False
SIM_DISTURBANCE = np.array([0.0, 0.0, 0.0], dtype=np.float64)


# =============================================================================
# Neural dynamics wrapper
# =============================================================================

def make_neural_dynamics(model: T_Dynamics):
    def dynamics(x: jnp.ndarray, u: jnp.ndarray, t, *, parameter=None) -> jnp.ndarray:
        dtype = x.dtype
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
    w_ctrl = W[Dx:]
    xref = reference[t]
    dx = x - xref
    return jnp.dot(w_state * dx, dx) + jnp.dot(w_ctrl * u, u)


# =============================================================================
# Reference trajectory
# =============================================================================

def build_pushing_reference(x0, x_goal, n, contact_dist=20.0):
    t = jnp.linspace(0.0, 1.0, n + 1)
    obj_pos = (1.0 - t[:, None]) * x0[:2] + t[:, None] * x_goal[:2]
    obj_theta = x0[2] + t * (x_goal[2] - x0[2])

    push_dir = x_goal[:2] - x0[:2]
    push_unit = push_dir / (jnp.linalg.norm(push_dir) + 1e-8)
    pusher = obj_pos - contact_dist * push_unit[None, :]

    return jnp.concatenate([obj_pos, obj_theta[:, None], pusher], axis=1)


def slice_reference(X_ref_full: jnp.ndarray, sim_k: int, N: int) -> jnp.ndarray:
    T_full = X_ref_full.shape[0]
    indices = jnp.arange(sim_k, sim_k + N + 1).clip(0, T_full - 1)
    return X_ref_full[indices]


# =============================================================================
# Constraints / disturbance
# =============================================================================

def combine_constraints() -> Tuple[Any, int]:
    """
    Replace this with your repo's actual constraint builder if you already have one.
    This local version creates simple box constraints on x and u.
    """
    # Expected GenericMPC signature in your code uses `constraints=` and `num_constraints=`.
    # If your package already has a helper for this, use that instead.
    #
    # Here we define a callable returning:
    #   c(x, u, t) <= 0
    #
    # State box:
    #   x_i - upper <= 0
    #   lower - x_i <= 0
    #
    # Input box:
    #   u_i - upper <= 0
    #   lower - u_i <= 0

    x_lo = jnp.array(
        [OBJ_X_MIN, OBJ_Y_MIN, THETA_MIN, PUSHER_X_MIN, PUSHER_Y_MIN],
        dtype=jnp.float64,
    )
    x_hi = jnp.array(
        [OBJ_X_MAX, OBJ_Y_MAX, THETA_MAX, PUSHER_X_MAX, PUSHER_Y_MAX],
        dtype=jnp.float64,
    )
    u_lo = jnp.array(DU_MIN, dtype=jnp.float64)
    u_hi = jnp.array(DU_MAX, dtype=jnp.float64)

    def constraints_all(x, u, t, parameter=None):
        return jnp.concatenate([
            x - x_hi,
            x_lo - x,
            u - u_hi,
            u_lo - u,
        ])

    nc = 2 * Dx + 2 * Du
    return constraints_all, nc


def disturbance(t, parameter=None):
    return jnp.zeros((Dx,), dtype=jnp.float64)


# =============================================================================
# Dynamic feasibility check
# =============================================================================

def check_dynamic_feasibility(X_pred, U_pred, dynamics_fn):
    tol = 1e-3
    residuals = np.zeros((N, Dx), dtype=np.float64)
    norms = np.zeros(N, dtype=np.float64)

    for k in range(N):
        x_k = jnp.array(X_pred[k], dtype=jnp.float64)
        u_k = jnp.array(U_pred[k], dtype=jnp.float64)
        x_next = np.asarray(dynamics_fn(x_k, u_k, k, parameter=DT))
        residuals[k] = np.asarray(X_pred[k + 1]) - x_next
        norms[k] = float(np.linalg.norm(residuals[k]))

    max_norm = float(norms.max())
    mean_norm = float(norms.mean())
    feasible = max_norm < tol

    print("\n=== Dynamic Feasibility Check ===")
    print(f"  Horizon N      : {N}")
    print(f"  Tolerance      : {tol:.1e}")
    print(f"  Max  residual  : {max_norm:.4e}")
    print(f"  Mean residual  : {mean_norm:.4e}")
    print(f"  Worst step     : k={int(norms.argmax())}  ||r||={max_norm:.4e}")
    print(
        "  Per-state max  : "
        + "  ".join(
            f"{STATE_LABELS[i]}={np.abs(residuals[:, i]).max():.3e}"
            for i in range(Dx)
        )
    )
    print("=================================\n")
    return residuals, norms, max_norm, feasible


# =============================================================================
# Simulator adapter
# =============================================================================

def make_sim_param_dict() -> Dict[str, Any]:
    return {
        "save_img": SAVE_IMG,
        "enable_vis": ENABLE_VIS,
        "window_size": WINDOW_SIZE,
        "pusher_size": PUSHER_SIZE,
        # add any task-specific keys your simulator class requires here
        # e.g. stem_size, bar_size, obs_pos_list, obs_size_list, obs_norm, ...
    }


def make_sim_env() -> Any:
    """
    Instantiate the simulator from the external package.

    You may need to pass additional task-specific constructor args depending
    on your T_Sim class.
    """
    param_dict = make_sim_param_dict()
    env = T_Sim(param_dict, step_dt=1.0 / 60.0)
    return env


def env_to_mpc_state(env_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert simulator env state to the 5D MPC state:
        [obj_x, obj_y, obj_theta, pusher_x, pusher_y]
    """
    com = np.asarray(env_dict["com_pos"], dtype=np.float64).reshape(-1)
    ang = np.asarray(env_dict["angle"], dtype=np.float64).reshape(-1)
    pusher = np.asarray(env_dict["pusher_pos"], dtype=np.float64).reshape(-1)

    if com.size < 2:
        raise ValueError(f"Simulator returned invalid com_pos: shape={com.shape}")
    if ang.size < 1:
        raise ValueError(f"Simulator returned invalid angle: shape={ang.shape}")
    if pusher.size < 2:
        raise ValueError(f"Simulator returned invalid pusher_pos: shape={pusher.shape}")

    x = np.array([com[0], com[1], ang[0], pusher[0], pusher[1]], dtype=np.float64)
    return x


def reset_sim_to_state(env: Any, x0: np.ndarray) -> np.ndarray:
    """
    Initialize the simulator world from the MPC initial state.

    Assumes:
      - one pushed object
      - init_poses expects [[obj_x, obj_y, obj_theta]]
      - pusher_pos expects [pusher_x, pusher_y]
    """
    obj_pose = np.array([x0[0], x0[1], x0[2]], dtype=np.float64)
    pusher_pos = np.array([x0[3], x0[4]], dtype=np.float64)

    env.create_world(
        init_poses=[obj_pose],
        pusher_pos=pusher_pos,
    )

    env_dict = env.get_env_state(rel=False)
    return env_to_mpc_state(env_dict)


def simulator_step(env: Any, x_cur: np.ndarray, u_apply: np.ndarray, dt: float) -> np.ndarray:
    """
    True plant step through the simulator package.

    MPC control convention:
        u_apply = [d_pusher_x, d_pusher_y]

    Simulator convention:
        env.update((uxf, uyf), rel=False, n_sim_time=...) expects ABSOLUTE target
        pusher coordinates.

    So we convert:
        target_pusher = current_pusher + u_apply
    """
    x_cur = np.asarray(x_cur, dtype=np.float64).reshape(-1)
    u_apply = np.asarray(u_apply, dtype=np.float64).reshape(-1)

    if x_cur.shape[0] != Dx:
        raise ValueError(f"x_cur must have shape ({Dx},), got {x_cur.shape}")
    if u_apply.shape[0] != Du:
        raise ValueError(f"u_apply must have shape ({Du},), got {u_apply.shape}")

    pusher_cur = x_cur[3:5]
    target_pusher = pusher_cur + u_apply

    env_dict = env.update(
        (float(target_pusher[0]), float(target_pusher[1])),
        rel=False,
        n_sim_time=dt,
    )

    if env_dict is None:
        raise RuntimeError("Simulator update() returned None; pusher may not be initialized.")

    if USE_SIM_DISTURBANCE:
        env.force_update([SIM_DISTURBANCE.astype(np.float64)])

    x_next = env_to_mpc_state(env_dict)
    return x_next


# =============================================================================
# Main — receding horizon MPC loop
# =============================================================================

def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {MODEL_DIR}")
    model = load_model(MODEL_DIR)
    dynamics_fn = make_neural_dynamics(model)

    x0 = jnp.array(X0, dtype=jnp.float64)
    x_goal = jnp.array(X_GOAL, dtype=jnp.float64)

    print(f"x0:     {np.array(x0)}")
    print(f"x_goal: {np.array(x_goal)}")

    X_ref_full = build_pushing_reference(
        x0, x_goal, SIM_STEPS + N, contact_dist=CONTACT_DIST
    )

    constraints_all, nc = combine_constraints()

    # Replace these weight sizes if your cost uses a different W layout.
    cfg = jnp.array(
        [5.0, 5.0, 0.5, 1.0, 1.0, 0.05, 0.05], dtype=jnp.float64
    )

    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=1e-2,
        rho_max=1e6,
        max_iterations=400,
        rho_update_frequency=20,
        initial_rho=1.0,
    )
    sls_cfg = SLSConfig(
        max_sls_iterations=100,
        sls_primal_tol=1e-2,
        enable_fastsls=False,
        initialize_nominal=False,
        max_initial_sqp_iterations=30,
        warm_start=False,
        rti=False,
        enable_linearization_bounds=False,
    )
    sqp_cfg = SQPConfig(
        max_sqp_iterations=100,
        warm_start=False,
        feas_tol=0.01,
        step_tol=0.01,
        line_search=True,
    )

    Q_bar = jnp.broadcast_to(jnp.eye(Dx), (N + 1, Dx, Dx))
    R_bar = jnp.broadcast_to(jnp.eye(Du), (N, Du, Du))

    X_warm = jnp.zeros((N + 1, Dx), dtype=jnp.float64)
    U_warm = jnp.zeros((N, Du), dtype=jnp.float64)

    controller = GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics_fn,
        constraints=constraints_all,
        obstacles=jnp.zeros((0, 3), dtype=jnp.float64),
        cost=pushing_cost,
        Q_bar=Q_bar,
        R_bar=R_bar,
        num_constraints=nc,
        disturbance=disturbance,
        shift=1,
        X_in=X_warm,
        U_in=U_warm,
    )

    # -------------------------------------------------------------------------
    # Build simulator and initialize from x0
    # -------------------------------------------------------------------------
    env = make_sim_env()
    x_cur_np = reset_sim_to_state(env, np.asarray(x0))
    x_cur = jnp.array(x_cur_np, dtype=jnp.float64)

    # -------------------------------------------------------------------------
    # Closed-loop storage
    # -------------------------------------------------------------------------
    cl_states = np.full((SIM_STEPS + 1, Dx), np.nan, dtype=np.float64)
    cl_inputs = np.full((SIM_STEPS, Du), np.nan, dtype=np.float64)
    cl_tubes = np.full((SIM_STEPS + 1, Dx), np.nan, dtype=np.float64)

    cl_states[0] = np.asarray(x_cur)

    last_X_pred = None
    last_Phi_x = None
    last_E_prev = None

    # -------------------------------------------------------------------------
    # Receding-horizon MPC loop
    # -------------------------------------------------------------------------
    for sim_k in range(SIM_STEPS):
        print(f"\n{'=' * 60}")
        print(f"  MPC step {sim_k + 1:3d} / {SIM_STEPS}")
        print(f"  Current state: {np.array(x_cur)}")

        X_ref_k = slice_reference(X_ref_full, sim_k, N)

        u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u, E_prev = controller.run(
            x0=x_cur,
            reference=X_ref_k,
            parameter=DT,
        )

        last_X_pred = np.asarray(X_pred)
        last_Phi_x = Phi_x
        last_E_prev = E_prev

        tube_k = get_trajectory_tubes(Phi_x, E_prev)
        cl_tubes[sim_k] = np.asarray(tube_k[0])

        u_apply = np.asarray(u0, dtype=np.float64)
        cl_inputs[sim_k] = u_apply

        print(f"  Applying u = {u_apply}")

        # ---------------------------------------------------------------------
        # TRUE PLANT STEP: simulator, not neural dynamics
        # ---------------------------------------------------------------------
        x_next_np = simulator_step(env, np.asarray(x_cur), u_apply, DT)
        cl_states[sim_k + 1] = x_next_np
        print(f"  Next state:   {x_next_np}")

        x_cur = jnp.array(x_next_np, dtype=jnp.float64)

    cl_tubes[SIM_STEPS] = cl_tubes[SIM_STEPS - 1]

    print(f"\n{'=' * 60}")
    print("Closed-loop MPC simulation complete.")
    print(f"  Final state : {cl_states[SIM_STEPS]}")
    print(f"  Goal state  : {np.array(x_goal)}")
    print(f"  Final error : {np.abs(cl_states[SIM_STEPS] - np.array(x_goal))}")

    # -------------------------------------------------------------------------
    # Tracking error vs tube
    # -------------------------------------------------------------------------
    X_ref_np = np.asarray(X_ref_full[: SIM_STEPS + 1])
    disturbed = np.zeros((1, SIM_STEPS, Dx), dtype=np.float64)
    for k in range(SIM_STEPS):
        disturbed[0, k] = np.abs(X_ref_np[k + 1] - cl_states[k + 1])

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    xs_for_vis = cl_states[np.newaxis, : SIM_STEPS + 1, :]

    animate_rollouts_t_shape(
        xs=xs_for_vis,
        x0=np.asarray(x0),
        x_goal=np.asarray(x_goal),
        fps=FPS,
        filename=str(out_dir / "mpc_rollout_vis.gif"),
        dt=DT,
    )

    if last_X_pred is None or last_Phi_x is None or last_E_prev is None:
        raise RuntimeError("No MPC solve outputs were stored; cannot render final tube.")

    tube_final = get_trajectory_tubes(last_Phi_x, last_E_prev)

    animate_tube_t_shape(
        X_pred=last_X_pred,
        tube=np.asarray(tube_final),
        x_goal=np.asarray(x_goal),
        fps=FPS,
        filename=str(out_dir / "mpc_tube_vis.gif"),
        dt=DT,
    )

    plot_disturbance_vs_tube(
        disturbed=disturbed,
        tube=np.asarray(cl_tubes[:SIM_STEPS]),
        dt=DT,
        filename=str(out_dir / "mpc_tracking_error.png"),
        state_labels=STATE_LABELS,
    )

    print("\nAll done!")


if __name__ == "__main__":
    main()