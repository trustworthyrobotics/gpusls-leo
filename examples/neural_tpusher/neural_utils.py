import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx

from gpu_sls.external.ReachDev.envs.T_pushing.t_sim import generate_init_target_states
from gpu_sls.external.ReachDev.models.load import load_model
from gpu_sls.external.ReachDev.models.T_pushing.dt_dyn import T_Dynamics
from gpu_sls.external.ReachDev.planning.T_pushing.plan_utils import (
    make_rollout_and_reward_fns,
)
from gpu_sls.external.ReachDev.planning.planner import MPPIPlanner


def load_t_pushing_dt_dyn(dt_dyn_dir: str) -> T_Dynamics:
    return load_model(model_dir=dt_dyn_dir, model_type="dt_dyn", mode="best")


def get_problem_dims(config, dt_dyn: T_Dynamics):
    data_config = config["data"]
    planning_config = config["planning"]

    abs_pose = dt_dyn.abs_pose
    pred_mode = dt_dyn.pred_mode

    scale = float(data_config["scale"])
    state_dim = data_config["state_dim"]
    pose_dim = data_config["pose_dim"]
    action_dim = data_config["action_dim"]
    T_dim = state_dim if pred_mode == "state" else pose_dim

    horizon = planning_config["horizon"]
    max_steps = planning_config["max_steps"] + 1
    n_act_step = planning_config["n_act_step"]
    action_bound = planning_config["action_bound"]

    action_lower_lim = -action_bound * jnp.ones((action_dim,)) / scale
    action_upper_lim = action_bound * jnp.ones((action_dim,)) / scale

    return {
        "abs_pose": abs_pose,
        "pred_mode": pred_mode,
        "scale": scale,
        "state_dim": state_dim,
        "pose_dim": pose_dim,
        "action_dim": action_dim,
        "T_dim": T_dim,
        "horizon": horizon,
        "max_steps": max_steps,
        "n_act_step": n_act_step,
        "action_bound": action_bound,
        "action_lower_lim": action_lower_lim,
        "action_upper_lim": action_upper_lim,
    }


def wrap_dt_dyn_as_dynamics(dt_dyn):
    """
    Wrapper for T_Dynamics -> GenericMPC

    Assumes:
        z == optimizer state = [object_state, pusher_pos]
        u == pusher delta action
    """
    def dynamics(z: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray, *, parameter):
        z_batched = z[None, :]
        u_batched = u[None, :]
        z_next = dt_dyn.forward(z_batched, u_batched)[0]
        return z_next

    return dynamics


def make_tracking_cost(action_weight: float = 1e-2):
    """
    Quadratic tracking cost for GenericMPC refinement around an MPPI nominal.
    Tracks the full optimizer state z against reference[t].
    """
    def cost(W, reference, z, u, t):
        z_ref = reference[t]
        dz = z - z_ref
        return jnp.sum(W * dz**2) + action_weight * jnp.sum(u**2)

    return cost


def shift_warmstart(X: jnp.ndarray, U: jnp.ndarray):
    """
    Standard receding-horizon warmstart shift.
    """
    X_shift = jnp.concatenate([X[1:], X[-1:]], axis=0)
    U_shift = jnp.concatenate([U[1:], U[-1:]], axis=0)
    return X_shift, U_shift


def make_trans_fn(*, scale: float, pred_mode: str, state_dim: int):
    def trans_fn(env_dict):
        pusher_pos = jnp.array(env_dict["pusher_pos"]) / scale
        if pred_mode == "pose":
            obj_state = jnp.array(
                np.concatenate([env_dict["com_pos"] / scale, env_dict["angle"]], axis=0)
            )
            env_state = np.concatenate(
                [np.array(env_dict["com_pos"]), np.array(env_dict["angle"]), env_dict["pusher_pos"]],
                axis=0,
            )
        else:
            obj_state = jnp.array(env_dict["state"][:state_dim]) / scale
            env_state = np.concatenate(
                [env_dict["state"][:state_dim], env_dict["pusher_pos"]], axis=0
            )

        z = jnp.concatenate([obj_state, pusher_pos], axis=0)
        return z, env_state, pusher_pos

    return trans_fn


def make_target_object(
    *,
    pred_mode: str,
    target_pose,
    init_pose,
    scale: float,
    data_config,
):
    if pred_mode == "pose":
        target_obj = jnp.array(target_pose / scale)
        target_obj = target_obj.at[2].set(target_pose[2])
    else:
        _, target_state = generate_init_target_states(
            init_pose,
            target_pose,
            param_dict={
                "stem_size": data_config["stem_size"],
                "bar_size": data_config["bar_size"],
            },
        )
        target_obj = jnp.array(target_state / scale)

    return target_obj


def make_mppi_trajopt(
    *,
    config,
    dt_dyn,
    abs_pose,
    pred_mode,
    action_lower_lim,
    action_upper_lim,
):
    rollout_fn, reward_fn, _, _ = make_rollout_and_reward_fns(
        dt_dyn,
        config,
        abs_pose,
        pred_mode,
    )

    mppi_planner = MPPIPlanner(
        config,
        rollout_fn,
        reward_fn,
        action_lower_lim,
        action_upper_lim,
    )

    return eqx.filter_jit(mppi_planner.trajectory_optimization)


def warmup_mppi_jit(
    *,
    jit_mppi_trajopt,
    seed: int,
    T_dim: int,
    action_dim: int,
    horizon: int,
):
    dummy_z = jnp.zeros((T_dim + action_dim,))
    dummy_u = jnp.zeros((horizon, action_dim))
    dummy_target = jnp.zeros((T_dim,))
    dummy_pusher = jnp.zeros((action_dim,))
    dummy_key = jax.random.PRNGKey(seed)

    jit_mppi_trajopt(
        dummy_key,
        dummy_z,
        dummy_u,
        skip=True,
        target_state=dummy_target,
        pusher_pos=dummy_pusher,
    )
    jit_mppi_trajopt(
        dummy_key,
        dummy_z,
        dummy_u,
        skip=False,
        target_state=dummy_target,
        pusher_pos=dummy_pusher,
    )


def build_mppi_warmstart(
    *,
    z_cur: jnp.ndarray,
    pusher_pos: jnp.ndarray,
    target_obj: jnp.ndarray,
    horizon: int,
    nu: int,
    action_lower_lim: jnp.ndarray,
    action_upper_lim: jnp.ndarray,
    prev_U: jnp.ndarray | None = None,
    jit_mppi_trajopt=None,
    rng_key=None,
):
    """
    Returns:
        X_init: MPPI predicted state sequence
        U_init: MPPI action sequence
        planning_res: raw MPPI planner output
    """
    if jit_mppi_trajopt is None:
        raise ValueError("jit_mppi_trajopt must be provided.")

    if rng_key is None:
        raise ValueError("rng_key must be provided.")

    if prev_U is None:
        init_act_seq = jnp.zeros((horizon, nu))
    else:
        init_act_seq = jnp.concatenate([prev_U[1:], prev_U[-1:]], axis=0)

    planning_res = jit_mppi_trajopt(
        rng_key,
        z_cur,
        init_act_seq,
        skip=False,
        target_state=target_obj,
        pusher_pos=pusher_pos,
    )

    U_mppi = jnp.asarray(planning_res["act_seq"])
    X_mppi = jnp.asarray(planning_res["state_seq"])
    U_mppi = jnp.clip(U_mppi, action_lower_lim, action_upper_lim)

    return X_mppi, U_mppi, planning_res