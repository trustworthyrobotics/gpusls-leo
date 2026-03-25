import os
import time
import pickle
import datetime
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import jax
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx

from gpu_sls.external.ReachDev.envs.T_pushing.t_sim import generate_init_target_states, T_Sim
from gpu_sls.external.ReachDev.models.load import load_model
from gpu_sls.external.ReachDev.models.T_pushing.dt_dyn import T_Dynamics
from gpu_sls.external.ReachDev.planning.T_pushing.plan_utils import (
    generate_test_cases,
    get_abs_states,
    plot_cost_stat,
    plot_plan_from_poses,
    make_rollout_and_reward_fns,
)
from gpu_sls.external.ReachDev.planning.planner import MPPIPlanner

from gpu_sls.utils.constraint_utils import (
    make_state_box_constraints,
    make_constant_disturbance,
)

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
Y_MIN = 0.65

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


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="T_pushing.yaml")
def main(config: DictConfig):
    if "testing" in config:
        testing_config = config["testing"]
        mode = testing_config.get("mode", "certified")
        assert mode in {"certified", "regular"}
        with open_dict(config):
            config["test_models"] = testing_config[mode]

    data_config = config["data"]
    planning_config = config["planning"]
    # seed = config["settings"]["seed"]
    seed = 42

    dt_dyn_dir = config["test_models"]["dt_dyn_dir"]
    dt_dyn: T_Dynamics = load_model(model_dir=dt_dyn_dir, model_type="dt_dyn", mode="best")

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

    param_dict = {
        "stem_size": data_config["stem_size"],
        "bar_size": data_config["bar_size"],
        "pusher_size": data_config["pusher_size"],
        "save_img": True,
        "enable_vis": False,
        "window_size": data_config["window_size"],
        "show_pose_center": True,
        "y_constraint": Y_MIN,          # the actual constraint (world units)
        "constraint_scale": scale,    # converts world → pixels
        "constraint_color": (0, 0, 255, 255),  # red line
    }

    out_dir = "visualizations"
    os.makedirs(out_dir, exist_ok=True)

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
            env_state = np.concatenate([env_dict["state"][:state_dim], env_dict["pusher_pos"]], axis=0)

        z = jnp.concatenate([obj_state, pusher_pos], axis=0)
        return z, env_state, pusher_pos

    n = T_dim + action_dim
    nu = action_dim
    dt = 1.0

    dynamics = wrap_dt_dyn_as_dynamics(dt_dyn)
    cost = make_tracking_cost()

    W = jnp.concatenate([
        10.0 * jnp.ones((T_dim,)),
        0.1 * jnp.ones((action_dim,)),
    ])

    cfg = MPCConfig(
        n=n,
        nu=nu,
        N=horizon,
        W=W,
        u_ref=jnp.zeros((nu,)),
        dt=dt,
    )

    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=0,
        rho_max=1e3,
        max_iterations=400,
        rho_update_frequency=20,
        initial_rho=1.0,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=2,
        sls_primal_tol=1e-2,
        enable_fastsls=False,
        initialize_nominal=True,
        warm_start=False,
        rti=False,
        enable_linearization_bounds=False,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=100,
        warm_start=False,
        feas_tol=1e-2,
        step_tol=1e-4,
        line_search=True,
    )
    x_max = jnp.array([1000.0, 1000.0, 4 * jnp.pi, 1000.0, 1000.0], dtype=jnp.float64)
    x_min = -x_max
    # x_min = x_min.at[1].set(Y_MIN)   # enforce object y >= 0.5
    constraints_all = make_state_box_constraints(x_min, x_max)
    constraints = constraints_all
    # obstacles = jnp.array([
    #     [3.5, 1.8, 0.6],   # center obstacle
    # ], dtype=jnp.float64)
    obstacles = jnp.zeros((0,3 ))

    num_constraints = 10 + obstacles.shape[0]

    E_mag = 0.01
    alpha_sim = E_mag * dt
    disturbance = make_constant_disturbance(n=n, alpha=alpha_sim)

    controller = GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        constraints=constraints,
        obstacles=obstacles,
        cost=cost,
        Q_bar=jnp.broadcast_to(jnp.eye(n), (horizon + 1, n, n)),
        R_bar=jnp.broadcast_to(jnp.eye(nu), (horizon, nu, nu)),
        num_constraints=num_constraints,
        disturbance=disturbance,
        shift=1,
        X_in=jnp.zeros((horizon + 1, n)),
        U_in=jnp.zeros((horizon, nu)),
    )

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

    jit_mppi_trajopt = eqx.filter_jit(mppi_planner.trajectory_optimization)

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

    num_test = planning_config["num_test"]
    test_id = planning_config.get("test_id", 0)
    num_test = 1
    init_pusher_pos_list, init_pose_list, target_pose_list = generate_test_cases(
        seed, num_test, test_id=test_id
    )

    init_pusher_pos = init_pusher_pos_list[0]
    init_pose = init_pose_list[0]
    target_pose = target_pose_list[0]

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

    env = T_Sim(
        param_dict=param_dict,
        init_poses=[init_pose],
        target_poses=[target_pose],
        pusher_pos=init_pusher_pos,
    )

    for _ in range(2):
        env_dict = env.update((init_pusher_pos[0], init_pusher_pos[1]), rel=False)

    planning_res_list = []
    gt_states = []
    t = 0

    prev_X = None
    prev_U = None

    while t < max_steps:
        env_dict = env.get_env_state(not abs_pose)
        z_cur, env_state, pusher_pos = trans_fn(env_dict)
        gt_states.append(env_state)

        mppi_key = jax.random.PRNGKey(seed)

        X_init, U_init, planning_res = build_mppi_warmstart(
            z_cur=z_cur,
            pusher_pos=pusher_pos,
            target_obj=target_obj,
            horizon=horizon,
            nu=nu,
            action_lower_lim=action_lower_lim,
            action_upper_lim=action_upper_lim,
            prev_U=prev_U,
            jit_mppi_trajopt=jit_mppi_trajopt,
            rng_key=mppi_key,
        )

        # controller.X_in = X_init
        # controller.U_in = U_init

        X_ref = X_init.at[:, 1].set(
            jnp.maximum(X_init[:, 1], Y_MIN)
        )

        reference = X_ref
        # reference = X_init

        try:
            u0, X_pred, U_pred, *solver_info = controller.run(
                x0=z_cur,
                reference=reference,
                parameter=dt,
            )
        except Exception as e:
            print(f"[WARN] GenericMPC solve raised exception: {e}")
            u0, X_pred, U_pred = None, None, None
            solver_info = []

        if (
            u0 is None
            or X_pred is None
            or U_pred is None
            or not jnp.all(jnp.isfinite(X_pred))
            or not jnp.all(jnp.isfinite(U_pred))
            # or True
        ):
            print("[WARN] Falling back to MPPI warmstart.")
            X_pred = X_init
            U_pred = U_init
            u0 = U_pred[0]
            solver_status = "mppi_fallback"
        else:
            solver_status = "genericmpc"

        prev_X, prev_U = shift_warmstart(X_pred, U_pred)
        # prev_X, prev_U = shift_warmstart(X_init, U_init)

        print("=" * 60)
        print(f"Sim step {t}")
        print("=" * 60)
        print(f"Current position: {np.array(z_cur)}")
        print(f"Goal position:    {np.array(target_obj)}")
        print(f"MPPI u[0]:        {np.array(U_init[0])}")
        print(f"MPC  u[0]:        {np.array(u0)}")
        print(f"Status:           {solver_status}")
        print("=" * 60)

        planning_res_list.append({
            "time_step": t,
            "warmstart_act_seq": np.array(U_init) * scale,
            "warmstart_state_seq": np.array(X_init),
            "act_seq": np.array(U_pred) * scale,
            "state_seq": np.array(X_pred),
            "target_state": np.array(target_obj),
            "planning_res": planning_res,
            "solver_status": solver_status,
        })

        for k in range(n_act_step):
            if t >= max_steps:
                break

            action = np.array(U_pred[k]) * scale
            next_pusher_pos = np.array(pusher_pos) * scale + action

            env_dict = env.update(
                (next_pusher_pos[0], next_pusher_pos[1]),
                rel=False,
                n_sim_time=1,
            )

            z_next, env_state, pusher_pos = trans_fn(env_dict)
            t += 1

    env.save_gif(os.path.join(out_dir, "sim_vis.gif"), fps=10)
    env.close()


if __name__ == "__main__":
    main()