import sys
import os
import time
import pickle
import datetime
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
LINEARIZATION_ERROR = os.path.join(
    ROOT, "src", "gpu_sls", "external", "linearization_error"
)
CROWN_REACH = os.path.join(
    ROOT, "src", "gpu_sls", "external", "ReachDev", "CROWN_Reach"
)

REACH_DEV = os.path.join(
    ROOT, "src", "gpu_sls", "external", "ReachDev"
)

sys.path.insert(0, LINEARIZATION_ERROR)
sys.path.insert(0, CROWN_REACH)
sys.path.insert(0, REACH_DEV)

import jax
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, "/home/jeff/trustworthroboticsgroup/gpu_sls/src/gpu_sls/external/ReachDev")
sys.path.insert(0, "/home/jeff/trustworthroboticsgroup/gpu_sls/src/gpu_sls/external/ReachDev/CROWN_Reach")

from gpu_sls.external.ReachDev.envs.T_pushing.t_sim import T_Sim
from gpu_sls.external.ReachDev.planning.T_pushing.plan_utils import (
    generate_test_cases,
    get_abs_states,
    plot_cost_stat,
    plot_plan_from_poses,
)

from gpu_sls.utils.constraint_utils import (
    make_state_box_constraints,
    make_constant_disturbance,
)
from neural_utils import (
    load_t_pushing_dt_dyn,
    get_problem_dims,
    wrap_dt_dyn_as_dynamics,
    make_tracking_cost,
    shift_warmstart,
    make_trans_fn,
    make_target_object,
    make_mppi_trajopt,
    warmup_mppi_jit,
    build_mppi_warmstart,
)

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig

Y_MIN = 0.70
X_MAX = 3.0


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
    seed = config["settings"]["seed"]
    seed = 42

    dt_dyn_dir = config["test_models"]["dt_dyn_dir"]
    dt_dyn = load_t_pushing_dt_dyn(dt_dyn_dir)

    dims = get_problem_dims(config, dt_dyn)
    abs_pose = dims["abs_pose"]
    pred_mode = dims["pred_mode"]
    scale = dims["scale"]
    state_dim = dims["state_dim"]
    pose_dim = dims["pose_dim"]
    action_dim = dims["action_dim"]
    T_dim = dims["T_dim"]
    horizon = dims["horizon"]
    max_steps = dims["max_steps"]
    n_act_step = dims["n_act_step"]
    action_bound = dims["action_bound"]
    action_lower_lim = dims["action_lower_lim"]
    action_upper_lim = dims["action_upper_lim"]

    param_dict = {
        "stem_size": data_config["stem_size"],
        "bar_size": data_config["bar_size"],
        "pusher_size": data_config["pusher_size"],
        "save_img": True,
        "enable_vis": False,
        "window_size": data_config["window_size"],
        "show_pose_center": True,
        "y_constraint": Y_MIN,  # the actual constraint (world units)
        "constraint_scale": scale,  # converts world → pixels
        "constraint_color": (255, 0, 0, 255),  # red line
        "x_constraint": X_MAX,
    }

    out_dir = "visualizations"
    os.makedirs(out_dir, exist_ok=True)

    trans_fn = make_trans_fn(
        scale=scale,
        pred_mode=pred_mode,
        state_dim=state_dim,
    )

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
        enable_linearization_gradients=False,
        lambda_rem=0.0,
        remainder_uses_time=False,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=100,
        warm_start=False,
        feas_tol=1e-2,
        step_tol=1e-4,
        line_search=False,
    )

    x_max = jnp.array([1000.0, 1000.0, 4 * jnp.pi, 1000.0, 1000.0], dtype=jnp.float64)
    x_min = -x_max
    x_min = x_min.at[1].set(Y_MIN)
    constraints_all = make_state_box_constraints(x_min, x_max)
    constraints = constraints_all

    obstacles = jnp.zeros((0, 3))

    num_constraints = 10 + obstacles.shape[0]

    E_mag = 0.01
    alpha_sim = E_mag * dt
    disturbance = make_constant_disturbance(n=n, alpha=alpha_sim)

    disturbance_center = jnp.full((horizon + 1, n), 0.0)

    R_bar = jnp.broadcast_to(jnp.eye(nu), (horizon, nu, nu))
    Q_single = jnp.diag(jnp.array([10.0, 10.0, 0.1, 100.0, 100.0]))
    Q_bar = jnp.broadcast_to(Q_single, (horizon + 1, n, n))

    controller = GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        constraints=constraints,
        obstacles=obstacles,
        disturbance_center=disturbance_center,
        cost=cost,
        Q_bar=Q_bar,
        R_bar=R_bar,
        num_constraints=num_constraints,
        disturbance=disturbance,
        shift=1,
        X_in=jnp.zeros((horizon + 1, n)),
        U_in=jnp.zeros((horizon, nu)),
        neural_dynamics=True,
        model_dir=dt_dyn_dir,
    )

    jit_mppi_trajopt = make_mppi_trajopt(
        config=config,
        dt_dyn=dt_dyn,
        abs_pose=abs_pose,
        pred_mode=pred_mode,
        action_lower_lim=action_lower_lim,
        action_upper_lim=action_upper_lim,
    )

    warmup_mppi_jit(
        jit_mppi_trajopt=jit_mppi_trajopt,
        seed=seed,
        T_dim=T_dim,
        action_dim=action_dim,
        horizon=horizon,
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

    target_obj = make_target_object(
        pred_mode=pred_mode,
        target_pose=target_pose,
        init_pose=init_pose,
        scale=scale,
        data_config=data_config,
    )

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

        X_ref = X_init.at[:, 1].set(
            jnp.maximum(X_init[:, 1], Y_MIN)
        )
        # X_ref = X_ref.at[:, 0].set(
        #     jnp.minimum(X_ref[:, 0], X_MAX)
        # )

        reference = X_ref

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

    np.savez(
        os.path.join(out_dir, "full_run_visualization_data.npz"),
        gt_states=np.asarray(gt_states, dtype=object),
        planning_res_list=np.asarray(planning_res_list, dtype=object),
        init_pose=np.asarray(init_pose),
        target_pose=np.asarray(target_pose),
        init_pusher_pos=np.asarray(init_pusher_pos),
        scale=np.array(scale, dtype=np.float64),
        Y_MIN=np.array(Y_MIN, dtype=np.float64),
        X_MAX=np.array(X_MAX, dtype=np.float64),
        stem_size=np.array(param_dict["stem_size"], dtype=np.float64),
        bar_size=np.array(param_dict["bar_size"], dtype=np.float64),
        pusher_size=np.array(param_dict["pusher_size"], dtype=np.float64),
        window_size=np.asarray(param_dict["window_size"]),
        y_constraint=np.array(param_dict["y_constraint"], dtype=np.float64),
        x_constraint=np.array(param_dict["x_constraint"], dtype=np.float64),
        dt=np.array(dt, dtype=np.float64),
        horizon=np.array(horizon, dtype=np.int32),
        n_act_step=np.array(n_act_step, dtype=np.int32),
    )

    env.save_gif(os.path.join(out_dir, "sim_vis.gif"), fps=10)
    env.close()


if __name__ == "__main__":
    main()