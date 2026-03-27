import sys
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

from gpu_sls.utils.sls_visual import get_trajectory_tubes, plot_tube_graph

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig

Y_MIN = 0.70
X_MAX = 10.0
NUM_RANDOM = 5


def step_with_disturbance(
    key: jax.Array,
    x: jnp.ndarray,
    u: jnp.ndarray,
    E: jnp.ndarray,
    dt: float,
    i: int,
    dynamics,
) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    """
    x_{k+1} = f(x_k, u_k) + E w

    For rollout indices:
      0 .. NUM_RANDOM-1 : random disturbances
      NUM_RANDOM ..     : all adversarial corners in {-1, +1}^n
    """
    x_nom = dynamics(x, u, 0, parameter=dt)

    n = x.shape[0]
    dtype = x.dtype

    key, subkey = jax.random.split(key)
    w_random = jax.random.uniform(
        subkey, (n,), dtype=dtype, minval=-1.0, maxval=1.0
    )

    combo_indices = jnp.arange(2**n)
    bits = ((combo_indices[:, None] >> jnp.arange(n)) & 1)
    adv_table = (2.0 * bits - 1.0).astype(dtype)

    num_adv = adv_table.shape[0]
    adv_idx = jnp.clip(i - NUM_RANDOM, 0, num_adv - 1)
    w_adv = adv_table[adv_idx]

    w = jnp.where(i >= NUM_RANDOM, w_adv, w_random)

    x_next = x_nom + E @ w
    return key, x_next, w


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name="T_pushing.yaml",
)
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
        "y_constraint": Y_MIN,
        "constraint_scale": scale,
        "constraint_color": (255, 0, 0, 255),
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
        max_sls_iterations=5,
        sls_primal_tol=1e-2,
        enable_fastsls=True,
        initialize_nominal=True,
        max_initial_sqp_iterations=100,
        warm_start=False,
        rti=False,
        enable_linearization_bounds=True,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=10,
        warm_start=False,
        feas_tol=1e-2,
        step_tol=1e-4,
        line_search=False,
    )

    x_max = jnp.array(
        [1000.0, 1000.0, 4 * jnp.pi, 1000.0, 1000.0],
        dtype=jnp.float64,
    )
    x_min = -x_max
    x_min = x_min.at[1].set(Y_MIN)
    x_max = x_max.at[0].set(X_MAX)
    constraints_all = make_state_box_constraints(x_min, x_max)
    constraints = constraints_all

    obstacles = jnp.zeros((0, 3))
    num_constraints = 2 * n + obstacles.shape[0]

    key = jax.random.PRNGKey(0)

    E_mag = 0.005
    alpha_sim = E_mag * dt
    E_sim = alpha_sim * jnp.eye(n, dtype=jnp.float64)
    disturbance = make_constant_disturbance(n=n, alpha=alpha_sim)

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

    num_test = 1
    test_id = planning_config.get("test_id", 0)
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
        env.update((init_pusher_pos[0], init_pusher_pos[1]), rel=False)

    planning_res_list = []
    gt_states = []
    t = 0
    prev_X = None
    prev_U = None

    # Save first successful SLS solve for tube / adversarial rollout visualization
    first_sls_result = None

    while t < max_steps:
        env_dict = env.get_env_state(not abs_pose)
        z_cur, env_state, pusher_pos = trans_fn(env_dict)
        gt_states.append(env_state)

        mppi_key = jax.random.PRNGKey(seed + t)

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

        X_ref = X_init.at[:, 1].set(jnp.maximum(X_init[:, 1], Y_MIN))
        X_ref = X_ref.at[:, 0].set(jnp.minimum(X_ref[:, 0], X_MAX))
        reference = X_ref

        try:
            u0, X_pred, U_pred, V, backoffs, Phi_x, Phi_u, EN = controller.run(
                x0=z_cur,
                reference=reference,
                parameter=dt,
            )
            solver_status = "sls"
        except Exception as e:
            print(f"[WARN] SLS solve raised exception: {e}")
            u0, X_pred, U_pred = None, None, None
            Phi_x, Phi_u, EN = None, None, None
            solver_status = "failed"

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

        if solver_status == "sls" and first_sls_result is None:
            first_sls_result = {
                "z_cur": z_cur,
                "X_pred": X_pred,
                "U_pred": U_pred,
                "Phi_x": Phi_x,
                "Phi_u": Phi_u,
                "EN": EN,
            }

        prev_X, prev_U = shift_warmstart(X_pred, U_pred)

        print("=" * 60)
        print(f"Sim step {t}")
        print("=" * 60)
        print(f"Current position: {np.array(z_cur)}")
        print(f"Goal position:    {np.array(target_obj)}")
        print(f"MPPI u[0]:        {np.array(U_init[0])}")
        print(f"SLS  u[0]:        {np.array(u0)}")
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
            if t >= max_steps or k >= U_pred.shape[0]:
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

    # Save environment visualization like the MPC script
    env.save_gif(os.path.join(out_dir, "sim_vis.gif"), fps=10)

    # Also run the SLS tube/disturbance visualization from the first successful SLS solve
    if first_sls_result is not None:
        z_cur = first_sls_result["z_cur"]
        X_pred = first_sls_result["X_pred"]
        U_pred = first_sls_result["U_pred"]
        Phi_x = first_sls_result["Phi_x"]
        Phi_u = first_sls_result["Phi_u"]
        EN = first_sls_result["EN"]

        NUM_ADV = 2 ** n
        N_ROLLOUTS = NUM_RANDOM + NUM_ADV
        T_steps = horizon

        xs = np.full((N_ROLLOUTS, T_steps, n), np.nan, dtype=np.float64)
        disturbed = np.full((N_ROLLOUTS, T_steps, n), np.nan, dtype=np.float64)
        stop_steps = np.full((N_ROLLOUTS,), T_steps, dtype=np.int32)

        for i in range(N_ROLLOUTS):
            disturbance_history = [jnp.zeros((n,), dtype=jnp.float64)]
            x = z_cur
            jax.debug.print("Rolling out iteration {}", i)

            for k in range(T_steps):
                disturbance_feedback = jnp.zeros((nu,), dtype=jnp.float64)
                for j in range(k + 1):
                    disturbance_feedback = (
                        disturbance_feedback + Phi_u[k, j] @ disturbance_history[j]
                    )

                u = U_pred[k] + disturbance_feedback

                key, x, w = step_with_disturbance(
                    key, x, u, E_sim, dt, i, dynamics=dynamics
                )

                err = np.abs(np.asarray(X_pred[k + 1] - x))
                disturbed[i, k, :] = err
                disturbance_history.append(E_sim @ w)
                xs[i, k] = np.asarray(x)

        tube = get_trajectory_tubes(Phi_x, EN)
        lower = X_pred - tube
        upper = X_pred + tube

        plot_tube_graph(
            disturbed,
            tube,
            dt,
            output_folder=os.getcwd(),
            filename="disturbance_vs_tube_size.png",
        )

        np.savez(
            os.path.join(out_dir, "sls_rollout_bundle.npz"),
            xs=xs,
            disturbed=disturbed,
            tube=np.asarray(tube),
            plan=np.asarray(X_pred),
            lower=np.asarray(lower),
            upper=np.asarray(upper),
            dt=float(dt),
        )

    env.close()


if __name__ == "__main__":
    main()