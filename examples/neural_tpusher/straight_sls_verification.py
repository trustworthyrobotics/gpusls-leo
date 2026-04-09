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

import os
import numpy as np
import pymunk

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_tubes_and_rollout_cloud(
    X_pred,
    lower,
    upper,
    xs,
    save_path="visualizations/tubes_rollout_cloud.png",
    cloud_times=(0, 5, 9),
    has_pusher=True,
):
    """
    Make one PNG:
      - plot all tube boxes in the object xy plane
      - overlay rollout state clouds at selected times

    Parameters
    ----------
    X_pred : array, shape (T+1, n)
        Nominal predicted trajectory.
    lower : array, shape (T+1, n)
        Tube lower bounds.
    upper : array, shape (T+1, n)
        Tube upper bounds.
    xs : array, shape (n_rollouts, T, n)
        Rollout trajectories.
    save_path : str
        Output PNG path.
    cloud_times : iterable of int
        Horizon indices to show rollout clouds for.
    has_pusher : bool
        Unused here, but kept for consistency with your codebase.
    """
    X_pred = np.asarray(X_pred)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    xs = np.asarray(xs)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # -------------------------
    # Plot nominal object path
    # object x/y assumed to be state indices 0,1
    # -------------------------
    ax.plot(
        X_pred[:, 0],
        X_pred[:, 1],
        "-k",
        linewidth=2.0,
        label="Nominal trajectory",
        zorder=3,
    )

    # -------------------------
    # Plot all tube boxes in xy
    # -------------------------
    for k in range(lower.shape[0]):
        x_min = lower[k, 0]
        y_min = lower[k, 1]
        width = upper[k, 0] - lower[k, 0]
        height = upper[k, 1] - lower[k, 1]

        rect = Rectangle(
            (x_min, y_min),
            width,
            height,
            fill=False,
            edgecolor="tab:blue",
            linewidth=1.2,
            alpha=0.7,
            zorder=1,
        )
        ax.add_patch(rect)

    # -------------------------
    # Plot rollout clouds at selected times
    # xs has shape (n_rollouts, T_steps, n)
    # -------------------------
    cloud_times = [t for t in cloud_times if 0 <= t < xs.shape[1]]

    markers = ["o", "s", "^", "D", "x", "*"]
    colors = ["tab:red", "tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:pink"]

    for idx, t in enumerate(cloud_times):
        ax.scatter(
            xs[:, t, 0],   # object x
            xs[:, t, 1],   # object y
            s=28,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            alpha=0.75,
            label=fr"Rollout cloud at $t={t}$",
            zorder=4,
        )

    # start / end markers
    ax.scatter(
        X_pred[0, 0], X_pred[0, 1],
        color="black", s=60, marker="o", zorder=5, label="Start"
    )
    ax.scatter(
        X_pred[-1, 0], X_pred[-1, 1],
        color="black", s=70, marker="*", zorder=5, label="End"
    )

    ax.set_xlabel("Object x")
    ax.set_ylabel("Object y")
    ax.set_title("Tube boxes with rollout clouds")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def visualize_x_sequence(env, xs, gif_path="output/x_rollout.gif", fps=10, has_pusher=True, scale=100):
    """
    Render a GIF from a sequence of states without simulating physics.

    Parameters
    ----------
    env : simulator instance
        Existing T_Sim / Base_Sim-style environment.
    xs : array-like, shape (T, nx)
        Sequence of states.
        Expected format:
            if has_pusher=True:
                [obj_x, obj_y, obj_theta, pusher_x, pusher_y]
            if has_pusher=False:
                [obj_x, obj_y, obj_theta]
    gif_path : str
        Output path for the gif.
    fps : int
        Frames per second for the gif.
    has_pusher : bool
        Whether xs includes pusher position.

    Returns
    -------
    None
    """
    xs = np.asarray(xs)

    if xs.ndim != 2:
        raise ValueError(f"xs must have shape (T, nx), got {xs.shape}")

    expected_nx = 5 if has_pusher else 3
    if xs.shape[1] < expected_nx:
        raise ValueError(
            f"xs must have at least {expected_nx} columns when has_pusher={has_pusher}, "
            f"got shape {xs.shape}"
        )

    if not getattr(env, "SAVE_IMG", False):
        raise ValueError("env.SAVE_IMG must be True so frames are stored for GIF export.")

    os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)

    # clear any old frames
    env.image_list = []

    for x in xs:
        # teleport object
        body = env.obj_list[0][0]
        body.position = pymunk.Vec2d(float(x[0] * scale), float(x[1] * scale))
        body.angle = float(x[2])
        body.velocity = (0.0, 0.0)
        body.angular_velocity = 0.0

        # teleport pusher if included
        if has_pusher:
            px, py = float(x[3] * scale), float(x[4] * scale)

            if env.pusher_body is None:
                env.add_pusher((px, py))
            else:
                env.pusher_body.position = pymunk.Vec2d(px, py)
                env.pusher_body.velocity = (0.0, 0.0)

        env.render()

    env.save_gif(gif_path, fps=fps)

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

    # Random disturbance
    key, subkey = jax.random.split(key)
    w_random = jax.random.uniform(
        subkey, (n,), dtype=dtype, minval=-1.0, maxval=1.0
    )

    # All corners of {-1, +1}^n
    combo_indices = jnp.arange(2**n)
    bits = ((combo_indices[:, None] >> jnp.arange(n)) & 1)
    adv_table = (2.0 * bits - 1.0).astype(dtype)

    num_adv = adv_table.shape[0]
    adv_idx = jnp.clip(i - NUM_RANDOM, 0, num_adv - 1)
    w_adv = adv_table[adv_idx]

    w = jnp.where(i >= NUM_RANDOM, w_adv, w_random)

    x_next = x_nom + E @ w
    return key, x_next, w


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
        enable_linearization_gradients=False,
        lambda_rem=0.0
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=40,
        warm_start=False,
        feas_tol=1e-2,
        step_tol=1e-4,
        line_search=True,
    )

    x_max = jnp.array([1000.0, 1000.0, 4 * jnp.pi, 1000.0, 1000.0], dtype=jnp.float64)
    x_min = -x_max
    # x_min = x_min.at[1].set(Y_MIN)
    constraints_all = make_state_box_constraints(x_min, x_max)
    constraints = constraints_all

    obstacles = jnp.zeros((0, 3))

    num_constraints = 2 * n + obstacles.shape[0]

    key = jax.random.PRNGKey(0)

    E_mag = 0.005
    alpha_sim = E_mag * dt
    E_sim = alpha_sim * jnp.eye(n, dtype=jnp.float64)
    disturbance = make_constant_disturbance(n=n, alpha=alpha_sim)

    # Q_bar = jnp.broadcast_to(jnp.eye(n), (horizon + 1, n, n))
    R_bar = jnp.broadcast_to(jnp.eye(nu), (horizon, nu, nu))
    Q_single = jnp.diag(jnp.array([10.0, 10.0, 0.1, 100.0, 100.0]))
    Q_bar = jnp.broadcast_to(Q_single, (horizon + 1, n, n))
    # R_single = jnp.diag(jnp.array([0.01, 0.01]))
    # R_bar = jnp.broadcast_to(R_single, (horizon, nu, nu))

    disturbance_center = jnp.full((horizon + 1, n), 0.0)

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
    jax.debug.print("{}", init_pusher_pos)
    init_pusher_pos[0] = 229
    init_pusher_pos[1] = 278
    init_pose = init_pose_list[0]
    # 2.38488095 2.29900584 0.61187779 2.2857082  2.77715499
    init_pose[0] = 238
    init_pose[1] = 230
    init_pose[2] = 0.611
    jax.debug.print("{}", init_pose)
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
    X_ref = X_ref.at[:, 0].set(
        jnp.minimum(X_ref[:, 0], X_MAX)
    )

    reference = X_ref

    u0, X_pred, U_pred, V, backoffs, Phi_x, Phi_u, EN, r_centerN = controller.run(
        x0=z_cur,
        reference=reference,
        parameter=dt,
    )

    NUM_ADV = 2**n
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
                disturbance_feedback = disturbance_feedback + Phi_u[k, j] @ disturbance_history[j]

            u = U_pred[k] + disturbance_feedback

            key, x, w = step_with_disturbance(
                key, x, u, E_sim, dt, i, dynamics=dynamics
            )
            jax.debug.print("Current state: {}", x)

            err = np.abs(np.asarray(x))

            disturbed[i, k, :] = err
            disturbance_history.append(E_sim @ w)
            xs[i, k] = np.asarray(x)

    env = T_Sim(
        param_dict=param_dict,
        init_poses=[init_pose],
        target_poses=[target_pose],
        pusher_pos=init_pusher_pos,
    )

    jax.debug.print(
        "nonzero values: {}",
        r_centerN[r_centerN != 0]
    )

    visualize_x_sequence(
        env,
        xs=xs[35],
        gif_path="visualizations/x_rollout.gif",
        fps=10,
        has_pusher=True,
    )

    tube = get_trajectory_tubes(Phi_x, EN)
    tube_center_shift = jnp.einsum("kjxn,jn->kx", Phi_x, r_centerN)
    shift = np.asarray(tube_center_shift)
    lower = X_pred - tube + shift
    upper = X_pred + tube + shift

    plot_tubes_and_rollout_cloud(
        X_pred=np.asarray(X_pred),
        lower=np.asarray(lower),
        upper=np.asarray(upper),
        xs=np.asarray(xs),
        save_path="visualizations/tubes_rollout_cloud.png",
        cloud_times=(0, 5, 9),
        has_pusher=True,
    )

    plot_tube_graph(
        disturbed,
        lower,
        upper,
        dt,
        output_folder=os.getcwd(),
        filename="disturbance_vs_tube_size.png",
    )

    save_npz_path = os.path.join(out_dir, "tube_graph_bundle.npz")
    np.savez(
        save_npz_path,

        # core rollout data
        xs=np.asarray(xs),                         # (n_rollouts, T, n)
        disturbed=np.asarray(disturbed),           # abs rollout values used by plot_tube_graph
        stop_steps=np.asarray(stop_steps),

        # nominal MPC outputs
        z_cur=np.asarray(z_cur),
        reference=np.asarray(reference),
        u0=np.asarray(u0),
        X_pred=np.asarray(X_pred),                 # (N+1, n)
        U_pred=np.asarray(U_pred),                 # (N, nu)
        V=np.asarray(V),
        backoffs=np.asarray(backoffs),

        # tube / SLS quantities
        Phi_x=np.asarray(Phi_x),
        Phi_u=np.asarray(Phi_u),
        EN=np.asarray(EN),
        r_centerN=np.asarray(r_centerN),
        tube=np.asarray(tube),
        shift=np.asarray(shift),
        lower=np.asarray(lower),
        upper=np.asarray(upper),

        # disturbance / model info
        E_sim=np.asarray(E_sim),
        disturbance_center=np.asarray(disturbance_center),

        # problem metadata
        dt=np.array(dt),
        horizon=np.array(horizon),
        n=np.array(n),
        nu=np.array(nu),
        T_dim=np.array(T_dim),
        action_dim=np.array(action_dim),
        Y_MIN=np.array(Y_MIN),
        X_MAX=np.array(X_MAX),

        # environment / case reconstruction info
        init_pusher_pos=np.asarray(init_pusher_pos),
        init_pose=np.asarray(init_pose),
        target_pose=np.asarray(target_pose),
        obstacles=np.asarray(obstacles),

        # optional plotting metadata
        cloud_times=np.asarray([0, 5, 9]),
        has_pusher=np.array(True),
        scale=np.array(scale),
        param_dict=np.array(param_dict, dtype=object),
    )

    print(f"Saved visualization bundle to: {save_npz_path}")

    env.close()


if __name__ == "__main__":
    main()