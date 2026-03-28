import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from gpu_sls.utils.sls_visual import get_trajectory_tubes


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from gpu_sls.utils.sls_visual import get_trajectory_tubes


def plot_plan_xy_with_tubes(
    X_pred,
    x0_quads,
    xg_quads,
    obstacles,
    Phi_x,
    E_prev,
    N_QUADS,
    SINGLE_N,
    xs=None,                  # (n_rollouts, T, N) or None
    filename="multi_quadrotor_plan_tubes.png",
    stride=1,
    box_alpha=0.15,
    rollout_alpha=0.25,
    rollout_linewidth=1.0,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    tube = get_trajectory_tubes(Phi_x, E_prev)

    X_pred = np.asarray(X_pred)
    tube = np.asarray(tube)
    x0_quads = np.asarray(x0_quads)
    xg_quads = np.asarray(xg_quads)
    obstacles = np.asarray(obstacles)

    if xs is not None:
        xs = np.asarray(xs)

    T = X_pred.shape[0]

    for i in range(N_QUADS):
        s = slice(i * SINGLE_N, (i + 1) * SINGLE_N)

        x_pred_i = X_pred[:, s]      # (T+1, SINGLE_N)
        tube_i = tube[:, s]          # (T+1, SINGLE_N)
        xy = x_pred_i[:, :2]

        # nominal plan
        line = ax.plot(
            xy[:, 0], xy[:, 1],
            linewidth=2,
            label=f"Quad {i}"
        )[0]
        color = line.get_color()

        # rollout trajectories
        if xs is not None:
            # xs shape: (n_rollouts, T, N)
            xs_i = xs[:, :, s]       # (n_rollouts, T, SINGLE_N)

            for r in range(xs_i.shape[0]):
                rollout_xy = xs_i[r, :, :2]
                ax.plot(
                    rollout_xy[:, 0],
                    rollout_xy[:, 1],
                    linewidth=rollout_linewidth,
                    alpha=rollout_alpha,
                    color=color,
                )

        # tube boxes
        for k in range(0, T, stride):
            cx, cy = xy[k]
            wx, wy = tube_i[k, 0], tube_i[k, 1]

            rect = Rectangle(
                (cx - wx, cy - wy),
                2 * wx,
                2 * wy,
                fill=True,
                alpha=box_alpha,
                edgecolor=None,
                facecolor=color,
            )
            ax.add_patch(rect)

        # start / goal
        ax.scatter(x0_quads[i, 0], x0_quads[i, 1], marker="o", s=80, color=color)
        ax.scatter(xg_quads[i, 0], xg_quads[i, 1], marker="x", s=100, color=color)

    # obstacles
    for j in range(obstacles.shape[0]):
        cx, cy, r = obstacles[j]
        circ = Circle((cx, cy), r, fill=False, linewidth=2, linestyle="--")
        ax.add_patch(circ)
        ax.scatter(cx, cy, marker="+", s=120)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Multi-Quadrotor Trajectories, Tubes, and Rollouts")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_tube_graph_multiquadrotor(
    disturbed,
    tube,
    dt,
    N_QUADS,
    SINGLE_N,
    filename: str = "disturbance_vs_tube_size_multiquadrotor.png",
    state_labels: list | None = None,
):
    """
    Plot deviation vs tube size for a multi-quadrotor state.

    disturbed: (n_rollouts, T, N_QUADS * SINGLE_N)
    tube:      (T+1, N_QUADS * SINGLE_N)

    Produces one subplot per state dimension per quad.
    """
    disturbed = np.asarray(disturbed)
    tube = np.asarray(tube)

    if disturbed.ndim != 3:
        raise ValueError(
            f"disturbed has shape {disturbed.shape}. Expected (n_rollouts, T, n_states)."
        )

    n_rollouts, T, n_states = disturbed.shape
    expected_n = N_QUADS * SINGLE_N

    if n_states != expected_n:
        raise ValueError(
            f"disturbed has last dimension {n_states}, but expected "
            f"N_QUADS * SINGLE_N = {N_QUADS} * {SINGLE_N} = {expected_n}."
        )

    if tube.ndim != 2 or tube.shape[1] != n_states:
        raise ValueError(
            f"tube has shape {tube.shape}. Expected (T+1, {n_states})."
        )

    tube_trim = tube[1:, :]
    if tube_trim.shape[0] != T:
        raise ValueError(
            f"tube[1:] has length {tube_trim.shape[0]}, but disturbed time dimension is {T}."
        )

    default_labels_12 = [
        ("px", "meters"),
        ("py", "meters"),
        ("pz", "meters"),
        ("phi", "radians"),
        ("theta", "radians"),
        ("psi", "radians"),
        ("vx", "m/s"),
        ("vy", "m/s"),
        ("vz", "m/s"),
        ("p", "rad/s"),
        ("q", "rad/s"),
        ("r", "rad/s"),
    ]

    if state_labels is None:
        if SINGLE_N == 12:
            state_labels = default_labels_12
        else:
            state_labels = [(f"x{i}", "units") for i in range(SINGLE_N)]

    if len(state_labels) != SINGLE_N:
        raise ValueError(
            f"state_labels must have length SINGLE_N={SINGLE_N}, got {len(state_labels)}."
        )

    t = np.arange(T) * dt

    nrows = N_QUADS * SINGLE_N
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(12, 2.2 * nrows + 2),
        sharex=True,
    )

    if nrows == 1:
        axes = [axes]

    for q in range(N_QUADS):
        base = q * SINGLE_N

        for j in range(SINGLE_N):
            idx = base + j
            ax = axes[idx]

            state_name, state_unit = state_labels[j]
            tube_i = tube_trim[:, idx]
            dev_all = disturbed[:, :, idx]

            ax.plot(
                t,
                tube_i,
                linewidth=3,
                label=f"tube size ({state_name}, quad {q})",
            )

            for r_idx, dev in enumerate(dev_all):
                m = np.isfinite(dev)
                ax.plot(
                    t[m],
                    dev[m],
                    alpha=0.5,
                    label=f"|{state_name} - nominal| (rollouts)" if r_idx == 0 else None,
                )

            ax.set_ylabel(state_unit)
            ax.set_title(f"Quad {q} — {state_name}: Deviation vs Tube Size")
            ax.grid(True)
            ax.legend(loc="best")

    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)