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
    r_centerN,
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

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_tube_graph_multiquadrotor(
    disturbed,
    X_pred,
    Phi_x,
    r_centerN,
    tube,
    dt,
    N_QUADS,
    SINGLE_N,
    filename: str = "disturbance_vs_tube_size_multiquadrotor.png",
    state_labels: list | None = None,
):
    """
    Plot actual rollout states against the off-centered tube bounds for a
    multi-quadrotor system.

    disturbed:  (n_rollouts, T,   N_QUADS * SINGLE_N)   actual rollout states
    X_pred:     (T+1,       N_QUADS * SINGLE_N)         nominal plan
    Phi_x:      (T+1, T+1,  N_QUADS * SINGLE_N, nw)
    r_centerN:  (T+1, nw)                               propagated disturbance center
    tube:       (T+1,       N_QUADS * SINGLE_N)         symmetric half-widths
    """
    disturbed = np.asarray(disturbed)
    X_pred = np.asarray(X_pred)
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

    if X_pred.ndim != 2 or X_pred.shape != (T + 1, n_states):
        raise ValueError(
            f"X_pred has shape {X_pred.shape}. Expected ({T + 1}, {n_states})."
        )

    if tube.ndim != 2 or tube.shape != (T + 1, n_states):
        raise ValueError(
            f"tube has shape {tube.shape}. Expected ({T + 1}, {n_states})."
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

    # Off-centered tube center shift: [T+1, n_states]
    tube_center_shift = np.asarray(
        jnp.einsum("kjxn,jn->kx", Phi_x, r_centerN)
    )

    # Tube center and bounds in absolute state coordinates
    tube_center = X_pred + tube_center_shift
    lower = tube_center - tube
    upper = tube_center + tube

    # Rollouts are length T, so trim the T+1 trajectories/bounds to match
    nominal_trim = X_pred[1:, :]
    center_trim = tube_center[1:, :]
    lower_trim = lower[1:, :]
    upper_trim = upper[1:, :]
    tube_trim = tube[1:, :]

    t = np.arange(T) * dt

    nrows = N_QUADS * SINGLE_N
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(12, 2.4 * nrows + 2),
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

            nominal_i = nominal_trim[:, idx]
            center_i = center_trim[:, idx]
            lower_i = lower_trim[:, idx]
            upper_i = upper_trim[:, idx]
            tube_i = tube_trim[:, idx]
            rollout_all = disturbed[:, :, idx]

            # nominal trajectory
            ax.plot(
                t,
                nominal_i,
                linestyle="--",
                linewidth=2,
                label=f"nominal ({state_name}, quad {q})",
            )

            # shifted tube center
            ax.plot(
                t,
                center_i,
                linewidth=2,
                label=f"tube center ({state_name}, quad {q})",
            )

            # off-centered tube bounds
            ax.plot(
                t,
                lower_i,
                linewidth=2,
                label=f"lower bound ({state_name}, quad {q})",
            )
            ax.plot(
                t,
                upper_i,
                linewidth=2,
                label=f"upper bound ({state_name}, quad {q})",
            )

            # fill between bounds
            ax.fill_between(
                t,
                lower_i,
                upper_i,
                alpha=0.2,
            )

            # optional symmetric tube size around shifted center, for reference
            # this is just the half-width, not an absolute state trajectory
            # ax.plot(t, tube_i, linewidth=2, label=f"tube half-width ({state_name}, quad {q})")

            # rollout trajectories
            for r_idx, rollout in enumerate(rollout_all):
                m = np.isfinite(rollout)
                ax.plot(
                    t[m],
                    rollout[m],
                    alpha=0.5,
                    label=f"rollouts ({state_name})" if r_idx == 0 else None,
                )

            ax.set_ylabel(state_unit)
            ax.set_title(f"Quad {q} — {state_name}: Off-Centered Tube vs Rollouts")
            ax.grid(True)
            ax.legend(loc="best")

    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)