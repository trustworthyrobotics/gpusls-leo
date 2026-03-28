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
    filename="multi_quadrotor_plan_tubes.png",
    stride=1,            # plot every k steps (important for clarity)
    box_alpha=0.15,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    tube = get_trajectory_tubes(Phi_x, E_prev)

    X_pred = np.asarray(X_pred)
    tube = np.asarray(tube)
    x0_quads = np.asarray(x0_quads)
    xg_quads = np.asarray(xg_quads)
    obstacles = np.asarray(obstacles)

    T = X_pred.shape[0]

    for i in range(N_QUADS):
        s = slice(i * SINGLE_N, (i + 1) * SINGLE_N)

        x_pred_i = X_pred[:, s]
        tube_i = tube[:, s]

        xy = x_pred_i[:, :2]

        line = ax.plot(xy[:, 0], xy[:, 1], linewidth=2, label=f"Quad {i}")[0]
        color = line.get_color()

        # -----------------------------
        # Draw square tube at each timestep
        # -----------------------------
        for k in range(0, T, stride):
            cx, cy = xy[k]
            wx, wy = tube_i[k, 0], tube_i[k, 1]

            rect = Rectangle(
                (cx - wx, cy - wy),   # bottom-left
                2 * wx,               # width
                2 * wy,               # height
                fill=True,
                alpha=box_alpha,
                edgecolor=None,
                facecolor=color,
            )
            ax.add_patch(rect)

        # start / goal
        ax.scatter(x0_quads[i, 0], x0_quads[i, 1], marker="o", s=80, color=color)
        ax.scatter(xg_quads[i, 0], xg_quads[i, 1], marker="x", s=100, color=color)

    # -----------------------------
    # Obstacles
    # -----------------------------
    for j in range(obstacles.shape[0]):
        cx, cy, r = obstacles[j]
        circ = Circle((cx, cy), r, fill=False, linewidth=2, linestyle="--")
        ax.add_patch(circ)
        ax.scatter(cx, cy, marker="+", s=120)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Multi-Quadrotor Trajectories with Tube Squares")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)