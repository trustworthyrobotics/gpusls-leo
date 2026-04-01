import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection


# ============================================================
# CONFIG
# ============================================================
NPZ_PATH = "visualizations/t_pushing_rollout_visualization_data.npz"
OUTPUT_PNG = "visualizations/tubes_and_selected_T_rollouts.png"

N_ROLLOUTS_TO_PLOT = 5
TRAJ_LINEWIDTH = 1.6
T_LINEWIDTH = 0.9
T_ALPHA = 0.70

# Which times to place T shapes along each selected rollout.
# If None, they will be chosen automatically.
T_STAMPS = None

# Draw every kth tube box
TUBE_STRIDE = 1

# Tube appearance
TUBE_FACE_RGBA = (0.1, 0.4, 1.0, 0.14)
TUBE_EDGE_RGBA = (0.1, 0.4, 1.0, 0.85)

# trajectory colors for up to 5 rollouts
ROLLOUT_COLORS = [
    (0.85, 0.15, 0.15, 1.0),
    (0.15, 0.65, 0.20, 1.0),
    (0.95, 0.55, 0.05, 1.0),
    (0.55, 0.25, 0.85, 1.0),
    (0.10, 0.65, 0.75, 1.0),
]


# ============================================================
# T SHAPE TEMPLATE
# ============================================================
def make_T_polygons_local(
    stem_size,
    bar_size,
):
    """
    Build a simple T shape in the object's local frame using two rectangles:
      - vertical stem centered under the bar
      - horizontal top bar

    Returns
    -------
    polys_local : list[np.ndarray]
        List of (m,2) polygons in world units.
    """
    stem_w, stem_h = float(stem_size[0]), float(stem_size[1])
    bar_w, bar_h = float(bar_size[0]), float(bar_size[1])

    # Put the bar on top, stem below it, centered.
    # Local origin is approximately center of the assembled T.
    y_bar_center = stem_h / 2.0
    y_stem_center = -bar_h / 2.0

    stem = np.array([
        [-stem_w / 2.0, y_stem_center - stem_h / 2.0],
        [ stem_w / 2.0, y_stem_center - stem_h / 2.0],
        [ stem_w / 2.0, y_stem_center + stem_h / 2.0],
        [-stem_w / 2.0, y_stem_center + stem_h / 2.0],
    ], dtype=np.float64)

    bar = np.array([
        [-bar_w / 2.0, y_bar_center - bar_h / 2.0],
        [ bar_w / 2.0, y_bar_center - bar_h / 2.0],
        [ bar_w / 2.0, y_bar_center + bar_h / 2.0],
        [-bar_w / 2.0, y_bar_center + bar_h / 2.0],
    ], dtype=np.float64)

    return [stem, bar]


def transform_polygon(poly_local, pose):
    """
    pose = [x, y, theta]
    """
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return poly_local @ R.T + np.array([x, y], dtype=np.float64)


# ============================================================
# ROLLOUT SELECTION
# ============================================================
def select_rollout_indices(xs, num_to_plot=5):
    """
    Spread the chosen rollouts roughly across the available set.
    """
    n_rollouts = xs.shape[0]
    if n_rollouts <= num_to_plot:
        return np.arange(n_rollouts, dtype=int)

    return np.linspace(0, n_rollouts - 1, num_to_plot, dtype=int)


def choose_t_stamps(T_steps, supplied=None):
    """
    Pick a few positions along the rollout to stamp T shapes.
    """
    if supplied is not None:
        return [int(t) for t in supplied if 0 <= int(t) < T_steps]

    # 4 evenly spaced times including start and near-end
    stamps = np.linspace(0, T_steps - 1, 4, dtype=int)
    return sorted(set(stamps.tolist()))


# ============================================================
# MAIN PLOT
# ============================================================
def main():
    data = np.load(NPZ_PATH, allow_pickle=True)

    xs = np.asarray(data["xs"])                       # (n_rollouts, T_steps, n)
    X_pred = np.asarray(data["X_pred"])               # (T_steps or T_steps+1, n)
    lower = np.asarray(data["lower"])                 # (T_steps+1, n)
    upper = np.asarray(data["upper"])                 # (T_steps+1, n)
    background_img = np.asarray(data["background_img"])
    scale = float(np.asarray(data["scale"]))
    param_dict = data["param_dict"].item()

    # infer world extents from saved background image + window size
    H, W = background_img.shape[:2]
    xlim = (0.0, W / scale)
    ylim = (0.0, H / scale)

    # Build local T polygons from saved geometry
    stem_size = param_dict["stem_size"]
    bar_size = param_dict["bar_size"]
    polys_local = make_T_polygons_local(stem_size=stem_size, bar_size=bar_size)

    rollout_indices = select_rollout_indices(xs, num_to_plot=N_ROLLOUTS_TO_PLOT)
    T_steps = xs.shape[1]
    t_stamps = choose_t_stamps(T_steps, supplied=T_STAMPS)

    os.makedirs(os.path.dirname(OUTPUT_PNG) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    # background
    ax.imshow(
        np.flipud(background_img),
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        zorder=0,
    )

    # --------------------------------------------------------
    # Tube rectangles in x-y
    # --------------------------------------------------------
    tube_patches = []
    for k in range(0, lower.shape[0], TUBE_STRIDE):
        x_min = float(lower[k, 0])
        y_min = float(lower[k, 1])
        width = float(upper[k, 0] - lower[k, 0])
        height = float(upper[k, 1] - lower[k, 1])
        tube_patches.append(Rectangle((x_min, y_min), width, height))

    if tube_patches:
        tube_collection = PatchCollection(
            tube_patches,
            facecolor=TUBE_FACE_RGBA,
            edgecolor=TUBE_EDGE_RGBA,
            linewidth=1.0,
            zorder=1,
        )
        ax.add_collection(tube_collection)

    # nominal trajectory
    ax.plot(
        X_pred[:, 0],
        X_pred[:, 1],
        color="black",
        linewidth=2.0,
        zorder=2,
    )

    # --------------------------------------------------------
    # Selected rollout trajectories + T stamps
    # --------------------------------------------------------
    for cidx, ridx in enumerate(rollout_indices):
        color = ROLLOUT_COLORS[cidx % len(ROLLOUT_COLORS)]
        traj = xs[ridx]

        # plot xy trajectory
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=color,
            linewidth=TRAJ_LINEWIDTH,
            alpha=0.95,
            zorder=3,
        )

        # mark start/end
        ax.scatter(
            traj[0, 0], traj[0, 1],
            s=18, color=color, zorder=4,
        )
        ax.scatter(
            traj[-1, 0], traj[-1, 1],
            s=28, color=color, marker="x", zorder=4,
        )

        # plot T shapes at selected times
        t_patches = []
        for t in t_stamps:
            pose = traj[t, :3]
            for poly_local in polys_local:
                poly_world = transform_polygon(poly_local, pose)
                t_patches.append(Polygon(poly_world, closed=True))

        if t_patches:
            coll = PatchCollection(
                t_patches,
                facecolor=(0.0, 0.0, 0.0, 0.0),
                edgecolor=(color[0], color[1], color[2], T_ALPHA),
                linewidth=T_LINEWIDTH,
                zorder=5,
            )
            ax.add_collection(coll)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Tubes with selected rollout T poses")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved figure to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()