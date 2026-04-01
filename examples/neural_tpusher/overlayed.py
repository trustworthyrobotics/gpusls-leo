import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import math


# ============================================================
# CONFIG
# ============================================================
# NPZ_PATH = "visualizations/tube_graph_bundle_straight.npz"
NPZ_PATH = "visualizations/t_pushing_rollout_visualization_data_copy.npz"
OUTPUT_PNG = "visualizations/tubes_and_T_panels.png"

ROLLOUT_INDICES = np.arange(100)

N_ROLLOUTS_TO_PLOT = 5
TRAJ_LINEWIDTH = 1.6

T_LINEWIDTH = 2.2

# Much lighter opacity for T outlines
T_ALPHA_MIN = 0.05
T_ALPHA_MAX = 0.6

T_STAMPS = np.arange(5) * 2

TUBE_FRAMES = None

# Tube styling
TUBE_FACE_ALPHA = 0.10
TUBE_EDGE_ALPHA = 0.45
TUBE_EDGE_LINEWIDTH = 1.0

# same timestamp => same color for tube + T
TIME_COLORS = [
    (0.85, 0.15, 0.15),  # red
    (0.15, 0.60, 0.20),  # green
    (0.95, 0.55, 0.05),  # orange
    (0.55, 0.25, 0.85),  # purple
    (0.10, 0.60, 0.75),  # cyan
    (0.90, 0.80, 0.10),  # yellow
    (0.55, 0.35, 0.20),  # brown
]


# ============================================================
# GEOMETRY
# ============================================================
def calculate_com(xs, ys, masses):
    masses = np.asarray(masses, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    return np.sum(xs * masses) / np.sum(masses), np.sum(ys * masses) / np.sum(masses)


def get_t_comp_centers(stem_size, bar_size):
    w_s, h_s = stem_size
    w_b, h_b = bar_size

    x_s, y_s = 0.0, h_s / 2.0
    x_b, y_b = 0.0, h_s + h_b / 2.0

    m_s = w_s * h_s
    m_b = w_b * h_b

    x_m, y_m = calculate_com([x_s, x_b], [y_s, y_b], [m_s, m_b])
    return np.array([x_s, y_s]), np.array([x_b, y_b]), np.array([x_m, y_m])


def make_t_outline_local_from_param_dict(param_dict, scale=1.0):
    """
    Construct a single outer-outline polygon for the T shape, in the same
    COM-centered local frame used by the simulator.

    This avoids the internal line caused by drawing the stem and bar as two
    separate outlined rectangles.
    """
    stem_size = np.asarray(param_dict["stem_size"], dtype=np.float64) / scale
    bar_size = np.asarray(param_dict["bar_size"], dtype=np.float64) / scale

    w_s, h_s = stem_size
    w_b, h_b = bar_size

    _, _, com = get_t_comp_centers(stem_size, bar_size)

    y0 = 0.0
    y1 = h_s
    y2 = h_s + h_b

    poly = np.array([
        [-w_s / 2.0, y0],
        [ w_s / 2.0, y0],
        [ w_s / 2.0, y1],
        [ w_b / 2.0, y1],
        [ w_b / 2.0, y2],
        [-w_b / 2.0, y2],
        [-w_b / 2.0, y1],
        [-w_s / 2.0, y1],
    ], dtype=np.float64)

    poly = poly - com
    return poly


def transform_polygon(poly, pose):
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return poly @ R.T + np.array([x, y], dtype=np.float64)


# ============================================================
# HELPERS
# ============================================================
def get_plot_limits(xs, X_pred=None, pad=1.5):
    pts = [xs[..., :2].reshape(-1, 2)]
    if X_pred is not None:
        pts.append(X_pred[..., :2].reshape(-1, 2))
    pts = np.concatenate(pts, axis=0)

    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)

    dx = xmax - xmin
    dy = ymax - ymin

    return (
        xmin - pad * dx, xmax + pad * dx,
        ymin - pad * dy, ymax + pad * dy,
    )


def get_stamp_alpha(idx, n, alpha_min, alpha_max):
    if n <= 1:
        return alpha_max
    return alpha_min + (alpha_max - alpha_min) * idx / (n - 1)


def get_time_color(idx):
    return TIME_COLORS[idx % len(TIME_COLORS)]


# ============================================================
# MAIN
# ============================================================
def main():
    data = np.load(NPZ_PATH, allow_pickle=True)

    xs = np.asarray(data["xs"])
    X_pred = np.asarray(data["X_pred"])
    tube = np.asarray(data["tube"])
    scale = float(np.asarray(data["scale"]))
    param_dict = data["param_dict"].item()

    t_outline_local = make_t_outline_local_from_param_dict(param_dict, scale)

    rollout_indices = [i for i in ROLLOUT_INDICES if i < xs.shape[0]]
    if N_ROLLOUTS_TO_PLOT is not None:
        rollout_indices = rollout_indices[:N_ROLLOUTS_TO_PLOT]

    xmin, xmax, ymin, ymax = get_plot_limits(xs, X_pred)

    ordered_frames = [t for t in T_STAMPS if 0 <= t < len(X_pred)]

    fig, ax = plt.subplots(figsize=(7, 7))

    # --------------------------------------------------------
    # PLOT IN TIME ORDER:
    #   all disturbed T's at time t
    #   then nominal tube at time t
    #   then nominal T at time t
    # --------------------------------------------------------
    for idx, t in enumerate(ordered_frames):
        color = get_time_color(idx)
        alpha = get_stamp_alpha(idx, len(ordered_frames), T_ALPHA_MIN, T_ALPHA_MAX)

        # ----------------------------------------------------
        # 1) plot the nominal tube first
        # ----------------------------------------------------
        cx, cy = X_pred[t, 0], X_pred[t, 1]
        hx, hy = tube[t, 0], tube[t, 1]

        rect = Polygon(
            np.array([
                [cx - hx, cy - hy],
                [cx + hx, cy - hy],
                [cx + hx, cy + hy],
                [cx - hx, cy + hy],
            ]),
            closed=True,
        )

        coll = PatchCollection(
            [rect],
            facecolor=[(*color, TUBE_FACE_ALPHA)],
            edgecolor=[(*color, TUBE_EDGE_ALPHA)],
            linewidth=TUBE_EDGE_LINEWIDTH,
            zorder=1,
        )
        ax.add_collection(coll)

        # ----------------------------------------------------
        # 2) plot the nominal T next
        # ----------------------------------------------------
        pose_nom = X_pred[t, :3]
        poly_world_nom = transform_polygon(t_outline_local, pose_nom)

        coll = PatchCollection(
            [Polygon(poly_world_nom, closed=True)],
            facecolor=(0, 0, 0, 0),
            edgecolor=(*color, 1.0),
            linewidth=3.5,
            zorder=2,
        )
        ax.add_collection(coll)

        # ----------------------------------------------------
        # 3) plot all disturbed T's last
        # ----------------------------------------------------
        for ridx in rollout_indices:
            traj = xs[ridx]
            if t >= len(traj):
                continue

            if t == 0:
                pose = X_pred[0, :3]
            else:
                pose = traj[t - 1, :3]

            poly_world = transform_polygon(t_outline_local, pose)

            coll = PatchCollection(
                [Polygon(poly_world, closed=True)],
                facecolor=(0, 0, 0, 0),
                edgecolor=(*color, alpha),
                linewidth=T_LINEWIDTH,
                zorder=3,
            )
            ax.add_collection(coll)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", OUTPUT_PNG)


if __name__ == "__main__":
    main()