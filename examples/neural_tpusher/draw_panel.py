import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import math


# ============================================================
# CONFIG
# ============================================================
NPZ_PATH = "visualizations/t_pushing_rollout_visualization_data_copy.npz"
OUTPUT_PNG = "visualizations/tubes_and_T_panels.png"

ROLLOUT_INDICES = np.arange(100)

N_ROLLOUTS_TO_PLOT = 5
TRAJ_LINEWIDTH = 1.6

T_LINEWIDTH = 2.2

T_ALPHA_MIN = 0.2
T_ALPHA_MAX = 0.5

T_STAMPS = np.arange(5) * 2

TUBE_FRAMES = None

TUBE_FACE_RGBA = (0.1, 0.4, 1.0, 0.22)
TUBE_EDGE_RGBA = (0.1, 0.4, 1.0, 0.95)
TUBE_EDGE_LINEWIDTH = 1.2

ROLLOUT_COLORS = [
    (0.85, 0.15, 0.15, 1.0),
    (0.15, 0.65, 0.20, 1.0),
    (0.95, 0.55, 0.05, 1.0),
    (0.55, 0.25, 0.85, 1.0),
    (0.10, 0.65, 0.75, 1.0),
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

    # Original component centers before COM shift
    _, _, com = get_t_comp_centers(stem_size, bar_size)

    # T in the temporary frame:
    # stem bottom at y = 0
    # bar sits on top of stem
    y0 = 0.0
    y1 = h_s
    y2 = h_s + h_b

    # Single merged T outline in CCW order
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

    # Shift to COM-centered frame
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

    # Use a single merged T outline
    t_outline_local = make_t_outline_local_from_param_dict(param_dict, scale)

    rollout_indices = [i for i in ROLLOUT_INDICES if i < xs.shape[0]]

    xmin, xmax, ymin, ymax = get_plot_limits(xs, X_pred)

    # --------------------------------------------------------
    # GRID LAYOUT
    # --------------------------------------------------------
    n = len(T_STAMPS)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)

    # --------------------------------------------------------
    # LOOP OVER TIME PANELS
    # --------------------------------------------------------
    for idx, t in enumerate(T_STAMPS):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        # --- disturbed T's ---
        for ridx in rollout_indices:
            traj = xs[ridx]
            if t >= len(traj):
                continue

            color = ROLLOUT_COLORS[ridx % len(ROLLOUT_COLORS)]

            if t == 0:
                pose = X_pred[0, :3]
            else:
                pose = traj[t - 1, :3]

            poly_world = transform_polygon(t_outline_local, pose)

            coll = PatchCollection(
                [Polygon(poly_world, closed=True)],
                facecolor=(0, 0, 0, 0),
                edgecolor=(color[0], color[1], color[2], 0.35),
                linewidth=T_LINEWIDTH,
            )
            ax.add_collection(coll)

        # --- nominal (BLACK, ON TOP) ---
        if t < len(X_pred):
            pose = X_pred[t, :3]
            poly_world = transform_polygon(t_outline_local, pose)

            coll = PatchCollection(
                [Polygon(poly_world, closed=True)],
                facecolor=(0, 0, 0, 0),
                edgecolor=(0, 0, 0, 1),
                linewidth=T_LINEWIDTH + 0.8,
                zorder=10,
            )
            ax.add_collection(coll)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t}")
        ax.set_xticks([])
        ax.set_yticks([])

    # turn off unused axes
    for i in range(n, nrows * ncols):
        r = i // ncols
        c = i % ncols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", OUTPUT_PNG)


if __name__ == "__main__":
    main()