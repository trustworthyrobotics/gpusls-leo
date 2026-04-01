import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


# ============================================================
# CONFIG
# ============================================================
NPZ_PATH = "visualizations/full_run_visualization_data_save.npz"
OUTPUT_PNG = "visualizations/gt_T_and_pusher_style.png"

T_STAMPS = np.arange(15)

TRAJ_LINEWIDTH = 1.6
T_LINEWIDTH = 2.2
PUSHER_LINEWIDTH = 1.8

T_ALPHA_MIN = 0.05
T_ALPHA_MAX = 0.6

TIME_COLORS = [
    (0.85, 0.15, 0.15),
    (0.15, 0.60, 0.20),
    (0.95, 0.55, 0.05),
    (0.55, 0.25, 0.85),
    (0.10, 0.60, 0.75),
    (0.90, 0.80, 0.10),
    (0.55, 0.35, 0.20),
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


def make_t_outline_local(stem_size, bar_size):
    stem_size = np.asarray(stem_size, dtype=np.float64)
    bar_size = np.asarray(bar_size, dtype=np.float64)

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

    return poly - com


def make_pusher_outline_local(pusher_size):
    pusher_size = np.asarray(pusher_size, dtype=np.float64)

    if pusher_size.size == 1:
        w = h = float(pusher_size.item())
    elif pusher_size.size == 2:
        w, h = pusher_size
    else:
        raise ValueError(f"Unexpected pusher_size shape/value: {pusher_size}")

    return np.array([
        [-w / 2.0, -h / 2.0],
        [ w / 2.0, -h / 2.0],
        [ w / 2.0,  h / 2.0],
        [-w / 2.0,  h / 2.0],
    ], dtype=np.float64)


def transform_polygon(poly, pose):
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return poly @ R.T + np.array([x, y], dtype=np.float64)


def transform_polygon_xy(poly, xy):
    x, y = xy
    return poly + np.array([x, y], dtype=np.float64)


# ============================================================
# HELPERS
# ============================================================
def get_time_color(idx):
    return TIME_COLORS[idx % len(TIME_COLORS)]


def get_stamp_alpha(idx, n, alpha_min, alpha_max):
    if n <= 1:
        return alpha_max
    return alpha_min + (alpha_max - alpha_min) * idx / (n - 1)


def get_plot_limits(gt_states, pusher_xy=None, pad=1.5):
    pts = [np.asarray(gt_states[:, :2], dtype=np.float64)]
    if pusher_xy is not None:
        pts.append(np.asarray(pusher_xy, dtype=np.float64))

    pts = np.concatenate(pts, axis=0)

    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)

    dx = xmax - xmin
    dy = ymax - ymin

    if dx <= 1e-12:
        dx = 1.0
    if dy <= 1e-12:
        dy = 1.0

    return (
        xmin - pad * dx, xmax + pad * dx,
        ymin - pad * dy, ymax + pad * dy,
    )


def extract_pusher_positions(data, gt_states):
    if "pusher_positions" in data:
        pusher_xy = np.asarray(data["pusher_positions"], dtype=np.float64)
    elif "gt_pusher_positions" in data:
        pusher_xy = np.asarray(data["gt_pusher_positions"], dtype=np.float64)
    else:
        # Fall back to assuming the pusher position is stored in gt_states[:, 3:5]
        if gt_states.shape[1] < 5:
            raise KeyError(
                "Could not find saved pusher positions in the NPZ, and gt_states "
                "does not have columns 3:5 available for fallback."
            )
        pusher_xy = gt_states[:, 3:5]

    if pusher_xy.ndim != 2 or pusher_xy.shape[1] != 2:
        raise ValueError(f"Unexpected pusher position shape: {pusher_xy.shape}")

    return pusher_xy


# ============================================================
# MAIN
# ============================================================
def main():
    data = np.load(NPZ_PATH, allow_pickle=True)

    gt_states = np.stack([np.asarray(s, dtype=np.float64) for s in data["gt_states"]])

    stem_size = np.asarray(data["stem_size"], dtype=np.float64)
    bar_size = np.asarray(data["bar_size"], dtype=np.float64)
    pusher_size = np.asarray(data["pusher_size"], dtype=np.float64)

    pusher_xy = extract_pusher_positions(data, gt_states)

    t_outline_local = make_t_outline_local(stem_size, bar_size)
    pusher_outline_local = make_pusher_outline_local(pusher_size)

    ordered_frames = [t for t in T_STAMPS if t < len(gt_states) and t < len(pusher_xy)]
    xmin, xmax, ymin, ymax = get_plot_limits(gt_states, pusher_xy)

    fig, ax = plt.subplots(figsize=(7, 7))

    # GT object trajectory
    # ax.plot(
    #     gt_states[:, 0],
    #     gt_states[:, 1],
    #     linewidth=TRAJ_LINEWIDTH,
    #     alpha=0.7,
    #     zorder=1,
    # )

    # # Optional: pusher trajectory
    # ax.plot(
    #     pusher_xy[:, 0],
    #     pusher_xy[:, 1],
    #     linewidth=1.2,
    #     alpha=0.35,
    #     zorder=1,
    # )

    # Time-ordered overlays
    for idx, t in enumerate(ordered_frames):
        color = get_time_color(idx)
        alpha = get_stamp_alpha(idx, len(ordered_frames), T_ALPHA_MIN, T_ALPHA_MAX)

        # T
        pose_T = gt_states[t, :3]
        poly_T = transform_polygon(t_outline_local, pose_T)
        coll_T = PatchCollection(
            [Polygon(poly_T, closed=True)],
            facecolor=(0, 0, 0, 0),
            edgecolor=(*color, alpha),
            linewidth=T_LINEWIDTH,
            zorder=2,
        )
        ax.add_collection(coll_T)

        # Pusher
        poly_p = transform_polygon_xy(pusher_outline_local, pusher_xy[t])
        coll_p = PatchCollection(
            [Polygon(poly_p, closed=True)],
            facecolor=(0, 0, 0, 0),
            edgecolor=(*color, alpha),
            linewidth=PUSHER_LINEWIDTH,
            zorder=3,
        )
        ax.add_collection(coll_p)

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
    print("stem_size:", stem_size)
    print("bar_size:", bar_size)
    print("pusher_size:", pusher_size)
    print("gt_states shape:", gt_states.shape)
    print("pusher_xy shape:", pusher_xy.shape)


if __name__ == "__main__":
    main()