"""
t_pushing_vis.py
================
Visualization helpers for T-pushing MPC / SLS tubes.

Ground-truth conventions matched:
- object pose is the T-shape COM pose: [x, y, theta]
- T geometry uses the same COM construction as the planning code
- final outline uses the same 8-vertex merge order as merge_t_shape()
- default sizes come from config:
    stem_size   = [30, 90]
    bar_size    = [120, 30]
    pusher_size = 5
    scale       = 100

Public API
----------
poses_to_polys(poses)
box_corners_nd(lo, hi)
animate_rollouts_t_shape(xs, x0, x_goal, fps, filename, *, dt, ...)
animate_tube_t_shape(X_pred, tube, x_goal, fps, filename, *, dt, ...)
plot_disturbance_vs_tube(disturbed, tube, dt, filename, *, state_labels)
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter


# ---------------------------------------------------------------------------
# Defaults matched to config
# ---------------------------------------------------------------------------
_DEFAULT_SCALE = 500.0
_DEFAULT_STEM_SIZE = (30.0, 90.0)
_DEFAULT_BAR_SIZE = (120.0, 30.0)
_DEFAULT_PUSHER_SIZE = 5.0

_STEM_W = _DEFAULT_STEM_SIZE[0] / _DEFAULT_SCALE
_STEM_H = _DEFAULT_STEM_SIZE[1] / _DEFAULT_SCALE
_BAR_W = _DEFAULT_BAR_SIZE[0] / _DEFAULT_SCALE
_BAR_H = _DEFAULT_BAR_SIZE[1] / _DEFAULT_SCALE
_PUSHER_R = _DEFAULT_PUSHER_SIZE / _DEFAULT_SCALE


# =============================================================================
# Geometry helpers
# =============================================================================

def get_t_comp_centers_w_com(stem_size, bar_size):
    """
    Component centers relative to the composite T COM.

    Canonical pre-COM-shift construction:
      stem center = (0, stem_h / 2)
      bar center  = (0, stem_h + bar_h / 2)
    """
    stem_w, stem_h = map(float, stem_size)
    bar_w, bar_h = map(float, bar_size)

    c_s = np.array([0.0, stem_h / 2.0], dtype=np.float64)
    c_b = np.array([0.0, stem_h + bar_h / 2.0], dtype=np.float64)

    area_s = stem_w * stem_h
    area_b = bar_w * bar_h
    com = (area_s * c_s + area_b * c_b) / (area_s + area_b)

    return c_s - com, c_b - com


def _rect_vertices_centered(w, h):
    """
    Rectangle vertices in local frame, ordered [BL, BR, TR, TL].
    """
    hw = w / 2.0
    hh = h / 2.0
    return np.array(
        [
            [-hw, -hh],
            [ hw, -hh],
            [ hw,  hh],
            [-hw,  hh],
        ],
        dtype=np.float64,
    )


def _rot2(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _transform_vertices(vertices, center_xy, theta):
    R = _rot2(theta)
    return (R @ vertices.T).T + np.asarray(center_xy, dtype=np.float64)


def _t_shape_parts(
    cx,
    cy,
    theta,
    stem_w=_STEM_W,
    stem_h=_STEM_H,
    bar_w=_BAR_W,
    bar_h=_BAR_H,
):
    """
    Returns (stem_world, bar_world), each shape (4, 2).
    """
    c_s, c_b = get_t_comp_centers_w_com((stem_w, stem_h), (bar_w, bar_h))

    stem_local = _rect_vertices_centered(stem_w, stem_h)
    bar_local = _rect_vertices_centered(bar_w, bar_h)

    stem_world = _transform_vertices(stem_local + c_s[None, :], (cx, cy), theta)
    bar_world = _transform_vertices(bar_local + c_b[None, :], (cx, cy), theta)
    return stem_world, bar_world


def _merge_t_shape(stem_vertices, bar_vertices):
    """
    Same 8-point merge order as the ground-truth merge_t_shape():
      stem[BL, BR, TR], bar[BR, TR, TL, BL], stem[TL]
    """
    return np.concatenate(
        [
            stem_vertices[..., 0:1, :],
            stem_vertices[..., 1:2, :],
            stem_vertices[..., 2:3, :],
            bar_vertices[..., 1:2, :],
            bar_vertices[..., 2:3, :],
            bar_vertices[..., 3:4, :],
            bar_vertices[..., 0:1, :],
            stem_vertices[..., 3:4, :],
        ],
        axis=-2,
    )


def _t_shape_outline(
    cx,
    cy,
    theta,
    stem_w=_STEM_W,
    stem_h=_STEM_H,
    bar_w=_BAR_W,
    bar_h=_BAR_H,
):
    stem, bar = _t_shape_parts(
        cx,
        cy,
        theta,
        stem_w=stem_w,
        stem_h=stem_h,
        bar_w=bar_w,
        bar_h=bar_h,
    )
    return _merge_t_shape(stem, bar)


def poses_to_polys(
    poses,
    stem_w=_STEM_W,
    stem_h=_STEM_H,
    bar_w=_BAR_W,
    bar_h=_BAR_H,
):
    """
    poses : (..., 3)  [x, y, theta]
    returns : (..., 8, 2)
    """
    poses = np.asarray(poses, dtype=np.float64)
    flat = poses.reshape(-1, 3)

    polys = np.stack(
        [
            _t_shape_outline(
                p[0],
                p[1],
                p[2],
                stem_w=stem_w,
                stem_h=stem_h,
                bar_w=bar_w,
                bar_h=bar_h,
            )
            for p in flat
        ],
        axis=0,
    )
    return polys.reshape(poses.shape[:-1] + (8, 2))


def box_corners_nd(lo, hi):
    """
    lo, hi : (N, D)
    returns : (2^D, N, D)
    """
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    if lo.shape != hi.shape:
        raise ValueError(f"lo and hi must have same shape, got {lo.shape} vs {hi.shape}")

    N, D = lo.shape
    n_corners = 2 ** D
    out = np.empty((n_corners, N, D), dtype=lo.dtype)

    for i in range(n_corners):
        for d in range(D):
            out[i, :, d] = hi[:, d] if ((i >> d) & 1) else lo[:, d]
    return out


# =============================================================================
# Shared plotting helpers
# =============================================================================

def _compute_window_center_and_half(x0_xy, goal_xy, min_half=1.0, margin=0.75):
    pts = np.vstack([x0_xy, goal_xy]).astype(np.float64)
    center = pts.mean(axis=0)
    span = np.max(np.abs(pts - center[None, :]), axis=0)
    half = max(min_half, float(np.max(span)) + margin)
    return center[0], center[1], half


def _style_dark_ax(ax, cx, cy, window_half, xlabel="x [m]", ylabel="y [m]"):
    ax.set_facecolor("#0d0d0d")
    for sp in ax.spines.values():
        sp.set_color("#333333")
    ax.tick_params(colors="#aaaaaa")
    ax.set_xlim(cx - window_half, cx + window_half)
    ax.set_ylim(cy - window_half, cy + window_half)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=10)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.12)


# =============================================================================
# Plot 1: closed-loop rollout animation
# =============================================================================

def animate_rollouts_t_shape(
    xs,           # (n_rollouts, T, Dx)
    x0,           # (Dx,)
    x_goal,       # (Dx,)
    fps,
    filename,
    *,
    dt,
    window_half=None,
    pusher_r=_PUSHER_R,
    stem_w=_STEM_W,
    stem_h=_STEM_H,
    bar_w=_BAR_W,
    bar_h=_BAR_H,
):
    """
    Animated rollout visualization.

    Assumes state layout:
      state[:3] = [obj_x, obj_y, obj_theta]
      state[3:5] = [pusher_x, pusher_y]
    """
    xs = np.asarray(xs, dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64)
    x_goal = np.asarray(x_goal, dtype=np.float64)

    if xs.ndim != 3:
        raise ValueError(f"xs must have shape (n_rollouts, T, Dx), got {xs.shape}")
    if xs.shape[-1] < 5:
        raise ValueError("animate_rollouts_t_shape expects Dx >= 5.")

    n_rollouts, n_steps, _ = xs.shape

    polys = poses_to_polys(
        xs[:, :, :3],
        stem_w=stem_w,
        stem_h=stem_h,
        bar_w=bar_w,
        bar_h=bar_h,
    )
    goal_poly = poses_to_polys(
        x_goal[:3][None],
        stem_w=stem_w,
        stem_h=stem_h,
        bar_w=bar_w,
        bar_h=bar_h,
    )[0]

    if window_half is None:
        cx, cy, window_half = _compute_window_center_and_half(x0[:2], x_goal[:2])
    else:
        cx, cy = float(x0[0]), float(x0[1])

    cmap = matplotlib.colormaps["plasma"]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    fig.patch.set_facecolor("#0d0d0d")
    _style_dark_ax(ax, cx, cy, window_half)
    ax.set_title("T-Pushing MPC Rollout", color="#ffffff", fontsize=11, fontweight="bold")

    goal_patch = plt.Polygon(
        goal_poly,
        facecolor="#ff4488",
        edgecolor="#ffffff",
        alpha=0.35,
        lw=0.8,
        zorder=1,
    )
    ax.add_patch(goal_patch)

    trace_lines = []
    body_patches = []
    pusher_patches = []

    for i in range(n_rollouts):
        color = "#00e5ff" if i == 0 else cmap(i / max(n_rollouts - 1, 1))

        line, = ax.plot(
            [],
            [],
            color=color,
            alpha=0.95 if i == 0 else 0.22,
            lw=1.8 if i == 0 else 0.8,
            zorder=2,
        )
        trace_lines.append(line)

        body = plt.Polygon(
            polys[i, 0],
            facecolor=color,
            edgecolor="#ffffff" if i == 0 else "none",
            lw=0.6,
            alpha=0.88 if i == 0 else 0.24,
            zorder=4 if i == 0 else 3,
        )
        ax.add_patch(body)
        body_patches.append(body)

        pusher = plt.Circle(
            (xs[i, 0, 3], xs[i, 0, 4]),
            pusher_r,
            color="#ffdd55" if i == 0 else "#bbbbbb",
            alpha=0.95 if i == 0 else 0.25,
            zorder=5,
        )
        ax.add_patch(pusher)
        pusher_patches.append(pusher)

    ax.legend(
        handles=[
            mpatches.Patch(color="#00e5ff", label="rollout"),
            mpatches.Patch(color="#ff4488", alpha=0.5, label="goal"),
        ],
        facecolor="#1a1a1a",
        labelcolor="#cccccc",
        fontsize=8,
        loc="upper left",
    )

    time_text = ax.text(
        0.02,
        0.04,
        "",
        transform=ax.transAxes,
        color="#aaaaaa",
        fontsize=9,
        va="bottom",
    )

    def update(frame):
        artists = [goal_patch, time_text]
        for i in range(n_rollouts):
            trace_lines[i].set_data(xs[i, :frame + 1, 0], xs[i, :frame + 1, 1])
            body_patches[i].set_xy(polys[i, frame])
            pusher_patches[i].center = (xs[i, frame, 3], xs[i, frame, 4])
            artists.extend([trace_lines[i], body_patches[i], pusher_patches[i]])

        time_text.set_text(f"t = {frame * dt:.2f} s")
        return artists

    ani = FuncAnimation(
        fig,
        update,
        frames=n_steps,
        interval=int(1000 / fps),
        blit=True,
        repeat=False,
    )
    ani.save(
        filename,
        writer=PillowWriter(fps=fps),
        savefig_kwargs={"facecolor": fig.get_facecolor()},
    )
    plt.close(fig)
    print(f"Saved: {filename}")


# =============================================================================
# Plot 2: SLS tube animation
# =============================================================================

def animate_tube_t_shape(
    X_pred,       # (N+1, Dx)
    tube,         # (N+1, Dx) or (N, Dx)
    x_goal,       # (Dx,)
    fps,
    filename,
    *,
    dt,
    window_half=None,
    pusher_r=_PUSHER_R,
    stem_w=_STEM_W,
    stem_h=_STEM_H,
    bar_w=_BAR_W,
    bar_h=_BAR_H,
):
    """
    Animate one predicted nominal trajectory together with a box tube over
    [obj_x, obj_y, obj_theta].

    The body tube is rendered by converting all 2^3 box corners of
    [x, y, theta] into T-shape polygons.
    """
    X_pred = np.asarray(X_pred, dtype=np.float64)
    tube = np.asarray(tube, dtype=np.float64)
    x_goal = np.asarray(x_goal, dtype=np.float64)

    if X_pred.ndim != 2:
        raise ValueError(f"X_pred must have shape (T, Dx), got {X_pred.shape}")
    if X_pred.shape[1] < 5:
        raise ValueError("animate_tube_t_shape expects Dx >= 5.")

    n_steps = X_pred.shape[0]

    if tube.shape[0] == n_steps:
        tube_use = tube
    elif tube.shape[0] == n_steps + 1:
        tube_use = tube[:n_steps]
    else:
        raise ValueError(
            f"tube has incompatible first dimension {tube.shape[0]}; "
            f"expected {n_steps} or {n_steps + 1}"
        )

    lo = X_pred[:, :3] - tube_use[:, :3]
    hi = X_pred[:, :3] + tube_use[:, :3]
    corners = box_corners_nd(lo, hi)  # (8, T, 3)

    corner_polys = poses_to_polys(
        corners,
        stem_w=stem_w,
        stem_h=stem_h,
        bar_w=bar_w,
        bar_h=bar_h,
    )
    nom_polys = poses_to_polys(
        X_pred[:, :3],
        stem_w=stem_w,
        stem_h=stem_h,
        bar_w=bar_w,
        bar_h=bar_h,
    )
    goal_poly = poses_to_polys(
        x_goal[:3][None],
        stem_w=stem_w,
        stem_h=stem_h,
        bar_w=bar_w,
        bar_h=bar_h,
    )[0]

    if window_half is None:
        cx, cy, window_half = _compute_window_center_and_half(X_pred[0, :2], x_goal[:2])
    else:
        cx, cy = float(X_pred[0, 0]), float(X_pred[0, 1])

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    fig.patch.set_facecolor("#0d0d0d")
    _style_dark_ax(ax, cx, cy, window_half)
    ax.set_title("T-Pushing SLS Tube", color="#ffffff", fontsize=11, fontweight="bold")

    goal_patch = plt.Polygon(
        goal_poly,
        facecolor="#ff4488",
        edgecolor="#ffffff",
        alpha=0.35,
        lw=0.8,
        zorder=1,
    )
    ax.add_patch(goal_patch)

    nom_line, = ax.plot([], [], color="#ffffff", lw=1.2, ls="--", alpha=0.55, zorder=2)

    corner_patches = []
    for c in range(corners.shape[0]):
        patch = plt.Polygon(
            corner_polys[c, 0],
            facecolor="#ff6b35",
            edgecolor="none",
            alpha=0.18,
            zorder=3,
        )
        ax.add_patch(patch)
        corner_patches.append(patch)

    nom_patch = plt.Polygon(
        nom_polys[0],
        facecolor="#00e5ff",
        edgecolor="#ffffff",
        lw=0.8,
        alpha=0.88,
        zorder=5,
    )
    ax.add_patch(nom_patch)

    nom_pusher = plt.Circle(
        (float(X_pred[0, 3]), float(X_pred[0, 4])),
        pusher_r,
        color="#ffdd55",
        alpha=0.95,
        zorder=6,
    )
    ax.add_patch(nom_pusher)

    ax.legend(
        handles=[
            mpatches.Patch(color="#00e5ff", label="nominal"),
            mpatches.Patch(color="#ff6b35", alpha=0.5, label="tube corners"),
            mpatches.Patch(color="#ff4488", alpha=0.5, label="goal"),
        ],
        facecolor="#1a1a1a",
        labelcolor="#cccccc",
        fontsize=8,
        loc="upper left",
    )

    time_text = ax.text(
        0.02,
        0.04,
        "",
        transform=ax.transAxes,
        color="#aaaaaa",
        fontsize=9,
        va="bottom",
    )

    def update(frame):
        artists = [goal_patch, nom_line, nom_patch, nom_pusher, time_text]

        nom_line.set_data(X_pred[:frame + 1, 0], X_pred[:frame + 1, 1])

        for c in range(len(corner_patches)):
            corner_patches[c].set_xy(corner_polys[c, frame])
            artists.append(corner_patches[c])

        nom_patch.set_xy(nom_polys[frame])
        nom_pusher.center = (X_pred[frame, 3], X_pred[frame, 4])

        time_text.set_text(f"t = {frame * dt:.2f} s")
        return artists

    ani = FuncAnimation(
        fig,
        update,
        frames=n_steps,
        interval=int(1000 / fps),
        blit=True,
        repeat=False,
    )
    ani.save(
        filename,
        writer=PillowWriter(fps=fps),
        savefig_kwargs={"facecolor": fig.get_facecolor()},
    )
    plt.close(fig)
    print(f"Saved: {filename}")


# =============================================================================
# Plot 3: disturbance vs tube size
# =============================================================================

def plot_disturbance_vs_tube(
    disturbed,    # (n_rollouts, N, Dx)
    tube,         # (N+1, Dx) or (N, Dx)
    dt,
    filename,
    *,
    state_labels=None,
):
    disturbed = np.asarray(disturbed, dtype=np.float64)
    tube = np.asarray(tube, dtype=np.float64)

    if disturbed.ndim != 3:
        raise ValueError(f"disturbed must have shape (n_rollouts, N, Dx), got {disturbed.shape}")

    n_rollouts, n_steps, Dx = disturbed.shape
    t = np.arange(n_steps) * dt

    if state_labels is None:
        default = ["obj x", "obj y", "obj theta", "pusher x", "pusher y"]
        if Dx <= len(default):
            state_labels = default[:Dx]
        else:
            state_labels = default + [f"state {i}" for i in range(len(default), Dx)]

    if tube.shape[0] == n_steps + 1:
        tube_plot = tube[1:n_steps + 1]
    elif tube.shape[0] == n_steps:
        tube_plot = tube
    else:
        raise ValueError(
            f"tube has incompatible first dimension {tube.shape[0]}; "
            f"expected {n_steps} or {n_steps + 1}"
        )

    colors = [
        "#00e5ff",
        "#ff6b35",
        "#c77dff",
        "#ffdd55",
        "#44ff88",
        "#66ccff",
        "#ff8888",
        "#99ff99",
    ]

    fig, axes = plt.subplots(Dx, 1, figsize=(10, 2.5 * Dx), sharex=True)
    if Dx == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0d0d0d")

    for i, ax in enumerate(axes):
        color = colors[i % len(colors)]
        label = state_labels[i]

        ax.set_facecolor("#111111")
        for sp in ax.spines.values():
            sp.set_color("#222222")
        ax.tick_params(colors="#888888", labelsize=8)

        for r in range(n_rollouts):
            ax.plot(t, disturbed[r, :, i], color=color, alpha=0.15, lw=0.6)

        ax.plot(t, tube_plot[:, i], color="#ff4488", lw=1.8, label="SLS tube", zorder=5)
        ax.set_ylabel(label, color="#aaaaaa", fontsize=9)

    axes[-1].set_xlabel("time [s]", color="#aaaaaa", fontsize=10)
    axes[0].set_title("T-Pushing: Tracking Error vs SLS Tube", color="#ffffff", fontsize=12, fontweight="bold")
    axes[0].legend(facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")