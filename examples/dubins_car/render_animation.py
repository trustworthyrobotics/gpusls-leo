"""
animate_dubins_rollouts_from_npz.py

Standalone visualization script for Dubins robust rollout data saved via np.savez_compressed.

Expected NPZ keys from your current save block:
  xs                    (R, T, 3) rollout states [px, py, theta]
  lower, upper           (N+1, 3) tube lower/upper bounds
  plans_xy/lower_xy/...  optional plotting arrays
  X_pred                 (N+1, 3) nominal plan
  obstacles              (n_obs, 3) [cx, cy, r]
  dt                     scalar

Example:
  python animate_dubins_rollouts_from_npz.py \
      --npz dubins_rollout_render_data.npz \
      --sprite car_removed.png \
      --out rollouts_sprite.mp4

For a GIF:
  python animate_dubins_rollouts_from_npz.py --npz dubins_rollout_render_data.npz --sprite car_removed.png --out rollouts_sprite.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================================================================
# Style
# =============================================================================
plt.rcParams.update({
    "font.size": 18,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False, 
    "ps.fonttype": 42,
})


# =============================================================================
# Sprite helpers: adapted from your imshow + Affine2D pattern
# =============================================================================
def load_sprite_rgba(path: str | Path) -> np.ndarray:
    img = plt.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img, np.ones_like(img)], axis=-1)
    elif img.shape[-1] == 3:
        img = np.dstack([img, np.ones(img.shape[:2], dtype=img.dtype)])
    return img


def make_sprite_artist(
    ax: plt.Axes,
    img_rgba: np.ndarray,
    x: float,
    y: float,
    theta: float,
    length: float,
    width: float,
    alpha: float,
    zorder: int = 8,
    interpolation: str = "nearest",
) -> dict[str, Any]:
    extent = (-length / 2.0, length / 2.0, -width / 2.0, width / 2.0)
    im = ax.imshow(
        img_rgba,
        extent=extent,
        origin="upper",
        interpolation=interpolation,
        alpha=alpha,
        zorder=zorder,
    )
    tf = Affine2D().rotate(theta).translate(x, y)
    im.set_transform(tf + ax.transData)
    return {"im": im, "tf": tf}


def update_sprite_artist(sprite: dict[str, Any], x: float, y: float, theta: float) -> None:
    tf: Affine2D = sprite["tf"]
    tf.clear()
    tf.rotate(theta)
    tf.translate(x, y)


def set_sprite_visible(sprite: dict[str, Any], visible: bool) -> None:
    sprite["im"].set_visible(visible)


# =============================================================================
# NPZ loading / shape normalization
# =============================================================================
def get_key(data: np.lib.npyio.NpzFile, *names: str, required: bool = True, default=None):
    for name in names:
        if name in data.files:
            return data[name]
    if required:
        raise KeyError(f"Missing required NPZ key. Tried: {names}. Available: {data.files}")
    return default


def normalize_plan_tubes(data: np.lib.npyio.NpzFile):
    """
    Returns:
      plan_xy   (N+1, 2)
      lower_xy  (N+1, 2)
      upper_xy  (N+1, 2)
    """
    if "plans_xy" in data.files:
        plans_xy = np.asarray(data["plans_xy"])
        plan_xy = plans_xy[0] if plans_xy.ndim == 3 else plans_xy
    else:
        plan_xy = np.asarray(get_key(data, "X_pred"))[:, :2]

    if "lowers_xy" in data.files and "uppers_xy" in data.files:
        lowers_xy = np.asarray(data["lowers_xy"])
        uppers_xy = np.asarray(data["uppers_xy"])
        lower_xy = lowers_xy[0] if lowers_xy.ndim == 3 else lowers_xy
        upper_xy = uppers_xy[0] if uppers_xy.ndim == 3 else uppers_xy
    else:
        lower_xy = np.asarray(get_key(data, "lower"))[:, :2]
        upper_xy = np.asarray(get_key(data, "upper"))[:, :2]

    return plan_xy, lower_xy, upper_xy


def finite_xy(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.zeros((0, 2))
    xy = arr[..., :2].reshape(-1, 2)
    mask = np.isfinite(xy).all(axis=1)
    return xy[mask]


# =============================================================================
# Animation
# =============================================================================
def save_rollout_sprite_animation(
    npz_path: str | Path,
    sprite_path: str | Path,
    out_path: str | Path = "rollouts_sprite.mp4",
    fps: int | None = None,
    stride: int = 1,
    tube_stride: int = 1,
    margin: float = 0.25,
    sprite_length: float = 0.09,
    sprite_width: float = 0.045,
    sprite_alpha: float = 0.85,
    rollout_alpha: float = 0.45,
    trail_lw: float = 1.25,
    show_all_trails: bool = True,
    show_nominal: bool = True,
    dpi: int = 250,
) -> None:
    data = np.load(npz_path, allow_pickle=False)

    xs = np.asarray(get_key(data, "xs"), dtype=float)       # (R,T,3)
    if xs.ndim != 3 or xs.shape[-1] < 3:
        raise ValueError(f"Expected xs with shape (R,T,3); got {xs.shape}")

    obstacles = np.asarray(get_key(data, "obstacles", required=False, default=np.zeros((0, 3))), dtype=float)
    if obstacles.size == 0:
        centers = np.zeros((0, 2))
        radii = np.zeros((0,))
    else:
        centers = obstacles[:, :2]
        radii = obstacles[:, 2]

    dt = float(np.asarray(get_key(data, "dt", required=False, default=np.array(0.1))).reshape(()))
    if fps is None:
        fps = max(1, int(round(1.0 / dt)))

    plan_xy, lower_xy, upper_xy = normalize_plan_tubes(data)
    sprite_rgba = load_sprite_rgba(sprite_path)

    # Downsample frames only; keep plotted trails continuous up to the selected frame index.
    stride = max(int(stride), 1)
    frame_ids = np.arange(0, xs.shape[1], stride)
    if frame_ids[-1] != xs.shape[1] - 1:
        frame_ids = np.append(frame_ids, xs.shape[1] - 1)

    # Axis limits from rollouts, plan, tube, obstacles.
    points = [finite_xy(xs), finite_xy(plan_xy), finite_xy(lower_xy), finite_xy(upper_xy)]
    if centers.size:
        obstacle_bounds = np.vstack([centers - radii[:, None], centers + radii[:, None]])
        points.append(obstacle_bounds)
    all_xy = np.vstack([p for p in points if p.size])
    xmin, ymin = all_xy.min(axis=0) - margin
    xmax, ymax = all_xy.max(axis=0) + margin

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(xmin), float(xmax))
    ax.set_ylim(float(ymin), float(ymax))
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_xlabel("$p_x$")
    ax.set_ylabel("$p_y$")
    ax.grid(True, alpha=0.25)

    # Obstacles.
    for c, r in zip(centers, radii):
        ax.add_patch(
            Circle(
                (float(c[0]), float(c[1])),
                float(r),
                facecolor="red",
                linewidth=2,
                alpha=0.45,
                zorder=1,
            )
        )

    # Robust tube rectangles.
    tube_boxes = PatchCollection([], alpha=0.16, zorder=2)
    ax.add_collection(tube_boxes)

    # Nominal plan.
    if show_nominal:
        nominal_line, = ax.plot(plan_xy[:, 0], plan_xy[:, 1], "--", lw=2.0, alpha=0.9, label="Nominal plan", zorder=3)
    else:
        nominal_line, = ax.plot([], [], "--", lw=2.0, alpha=0.9, label="Nominal plan", zorder=3)

    # Rollout trails: one trail and one sprite per rollout.
    n_rollouts = xs.shape[0]
    trail_lines = []
    sprites = []
    for r in range(n_rollouts):
        line, = ax.plot(
            [],
            [],
            color="darkorange",
            lw=trail_lw,
            alpha=0.75,
            zorder=4,
        )
        trail_lines.append(line)

        # Initialize at first finite pose, fallback to first row.
        finite_mask = np.isfinite(xs[r, :, :3]).all(axis=1)
        if finite_mask.any():
            init_idx = int(np.argmax(finite_mask))
            x0, y0, th0 = xs[r, init_idx, :3]
        else:
            x0, y0, th0 = 0.0, 0.0, 0.0

        sprite = make_sprite_artist(
            ax,
            sprite_rgba,
            float(x0),
            float(y0),
            float(th0),
            length=sprite_length,
            width=sprite_width,
            alpha=sprite_alpha,
            zorder=8,
        )
        set_sprite_visible(sprite, False)
        sprites.append(sprite)

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    legend_handles = [
        Patch(
            facecolor="red",
            edgecolor="red",
            alpha=0.45,
            label="Obstacle",
        ),
        Patch(
            facecolor="tab:blue",
            edgecolor="tab:blue",
            alpha=0.16,
            label="Robust tube",
        ),
        Line2D(
            [0], [0],
            color="darkorange",
            lw=2,
            label="Rollouts",
        ),
    ]

    if show_nominal:
        legend_handles.append(
            Line2D(
                [0], [0],
                color="C0",
                linestyle="--",
                lw=2,
                label="Nominal plan",
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="lower right",
        framealpha=0.9,
    )

    # Prebuild tube boxes once because your NPZ stores a single robust tube.
    rects = []
    for k in range(0, min(lower_xy.shape[0], upper_xy.shape[0]), max(int(tube_stride), 1)):
        lo = lower_xy[k]
        up = upper_xy[k]
        w = up[0] - lo[0]
        h = up[1] - lo[1]
        if np.isfinite([lo[0], lo[1], w, h]).all() and w >= 0.0 and h >= 0.0:
            rects.append(Rectangle((float(lo[0]), float(lo[1])), float(w), float(h)))
    tube_boxes.set_paths(rects)

    def init():
        for line in trail_lines:
            line.set_data([], [])
        for sprite in sprites:
            set_sprite_visible(sprite, False)
        time_text.set_text("")
        return [*trail_lines, *[s["im"] for s in sprites], tube_boxes, nominal_line, time_text]

    def update(frame_number: int):
        t = int(frame_ids[frame_number])

        for r in range(n_rollouts):
            valid = np.isfinite(xs[r, : t + 1, :3]).all(axis=1)
            if not valid.any():
                trail_lines[r].set_data([], [])
                set_sprite_visible(sprites[r], False)
                continue

            idxs = np.where(valid)[0]
            if show_all_trails:
                trail_lines[r].set_data(xs[r, idxs, 0], xs[r, idxs, 1])
            else:
                trail_lines[r].set_data([], [])

            last = idxs[-1]
            x, y, th = xs[r, last, :3]
            update_sprite_artist(sprites[r], float(x), float(y), float(th))
            set_sprite_visible(sprites[r], True)

        return [*trail_lines, *[s["im"] for s in sprites], tube_boxes, nominal_line, time_text]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        init_func=init,
        blit=True,
        interval=int(round(1000.0 / fps)),
    )

    out_path = Path(out_path)
    ext = out_path.suffix.lower()
    if ext == ".mp4":
        if not animation.FFMpegWriter.isAvailable():
            raise RuntimeError("ffmpeg is not available. Use --out rollouts_sprite.gif or install ffmpeg.")
        writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        ani.save(out_path, writer=writer, dpi=dpi)
    elif ext == ".gif":
        writer = animation.PillowWriter(fps=fps)
        ani.save(out_path, writer=writer, dpi=dpi)
    else:
        raise ValueError("Output must end in .mp4 or .gif")

    plt.close(fig)
    print(f"[Saved] {out_path}")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate Dubins rollout NPZ with car sprites.")
    parser.add_argument("--npz", default="dubins_rollout_render_data.npz", help="Path to rollout NPZ file.")
    parser.add_argument("--sprite", default="car_removed.png", help="Path to transparent car sprite PNG.")
    parser.add_argument("--out", default="rollouts_sprite.gif", help="Output .mp4 or .gif path.")
    parser.add_argument("--fps", type=int, default=None, help="Animation FPS. Defaults to round(1/dt).")
    parser.add_argument("--stride", type=int, default=1, help="Render every kth simulation step.")
    parser.add_argument("--tube-stride", type=int, default=1, help="Draw every kth tube rectangle.")
    parser.add_argument("--margin", type=float, default=0.25, help="Axis margin around data.")
    parser.add_argument("--sprite-length", type=float, default=0.15, help="Car sprite length in world units.")
    parser.add_argument("--sprite-width", type=float, default=0.075, help="Car sprite width in world units.")
    parser.add_argument("--sprite-alpha", type=float, default=0.85, help="Car sprite opacity.")
    parser.add_argument("--rollout-alpha", type=float, default=0.45, help="Trail opacity.")
    parser.add_argument("--dpi", type=int, default=250, help="Output DPI.")
    parser.add_argument("--hide-trails", action="store_true", help="Hide rollout trails and show only moving sprites.")
    parser.add_argument("--hide-nominal", action="store_true", help="Hide nominal planned trajectory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_rollout_sprite_animation(
        npz_path=args.npz,
        sprite_path=args.sprite,
        out_path=args.out,
        fps=30,
        stride=args.stride,
        tube_stride=args.tube_stride,
        margin=args.margin,
        sprite_length=args.sprite_length,
        sprite_width=args.sprite_width,
        sprite_alpha=args.sprite_alpha,
        rollout_alpha=args.rollout_alpha,
        show_all_trails=not args.hide_trails,
        show_nominal=not args.hide_nominal,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
