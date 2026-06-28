from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import animation


NPZ_PATH = "dubins_rollout_render_data.npz"
OUT_PATH = "disturbance_vs_bounds.gif"

FPS = 30
STRIDE = 1
DPI = 600
BITRATE = 15000

PLOT_ABSOLUTE_STATE = True

ROLLOUT_COLOR = "#ff8c00"
BOUND_COLOR = "#0057b8"
FILL_COLOR = "#4f8fd6"

FILL_ALPHA = 0.22
ROLLOUT_ALPHA = 0.95
ROLLOUT_LW = 0.9
BOUND_LW = 2.0

STATE_LABELS = [r"$p_x$", r"$p_y$", r"$\theta$"]


plt.rcParams.update({
    "font.size": 18,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False, 
    "ps.fonttype": 42,
})


def get_key(data: np.lib.npyio.NpzFile, *names: str, required: bool = True, default=None):
    for name in names:
        if name in data.files:
            return data[name]
    if required:
        raise KeyError(f"Missing required NPZ key. Tried: {names}. Available: {data.files}")
    return default


def finite_minmax(*arrays: np.ndarray, pad_frac: float = 0.08, min_pad: float = 1e-3):
    vals = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)

    if not vals:
        return -1.0, 1.0

    vals = np.concatenate(vals)
    lo = float(vals.min())
    hi = float(vals.max())
    pad = max((hi - lo) * pad_frac, min_pad)
    return lo - pad, hi + pad


def normalize_time_series(data: np.lib.npyio.NpzFile):
    xs = np.asarray(get_key(data, "disturbed", "xs"), dtype=float)
    if xs.ndim != 3 or xs.shape[-1] < 3:
        raise ValueError(f"Expected disturbed/xs with shape (R,T,3); got {xs.shape}")

    lower = np.asarray(get_key(data, "lower"), dtype=float)
    upper = np.asarray(get_key(data, "upper"), dtype=float)
    x_pred = np.asarray(get_key(data, "X_pred"), dtype=float)

    dt = float(np.asarray(get_key(data, "dt", required=False, default=np.array(0.1))).reshape(()))

    T = xs.shape[1]

    lower_post = lower[1 : T + 1, :3]
    upper_post = upper[1 : T + 1, :3]
    pred_post = x_pred[1 : T + 1, :3]

    if PLOT_ABSOLUTE_STATE:
        y = xs[:, :, :3]
        lo = lower_post
        hi = upper_post
        ylabel_prefix = "state"
    else:
        y = xs[:, :, :3] - pred_post[None, :, :]
        lo = lower_post - pred_post
        hi = upper_post - pred_post
        ylabel_prefix = "deviation"

    T = min(y.shape[1], lo.shape[0], hi.shape[0])
    y = y[:, :T, :]
    lo = lo[:T, :]
    hi = hi[:T, :]

    time = np.arange(T, dtype=float) * dt

    frame_ids = np.arange(0, T, max(int(STRIDE), 1))
    if frame_ids[-1] != T - 1:
        frame_ids = np.append(frame_ids, T - 1)

    return time, y, lo, hi, dt, ylabel_prefix, frame_ids


def main() -> None:
    data = np.load(NPZ_PATH, allow_pickle=False)
    time, y, lower, upper, dt, ylabel_prefix, frame_ids = normalize_time_series(data)

    n_rollouts, T, n_state = y.shape
    n_plot = min(3, n_state)

    fig, axes = plt.subplots(n_plot, 1, figsize=(10, 9.5), sharex=True)
    if n_plot == 1:
        axes = [axes]

    rollout_lines: list[list[plt.Line2D]] = []
    lower_lines = []
    upper_lines = []
    fill_patches = []

    for d, ax in enumerate(axes):
        ax.grid(True, alpha=0.25)

        ylo, yhi = finite_minmax(y[:, :, d], lower[:, d], upper[:, d])
        ax.set_xlim(float(time[0]), float(time[-1]))
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel(f"{STATE_LABELS[d]} {ylabel_prefix}")

        lower_line, = ax.plot(
            [],
            [],
            color=BOUND_COLOR,
            lw=BOUND_LW,
            zorder=6,
            solid_capstyle="round",
            label="Lower bound",
        )

        upper_line, = ax.plot(
            [],
            [],
            color=BOUND_COLOR,
            lw=BOUND_LW,
            zorder=6,
            solid_capstyle="round",
            label="Upper bound",
        )

        for ln in (lower_line, upper_line):
            ln.set_path_effects([
                pe.Stroke(linewidth=BOUND_LW + 1.0, foreground="white"),
                pe.Normal(),
            ])

        lower_lines.append(lower_line)
        upper_lines.append(upper_line)
        fill_patches.append(None)

        dim_rollout_lines = []
        for _ in range(n_rollouts):
            line, = ax.plot(
                [],
                [],
                color=ROLLOUT_COLOR,
                alpha=ROLLOUT_ALPHA,
                lw=ROLLOUT_LW,
                zorder=2,
            )
            dim_rollout_lines.append(line)

        rollout_lines.append(dim_rollout_lines)

    axes[-1].set_xlabel("time [s]")

    handles = [
        lower_lines[0],
        rollout_lines[0][0],
    ]
    labels = [
        "Tube bound",
        "Rollouts",
    ]
    axes[0].legend(handles, labels, loc="lower right", framealpha=0.9, bbox_to_anchor=(1.0, 0.0))

    time_text = axes[0].text(
        0.015,
        0.94,
        "",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        zorder=10,
    )

    def init():
        artists = []

        for d, ax in enumerate(axes):
            lower_lines[d].set_data(time, lower[:, d])
            upper_lines[d].set_data(time, upper[:, d])

            fill_patches[d] = ax.fill_between(
                time,
                lower[:, d],
                upper[:, d],
                color=FILL_COLOR,
                alpha=FILL_ALPHA,
                zorder=1,
            )

            artists.extend([
                fill_patches[d],
                lower_lines[d],
                upper_lines[d],
            ])

            for r in range(n_rollouts):
                rollout_lines[d][r].set_data([], [])
                artists.append(rollout_lines[d][r])

        time_text.set_text("")
        artists.append(time_text)
        return artists

    def update(frame_number: int):
        t_idx = int(frame_ids[frame_number])
        artists = []

        for d in range(n_plot):
            artists.extend([
                fill_patches[d],
                lower_lines[d],
                upper_lines[d],
            ])

            for r in range(n_rollouts):
                valid = np.isfinite(y[r, : t_idx + 1, d])

                if not valid.any():
                    rollout_lines[d][r].set_data([], [])
                else:
                    rollout_lines[d][r].set_data(
                        time[: t_idx + 1][valid],
                        y[r, : t_idx + 1, d][valid],
                    )

                artists.append(rollout_lines[d][r])

        artists.append(time_text)
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        init_func=init,
        blit=False,
        interval=int(round(1000.0 / FPS)),
    )

    out_path = Path(OUT_PATH)

    if out_path.suffix.lower() == ".mp4":
        if not animation.FFMpegWriter.isAvailable():
            raise RuntimeError("ffmpeg is not available. Change OUT_PATH to .gif or install ffmpeg.")
        writer = animation.FFMpegWriter(fps=FPS, bitrate=BITRATE)
        ani.save(out_path, writer=writer, dpi=DPI)

    elif out_path.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=FPS)
        ani.save(out_path, writer=writer, dpi=DPI)

    else:
        raise ValueError("OUT_PATH must end in .mp4 or .gif")

    plt.close(fig)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()