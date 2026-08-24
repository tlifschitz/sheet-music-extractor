"""Generate the "How it works" figures in the README.

Each figure shows both what the detector *sees* (real frames, annotated) and
what it *computes* (the actual signal, with the actual constants marked). The
signals come from a real run of `extract_bars`, collected through its
`observer` hook, so a figure cannot quietly disagree with the algorithm it
documents: retune a constant and the next run redraws the annotation.

The checked-in figures were made from

    videos/Billie Eilish - What Was I Made For？ - Accurate Piano Tutorial with Sheet Music.mp4

which is not in the repository — `videos/` is gitignored. That video is a good
subject because it opens on a dark intro card and fades the paper in over some
forty frames, so the brightness threshold is crossed gradually, on camera.
Another video will produce the same four figures at different frame numbers.

    python docs/make_figures.py "videos/<name>.mp4"

Takes about three minutes on a 1080p60 source: one full decode for the signals
and the staff lines, then a seek per illustrated frame.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # written to a file, never shown; same reason as --debug-gif

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from video2sheet import pipeline as v

# Echoes the debug overlay, so the figures and the GIF can be read as one
# legend: red is what the detector believes, black is a threshold it tests
# against. BLUE is for a measured signal, which the overlay has no way to draw.
RED = "#d92b2b"
BLACK = "#1a1a1a"
BLUE = "#2b6cb0"
GREEN = "#2f8f46"  # a half, at the moment it is captured
MUTED = "#98a2b3"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# --- Reading the video ------------------------------------------------------


def run(video_path):
    """Decode once, keeping the per-frame signals and the captured staff lines."""
    log = []
    bars, _ = v.extract_bars(video_path, observer=log.append)
    if not bars:
        raise SystemExit(
            "No staff lines were captured, so there is nothing to illustrate. "
            "See --debug-gif."
        )
    return log, bars


def frames_at(video_path, indices):
    """Fetch the handful of frames the panels draw, by seeking.

    The observer deliberately records scalars only: holding on to a thousand
    1080p frames to pick four out of them would cost gigabytes.
    """
    cap = cv2.VideoCapture(str(video_path))
    out = {}
    for i in sorted(set(indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i - 1)  # the observer counts from 1
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"Could not seek to frame {i}")
        out[i] = frame
    cap.release()
    return out


def first(log, event):
    return next(r for r in log if event in r["events"])


def rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def show(ax, frame, title=None, keep_x=False):
    h, w = frame.shape[:2]
    ax.imshow(rgb(frame), extent=[0, w, h, 0], aspect="auto", interpolation="antialiased")
    ax.set_yticks([])
    if keep_x:
        # set_xticks([]) would propagate through sharex and strip the scale off
        # the plot underneath, which is the one panel that needs it.
        ax.tick_params(axis="x", labelbottom=False, length=0)
    else:
        ax.set_xticks([])
    if title:
        ax.set_title(title)
    return ax


def label(ax, x, y, text, color, size=8, **kw):
    ax.text(
        x, y, text, color=color, fontsize=size, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=0.8, alpha=0.9),
        **kw,
    )


# --- 1. Find the staff ------------------------------------------------------


def figure_find_the_staff(video_path, log, out):
    """Three frames across the fade-in, plus the two signals that judge them."""
    acquired = first(log, "acquire")
    n = acquired["frame"]
    # Two rejected frames from the fade, spread across it, and the frame that
    # finally crosses. Picking them from the log rather than hardcoding keeps
    # the figure honest on a video that fades faster or not at all.
    dark = log[min(n // 2, n - 1)]
    rising = max(
        (r for r in log[:n] if r["brightness"] < v.MIN_CORNER_BRIGHTNESS),
        key=lambda r: r["brightness"],
        default=dark,
    )
    picks = [dark, rising, acquired]
    frames = frames_at(video_path, [r["frame"] for r in picks])

    fig = plt.figure(figsize=(13, 7.6), dpi=110)
    grid = fig.add_gridspec(2, 6, height_ratios=[1.25, 1], hspace=0.28, wspace=0.55)

    k = v.PENTAGRAM_DETECTION_BAR_WIDTH
    for i, rec in enumerate(picks):
        ax = fig.add_subplot(grid[0, 2 * i : 2 * i + 2])
        frame = frames[rec["frame"]]
        h, w = frame.shape[:2]
        show(ax, frame)

        # The strip the detector inspects, and the corner patch it averages.
        ax.add_patch(Rectangle((0, 0), k, h, fill=False, ec=BLUE, lw=1.6))
        ax.add_patch(Rectangle((0, 0), k, k, fill=False, ec=RED, lw=1.6))

        # Both regions are 20 px of 1920, so they are near-invisible at this
        # size and there is nothing to see inside them anyway — a flat patch of
        # paper. What matters is the number each one measures, so both get a
        # leader line out to it rather than a magnified crop.
        ax.annotate(
            f"{k}×{k} corner\nmean {rec['brightness']:.0f}",
            xy=(k, k * 0.6), xytext=(w * 0.26, h * 0.10),
            color=RED, fontsize=8, fontweight="bold", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=RED, lw=0.8),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
        )
        ax.annotate(
            f"{k}-px strip",
            xy=(k, h * 0.72), xytext=(w * 0.20, h * 0.72),
            color=BLUE, fontsize=8, fontweight="bold", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=BLUE, lw=0.8),
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2),
        )

        passes = rec["brightness"] > v.MIN_CORNER_BRIGHTNESS
        ax.set_title(
            f"frame {rec['frame']} — corner {rec['brightness']:.0f}"
            f"{' >' if passes else ' <'} {v.MIN_CORNER_BRIGHTNESS}\n"
            + ("staff acquired" if passes else "not paper yet, stay in state 0"),
            color=GREEN if passes else BLACK,
        )
        if passes:
            y = rec["boundary_y"]
            ax.axhline(y, color=RED, lw=2)
            label(ax, w * 0.55, y - 26, f"boundary_y = {y}", RED, va="bottom")

    # The brightness the state-0 check thresholds, over the whole fade.
    ax = fig.add_subplot(grid[1, 0:3])
    span = log[: n + n // 2]
    ax.plot([r["frame"] for r in span], [r["brightness"] for r in span], color=BLUE, lw=1.5)
    ax.axhline(v.MIN_CORNER_BRIGHTNESS, color=BLACK, ls="--", lw=1.2)
    ax.text(
        span[0]["frame"], v.MIN_CORNER_BRIGHTNESS + 5,
        f"MIN_CORNER_BRIGHTNESS = {v.MIN_CORNER_BRIGHTNESS}", color=BLACK, fontsize=8,
    )
    ax.plot([n], [acquired["brightness"]], "o", color=RED, ms=7, zorder=5)
    ax.annotate(
        f"frame {n}: state 0 → 3",
        xy=(n, acquired["brightness"]), xytext=(n + len(span) * 0.06, 120),
        color=RED, fontsize=8,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
    )
    for rec in picks[:2]:
        ax.axvline(rec["frame"], color=MUTED, ls=":", lw=1)
    ax.set_xlabel("frame")
    ax.set_ylabel("mean corner brightness")
    ax.set_title("Is this white paper, or a title card?")
    ax.set_ylim(-10, 265)

    # Where inside the strip the paper stops. Rows run down the y axis, so the
    # profile reads in the same orientation as the frames above it.
    frame = frames[acquired["frame"]]
    strip = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :k]
    profile = strip.mean(axis=1)
    diffs = np.abs(np.diff(profile))
    rows = np.arange(len(profile))
    y = acquired["boundary_y"]

    for column, (values, xs, name) in enumerate(
        [(profile, rows, "brightness down the strip"), (diffs, rows[:-1], "|np.diff|")]
    ):
        ax = fig.add_subplot(grid[1, 3 + column])
        ax.plot(values, xs, color=BLUE, lw=1.2)
        ax.axhline(y, color=RED, lw=1.6)
        # argmax deliberately skips the first and last k rows.
        ax.axhspan(0, k, color=MUTED, alpha=0.35)
        ax.axhspan(len(profile) - k, len(profile), color=MUTED, alpha=0.35)
        ax.set_ylim(len(profile), 0)
        ax.set_title(name)
        ax.set_ylabel("row" if column == 0 else "")
        if column:
            label(ax, diffs.max() * 0.5, y - 40, f"sharpest change\nrow {y}", RED)
        else:
            ax.text(
                8, k + 26, f"first/last {k} rows\nignored", color=MUTED, fontsize=7,
                va="top",
            )

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --- 2. Track the playhead --------------------------------------------------


def figure_track_the_playhead(video_path, log, out):
    """The greyscale invariant, and the one-pass argmax it makes possible."""
    tracked = [r for r in log if r["x"] is not None and r["boundary_y"]]
    # Mid-sweep: the cursor is clear of both thresholds and of the frame edges,
    # so the saturation peak is unambiguous and nothing is clipped.
    target = 0.42 * max(r["x"] for r in tracked)
    rec = min(tracked, key=lambda r: abs(r["x"] - target))
    frame = frames_at(video_path, [rec["frame"]])[rec["frame"]]

    h, w = frame.shape[:2]
    y = rec["boundary_y"]
    staff = frame[:y, :]
    sat_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    sat_staff = sat_full[:y, :]
    column_sat = sat_staff.mean(axis=0)
    kernel_size = min(int(round(w * v.SMOOTHING_KERNEL_RATIO)), v.MAX_SMOOTHING_KERNEL)
    smoothed = np.convolve(column_sat, np.ones(kernel_size) / kernel_size, mode="same")

    fig, axes = plt.subplots(
        4, 1, figsize=(11.5, 12), dpi=104, sharex=True,
        gridspec_kw=dict(height_ratios=[h, h, y * 1.6, h * 0.62], hspace=0.16),
    )

    show(axes[0], frame, keep_x=True)
    axes[0].axhline(y, color=RED, lw=2)
    axes[0].axvline(rec["x"], color=RED, lw=2)
    label(axes[0], rec["x"] + 20, 40, f"playhead x = {rec['x']}", RED, va="top")
    label(axes[0], w * 0.62, y - 16, f"boundary_y = {y}", RED, va="bottom")
    axes[0].set_title(f"frame {rec['frame']}")

    axes[1].imshow(sat_full, cmap="magma", vmin=0, vmax=255,
                   extent=[0, w, h, 0], aspect="auto")
    axes[1].axhline(y, color="white", lw=1.6, ls="--")
    axes[1].set_yticks([])
    axes[1].tick_params(axis="x", labelbottom=False, length=0)
    axes[1].set_title(
        f"HSV saturation, whole frame — the falling notes reach {sat_full.max()}, "
        "which is why the crop comes first"
    )

    axes[2].imshow(sat_staff, cmap="magma", vmin=0, vmax=255,
                   extent=[0, w, y, 0], aspect="auto")
    axes[2].set_yticks([])
    axes[2].tick_params(axis="x", labelbottom=False, length=0)
    axes[2].set_title(
        f"the same channel above the boundary — median {np.median(sat_staff):.1f}: "
        "printed music is greyscale, so the cursor is nearly all that is left"
    )

    ax = axes[3]
    ax.plot(column_sat, color=MUTED, lw=0.9, label="mean saturation per column")
    ax.plot(smoothed, color=BLUE, lw=1.3, label=f"smoothed ({kernel_size} px)")
    ax.axhline(
        v.BAR_POSITION_SATURATION_THRESHOLD, color=BLACK, ls="--", lw=1.2,
        label=f"BAR_POSITION_SATURATION_THRESHOLD = {v.BAR_POSITION_SATURATION_THRESHOLD}",
    )
    ax.axvline(rec["x"], color=RED, lw=1.6)
    ax.plot([rec["x"]], [smoothed[rec["x"]]], "o", color=RED, ms=6, zorder=5)
    ax.annotate(
        f"argmax → x = {rec['x']}\npeak {smoothed.max():.1f}",
        xy=(rec["x"], smoothed.max()), xytext=(rec["x"] + w * 0.04, smoothed.max() * 0.78),
        color=RED, fontsize=8, arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
    )
    # Being honest about the smaller peaks: a few coloured marks in the score
    # clear the floor too. The threshold only answers "is a cursor on screen at
    # all"; which column it is at is settled by the argmax, not by the floor.
    ax.text(
        w * 0.56, smoothed.max() * 0.52,
        "the smaller peaks are coloured marks in the score.\n"
        "The threshold only asks whether a cursor is on screen —\n"
        "the argmax is what places it.",
        fontsize=7.5, color=MUTED, va="top",
    )
    ax.set_xlim(0, w)
    ax.set_xlabel("column (x)")
    ax.set_ylabel("saturation")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --- 3. Split capture -------------------------------------------------------


STATE_LABELS = {
    0: "0 · looking for the staff",
    1: "1 · armed, waiting for th1",
    2: "2 · right half held, waiting for th2",
    3: "3 · emitted, waiting for the wrap",
}
STATE_COLORS = {0: "#eceff3", 1: "#e8f1fb", 2: "#fdf2e3", 3: "#eaf6ec"}


def figure_split_capture(video_path, log, out):
    """One sweep, frozen at the two moments that matter."""
    arm = first(log, "arm")
    fire = next(r for r in log if "fire" in r["events"] and r["frame"] > arm["frame"])
    start = first(log, "acquire")["frame"]
    end = min(len(log), fire["frame"] + (fire["frame"] - arm["frame"]) // 3)
    span = [r for r in log if start <= r["frame"] <= end]

    frames = frames_at(video_path, [arm["frame"], fire["frame"]])
    w = frames[arm["frame"]].shape[1]
    th1 = int(round(w * v.PLAYHEAD_ARM_RATIO))
    th2 = int(round(w * v.PLAYHEAD_FIRE_RATIO))
    mid = int(round(w * 0.5))

    cut_right = frames[arm["frame"]][: arm["boundary_y"], mid:]
    cut_left = frames[fire["frame"]][: fire["boundary_y"], :mid]
    merged = np.hstack((cut_left, cut_right))

    fig = plt.figure(figsize=(13, 10), dpi=110)
    # Roomy hspace: the state legend hangs below the trace axes, in the gap.
    grid = fig.add_gridspec(3, 2, height_ratios=[1.15, 0.85, 0.62], hspace=0.55, wspace=0.06)

    # The sweep, as the state machine sees it.
    ax = fig.add_subplot(grid[0, :])
    runs, run_start = [], span[0]
    for prev, rec in zip(span, span[1:]):
        if rec["state"] != prev["state"]:
            runs.append((run_start["frame"], rec["frame"], prev["state"]))
            run_start = rec
    runs.append((run_start["frame"], span[-1]["frame"], span[-1]["state"]))
    for lo, hi, state in runs:
        ax.axvspan(lo, hi, color=STATE_COLORS[state], zorder=0)
        ax.text(
            (lo + hi) / 2, w * 0.985, str(state), ha="center", va="top",
            fontsize=11, fontweight="bold", color=MUTED, zorder=1,
        )

    xs = [r["frame"] for r in span if r["x"] is not None]
    ys = [r["x"] for r in span if r["x"] is not None]
    ax.plot(xs, ys, color=BLUE, lw=1.8, zorder=3)
    for value, name in [(th1, f"th1 = {th1}  ({v.PLAYHEAD_ARM_RATIO:.0%})"),
                        (th2, f"th2 = {th2}  ({v.PLAYHEAD_FIRE_RATIO:.0%})")]:
        ax.axhline(value, color=BLACK, ls="--", lw=1.2, zorder=2)
        ax.text(span[0]["frame"] + 4, value + w * 0.012, name, fontsize=8, color=BLACK)
    ax.axhline(mid, color=MUTED, ls=":", lw=1.1, zorder=2)
    ax.text(span[0]["frame"] + 4, mid + w * 0.012, f"mid = {mid}", fontsize=8, color=MUTED)

    # The arm marker sits low with empty space above and right of it; the fire
    # marker is near the top, so its callout drops into the gap under the
    # trace. Both carry a backing box, since the trace runs behind them.
    for rec, text, dx, dy in [
        (arm, "grab the RIGHT half", 0.05, w * 0.15),
        (fire, "grab the LEFT half,\nstitch and emit", -0.20, -w * 0.30),
    ]:
        ax.plot([rec["frame"]], [rec["x"]], "o", color=RED, ms=8, zorder=5)
        ax.annotate(
            f"frame {rec['frame']} · x = {rec['x']}\n{text}",
            xy=(rec["frame"], rec["x"]),
            xytext=(rec["frame"] + (end - start) * dx, rec["x"] + dy),
            color=RED, fontsize=8.5, fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.3),
        )
    ax.set_xlim(start, end)
    ax.set_ylim(0, w)
    ax.set_xlabel("frame")
    ax.set_ylabel("playhead x")
    ax.set_title("One sweep. The state is the shaded band; the two markers are the captures.")
    # Below the axes: every corner inside them is either trace or callout.
    ax.legend(
        handles=[Rectangle((0, 0), 1, 1, fc=STATE_COLORS[s], ec=MUTED, lw=0.5)
                 for s in sorted(STATE_COLORS)],
        labels=[STATE_LABELS[s] for s in sorted(STATE_COLORS)],
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=7.5,
        frameon=False, handlelength=1.4, columnspacing=1.4,
    )

    # The same two frames, cropped to the staff, with the captured half lit up.
    for column, (rec, lo, hi, name) in enumerate(
        [(arm, mid, w, "RIGHT half"), (fire, 0, mid, "LEFT half")]
    ):
        ax = fig.add_subplot(grid[1, column])
        staff = frames[rec["frame"]][: rec["boundary_y"], :]
        show(ax, staff)
        sh = staff.shape[0]
        ax.add_patch(Rectangle((lo, 0), hi - lo, sh, color=GREEN, alpha=0.16))
        ax.add_patch(Rectangle((lo, 0), hi - lo, sh, fill=False, ec=GREEN, lw=2))
        ax.axvline(rec["x"], color=RED, lw=2)
        label(ax, (lo + hi) / 2, sh * 0.12, f"{name} captured", GREEN, ha="center")
        ax.set_title(
            f"frame {rec['frame']} — cursor at {rec['x']}, "
            f"{'still left of' if rec['x'] < mid else 'well past'} the middle"
        )

    ax = fig.add_subplot(grid[2, :])
    show(ax, merged)
    ax.axvline(mid, color=GREEN, ls="--", lw=1.6)
    label(ax, mid + 14, merged.shape[0] * 0.16, "seam — same rendering, so it lines\n"
          "up to the pixel", GREEN, va="top")
    ax.set_title(
        f"np.hstack of the two halves: one staff line, {merged.shape[1]}×{merged.shape[0]}, "
        "with the cursor in neither"
    )

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --- 4. Lay out pages -------------------------------------------------------


MAX_THUMBS = 4


def _justification_gap(head_height, line_heights):
    """Where the first justification gap falls on a page, and how tall it is.

    Mirrors the arithmetic in `_justify`, which spreads the slack rather than
    letting it pool at the foot. Returns None when there is nothing to spread.
    """
    if len(line_heights) < 2:
        return None
    used = head_height + sum(line_heights)
    slack = v.A4_HEIGHT_PX - v.BOTTOM_MARGIN_PX - used
    gap = min(max(slack // (len(line_heights) - 1), 0), v.MAX_EXTRA_GAP_PX)
    if gap == 0:
        return None
    return head_height + line_heights[0], gap


def figure_lay_out_pages(bars, title, artist, out):
    """What the page packer does with the staff lines, and what --dense changes."""
    balanced = v.build_pages(bars, title, artist, balance=True)
    dense = v.build_pages(bars, title, artist, balance=False)

    heights = [
        int(bar.shape[0] * (v.A4_WIDTH_PX / bar.shape[1])) for bar in bars
    ]
    counts = {
        "default": v._balanced(heights, v.TITLE_BLOCK_HEIGHT_PX, v.TOP_MARGIN_PX),
        "--dense": v._pack(heights, v.TITLE_BLOCK_HEIGHT_PX, v.TOP_MARGIN_PX),
    }

    shown = min(MAX_THUMBS, max(len(balanced), len(dense)))
    fig = plt.figure(figsize=(13, 8.2), dpi=110)
    grid = fig.add_gridspec(2, shown + 1, width_ratios=[1] * shown + [1.5], wspace=0.22,
                            hspace=0.22)

    for row, (name, pages) in enumerate([("default", balanced), ("--dense", dense)]):
        for column in range(shown):
            ax = fig.add_subplot(grid[row, column])
            ax.set_xticks([])
            ax.set_yticks([])
            if column >= len(pages):
                ax.axis("off")
                continue
            page = v.pad_to_a4(pages[column])
            ax.imshow(page, cmap="gray", vmin=0, vmax=255,
                      extent=[0, v.A4_WIDTH_PX, v.A4_HEIGHT_PX, 0], aspect="auto",
                      interpolation="antialiased")
            for spine in ax.spines.values():
                spine.set_edgecolor(MUTED)
            ax.set_title(
                f"{name} · page {column + 1}"
                + (f" of {len(pages)}" if column + 1 == shown else "")
                + f" — {counts[name][column]} lines",
                fontsize=8.5,
            )
            # One constant per page, on the balanced row, so the labels never
            # stack on top of each other. Each is annotated on the page where
            # it actually applies: the title block only exists on page one, the
            # plain top margin only on the pages after it.
            if row:
                continue
            centre = v.A4_WIDTH_PX / 2
            if column == 0:
                ax.axhline(v.TITLE_BLOCK_HEIGHT_PX, color=BLUE, lw=1.2, ls="--")
                label(ax, centre, v.TITLE_BLOCK_HEIGHT_PX + 70,
                      f"TITLE_BLOCK_HEIGHT_PX\n= {v.TITLE_BLOCK_HEIGHT_PX}", BLUE,
                      size=6.5, ha="center", va="top")
                gap = _justification_gap(
                    v.TITLE_BLOCK_HEIGHT_PX, heights[: counts["default"][0]]
                )
                if gap:
                    top, size = gap
                    ax.axhspan(top, top + size, color=GREEN, alpha=0.5)
                    ax.annotate(
                        f"slack spread between\nthe lines: {size} px",
                        xy=(v.A4_WIDTH_PX * 0.80, top + size / 2),
                        xytext=(v.A4_WIDTH_PX * 0.16, top + 560),
                        color=GREEN, fontsize=6.5, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2),
                    )
            elif column == 1:
                ax.axhline(v.TOP_MARGIN_PX, color=BLUE, lw=1.2, ls="--")
                label(ax, centre, v.TOP_MARGIN_PX + 70,
                      f"TOP_MARGIN_PX = {v.TOP_MARGIN_PX}", BLUE, size=6.5,
                      ha="center", va="top")
            elif column == 2:
                limit = v.A4_HEIGHT_PX - v.BOTTOM_MARGIN_PX
                ax.axhline(limit, color=BLACK, lw=1.2, ls="--")
                label(ax, v.A4_WIDTH_PX * 0.02, limit - 70,
                      "A4_HEIGHT_PX − BOTTOM_MARGIN_PX\n"
                      f"= {limit}: nothing may cross", BLACK, size=6.5,
                      ha="left", va="bottom")

    # The packing decision itself, which is hard to read off a thumbnail.
    ax = fig.add_subplot(grid[:, shown])
    width = 0.38
    for offset, (name, colour) in zip(
        (-width / 2, width / 2), [("default", BLUE), ("--dense", MUTED)]
    ):
        values = counts[name]
        ax.bar(
            np.arange(len(values)) + offset, values, width, label=name, color=colour,
        )
        for i, value in enumerate(values):
            ax.text(i + offset, value + 0.08, str(value), ha="center", fontsize=8,
                    color=colour, fontweight="bold")
    ax.set_xticks(np.arange(max(len(c) for c in counts.values())))
    ax.set_xticklabels(
        [f"p{i + 1}" for i in range(max(len(c) for c in counts.values()))]
    )
    ax.set_ylabel("staff lines on the page")
    ax.set_title(
        f"{len(bars)} staff lines over {len(balanced)} pages.\n"
        "Balancing spends the same paper, more evenly."
    )
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(max(c) for c in counts.values()) + 1)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --- 5. The barline artifact ------------------------------------------------


def figure_barline_artifact(video_path, log, out):
    """Why a black barline reports more saturation than the coloured cursor.

    HSV saturation is (max - min) / max, so the denominator collapses on dark
    pixels and a few units of compression noise across the channels read as
    strong colour. This is the one rival the greyscale invariant does not
    dispose of, and it is invisible on screen — hence a figure.
    """
    tracked = [r for r in log if r["x"] is not None and r["boundary_y"]]
    target = 0.42 * max(r["x"] for r in tracked)
    rec = min(tracked, key=lambda r: abs(r["x"] - target))  # same frame as figure 2
    frame = frames_at(video_path, [rec["frame"]])[rec["frame"]]

    y = rec["boundary_y"]
    crop = frame[:y, :]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1].astype(float), hsv[:, :, 2].astype(float)

    # The offender: the column with the most full-height saturation that is
    # also dark. Searching rather than hardcoding, so this works on any video.
    column_median = np.array([np.median(sat[:, x]) for x in range(crop.shape[1])])
    dark = val.mean(axis=0) < 200
    if not dark.any():
        print(f"no dark column in frame {rec['frame']}; skipping {out.name}")
        return
    bar = int(np.where(dark, column_median, 0).argmax())
    lo, hi = bar - 15, bar + 29

    fig = plt.figure(figsize=(13, 6.4), dpi=110)
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.8], wspace=0.32)

    for column, (image, cmap, vmax, title) in enumerate([
        (cv2.cvtColor(crop[:, lo:hi], cv2.COLOR_BGR2RGB), None, None,
         "what you see\na black barline"),
        (val[:, lo:hi], "gray", 255, "V (brightness)\nthe line is near zero"),
        (sat[:, lo:hi], "magma", 80, "S (saturation)\nthe line blazes"),
    ]):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=vmax,
                  extent=[lo, hi, y, 0], aspect="auto", interpolation="nearest")
        ax.axvline(bar, color=RED, lw=0.8, ls=":")
        ax.set_yticks([])
        ax.set_xticks([lo, bar, hi - 1])
        ax.set_title(title, fontsize=9)

    ax = fig.add_subplot(grid[0, 3])
    xs = np.arange(max(0, bar - 30), min(crop.shape[1], bar + 30))
    ax.plot(xs, column_median[xs], color=RED, lw=1.4, label="median saturation")
    ax.plot(xs, val.mean(axis=0)[xs], color=BLACK, lw=1.4, label="mean brightness")
    ax.axvline(bar, color=RED, lw=0.8, ls=":")
    ax.set_xlabel("column (x)")
    ax.legend(fontsize=8, loc="center right")
    ax.set_title("the saturation spike is exactly\nthe brightness trough", fontsize=9)

    # The arithmetic, on one real pixel of each. Not the darkest pixel of the
    # line: at pure black every channel ties, S degenerates to 0, and the point
    # is lost. A representative dark pixel is where the ratio misbehaves.
    dim = np.flatnonzero((val[:, bar] > 0) & (val[:, bar] < 100))
    row = int(dim[np.argsort(sat[dim, bar])[len(dim) // 2]]) if len(dim) else 0
    ink = tuple(int(c) for c in crop[row, bar])
    cursor = tuple(int(c) for c in crop[crop.shape[0] // 2, rec["x"]])
    fig.suptitle(
        f"frame {rec['frame']}, x = {bar}: a barline out-saturates the cursor.\n"
        f"S is (max − min) / max, so ink BGR{ink} scores "
        f"{int(255 * (max(ink) - min(ink)) / max(max(ink), 1))} on a channel spread of "
        f"{max(ink) - min(ink)}, while the cursor's BGR{cursor} scores "
        f"{int(255 * (max(cursor) - min(cursor)) / max(max(cursor), 1))} on a spread of "
        f"{max(cursor) - min(cursor)}.",
        fontsize=9.5,
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --- Entry point ------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path, help="the tutorial video to illustrate")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path(__file__).parent,
        help="where to write the PNGs (default: docs/)",
    )
    parser.add_argument(
        "--only", type=int, choices=(1, 2, 3, 4, 5), action="append",
        help="regenerate only this figure (repeatable)",
    )
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"No such video: {args.video}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log, bars = run(args.video)
    title, artist = v.split_title(args.video.stem)
    wanted = set(args.only or (1, 2, 3, 4, 5))

    if 1 in wanted:
        figure_find_the_staff(args.video, log, args.output_dir / "step-1-find-the-staff.png")
    if 2 in wanted:
        figure_track_the_playhead(
            args.video, log, args.output_dir / "step-2-track-the-playhead.png"
        )
    if 3 in wanted:
        figure_split_capture(args.video, log, args.output_dir / "step-3-split-capture.png")
    if 4 in wanted:
        figure_lay_out_pages(
            bars, title, artist, args.output_dir / "step-4-lay-out-pages.png"
        )
    if 5 in wanted:
        figure_barline_artifact(
            args.video, log, args.output_dir / "barline-artifact.png"
        )


if __name__ == "__main__":
    main()
