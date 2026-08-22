"""Render the README's animated detector demo.

Replays a stretch of a tutorial video with the same overlay that
`video2sheet.py --show` draws, and highlights each half of the staff at the
moment it is captured, which is the part of the algorithm a still cannot show.

Needs a local tutorial video; the repository ships none.

    python tools/make_demo_gif.py "videos/Some Tutorial.mp4" -o docs/detector.gif
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import video2sheet as v  # noqa: E402

BLACK = (0, 0, 0)
RED = (0, 0, 255)
RIGHT_TINT = (90, 170, 60)  # BGR
LEFT_TINT = (170, 110, 60)
FLASH_FRAMES = 4


def label(frame, text, colour):
    cv2.putText(frame, text, (26, 54), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 7, cv2.LINE_AA)
    cv2.putText(frame, text, (26, 54), cv2.FONT_HERSHEY_DUPLEX, 1.3, colour, 2, cv2.LINE_AA)


def tint(frame, x0, x1, colour, alpha=0.28):
    patch = frame[:, x0:x1]
    wash = np.full_like(patch, colour, np.uint8)
    frame[:, x0:x1] = cv2.addWeighted(patch, 1 - alpha, wash, alpha, 0)


def render(video, start, end, step, width, crop_pad, frame_ms, hold_ms):
    cap = cv2.VideoCapture(str(video))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    kernel = min(
        int(round(frame_width * v.SMOOTHING_KERNEL_RATIO)), v.MAX_SMOOTHING_KERNEL
    )
    th1 = int(round(frame_width * v.PLAYHEAD_ARM_RATIO))
    th2 = int(round(frame_width * v.PLAYHEAD_FIRE_RATIO))
    mid = frame_width // 2

    state, captured, flash = 1, 0, None
    frames, durations = [], []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for index in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break

        brightness, boundary = v.detect_pentagram_boundary(frame)
        if brightness < v.MIN_CORNER_BRIGHTNESS:
            continue
        x = v.get_bar_position(frame[:boundary, :], kernel)

        if x is not None:
            if state == 1 and x > th1:
                state, flash = 2, ("right", FLASH_FRAMES)
            if state == 2 and x > th2:
                state, captured, flash = 3, captured + 1, ("left", FLASH_FRAMES)
            if state == 3 and x < th1:
                state = 1

        if (index - start) % step:
            continue

        view = frame[: boundary + crop_pad].copy()
        duration = frame_ms
        if flash and flash[1] > 0:
            side, left = flash
            # Hold on the first frame a label appears, so it can be read.
            if left == FLASH_FRAMES:
                duration = hold_ms
            if side == "right":
                tint(view, mid, frame_width, RIGHT_TINT)
                label(view, "right half captured", (60, 190, 70))
            else:
                tint(view, 0, mid, LEFT_TINT)
                label(view, f"left half captured  ->  staff line {captured}", (200, 130, 60))
            flash = (side, left - 1)

        cv2.line(view, (th1, 0), (th1, boundary), BLACK, 3)
        cv2.line(view, (th2, 0), (th2, boundary), BLACK, 3)
        cv2.line(view, (0, boundary), (frame_width, boundary), RED, 4)
        if x is not None:
            cv2.line(view, (x, 0), (x, boundary), RED, 4)

        height = round(view.shape[0] * width / view.shape[1])
        view = cv2.resize(view, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)))
        durations.append(duration)

    cap.release()
    return frames, durations


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("video", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("docs/detector.gif"))
    p.add_argument("--start", type=int, default=6010)
    p.add_argument("--end", type=int, default=6880)
    p.add_argument("--step", type=int, default=10, help="keep one frame in N")
    p.add_argument("--width", type=int, default=820)
    p.add_argument(
        "--crop-pad",
        type=int,
        default=16,
        help="rows kept below the staff boundary; keep it tight, the video draws "
        "lyric captions just underneath",
    )
    p.add_argument("--duration", type=int, default=70, help="ms per frame")
    p.add_argument(
        "--hold",
        type=int,
        default=1000,
        help="ms to pause on the frame where a capture label appears",
    )
    p.add_argument("--colors", type=int, default=96)
    args = p.parse_args()

    frames, durations = render(
        args.video, args.start, args.end, args.step, args.width, args.crop_pad,
        args.duration, args.hold,
    )
    if not frames:
        raise SystemExit("No frames rendered; check --start/--end and the video.")

    frames = [f.quantize(colors=args.colors, method=Image.MEDIANCUT) for f in frames]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size = args.output.stat().st_size / 1024
    total = sum(durations) / 1000
    print(
        f"{args.output}: {len(frames)} frames, {frames[0].size[0]}x{frames[0].size[1]}, "
        f"{size:.0f} KB, {total:.1f}s ({durations.count(args.hold)} pauses)"
    )


if __name__ == "__main__":
    main()
