"""Reconstruct printable sheet music from a piano-tutorial video.

The videos this targets show a scrolling staff in the upper part of the frame
and falling notes / a keyboard below it, with a coloured playhead sweeping
left to right across the staff. Each sweep covers one screenful of music.

The trick used here is that the playhead is never in the way if you capture
each half of the staff at the right moment: the right half is grabbed while
the playhead is still near the left, and the left half once it has moved past
the right. Stitching the two halves gives a clean, cursor-free staff image.
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# --- Tuning constants -------------------------------------------------------
# These are fitted to one channel's video layout. When extraction fails on a
# new video it is almost always these, not the algorithm, that need adjusting;
# run with --show to see what the detector is doing.

PLAYHEAD_ARM_RATIO = 0.25  # capture the right half once the playhead passes here
PLAYHEAD_FIRE_RATIO = 0.85  # capture the left half and emit once it passes here
SMOOTHING_KERNEL_RATIO = 0.005
MAX_SMOOTHING_KERNEL = 3
BAR_POSITION_SATURATION_THRESHOLD = 10
PENTAGRAM_DETECTION_BAR_WIDTH = 20
MIN_CORNER_BRIGHTNESS = 230
BOUNDARY_JITTER_TOLERANCE = 10

# --- Page layout ------------------------------------------------------------
A4_WIDTH_PX = 2480  # A4 at 300 DPI
A4_HEIGHT_PX = 3508
TOP_BOTTOM_MARGIN_PX = 200


def get_bar_position(frame, kernel_size):
    """Return the x of the playhead, or None if no coloured bar is visible.

    The staff is greyscale, so the playhead is the only strongly saturated
    thing in the crop: the column with peak mean saturation is the cursor.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    kernel = np.ones(kernel_size) / kernel_size
    smoothed_sum = np.convolve(np.mean(hsv[:, :, 1], axis=0), kernel, mode="same")

    if np.max(smoothed_sum) > BAR_POSITION_SATURATION_THRESHOLD:
        return int(np.argmax(smoothed_sum))
    return None


def detect_pentagram_boundary(frame):
    """Locate the row splitting the staff area from the falling-note area.

    Only a narrow strip at the left edge is inspected. Its mean brightness
    tells us whether we are looking at white paper at all (as opposed to a
    title card), and the sharpest vertical brightness change within it is the
    bottom edge of the sheet music.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray[:, :PENTAGRAM_DETECTION_BAR_WIDTH]

    vertical_brightness = np.mean(gray, axis=1)
    average_corner_brightness = np.mean(
        vertical_brightness[:PENTAGRAM_DETECTION_BAR_WIDTH]
    )

    diffs = np.diff(vertical_brightness)
    k = PENTAGRAM_DETECTION_BAR_WIDTH
    # Sharpest change, ignoring the first/last k rows.
    brightness_drop_y = int(np.argmax(np.abs(diffs[k:-k])) + k)

    return average_corner_brightness, brightness_drop_y


def extract_bars(video_path, show=False, dump_dir=None):
    """Decode the video and return one greyscale image per staff line."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Frames per second: {fps}")
    print(f"Resolution: {width} x {height}")

    kernel_size = min(int(round(width * SMOOTHING_KERNEL_RATIO)), MAX_SMOOTHING_KERNEL)
    th1 = int(round(width * PLAYHEAD_ARM_RATIO))
    th2 = int(round(width * PLAYHEAD_FIRE_RATIO))
    mid = int(round(width * 0.5))

    bars = []
    trace = {"x": [], "y": []}  # playhead position over time, for --plot

    state = 0
    boundary_y = None
    bar_position = None
    cut_right = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if state == 0:
            average_corner_brightness, brightness_drop_y = detect_pentagram_boundary(frame)
            if average_corner_brightness > MIN_CORNER_BRIGHTNESS:
                boundary_y = brightness_drop_y
                state = 1
                print(f"Found pentagram boundary {boundary_y}")
        else:
            average_corner_brightness, current_boundary_y = detect_pentagram_boundary(frame)

            if average_corner_brightness < MIN_CORNER_BRIGHTNESS:
                print(
                    f"Lost pentagram boundary "
                    f"({average_corner_brightness:.1f} < {MIN_CORNER_BRIGHTNESS}), "
                    f"resetting state"
                )
                boundary_y = None
                state = 0
                continue
            if abs(current_boundary_y - boundary_y) > BOUNDARY_JITTER_TOLERANCE:
                print(f"Boundary moved from {boundary_y} to {current_boundary_y}, re-anchoring")
                boundary_y = current_boundary_y

            staff = frame[:boundary_y, :]
            bar_position = get_bar_position(staff, kernel_size)

            if bar_position is not None:
                trace["x"].append(frame_idx)
                trace["y"].append(bar_position)

                # Deliberately a chain of ifs rather than elifs: a fast
                # playhead may cross more than one threshold in a single frame.
                if state == 1 and bar_position > th1:
                    cut_right = staff[:, mid:].copy()
                    state = 2
                if state == 2 and bar_position > th2:
                    cut_left = staff[:, :mid].copy()
                    merged = np.hstack((cut_left, cut_right))
                    bars.append(cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY))
                    if dump_dir is not None:
                        cv2.imwrite(str(dump_dir / f"{len(bars):03d}.png"), merged)
                    print(f"frame {frame_idx}: captured staff line {len(bars)}")
                    state = 3
                if state == 3 and bar_position < th1:
                    state = 1

        if show:
            if boundary_y is not None:
                cv2.line(frame, (th1, 0), (th1, boundary_y), (0, 0, 0), 1)
                cv2.line(frame, (th2, 0), (th2, boundary_y), (0, 0, 0), 1)
                cv2.line(frame, (0, boundary_y), (width, boundary_y), (0, 0, 255), 2)
                if bar_position is not None:
                    cv2.line(frame, (bar_position, 0), (bar_position, boundary_y), (0, 0, 255), 2)
            cv2.imshow("video2sheet", frame)
            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    return bars, trace


def build_pages(bars, title):
    """Scale staff lines to A4 width and stack them into full pages."""
    resized = []
    for bar in bars:
        h, w = bar.shape
        new_h = int(h * (A4_WIDTH_PX / w))
        resized.append(cv2.resize(bar, (A4_WIDTH_PX, new_h), interpolation=cv2.INTER_LINEAR))

    title_bar = np.ones((TOP_BOTTOM_MARGIN_PX, A4_WIDTH_PX), dtype=np.uint8) * 255
    font, font_scale, font_thickness = cv2.FONT_HERSHEY_PLAIN, 2, 3
    text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
    text_x = (A4_WIDTH_PX - text_size[0]) // 2
    text_y = (TOP_BOTTOM_MARGIN_PX + text_size[1]) // 2
    cv2.putText(
        title_bar, title, (text_x, text_y), font, font_scale, (0,), font_thickness, cv2.LINE_AA
    )

    pages = []
    current_page = [title_bar]
    current_height = TOP_BOTTOM_MARGIN_PX

    for bar in resized:
        if current_height + bar.shape[0] > A4_HEIGHT_PX - TOP_BOTTOM_MARGIN_PX and current_page:
            pages.append(np.vstack(current_page))
            current_page = [np.ones((TOP_BOTTOM_MARGIN_PX, A4_WIDTH_PX), dtype=np.uint8) * 255]
            current_height = TOP_BOTTOM_MARGIN_PX
        current_page.append(bar)
        current_height += bar.shape[0]

    if len(current_page) > 1:
        pages.append(np.vstack(current_page))

    return pages


def pad_to_a4(page):
    """Pad a short page with white so every sheet prints as a full A4."""
    if page.shape[0] >= A4_HEIGHT_PX:
        return page
    pad = A4_HEIGHT_PX - page.shape[0]
    return np.pad(page, ((0, pad), (0, 0)), mode="constant", constant_values=255)


def save_pdf(pages, output_path):
    images = [Image.fromarray(pad_to_a4(page)).convert("RGB") for page in pages]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path, save_all=True, append_images=images[1:])


def plot_trace(trace, video_path):
    import matplotlib.pyplot as plt

    plt.plot(trace["x"], trace["y"])
    plt.xlabel("frame")
    plt.ylabel("playhead x")
    plt.title(Path(video_path).stem)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path, help="path to the tutorial video")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "sheets",
        help="where to write the PDF (default: ./sheets next to this script)",
    )
    parser.add_argument("--show", action="store_true", help="live detector overlay; q to abort")
    parser.add_argument("--plot", action="store_true", help="plot playhead position afterwards")
    parser.add_argument("--no-open", action="store_true", help="do not open the PDF when done")
    parser.add_argument(
        "--dump-bars",
        type=Path,
        metavar="DIR",
        help="also write each captured staff line as a PNG (input for experiments/)",
    )
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"No such video: {args.video}")

    if args.dump_bars is not None:
        args.dump_bars.mkdir(parents=True, exist_ok=True)

    bars, trace = extract_bars(args.video, show=args.show, dump_dir=args.dump_bars)

    if not bars:
        raise SystemExit(
            "No staff lines were captured. The detector never completed a playhead "
            "sweep, which usually means the tuning constants at the top of this file "
            "do not match this video's layout. Re-run with --show to diagnose."
        )

    title = args.video.stem
    output_path = args.output_dir / f"{title}.pdf"
    pages = build_pages(bars, title)
    save_pdf(pages, output_path)
    print(f"Wrote {len(pages)} page(s) from {len(bars)} staff lines to {output_path}")

    if args.plot:
        plot_trace(trace, args.video)
    if not args.no_open:
        subprocess.Popen(["open", str(output_path)])


if __name__ == "__main__":
    main()
