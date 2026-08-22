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
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
TOP_MARGIN_PX = 200
# The foot of the page carries only the page number, so it needs less room
# than the head. Tying both to one constant used to reject a staff line that
# missed the reserve by a few millimetres.
BOTTOM_MARGIN_PX = 170

# --- Title block ------------------------------------------------------------
TITLE_BLOCK_HEIGHT_PX = 360  # taller than a plain margin: it holds the heading
TITLE_SIZE_PX = 112
ARTIST_SIZE_PX = 54
PAGE_NUMBER_SIZE_PX = 34
MAX_EXTRA_GAP_PX = 130  # ceiling on justification, so sparse pages do not sprawl
TITLE_MAX_WIDTH_RATIO = 0.86  # shrink the title rather than let it run to the edge
ARTIST_GREY = 90

# Junk that YouTube titles carry and a score should not.
BOILERPLATE_RE = re.compile(
    r"(accurate\s+)?piano\s+tutorial.*|with\s+sheet\s+music|\+\s*sheets?|synthesia|"
    r"\(?official\s+video\)?|tutorial",
    re.IGNORECASE,
)

# yt-dlp swaps characters that are illegal in filenames for fullwidth
# lookalikes; map them back so the heading reads normally.
FULLWIDTH = str.maketrans({"？": "?", "：": ":", "／": "/", "＂": '"', "＊": "*", "｜": "|", "＜": "<", "＞": ">"})


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


@lru_cache(maxsize=None)
def _font(size, bold=False):
    """A serif face that is identical on every platform.

    matplotlib ships DejaVu, so this needs no system font lookup and renders
    the same way locally and in CI. Falls back to Pillow's built-in scalable
    font if matplotlib is not installed.
    """
    try:
        import matplotlib

        name = "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf"
        path = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / name
        return ImageFont.truetype(str(path), size)
    except Exception:
        return ImageFont.load_default(size=size)


def split_title(stem):
    """Turn a downloaded filename into (title, artist).

    "Billie Eilish - What Was I Made For？ - Accurate Piano Tutorial with
    Sheet Music" becomes ("What Was I Made For?", "Billie Eilish"). Anything
    that does not look like "artist - song - boilerplate" is passed through
    unchanged as the title, with no artist line.
    """
    stem = stem.translate(FULLWIDTH)
    parts = [p.strip() for p in stem.split(" - ")]
    parts = [p for p in parts if p and not BOILERPLATE_RE.fullmatch(p.strip())]

    if len(parts) >= 2:
        return " - ".join(parts[1:]), parts[0]
    if len(parts) == 1:
        return parts[0], None
    return stem, None


def _draw_centred(draw, text, font, y, width, fill=0):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(((width - (right - left)) // 2 - left, y - top), text, font=font, fill=fill)
    return bottom - top


def _fitted_font(draw, text, size, bold, max_width):
    """Shrink the type until the line fits, rather than letting it overflow."""
    while size > 24:
        font = _font(size, bold)
        left, _, right, _ = draw.textbbox((0, 0), text, font=font)
        if right - left <= max_width:
            return font
        size -= 4
    return _font(size, bold)


def make_title_block(title, artist=None):
    """Render the heading that opens the score."""
    block = Image.new("L", (A4_WIDTH_PX, TITLE_BLOCK_HEIGHT_PX), 255)
    draw = ImageDraw.Draw(block)
    max_width = int(A4_WIDTH_PX * TITLE_MAX_WIDTH_RATIO)

    font = _fitted_font(draw, title, TITLE_SIZE_PX, True, max_width)
    title_height = _draw_centred(draw, title, font, TITLE_BLOCK_HEIGHT_PX // 3, A4_WIDTH_PX)

    if artist:
        font = _fitted_font(draw, artist, ARTIST_SIZE_PX, False, max_width)
        _draw_centred(
            draw,
            artist,
            font,
            TITLE_BLOCK_HEIGHT_PX // 3 + title_height + 46,
            A4_WIDTH_PX,
            fill=ARTIST_GREY,
        )

    return np.asarray(block)


def add_page_number(page, number, total):
    """Stamp "n / total" into the bottom margin."""
    if total < 2:
        return page
    img = Image.fromarray(page)
    draw = ImageDraw.Draw(img)
    _draw_centred(
        draw,
        f"{number} / {total}",
        _font(PAGE_NUMBER_SIZE_PX),
        A4_HEIGHT_PX - BOTTOM_MARGIN_PX // 2,
        A4_WIDTH_PX,
        fill=ARTIST_GREY,
    )
    return np.asarray(img)


def _pack(heights, first_start, rest_start, max_lines=None):
    """Greedily fill pages, returning how many staff lines land on each.

    `max_lines` optionally caps the lines per page, which is what lets the
    balancing pass even the pages out.
    """
    limit = A4_HEIGHT_PX - BOTTOM_MARGIN_PX
    pages, count, height = [], 0, first_start

    for h in heights:
        overflows = height + h > limit
        at_cap = max_lines is not None and count >= max_lines
        if (overflows or at_cap) and count:
            pages.append(count)
            count, height = 0, rest_start
        count += 1
        height += h

    if count:
        pages.append(count)
    return pages


def _fits(counts, heights, first_start, rest_start):
    """Would this split actually fit on the pages?"""
    limit = A4_HEIGHT_PX - BOTTOM_MARGIN_PX
    index = 0
    for page_number, count in enumerate(counts):
        start = first_start if page_number == 0 else rest_start
        if start + sum(heights[index : index + count]) > limit:
            return False
        index += count
    return True


def _balanced(heights, first_start, rest_start):
    """Spread the staff lines evenly, without spending an extra page.

    Plain greedy packing crams the early pages and leaves the last one nearly
    empty (6/6/3 for fifteen lines). This divides them as evenly as the page
    count allows (5/5/5), falling back to greedy if that does not fit.
    """
    pages = _pack(heights, first_start, rest_start)
    if len(pages) < 2:
        return pages

    base, extra = divmod(len(heights), len(pages))
    # The first page is shorter than the rest, since it carries the title, so
    # try handing the extra lines to the later pages first.
    for counts in ([base] * (len(pages) - extra) + [base + 1] * extra,
                   [base + 1] * extra + [base] * (len(pages) - extra)):
        if all(counts) and _fits(counts, heights, first_start, rest_start):
            return counts

    target = -(-len(heights) // len(pages))  # ceil
    for cap in range(target, max(pages) + 1):
        candidate = _pack(heights, first_start, rest_start, cap)
        if len(candidate) == len(pages):
            return candidate
    return pages


def _justify(head, lines):
    """Spread the leftover height between the staff lines.

    Balancing decides how many lines land on each page; without this the
    slack all collects at the foot and reads as a line that failed to fit.
    Engraved scores distribute it between the systems instead.
    """
    blocks = [head] + lines
    if len(lines) < 2:
        return np.vstack(blocks)

    used = head.shape[0] + sum(line.shape[0] for line in lines)
    slack = A4_HEIGHT_PX - BOTTOM_MARGIN_PX - used
    gap = min(max(slack // (len(lines) - 1), 0), MAX_EXTRA_GAP_PX)
    if gap == 0:
        return np.vstack(blocks)

    spacer = np.full((gap, A4_WIDTH_PX), 255, np.uint8)
    stacked = [head, lines[0]]
    for line in lines[1:]:
        stacked += [spacer, line]
    return np.vstack(stacked)


def build_pages(bars, title, artist=None, balance=True):
    """Scale staff lines to A4 width and stack them into pages.

    With `balance` the lines are spread evenly over the pages and the
    remaining height is distributed between them; without it, pages are
    packed as full as they go.
    """
    resized = []
    for bar in bars:
        h, w = bar.shape
        new_h = int(h * (A4_WIDTH_PX / w))
        resized.append(cv2.resize(bar, (A4_WIDTH_PX, new_h), interpolation=cv2.INTER_LINEAR))

    if not resized:
        return []

    title_block = make_title_block(title, artist)
    heights = [bar.shape[0] for bar in resized]

    counts = (_balanced if balance else _pack)(heights, title_block.shape[0], TOP_MARGIN_PX)

    pages, index = [], 0
    for page_number, count in enumerate(counts):
        head = title_block if page_number == 0 else np.full(
            (TOP_MARGIN_PX, A4_WIDTH_PX), 255, np.uint8
        )
        lines = resized[index : index + count]
        pages.append(_justify(head, lines) if balance else np.vstack([head] + lines))
        index += count

    return pages


def pad_to_a4(page):
    """Pad a short page with white so every sheet prints as a full A4."""
    if page.shape[0] >= A4_HEIGHT_PX:
        return page
    pad = A4_HEIGHT_PX - page.shape[0]
    return np.pad(page, ((0, pad), (0, 0)), mode="constant", constant_values=255)


def save_pdf(pages, output_path):
    images = [
        Image.fromarray(add_page_number(pad_to_a4(page), i, len(pages))).convert("RGB")
        for i, page in enumerate(pages, 1)
    ]

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
        "--dense",
        action="store_true",
        help="pack pages as full as possible instead of spreading lines evenly",
    )
    parser.add_argument("--title", help="override the heading (default: parsed from the filename)")
    parser.add_argument("--artist", help="override the artist line under the heading")
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

    parsed_title, parsed_artist = split_title(args.video.stem)
    title = args.title or parsed_title
    artist = args.artist if args.artist is not None else parsed_artist

    output_path = args.output_dir / f"{args.video.stem}.pdf"
    pages = build_pages(bars, title, artist, balance=not args.dense)
    save_pdf(pages, output_path)
    print(f"Wrote {len(pages)} page(s) from {len(bars)} staff lines to {output_path}")

    if args.plot:
        plot_trace(trace, args.video)
    if not args.no_open:
        subprocess.Popen(["open", str(output_path)])


if __name__ == "__main__":
    main()
