"""Synthetic tutorial videos for the end-to-end tests.

Reproduces the layout `extract_bars` expects — bright paper on top, a dark
falling-note band below, a saturated cursor sweeping across the staff — so the
suite exercises the real pipeline without shipping any video.

The geometry is chosen so MJPG compression cannot disturb the detector:

* 320x176, both multiples of 8, so there is no macroblock padding.
* The paper/notes boundary sits on row 112 (14 * 8), a block edge, with
  uniform blocks on either side. `detect_pentagram_boundary` then reports 111
  on every frame, and every captured staff line has an identical shape.
* Nothing — no mark, no cursor — ever enters columns 0..19, the strip the
  boundary detector reads. A cursor there would drag the mean brightness under
  MIN_CORNER_BRIGHTNESS and reset the state machine every lap.
* Marks are horizontal, so JPEG error lands in vertical frequencies and
  produces no chroma fringing that could read as a stray playhead.
"""

import cv2
import numpy as np
import pytest

WIDTH, HEIGHT = 320, 176
BOUNDARY_ROW = 112  # rows 0..111 are paper, 112.. are the falling-note band
DETECTED_BOUNDARY = BOUNDARY_ROW - 1  # what detect_pentagram_boundary reports

CURSOR_WIDTH = 8
CURSOR_BGR = (0, 0, 255)  # pure red: saturation 255
MARK_ROWS = (30, 50, 70, 90)  # fake staff lines
MARK_LEFT, MARK_RIGHT = 24, 312

# Never below 24: that keeps the cursor clear of the 20-column detection strip.
SWEEP_POSITIONS = tuple(range(24, WIDTH, 12))
FPS = 25.0


def _frame(cursor_x):
    """One frame of the fake tutorial, with the cursor at `cursor_x`."""
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    frame[:BOUNDARY_ROW] = 255
    for row in MARK_ROWS:
        frame[row : row + 3, MARK_LEFT:MARK_RIGHT] = 0
    frame[:BOUNDARY_ROW, cursor_x : cursor_x + CURSOR_WIDTH] = CURSOR_BGR
    return frame


def write_tutorial_video(path, sweeps, lead_in=()):
    """Write `sweeps` full playhead sweeps to `path`, returning it.

    `lead_in` is a sequence of cursor positions played before the first full
    sweep — used to drop the detector into the middle of a sweep.
    """
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write MJPG/AVI")

    for cursor_x in lead_in:
        writer.write(_frame(cursor_x))
    for _ in range(sweeps):
        for cursor_x in SWEEP_POSITIONS:
            writer.write(_frame(cursor_x))
    writer.release()
    return path


def peak_column_saturation(bgr):
    """Max mean column saturation — the quantity get_bar_position thresholds.

    Above BAR_POSITION_SATURATION_THRESHOLD means a cursor is present.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1], axis=0).max())


def _extract(video, tmp_path_factory, name):
    """Run the real pipeline over `video`, keeping the colour staff lines.

    The bars `extract_bars` returns are greyscale, so the cursor-free property
    cannot be checked on them. `--dump-bars` writes the colour originals, and
    those are what the assertions read.
    """
    from sheet_music_extractor import video2sheet as v

    dump_dir = tmp_path_factory.mktemp(name)
    bars, _ = v.extract_bars(video, dump_dir=dump_dir)
    return bars, dump_dir


@pytest.fixture(scope="session")
def sweeps():
    return 4


@pytest.fixture(scope="session")
def tutorial_video(tmp_path_factory, sweeps):
    path = tmp_path_factory.mktemp("video") / "Test Artist - Test Song - Piano Tutorial.avi"
    return write_tutorial_video(path, sweeps=sweeps)


@pytest.fixture(scope="session")
def extracted(tutorial_video, tmp_path_factory):
    """(bars, dump_dir) for the happy path. Session-scoped: decoding once."""
    return _extract(tutorial_video, tmp_path_factory, "bars")


@pytest.fixture(scope="session")
def midsweep_extracted(tmp_path_factory):
    """The detector acquires the staff with the playhead past both thresholds."""
    path = tmp_path_factory.mktemp("midsweep") / "midsweep.avi"
    # 288, 300 and 312 are all beyond PLAYHEAD_FIRE_RATIO * 320 = 272.
    write_tutorial_video(path, sweeps=2, lead_in=(288, 300, 312))
    return _extract(path, tmp_path_factory, "midsweep_bars")
