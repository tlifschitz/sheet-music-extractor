"""Detector tests against synthetic frames.

No video fixtures: every frame here is built with numpy, so the tests stay
fast, deterministic, and independent of any particular tutorial video.
"""

import cv2
import numpy as np
import pytest

from video2sheet import pipeline as v


def paper_frame(height=400, width=600, split=200, paper=250, dark=20):
    """White 'paper' above `split`, dark falling-note area below."""
    frame = np.full((height, width, 3), dark, np.uint8)
    frame[:split] = paper
    return frame


def saturated_column(x, height=100, width=400, grey=200):
    """A greyscale frame with a single strongly saturated column at `x`."""
    frame = np.full((height, width, 3), grey, np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, x, 1] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class TestPentagramBoundary:
    def test_finds_the_brightness_edge(self):
        brightness, y = v.detect_pentagram_boundary(paper_frame(split=200))
        assert brightness > v.MIN_CORNER_BRIGHTNESS
        # The edge sits on the last bright row.
        assert y == pytest.approx(199, abs=1)

    @pytest.mark.parametrize("split", [120, 200, 275, 310])
    def test_tracks_the_edge_wherever_it_is(self, split):
        _, y = v.detect_pentagram_boundary(paper_frame(split=split))
        assert y == pytest.approx(split - 1, abs=1)

    def test_dark_frame_reads_as_no_paper(self):
        """A title card must not be mistaken for a staff."""
        brightness, _ = v.detect_pentagram_boundary(paper_frame(paper=60, dark=40))
        assert brightness < v.MIN_CORNER_BRIGHTNESS

    def test_only_the_left_strip_is_inspected(self):
        """Bright content on the right must not affect the verdict."""
        frame = np.full((400, 600, 3), 20, np.uint8)
        frame[:, 300:] = 255
        brightness, _ = v.detect_pentagram_boundary(frame)
        assert brightness < v.MIN_CORNER_BRIGHTNESS


class TestBarPosition:
    def test_locates_the_saturated_column(self):
        # Smoothing biases the peak by up to a kernel width; that is expected.
        assert v.get_bar_position(saturated_column(137), 3) == pytest.approx(137, abs=2)

    @pytest.mark.parametrize("x", [40, 137, 250, 380])
    def test_follows_the_column_across_the_frame(self, x):
        assert v.get_bar_position(saturated_column(x), 3) == pytest.approx(x, abs=2)

    def test_returns_none_without_a_playhead(self):
        """A plain greyscale staff has no cursor to report."""
        flat = np.full((100, 400, 3), 200, np.uint8)
        assert v.get_bar_position(flat, 3) is None

    def test_faint_colour_is_below_threshold(self):
        frame = np.full((100, 400, 3), 200, np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, 137, 1] = 5  # far under BAR_POSITION_SATURATION_THRESHOLD
        faint = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        assert v.get_bar_position(faint, 3) is None
