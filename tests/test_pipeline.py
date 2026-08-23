"""End-to-end tests: a synthetic tutorial video through the real pipeline.

The rest of the suite covers the pieces in isolation. These cover the thing no
unit test can — that the pieces still fit together over time. The bug this
suite exists because of (a state machine that armed mid-sweep and stitched the
cursor into a staff line) was invisible to every unit test and was found by
eye, in a GIF.
"""

import sys

import cv2
import numpy as np
import pytest

from sheet_music_extractor import video2sheet as v

from conftest import (
    CURSOR_BGR,
    DETECTED_BOUNDARY,
    HEIGHT,
    WIDTH,
    peak_column_saturation,
    write_frames,
)


class TestHappyPath:
    def test_captures_one_staff_line_per_sweep(self, extracted, sweeps):
        """Exact, not >=: the synthetic video is deterministic, so an
        off-by-one here is a real defect, not noise."""
        bars, _ = extracted
        assert len(bars) == sweeps

    def test_no_captured_staff_line_contains_the_cursor(self, extracted):
        """The property the whole project rests on.

        If the playhead ends up inside a captured half, the split-capture
        timing failed and the score is corrupt. Checked on the --dump-bars
        PNGs because the bars extract_bars returns are already greyscale.
        """
        _, dump_dir = extracted
        pngs = sorted(dump_dir.glob("*.png"))
        assert pngs, "nothing was dumped, so this test would prove nothing"
        for png in pngs:
            saturation = peak_column_saturation(cv2.imread(str(png)))
            assert saturation < v.BAR_POSITION_SATURATION_THRESHOLD, (
                f"{png.name} still has a saturated column: {saturation:.1f}"
            )

    def test_the_measurement_can_see_a_cursor_at_all(self):
        """Control for the test above.

        Without this, desaturating everything by accident would make the
        cursor-free assertion pass forever while proving nothing.
        """
        frame = np.full((DETECTED_BOUNDARY, WIDTH, 3), 255, np.uint8)
        assert peak_column_saturation(frame) < v.BAR_POSITION_SATURATION_THRESHOLD

        frame[:, 100:108] = CURSOR_BGR
        assert peak_column_saturation(frame) > 200

    def test_every_staff_line_has_the_same_shape(self, extracted):
        """A boundary re-anchor between arming and firing once left the two
        halves at different heights, raising inside np.hstack."""
        bars, _ = extracted
        assert {bar.shape for bar in bars} == {(DETECTED_BOUNDARY, WIDTH)}

    def test_the_captured_staff_is_not_blank(self, extracted):
        """Catches capturing the wrong region entirely."""
        bars, _ = extracted
        assert bars[0].min() == 0  # the marks survived
        assert bars[0].max() == 255  # so did the paper


class TestMidSweepAcquisition:
    def test_does_not_stitch_the_cursor_into_a_staff_line(self, midsweep_extracted):
        """Regression for 51d1970.

        Entering at state 1 on boundary acquisition let a playhead that was
        already past both thresholds arm and fire in the same frame, stitching
        two halves grabbed at the same instant — with the cursor sitting inside
        one of them. Entering at state 3 waits for the playhead to wrap left.
        """
        bars, dump_dir = midsweep_extracted

        assert len(bars) == 2, "the partial opening sweep should be discarded"
        for png in sorted(dump_dir.glob("*.png")):
            saturation = peak_column_saturation(cv2.imread(str(png)))
            assert saturation < v.BAR_POSITION_SATURATION_THRESHOLD, (
                f"{png.name} was stitched mid-sweep: {saturation:.1f}"
            )


class TestLayoutFromRealBars:
    def test_lays_the_captured_lines_out_onto_a4(self, extracted):
        """Each 111x320 bar scales to 860 rows at A4 width, so with the 360-row
        title block only three fit above the 3338-row limit — four lines take
        two pages, balanced 2/2."""
        bars, _ = extracted
        pages = v.build_pages(bars, "Test Song", "Test Artist")

        assert len(pages) == 2
        assert all(page.shape[1] == v.A4_WIDTH_PX for page in pages)
        assert all(
            page.shape[0] <= v.A4_HEIGHT_PX - v.BOTTOM_MARGIN_PX for page in pages
        )


class TestCommandLine:
    def test_writes_a_pdf_into_cwd_relative_sheets(self, tutorial_video, tmp_path, monkeypatch):
        """Also pins the output-dir default.

        It used to be relative to the module file, which after packaging would
        write inside site-packages. Running from a scratch directory is what
        makes a regression here fail.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["video2sheet", str(tutorial_video), "--no-open"])

        v.main()

        pdf = tmp_path / "sheets" / f"{tutorial_video.stem}.pdf"
        assert pdf.is_file()
        data = pdf.read_bytes()
        assert data.startswith(b"%PDF-")
        assert data.rstrip().endswith(b"%%EOF")
        assert len(data) > 10_000, "an A4 page of raster, not a stub"

    def test_a_missing_video_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["video2sheet", str(tmp_path / "nope.mp4")])
        with pytest.raises(SystemExit, match="No such video"):
            v.main()

    def test_a_video_with_no_staff_exits_with_guidance(self, tmp_path, monkeypatch):
        """A title card only: bright paper is never found, so nothing is
        captured and the user is pointed at the tuning constants."""
        black = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        path = write_frames(tmp_path / "blank.avi", [black] * 10)

        monkeypatch.setattr(sys, "argv", ["video2sheet", str(path), "--no-open"])
        with pytest.raises(SystemExit, match="--debug-gif"):
            v.main()


class TestDebugGif:
    def test_writes_an_overlay_reel(self, tutorial_video, tmp_path):
        """--debug-gif is the only way to see the detector now that there is
        no live window, so it needs to actually produce a file."""
        frames = []
        v.extract_bars(tutorial_video, debug_frames=frames)
        assert frames

        out = tmp_path / "debug.gif"
        v.save_debug_gif(frames, out)
        assert out.read_bytes().startswith(b"GIF8")
