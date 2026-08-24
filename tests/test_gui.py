"""The window, minus the window.

Everything worth testing here is the glue: where the PDF goes, whether
progress actually advances, and whether a video of the wrong shape produces a
sentence a person can read rather than a traceback. None of that needs a
display, and CI does not have one.
"""

import numpy as np
import pytest

from sheet_music_extractor import gui

from conftest import HEIGHT, WIDTH, write_frames


class TestOutputPath:
    def test_lands_beside_the_source_video(self, tmp_path):
        """Not ./sheets: a double-clicked app's working directory is wherever
        the launcher happened to be, which the user cannot find again."""
        video = tmp_path / "Artist - Song.mp4"
        assert gui.output_path_for(video) == tmp_path / "Artist - Song.pdf"

    def test_keeps_dots_in_the_stem(self, tmp_path):
        video = tmp_path / "Song feat. Someone.mkv"
        assert gui.output_path_for(video).name == "Song feat. Someone.pdf"


class TestFrameCount:
    def test_counts_a_real_file(self, tutorial_video):
        assert gui.count_frames(tutorial_video) > 0

    def test_unreadable_file_is_unknown_rather_than_zero(self, tmp_path):
        """A falsy count drives an indeterminate bar; a 0 would divide by zero."""
        junk = tmp_path / "not-a-video.mp4"
        junk.write_bytes(b"nope")
        assert gui.count_frames(junk) is None


class TestConvert:
    def test_writes_a_pdf_beside_the_video(self, tutorial_video, sweeps):
        output, count = gui.convert(tutorial_video)
        assert output.exists()
        assert output.read_bytes().startswith(b"%PDF")
        assert count == sweeps

    def test_progress_advances_and_ends_at_one(self, tutorial_video):
        seen = []
        gui.convert(tutorial_video, on_progress=seen.append)
        assert seen and seen == sorted(seen)
        assert seen[-1] == 1.0
        assert all(0 <= f <= 1 for f in seen)

    def test_the_wrong_sort_of_video_explains_itself(self, tmp_path):
        """The failure a real user will hit. It must be a sentence, not a
        traceback, and must not mention a flag they cannot type."""
        black = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        path = write_frames(tmp_path / "blank.avi", [black] * 10)

        with pytest.raises(ValueError) as caught:
            gui.convert(path)
        message = str(caught.value)
        assert "cursor" in message
        assert "--" not in message


class TestWindow:
    def test_it_builds(self):
        """Cheap guard against a typo in a widget call. Skipped without a
        display, which is every CI runner."""
        tk = pytest.importorskip("tkinter")
        try:
            root = tk.Tk()
        except tk.TclError as error:
            pytest.skip(f"no display: {error}")
        try:
            gui.App(root)
        finally:
            root.destroy()
