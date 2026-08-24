"""Heading: filename parsing and rendering of the title block."""

import numpy as np
import pytest

from video2sheet import pipeline as v


class TestSplitTitle:
    @pytest.mark.parametrize(
        "stem,expected",
        [
            ("Coldplay - Yellow - Piano Tutorial with Sheet Music", ("Yellow", "Coldplay")),
            ("Ed Sheeran - Happier - Piano Tutorial + SHEETS", ("Happier", "Ed Sheeran")),
            ("Ludovico Einaudi - Experience", ("Experience", "Ludovico Einaudi")),
        ],
    )
    def test_strips_boilerplate_and_splits_artist(self, stem, expected):
        assert v.split_title(stem) == expected

    def test_restores_fullwidth_characters(self):
        """yt-dlp writes '？' because '?' is illegal in filenames."""
        title, artist = v.split_title("Billie Eilish - What Was I Made For？ - Piano Tutorial")
        assert title == "What Was I Made For?"
        assert artist == "Billie Eilish"

    def test_unsplittable_name_becomes_a_bare_title(self):
        title, artist = v.split_title("JUMPSUIT (twenty one pilots) - Piano Tutorial + SHEETS")
        assert title == "JUMPSUIT (twenty one pilots)"
        assert artist is None

    def test_song_titles_containing_a_dash_survive(self):
        title, artist = v.split_title("Artist - Some - Long - Name - Piano Tutorial")
        assert artist == "Artist"
        assert title == "Some - Long - Name"

    def test_nothing_but_boilerplate_falls_back_to_the_stem(self):
        assert v.split_title("Piano Tutorial") == ("Piano Tutorial", None)


class TestTitleBlock:
    def test_has_the_expected_shape(self):
        block = v.make_title_block("Yellow", "Coldplay")
        assert block.shape == (v.TITLE_BLOCK_HEIGHT_PX, v.A4_WIDTH_PX)

    def test_actually_draws_ink(self):
        blank = v.make_title_block("", None)
        drawn = v.make_title_block("Yellow", "Coldplay")
        assert (drawn < 128).sum() > 0
        assert (blank < 128).sum() == 0

    def test_the_artist_line_adds_ink(self):
        without = (v.make_title_block("Yellow", None) < 200).sum()
        with_artist = (v.make_title_block("Yellow", "Coldplay") < 200).sum()
        assert with_artist > without

    def test_title_is_far_larger_than_the_old_hershey_text(self):
        """The original heading was 25 px tall on a 2480 px page."""
        block = v.make_title_block("Yellow", None)
        rows = np.nonzero((block < 128).sum(axis=1))[0]
        assert rows.max() - rows.min() > 60

    def test_long_titles_are_shrunk_to_fit(self):
        """A very long heading must not run off the page edge."""
        block = v.make_title_block("A " * 60, None)
        cols = np.nonzero((block < 128).sum(axis=0))[0]
        assert cols.min() > 0
        assert cols.max() < v.A4_WIDTH_PX - 1

    def test_heading_stays_inside_the_block(self):
        block = v.make_title_block("Yellow", "Coldplay")
        assert (block[-1] == 255).all()


class TestPageNumbers:
    def test_single_page_is_not_numbered(self):
        page = np.full((v.A4_HEIGHT_PX, v.A4_WIDTH_PX), 255, np.uint8)
        assert (v.add_page_number(page, 1, 1) == 255).all()

    def test_multi_page_documents_are_numbered(self):
        page = np.full((v.A4_HEIGHT_PX, v.A4_WIDTH_PX), 255, np.uint8)
        stamped = v.add_page_number(page, 2, 5)
        assert (stamped < 200).sum() > 0
        # The number belongs in the bottom margin, not over the music.
        assert (stamped[: v.A4_HEIGHT_PX - v.BOTTOM_MARGIN_PX] == 255).all()
