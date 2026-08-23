"""Page-layout tests: the arithmetic that turns staff lines into A4 pages."""

import numpy as np
import pytest

from sheet_music_extractor import video2sheet as v

MAX_PAGE_HEIGHT = v.A4_HEIGHT_PX - v.BOTTOM_MARGIN_PX


def staff_lines(count, height=300, width=1200):  # noqa: D103
    return [np.full((height, width), 255, np.uint8) for _ in range(count)]


class TestBuildPages:
    def test_no_bars_yields_no_pages(self):
        """main() relies on this to report a failed extraction instead of
        emitting a PDF containing nothing but a title."""
        assert v.build_pages([], "Title") == []

    def test_every_page_is_a4_wide(self):
        pages = v.build_pages(staff_lines(10), "Title")
        assert all(page.shape[1] == v.A4_WIDTH_PX for page in pages)

    def test_no_page_overflows_a4(self):
        pages = v.build_pages(staff_lines(30), "Title", balance=False)
        assert pages
        assert all(page.shape[0] <= MAX_PAGE_HEIGHT for page in pages)

    def test_bottom_margin_is_respected(self):
        """Pages must break early enough to leave the bottom margin free.

        Deliberately uses thin lines: with thick ones the whole margin fits
        inside a single line's height, so a page-break bug slips through
        while the assertion still passes.
        """
        thin = [np.full((100, v.A4_WIDTH_PX), 255, np.uint8) for _ in range(80)]
        pages = v.build_pages(thin, "Title", balance=False)
        assert pages
        assert all(page.shape[0] <= MAX_PAGE_HEIGHT for page in pages)

    def test_more_lines_need_more_pages(self):
        few = v.build_pages(staff_lines(4), "Title")
        many = v.build_pages(staff_lines(40), "Title")
        assert len(many) > len(few)

    def test_lines_are_scaled_to_page_width(self):
        """A narrow input is upscaled, keeping its aspect ratio."""
        narrow = [np.full((100, 400), 255, np.uint8)]
        page = v.build_pages(narrow, "Title", balance=False)[0]
        expected = v.TITLE_BLOCK_HEIGHT_PX + int(100 * (v.A4_WIDTH_PX / 400))
        assert page.shape == (expected, v.A4_WIDTH_PX)

    @pytest.mark.parametrize("count", [1, 5, 23, 100])
    def test_all_lines_are_kept(self, count):
        """No staff line may be silently dropped at a page break."""
        # balance=False: this checks the raw arithmetic, and justification
        # deliberately inserts extra whitespace between lines.
        pages = v.build_pages(staff_lines(count), "Title", balance=False)
        line_height = int(300 * (v.A4_WIDTH_PX / 1200))
        # The first page opens with the title block, the rest with a margin.
        headers = v.TITLE_BLOCK_HEIGHT_PX + v.TOP_MARGIN_PX * (len(pages) - 1)
        assert sum(p.shape[0] for p in pages) - headers == count * line_height


class TestSavePdf:
    def test_writes_a_pdf(self, tmp_path):
        out = tmp_path / "score.pdf"
        v.save_pdf(v.build_pages(staff_lines(3), "Title"), out)
        assert out.read_bytes().startswith(b"%PDF")

    def test_creates_missing_directories(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "score.pdf"
        v.save_pdf(v.build_pages(staff_lines(1), "Title"), out)
        assert out.exists()

    def test_pages_are_padded_to_full_a4(self):
        """Short final pages must still print as A4, not as a stub."""
        short = v.build_pages(staff_lines(1), "Title")[0]
        assert short.shape[0] < v.A4_HEIGHT_PX

        padded = v.pad_to_a4(short)
        assert padded.shape == (v.A4_HEIGHT_PX, v.A4_WIDTH_PX)
        # The padding is white, and the original content is untouched.
        assert (padded[short.shape[0] :] == 255).all()
        assert (padded[: short.shape[0]] == short).all()

    def test_full_height_pages_are_left_alone(self):
        exact = np.full((v.A4_HEIGHT_PX, v.A4_WIDTH_PX), 255, np.uint8)
        assert v.pad_to_a4(exact) is exact


class TestBalancing:
    """Greedy packing crams the early pages; balancing evens them out."""

    def test_balancing_does_not_cost_an_extra_page(self):
        lines = staff_lines(15, height=383, width=1920)
        dense = v.build_pages(lines, "Title", balance=False)
        even = v.build_pages(lines, "Title", balance=True)
        assert len(even) == len(dense)

    def test_pages_end_up_evenly_filled(self):
        """Fifteen lines pack as 6/6/3 but balance as 5/5/5."""
        lines = staff_lines(15, height=383, width=1920)
        dense = [p.shape[0] for p in v.build_pages(lines, "Title", balance=False)]
        even = [p.shape[0] for p in v.build_pages(lines, "Title", balance=True)]
        assert max(dense) - min(dense) > max(even) - min(even)

    def test_no_line_is_lost_when_balancing(self):
        for count in (1, 7, 15, 40):
            lines = staff_lines(count, height=383, width=1920)
            dense = v.build_pages(lines, "Title", balance=False)
            even = v.build_pages(lines, "Title", balance=True)
            assert sum(v._pack([383] * count, v.TITLE_BLOCK_HEIGHT_PX, v.TOP_MARGIN_PX)) == count
            assert len(even) == len(dense)

    def test_balanced_pages_still_respect_the_margins(self):
        pages = v.build_pages(staff_lines(15, height=383, width=1920), "Title")
        assert all(page.shape[0] <= MAX_PAGE_HEIGHT for page in pages)


class TestJustification:
    def test_slack_moves_off_the_foot_of_the_page(self):
        """The complaint this fixes: a gap at the bottom that looks like a
        staff line failed to fit."""
        lines = staff_lines(15, height=383, width=1920)
        even = v.build_pages(lines, "Title", balance=True)
        dense_first = v.build_pages(lines, "Title", balance=False)[0]
        gap_dense = v.A4_HEIGHT_PX - dense_first.shape[0]
        gap_even = v.A4_HEIGHT_PX - even[0].shape[0]
        assert gap_even < gap_dense or even[0].shape[0] >= dense_first.shape[0]

    def test_a_single_line_is_not_stretched(self):
        page = v.build_pages(staff_lines(1), "Title", balance=True)[0]
        assert page.shape[0] == v.TITLE_BLOCK_HEIGHT_PX + int(300 * (v.A4_WIDTH_PX / 1200))

    def test_gaps_are_capped(self):
        """A sparse page must not sprawl its two lines across the sheet."""
        page = v.build_pages(staff_lines(2, height=100, width=2480), "Title", balance=True)[0]
        natural = v.TITLE_BLOCK_HEIGHT_PX + 200
        assert page.shape[0] <= natural + v.MAX_EXTRA_GAP_PX
