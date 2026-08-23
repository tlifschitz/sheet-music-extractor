"""URL handling in the download helper. No network access."""

import pytest

from sheet_music_extractor import youtube as yt


class TestChannelUrl:
    @pytest.mark.parametrize(
        "target,expected",
        [
            ("@SomeChannel", "https://www.youtube.com/@SomeChannel/videos"),
            ("UCIjyqJXAr_G420gKaYwN0ug", "https://www.youtube.com/channel/UCIjyqJXAr_G420gKaYwN0ug/videos"),
        ],
    )
    def test_bare_handles_and_ids_become_video_tabs(self, target, expected):
        assert yt.channel_url(target) == expected

    def test_appends_videos_tab_to_a_channel_url(self):
        assert yt.channel_url("https://www.youtube.com/@X") == "https://www.youtube.com/@X/videos"

    def test_trailing_slash_does_not_double_up(self):
        assert yt.channel_url("https://www.youtube.com/@X/") == "https://www.youtube.com/@X/videos"

    def test_existing_videos_tab_is_left_alone(self):
        url = "https://www.youtube.com/@X/videos"
        assert yt.channel_url(url) == url


class TestVideoUrlDetection:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=ss5dDq41L2k",
            "https://youtu.be/ss5dDq41L2k",
            "https://www.youtube.com/shorts/ss5dDq41L2k",
            "https://www.youtube.com/watch?v=ss5dDq41L2k&list=PLxxx",
        ],
    )
    def test_recognises_video_urls(self, url):
        match = yt.VIDEO_URL_RE.search(url)
        assert match and match.group(1) == "ss5dDq41L2k"

    @pytest.mark.parametrize(
        "target",
        [
            "@SomeChannel",
            "UCIjyqJXAr_G420gKaYwN0ug",
            "https://www.youtube.com/@SomeChannel/videos",
        ],
    )
    def test_channels_are_not_mistaken_for_videos(self, target):
        """Otherwise main() would try to download the channel page itself."""
        assert yt.VIDEO_URL_RE.search(target) is None


class TestFormatSelection:
    def test_video_only_is_the_default(self):
        """The pipeline never reads audio; fetching it is wasted bandwidth."""
        assert "bv*" in yt.VIDEO_ONLY_FORMAT
        assert "+ba" not in yt.VIDEO_ONLY_FORMAT

    def test_audio_variant_asks_for_sound(self):
        assert "ba" in yt.WITH_AUDIO_FORMAT
