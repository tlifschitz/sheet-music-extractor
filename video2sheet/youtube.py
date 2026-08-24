"""Fetch tutorial videos to feed into video2sheet.

Takes either a single video URL or a channel, in which case its uploads can be
filtered by title regex or sampled at random before downloading.

By default only the video stream is downloaded: video2sheet reads frames and
never touches the audio, so pulling the audio track just wastes bandwidth.
"""

import argparse
import random
import re
import sys
from pathlib import Path

from yt_dlp import YoutubeDL

VIDEO_URL_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/)([\w-]{11})")

# Prefer a progressive/video-only MP4 no larger than 1080p, then fall back.
VIDEO_ONLY_FORMAT = "bv*[ext=mp4][height<=1080]/bv*[height<=1080]/bv*/b"
WITH_AUDIO_FORMAT = "b[ext=mp4][height<=1080]/bv*[height<=1080]+ba/b"


def channel_url(target):
    """Accept a channel URL, an @handle, or a bare channel ID."""
    if target.startswith("http"):
        return target if "/videos" in target else target.rstrip("/") + "/videos"
    if target.startswith("@"):
        return f"https://www.youtube.com/{target}/videos"
    return f"https://www.youtube.com/channel/{target}/videos"


def list_channel_videos(target, limit=None):
    """Return [{title, url}] for a channel's uploads, without downloading."""
    opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    if limit:
        opts["playlistend"] = limit

    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(channel_url(target), download=False)

    videos = []

    def walk(node):
        for entry in node.get("entries") or []:
            if entry is None:
                continue
            if entry.get("_type") == "playlist":
                walk(entry)
            elif entry.get("id"):
                videos.append(
                    {
                        "title": entry.get("title") or entry["id"],
                        "url": f"https://www.youtube.com/watch?v={entry['id']}",
                    }
                )

    walk(info)
    return videos[:limit] if limit else videos


def download(url, output_dir, with_audio=False):
    """Download one video and return the resulting path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opts = {
        "format": WITH_AUDIO_FORMAT if with_audio else VIDEO_ONLY_FORMAT,
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return Path(ydl.prepare_filename(info))


def choose(videos):
    """Prompt for one of the listed videos."""
    for i, video in enumerate(videos, 1):
        print(f"{i:3}. {video['title']}")
    try:
        choice = int(input(f"Pick a video (1-{len(videos)}): "))
    except (ValueError, EOFError):
        raise SystemExit("Not a number.")
    if not 1 <= choice <= len(videos):
        raise SystemExit("Out of range.")
    return [videos[choice - 1]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="a video URL, or a channel URL / @handle / channel ID")
    parser.add_argument(
        "-o", "--output-dir", default="videos", help="download directory (default: videos)"
    )
    parser.add_argument("--match", metavar="REGEX", help="keep channel videos whose title matches")
    parser.add_argument("--random", type=int, metavar="N", help="keep N channel videos at random")
    parser.add_argument("--limit", type=int, help="only look at the newest N uploads")
    parser.add_argument("--list", action="store_true", help="list matches and exit")
    parser.add_argument("--all", action="store_true", help="download every match, no prompt")
    parser.add_argument("--with-audio", action="store_true", help="also fetch the audio track")
    args = parser.parse_args()

    if VIDEO_URL_RE.search(args.target):
        path = download(args.target, args.output_dir, args.with_audio)
        print(f"Saved {path}")
        return

    videos = list_channel_videos(args.target, args.limit)
    if not videos:
        raise SystemExit("No videos found for that channel.")
    print(f"Found {len(videos)} videos", file=sys.stderr)

    if args.match:
        pattern = re.compile(args.match, re.IGNORECASE)
        videos = [v for v in videos if pattern.search(v["title"])]
        if not videos:
            raise SystemExit("Nothing matched that pattern.")
    if args.random:
        videos = random.sample(videos, min(args.random, len(videos)))

    if args.list:
        for video in videos:
            print(f"{video['title']}\t{video['url']}")
        return

    if not args.all:
        videos = choose(videos)

    for video in videos:
        print(f"\n=> {video['title']}")
        download(video["url"], args.output_dir, args.with_audio)


if __name__ == "__main__":
    main()
