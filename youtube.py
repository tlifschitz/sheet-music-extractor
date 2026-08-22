import scrapetube
from pytubefix import YouTube
from typing import Optional
from tqdm import tqdm
import random
import re

def get_channel_videos(channel_id: str, limit: Optional[int] = 5, sleep: Optional[float] = 0.5) -> list[dict]:
    """
    Retrieves all videos from a YouTube channel using scrapetube.

    Args:
        channel_id (str): The YouTube channel ID.

    Returns:
        list of dict: Each dict contains 'title' and 'url' of a video.
    """
    videos = scrapetube.get_channel(channel_id, limit=limit, sleep=sleep)
    result = []
    for video in videos:
        title = video["title"]["runs"][0]["text"]
        video_id = video["videoId"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        result.append({"title": title, "url": url})
    return result



def download_youtube_video(url: str, path: Optional[str] = None) -> None:
    """
    Downloads a YouTube video from its URL with a tqdm progress bar.

    Args:
        url (str): The URL of the YouTube video to download.
        path (Optional[str]): The directory path where the video will be saved.
            If None, saves to the current working directory.

    Returns:
        None
    """
    yt = YouTube(url)
    print(f"Title: {yt.title}")
    ys = yt.streams.get_highest_resolution()

    
    total = ys.filesize
    pbar = tqdm(total=total, unit='B', unit_scale=True, desc="Downloading")

    def on_progress(stream, chunk, bytes_remaining):
        bytes_downloaded = total - bytes_remaining
        pbar.n = bytes_downloaded
        pbar.refresh()

    yt.register_on_progress_callback(on_progress)
    ys.download(output_path=path)
    pbar.close()


if __name__ == "__main__":

    channel_id = "UCIjyqJXAr_G420gKaYwN0ug"
    videos = get_channel_videos(channel_id, limit=99999999)

    if not videos:
        print("No videos found for this channel.")
        exit()

    print("Choose an option:")
    print("1. Select random sample of videos")
    print("2. Search videos by regex in title")
    option = input("Enter 1 or 2: ").strip()

    selected_videos = []

    if option == "1":
        try:
            n = int(input(f"How many random videos? (1-{len(videos)}): "))
            if 1 <= n <= len(videos):
                selected_videos = random.sample(videos, n)
            else:
                print("Invalid number.")
                exit()
        except ValueError:
            print("Invalid input.")
            exit()
    elif option == "2":
        pattern = input("Enter regex pattern to search in titles: ")
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            filtered = [v for v in videos if regex.search(v["title"])]
            if not filtered:
                print("No videos matched the pattern.")
                exit()
            selected_videos = filtered
        except re.error:
            print("Invalid regex pattern.")
            exit()
    else:
        print("Invalid option.")
        exit()

    videos = selected_videos
    if not videos:
        print("No videos found for this channel.")
    else:
        print("Select a video to download:")
        for idx, video in enumerate(videos, 1):
            print(f"{idx}. {video['title']}")
        try:
            choice = int(input(f"Enter a number (1-{len(videos)}): "))
            if 1 <= choice <= len(videos):
                selected_video = videos[choice - 1]
                download_youtube_video(selected_video["url"])
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input.")
