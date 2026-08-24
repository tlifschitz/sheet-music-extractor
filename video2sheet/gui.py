"""A window for people who will not open a terminal.

The pipeline itself is a CLI, which is the wrong shape for the person this is
actually for: someone who found an arrangement they liked and wants the PDF.
This is a file picker, a progress bar and an error box — no flags, no paths,
no Python.

Deliberately Tkinter rather than Qt or wx: it is in the standard library, so it
adds no dependency to a project that is otherwise four wheels, and PyInstaller
bundles it without help.
"""

import queue
import threading
import traceback
from pathlib import Path

import cv2

from . import pipeline as v

WINDOW_TITLE = "Video to Sheet Music"
# Formats OpenCV will actually decode. Anything else is offered under "all
# files" rather than hidden, since the list is not exhaustive.
VIDEO_TYPES = [
    ("Video files", "*.mp4 *.mkv *.webm *.mov *.avi *.m4v"),
    ("All files", "*.*"),
]


def output_path_for(video):
    """Where the PDF lands: beside the video it came from.

    The CLI writes to ./sheets relative to the working directory, which is a
    sensible default for a shell and a useless one for a double-clicked app —
    its working directory is wherever the launcher happened to be. Next to the
    source file is somewhere the user can actually find again.
    """
    video = Path(video)
    return video.parent / f"{video.stem}.pdf"


def count_frames(video):
    """Total frames, for the progress bar, or None if the container will not say.

    Some containers report 0 or a negative count rather than admitting they do
    not know; treat anything non-positive as unknown and run indeterminate.
    """
    capture = cv2.VideoCapture(str(video))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return total if total > 0 else None


def convert(video, on_progress=None):
    """Run the real pipeline, reporting progress as a fraction from 0 to 1.

    Progress rides on `extract_bars`'s observer hook, which already fires once
    per decoded frame. Decoding is nearly all of the runtime, so a bar driven
    by it is honest — page layout and PDF writing are the last blink.
    """
    video = Path(video)
    total = count_frames(video)

    def observer(record):
        if on_progress and total:
            on_progress(min(record["frame"] / total, 1.0))

    bars, _ = v.extract_bars(video, observer=observer)
    if not bars:
        raise ValueError(
            "No staff lines were found in that video.\n\n"
            "This tool reads one particular layout: sheet music scrolling "
            "across the top of the frame, with falling notes below it and a "
            "coloured cursor sweeping left to right. Videos shaped any other "
            "way will not work."
        )

    if on_progress:
        on_progress(1.0)
    title, artist = v.split_title(video.stem)
    output = output_path_for(video)
    v.save_pdf(v.build_pages(bars, title, artist), output)
    return output, len(bars)


class App:
    """The window. All Tk calls happen on the main thread; the pipeline does not.

    Tk is not thread-safe, so the worker thread never touches a widget. It puts
    messages on a queue and the main loop drains it on a timer.
    """

    def __init__(self, root):
        from tkinter import StringVar, ttk

        self.root = root
        self.events = queue.Queue()
        self.video = None
        self.busy = False

        root.title(WINDOW_TITLE)
        root.minsize(520, 0)
        root.resizable(True, False)

        frame = ttk.Frame(root, padding=20)
        frame.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        ttk.Label(
            frame, text="Turn a piano tutorial video into a printable score.",
            font=("TkDefaultFont", 13, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            frame, wraplength=470, foreground="#555",
            text="Pick the video file. The PDF is saved next to it and opens "
                 "when it is done.",
        ).grid(row=1, column=0, sticky="w", pady=(4, 14))

        self.choose = ttk.Button(frame, text="Choose a video…", command=self.pick)
        self.choose.grid(row=2, column=0, sticky="w")

        self.status = StringVar(value="No video chosen yet.")
        ttk.Label(frame, textvariable=self.status, wraplength=470).grid(
            row=3, column=0, sticky="w", pady=(12, 6)
        )

        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=1000)
        self.progress.grid(row=4, column=0, sticky="ew")

        self.run = ttk.Button(frame, text="Make the PDF", command=self.start,
                              state="disabled")
        self.run.grid(row=5, column=0, sticky="w", pady=(14, 0))

        self.root.after(100, self.drain)

    def pick(self):
        from tkinter import filedialog

        chosen = filedialog.askopenfilename(title="Choose a tutorial video",
                                            filetypes=VIDEO_TYPES)
        if not chosen:
            return
        self.video = Path(chosen)
        self.status.set(f"Ready: {self.video.name}")
        self.progress["value"] = 0
        self.run["state"] = "normal"

    def start(self):
        if self.busy or not self.video:
            return
        self.busy = True
        self.run["state"] = "disabled"
        self.choose["state"] = "disabled"
        self.status.set(f"Reading {self.video.name}…")
        threading.Thread(target=self.work, args=(self.video,), daemon=True).start()

    def work(self, video):
        """Off the main thread: nothing here may touch a widget."""
        try:
            output, count = convert(
                video, on_progress=lambda f: self.events.put(("progress", f))
            )
            self.events.put(("done", (output, count)))
        except Exception as error:  # noqa: BLE001 - anything here belongs in the box
            self.events.put(("failed", error))

    def drain(self):
        """Pump the worker's messages into the widgets, on the main thread."""
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    self.progress["value"] = payload * 1000
                elif kind == "done":
                    self.finish(*payload)
                elif kind == "failed":
                    self.fail(payload)
        except queue.Empty:
            pass
        self.root.after(100, self.drain)

    def finish(self, output, count):
        self.busy = False
        self.choose["state"] = "normal"
        self.run["state"] = "normal"
        self.progress["value"] = 1000
        self.status.set(f"Done — {count} staff lines saved to {output.name}")
        v.reveal(output)

    def fail(self, error):
        from tkinter import messagebox

        self.busy = False
        self.choose["state"] = "normal"
        self.run["state"] = "normal"
        self.progress["value"] = 0
        self.status.set("That did not work.")
        # A ValueError here is the pipeline's own "wrong sort of video", which
        # is written for a reader. Anything else is a bug, and the traceback
        # goes to the console for whoever is debugging it.
        if not isinstance(error, ValueError):
            traceback.print_exception(type(error), error, error.__traceback__)
        message = str(error) or f"{type(error).__name__}: something went wrong."
        messagebox.showerror(WINDOW_TITLE, message)


def main():
    import tkinter as tk

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
