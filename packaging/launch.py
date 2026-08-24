"""Entry point for the bundled app.

A separate module rather than pointing PyInstaller at the package: a frozen
build has no console, so anything that escapes main() would vanish silently.
This puts it in a dialog instead, which is the only place the user will look.
"""

import sys
import traceback


def self_test(video=None):
    """Prove the bundle works, without a human clicking anything.

    Run by the release workflow on the artifact it just built. A frozen app is
    exactly where imports go missing and data files land in the wrong place,
    and neither shows up until someone opens it.
    """
    from PIL import ImageFont

    from video2sheet import pipeline as v

    font = v._font(112, bold=True)
    if not isinstance(font, ImageFont.FreeTypeFont):
        raise SystemExit("self-test: the bundled DejaVu face was not found")
    print(f"self-test: font ok ({font.getname()})")

    if video:
        from video2sheet.gui import convert

        output, count = convert(video)
        print(f"self-test: {count} staff lines -> {output}")
    print("self-test: ok")


def main():
    try:
        if "--self-test" in sys.argv:
            rest = [a for a in sys.argv[1:] if a != "--self-test"]
            return self_test(rest[0] if rest else None)
        from video2sheet.gui import main as run
        run()
    except Exception:
        details = traceback.format_exc()
        print(details, file=sys.stderr)
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Video to Sheet Music",
                "The app could not start.\n\n" + details.strip().splitlines()[-1],
            )
        except Exception:
            pass
        raise SystemExit(1)


if __name__ == "__main__":
    main()
