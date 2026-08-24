# PyInstaller spec for the double-clickable app.
#
# Built by .github/workflows/release.yml on a tag, for macOS and Windows, and
# attached to the GitHub release. The point is a user who has no Python, no
# git and no terminal: they download one file and open it.
#
#     pyinstaller packaging/video2sheet.spec
#
# Run from the repository root; SPECPATH is this directory.

import sys
from pathlib import Path

ROOT = Path(SPECPATH).parent
NAME = "Video to Sheet Music"

# matplotlib is a runtime dependency of the pipeline for exactly two font
# files, which the title block is typeset with. Carrying the rest of it — the
# backends, the sample data, the test images — would add tens of megabytes to
# a download aimed at someone on a home connection, so the fonts come along
# and matplotlib itself does not.
import matplotlib

FONTS = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
# "fonts" is where _font_dirs() looks first when sys._MEIPASS is set.
datas = [
    (str(FONTS / name), "fonts")
    for name in ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf")
]

analysis = Analysis(
    [str(ROOT / "packaging" / "launch.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=["PIL._tkinter_finder"],
    excludes=[
        "matplotlib",  # see above: the fonts are bundled, the library is not
        "yt_dlp",      # the fetcher is a separate CLI; the app never imports it
        "pytest",
        "IPython",
        "tkinter.test",
    ],
    noarchive=False,
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    exclude_binaries=True,
    name=NAME,
    console=False,  # no terminal window behind the app
    disable_windowed_traceback=False,
    argv_emulation=sys.platform == "darwin",  # let a dropped file arrive as argv
)
collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    name=NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(
        collection,
        name=f"{NAME}.app",
        bundle_identifier="com.tlifschitz.video2sheet",
        info_plist={
            "NSHighResolutionCapable": True,
            "CFBundleShortVersionString": "0.1.0",
            # Declares the app as an opener for video files, so "Open With"
            # and dropping a file on the icon both work.
            "CFBundleDocumentTypes": [
                {
                    "CFBundleTypeName": "Video",
                    "CFBundleTypeRole": "Viewer",
                    "LSItemContentTypes": ["public.movie"],
                }
            ],
        },
    )
