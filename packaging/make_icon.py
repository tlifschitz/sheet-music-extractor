"""Draw the app icon and write the .icns and .ico the PyInstaller spec needs.

    python packaging/make_icon.py

The drawing is code rather than a checked-in design file so the icon can be
re-cut at any size — and edited — without a graphics editor. Its two outputs
*are* checked in: the Windows runner has no way to make a .icns, and neither
runner should have to install Pillow-with-fonts just to build an icon that
changes once a year.

The picture is the pipeline: a sheet of paper with a staff on it, above the
falling colour bars of the tutorial video, with each bar under the note it
turns into.
"""

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
SITE = HERE.parent / "site"

# Everything is drawn in this square, then rounded off and inset into the
# 1024px canvas macOS and Windows both want.
SIDE = 856
INSET = (1024 - SIDE) // 2
RADIUS = 192
SS = 4  # supersampling: draw big, shrink down, get smooth curves for free

INK = (22, 32, 43)          # the video's dark background, and the staff's ink
PAPER = (250, 249, 246)     # the scrolling sheet
STAFF = (43, 52, 64)
NOTE = (17, 24, 33)
CYAN = (53, 196, 232)       # the falling notes, in the two colours the
GREEN = (74, 222, 128)      # tutorials use for the left and right hand

PAPER_BOTTOM = 500          # where the sheet ends and the note area begins
STAFF_CENTRE = 250
STAFF_GAP = 44
STAFF_THICK = 10

# x, y, colour: a note on the staff and the bar below that becomes it.
VOICES = [
    (168, STAFF_CENTRE + STAFF_GAP, GREEN),
    (312, STAFF_CENTRE, GREEN),
    (452, STAFF_CENTRE - STAFF_GAP, CYAN),
    (596, STAFF_CENTRE - 2 * STAFF_GAP, CYAN),
    (712, STAFF_CENTRE - STAFF_GAP // 2, CYAN),
]
BAR_TOPS = [612, 560, 596, 548, 584]


def _note_head(draw, x, y, s):
    """A filled head, tilted the way an engraved one is."""
    rx, ry = 34, 26
    head = Image.new("RGBA", ((rx * 2 + 8) * s, (ry * 2 + 8) * s), (0, 0, 0, 0))
    ImageDraw.Draw(head).ellipse(
        [4 * s, 4 * s, (rx * 2 + 4) * s, (ry * 2 + 4) * s], fill=NOTE
    )
    head = head.rotate(20, resample=Image.BICUBIC)
    draw._image.alpha_composite(
        head, (int((x - rx - 4) * s), int((y - ry - 4) * s))
    )
    # Stem, up from the right of the head, as most of these notes would have.
    draw.rectangle(
        [(x + rx - 9) * s, (y - 122) * s, (x + rx) * s, (y - 4) * s], fill=NOTE
    )


def draw_icon():
    s = SS
    im = Image.new("RGBA", (SIDE * s, SIDE * s), INK)
    d = ImageDraw.Draw(im)

    d.rectangle([0, 0, SIDE * s, PAPER_BOTTOM * s], fill=PAPER)

    for i in range(5):
        y = STAFF_CENTRE + (i - 2) * STAFF_GAP
        d.rectangle(
            [70 * s, (y - STAFF_THICK / 2) * s,
             786 * s, (y + STAFF_THICK / 2) * s],
            fill=STAFF,
        )

    for (x, y, _), top in zip(VOICES, BAR_TOPS):
        _note_head(d, x, y, s)

    for (x, _, colour), top in zip(VOICES, BAR_TOPS):
        d.rounded_rectangle(
            [(x - 27) * s, top * s, (x + 27) * s, (SIDE + 40) * s],
            radius=24 * s,
            fill=colour,
        )

    im = im.resize((SIDE, SIDE), Image.LANCZOS)

    mask = Image.new("L", (SIDE * s, SIDE * s), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, SIDE * s - 1, SIDE * s - 1], radius=RADIUS * s, fill=255
    )
    im.putalpha(mask.resize((SIDE, SIDE), Image.LANCZOS))

    canvas = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    canvas.alpha_composite(im, (INSET, INSET))
    return canvas


def main():
    icon = draw_icon()
    icon.save(HERE / "icon.png")
    icon.save(HERE / "icon.icns")
    icon.save(
        HERE / "icon.ico",
        sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)],
    )
    print("wrote icon.png, icon.icns, icon.ico in", HERE)

    # The site wears the same face as the app. Emitted from here rather than
    # kept as separate art, so there is one drawing and the tab icon cannot
    # drift from the thing the visitor downloads.
    icon.save(SITE / "favicon.ico", sizes=[(s, s) for s in (16, 32, 48)])

    # iOS composites a home-screen icon over black and applies its own mask,
    # so the transparent rounded corners have to go: flatten onto the paper
    # colour and let the system round it.
    touch = Image.new("RGB", icon.size, PAPER)
    touch.paste(icon, mask=icon.getchannel("A"))
    touch.resize((180, 180), Image.LANCZOS).save(SITE / "apple-touch-icon.png")
    print("wrote favicon.ico, apple-touch-icon.png in", SITE)


if __name__ == "__main__":
    main()
