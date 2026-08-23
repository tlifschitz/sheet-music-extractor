"""Click a pixel to print its HSV value.

Used to pick BAR_POSITION_SATURATION_THRESHOLD for a video whose playhead is
an unusual colour. Point it at any frame or at a staff line dumped by
`video2sheet <video> --dump-bars bars/`.

Drawn with matplotlib rather than cv2.imshow: the project depends on the
headless OpenCV build, which has no GUI on Linux.
"""

import sys

import cv2
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "output.png"
img_bgr = cv2.imread(path)
if img_bgr is None:
    raise SystemExit(f"Could not read image: {path}")

img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)


def on_click(event):
    if event.xdata is None or event.ydata is None:
        return
    x, y = int(event.xdata), int(event.ydata)
    h, s, v = img_hsv[y, x]
    print(f"({x},{y})  H={h:3d}  S={s:3d}  V={v:3d}")


fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
ax.set_title(f"Click to inspect HSV — {path}")
ax.axis("off")
fig.canvas.mpl_connect("button_press_event", on_click)
plt.show()
