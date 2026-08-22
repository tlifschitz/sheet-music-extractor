import sys

import cv2
import numpy as np

# Load image: any video frame or a staff line dumped by --dump-bars.
path = sys.argv[1] if len(sys.argv) > 1 else "output.png"
img_bgr = cv2.imread(path)
if img_bgr is None:
    raise SystemExit(f"Could not read image: {path}")
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Click to inspect HSV
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = img_hsv[y, x]
        print(f"Clicked at ({x},{y}): HSV = {pixel}")

cv2.namedWindow("Click to inspect HSV")
cv2.setMouseCallback("Click to inspect HSV", on_mouse)
cv2.imshow("Click to inspect HSV", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()