# Experiments

Side explorations that are not part of the pipeline. Kept because they are
useful when re-tuning the detector, not because they are finished.

- **`borders.py`** — Hough-transform detection of bar lines in an extracted
  staff image. The idea was to split each staff line at its bar lines so the
  page layout could break on musical boundaries instead of screen boundaries.
  It detects lines reliably but the merging heuristic is unfinished
  (`merge_two_lines` currently returns its first argument unchanged).

  ```bash
  video2sheet <video> --dump-bars bars/
  python borders.py bars/001.png
  ```

- **`tuneblue.py`** — click a pixel in an image to print its HSV value. Used to
  pick `BAR_POSITION_SATURATION_THRESHOLD` for a new video style. Drawn with matplotlib, since the project depends on the headless OpenCV
  build. Point it at any frame or dumped staff line.
