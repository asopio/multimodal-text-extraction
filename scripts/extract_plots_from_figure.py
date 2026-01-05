"""Extract plot panels from a screenshot/bitmap of a PDF page.

This script was written for the specific screenshot in this repo (fig2_and_3.png),
which contains a 2x2 grid of plots plus captions underneath each row.

Approach (high level)
---------------------
1) Load the image as RGB and also compute a grayscale version.
2) Find the horizontal split between the *top* and *bottom* figure blocks by
   looking for a wide horizontal "blank" band (few dark pixels) around the
   middle of the page.
3) Split the image into four quadrants using (x_split = width/2, y_split).
4) For each quadrant:
   - Compute a 1D "ink" profile: count of dark pixels for each row and column.
   - Find the top/bottom/left/right bounds that contain plot content.
   - Try to remove captions by looking for an *internal* blank run between two
     content regions (plot area above, caption below).
   - Add a little padding so titles/legends aren't cut off.
5) Save each cropped panel.

How general is it?
------------------
This is a *heuristic* method. It's reasonably robust for:
- White/near-white page background.
- Dark plot content (axes, ticks, text, lines).
- A regular multi-panel layout (here: 2x2) with captions separated by whitespace.

It is NOT guaranteed to work on arbitrary PDF screenshots, because:
- Page backgrounds can be non-white, scanned/noisy, or colored.
- Layouts can be irregular (not a clean grid, varying panel sizes).
- Captions can touch the plot area (no blank separator), or be inside panels.
- There can be additional page elements (headers/footers, multiple columns).

For “any PDF document”, you usually need a layout-analysis step (connected
components/contours, text detection/OCR, or PDF-native vector extraction).

Usage
-----
pixi run python scripts/extract_plots_from_figure.py \
  --input fig2_and_3.png --outdir figs

It will write:
- figs/paper_top_left.png
- figs/paper_top_right.png
- figs/paper_bottom_left.png
- figs/paper_bottom_right.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_rgb(path: Path) -> np.ndarray:
    """Load an image as uint8 RGB using Pillow if available.

    We fall back to matplotlib if Pillow isn't installed.
    """

    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        import matplotlib.image as mpimg

        arr = mpimg.imread(path)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        return arr


def _save_rgb(path: Path, arr: np.ndarray) -> None:
    """Save a uint8 RGB image using Pillow if available."""

    try:
        from PIL import Image

        Image.fromarray(arr).save(path)
    except Exception:
        import matplotlib.pyplot as plt

        plt.imsave(path, arr)


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB to float32 grayscale (0..255)."""

    return (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    ).astype(np.float32)


def _smooth1d(x: np.ndarray, k: int) -> np.ndarray:
    """Simple moving-average smoother to make projection profiles less noisy."""

    k = max(3, int(k) | 1)  # odd window length
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def _find_y_split(gray: np.ndarray) -> int:
    """Find horizontal split between the top and bottom figure blocks.

    Idea: compute a row-wise "ink" count (# pixels darker than ~white). Then
    look for the longest low-ink run around the center of the page.
    """

    h, w = gray.shape

    # Count "not nearly white" pixels per row.
    row_ink = (gray < 240).sum(axis=1)
    row_ink = _smooth1d(row_ink, 31)

    # Rows with fewer dark pixels than this are considered "blank".
    low_thresh = max(3, int(0.01 * w))
    low = row_ink < low_thresh

    start = int(0.25 * h)
    end = int(0.75 * h)

    # Find contiguous runs of low==True.
    runs: list[tuple[int, int]] = []
    i = start
    while i < end:
        if low[i]:
            j = i
            while j < end and low[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1

    if runs:
        a, b = max(runs, key=lambda t: t[1] - t[0])
        return (a + b) // 2

    # Fallback if no clean blank band is found.
    return int(np.argmin(row_ink[start:end]) + start)


def _find_content_bounds_1d(
    counts: np.ndarray,
    *,
    blank_thr: float,
    content_thr: float,
    pad_lo: int,
    pad_hi: int,
    prefer_internal_blank_split: bool,
    min_blank_run: int = 10,
) -> tuple[int, int]:
    """Find [lo, hi] bounds for content using a 1D ink profile.

    - For rows: counts[r] = # dark pixels in row r.
    - For cols: counts[c] = # dark pixels in col c.

    When prefer_internal_blank_split=True, we try to cut *above* an internal
    blank run that separates the plot from the caption. That’s the key trick
    for removing captions.
    """

    n = len(counts)

    # First/last rows (or columns) that exceed the "content" threshold.
    idx = np.where(counts > content_thr)[0]
    if len(idx) == 0:
        return 0, n - 1

    lo = max(0, int(idx[0]) - pad_lo)
    hi_content = int(idx[-1])

    if prefer_internal_blank_split:
        blank = counts < blank_thr

        # Walk upwards from the bottom looking for an internal blank run.
        i = n - 1
        while i >= 0:
            if blank[i]:
                j = i
                while j >= 0 and blank[j]:
                    j -= 1
                run_start = j + 1
                run_end = i
                run_len = run_end - run_start + 1

                # We only trust a blank run if:
                # - it's long enough,
                # - it's not right near the top bound,
                # - there is content both above and below (i.e. it really splits two regions).
                if run_len >= min_blank_run and run_start > lo + 20:
                    below_has_content = (
                        np.any(counts[run_end + 1 :] > content_thr)
                        if run_end + 1 < n
                        else False
                    )
                    above_has_content = np.any(
                        counts[max(lo, run_start - 50) : run_start] > content_thr
                    )

                    if above_has_content and below_has_content:
                        # Cut at the start of that blank run (caption begins below it).
                        hi = run_start - 1
                        # Add padding, but never cross into the blank separator.
                        hi = min(n - 1, min(hi + pad_hi, run_start - 1))
                        return lo, hi

                i = j
            else:
                i -= 1

    # Default: last content row/col plus padding.
    hi = min(n - 1, hi_content + pad_hi)
    return lo, hi


def crop_2x2_panels(
    rgb: np.ndarray,
    *,
    dark_threshold: int = 235,
    pad_frac_top: float = 0.04,
    pad_frac_bottom: float = 0.06,
    pad_frac_side: float = 0.04,
) -> dict[str, np.ndarray]:
    """Crop four panels from an image that is laid out like a 2x2 grid.

    Returns dict with keys: top_left, top_right, bottom_left, bottom_right.
    """

    gray = _to_gray(rgb)
    H, W = gray.shape

    y_split = _find_y_split(gray)
    x_split = W // 2

    def crop_quadrant(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        q = gray[y0:y1, x0:x1]
        mask = q < float(dark_threshold)

        # Ink profiles (rows/cols)
        row = mask.sum(axis=1).astype(np.float32)
        col = mask.sum(axis=0).astype(np.float32)

        h, w = q.shape

        # Adaptive thresholds based on quadrant size.
        content_thr_row = max(10.0, 0.03 * w)
        content_thr_col = max(10.0, 0.03 * h)
        blank_thr_row = max(2.0, 0.004 * w)
        blank_thr_col = max(2.0, 0.004 * h)

        # Padding in pixels.
        pad_top = int(pad_frac_top * h) + 4
        pad_bottom = int(pad_frac_bottom * h) + 6
        pad_side = int(pad_frac_side * w) + 4

        top, bottom = _find_content_bounds_1d(
            row,
            blank_thr=blank_thr_row,
            content_thr=content_thr_row,
            pad_lo=pad_top,
            pad_hi=pad_bottom,
            prefer_internal_blank_split=True,
            min_blank_run=10,
        )
        left, right = _find_content_bounds_1d(
            col,
            blank_thr=blank_thr_col,
            content_thr=content_thr_col,
            pad_lo=pad_side,
            pad_hi=pad_side,
            prefer_internal_blank_split=False,
        )

        return rgb[y0 + top : y0 + bottom + 1, x0 + left : x0 + right + 1]

    return {
        "top_left": crop_quadrant(0, y_split, 0, x_split),
        "top_right": crop_quadrant(0, y_split, x_split, W),
        "bottom_left": crop_quadrant(y_split, H, 0, x_split),
        "bottom_right": crop_quadrant(y_split, H, x_split, W),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract 2x2 plot panels from a PDF screenshot image."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input image path")
    parser.add_argument(
        "--outdir", type=Path, default=Path("figs"), help="Output directory"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="paper_",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--dark-threshold",
        type=int,
        default=235,
        help="Pixel values below this (in grayscale) count as 'ink'",
    )
    parser.add_argument(
        "--pad-top",
        type=float,
        default=0.04,
        help="Top padding as a fraction of panel height",
    )
    parser.add_argument(
        "--pad-bottom",
        type=float,
        default=0.06,
        help="Bottom padding as a fraction of panel height",
    )
    parser.add_argument(
        "--pad-side",
        type=float,
        default=0.04,
        help="Side padding as a fraction of panel width",
    )

    args = parser.parse_args()

    rgb = _load_rgb(args.input)
    crops = crop_2x2_panels(
        rgb,
        dark_threshold=args.dark_threshold,
        pad_frac_top=args.pad_top,
        pad_frac_bottom=args.pad_bottom,
        pad_frac_side=args.pad_side,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Keep naming consistent with what we generated manually.
    mapping = {
        "top_left": "top_left",
        "top_right": "top_right",
        "bottom_left": "bottom_left",
        "bottom_right": "bottom_right",
    }

    for key, suffix in mapping.items():
        out = args.outdir / f"{args.prefix}{suffix}.png"
        _save_rgb(out, crops[key].astype(np.uint8))
        print(f"wrote {out} shape={crops[key].shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
