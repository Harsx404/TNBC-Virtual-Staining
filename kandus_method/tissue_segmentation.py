"""
tissue_segmentation.py - Kandu's Method: Tissue Compartment Segmentation (v2)
===============================================================================
Segments H&E (or PDL1 IHC) images into three compartments:
  Tumor cells (TC), Lymphocytes/immune (LC), Stroma (ST)

FIX vs v1 stain_analysis.py compartment code
--------------------------------------------
  OLD: Otsu on raw H channel; thresholds 80px (LC) and 500px (TC) too small.
       Classified ~99% of tissue as stroma, tumor_percent came out 0.6%.

  NEW:
    1. Detect ALL nuclei using adaptive thresholding (handles non-uniform staining)
    2. Watershed to separate touching nuclei
    3. Classify by RELATIVE nuclear area (percentile-based, not hard px thresholds)
       - Smallest 30% of nuclei -> Lymphocytes (small round dark cells)
       - Remaining 70%          -> Tumor cells (larger, pleomorphic nuclei)
    4. Tumor REGION includes the cytoplasm around tumor nuclei (dilated)
    5. Stroma = tissue - (tc_region + lc_region)

This gives biologically plausible results:
  tumor_percent  typically 0.20 - 0.60 in TNBC TMA cores
  immune_percent typically 0.05 - 0.30
  stroma_percent = remainder
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def segment_tissue(
    image_path:  str | Path,
    tissue_mask: Optional[np.ndarray] = None,
    proc_width:  int = 512,   # downsample to this width for speed; masks upscaled back
) -> dict:
    """
    Segment a histopathology image into Tumor / Lymphocyte / Stroma.

    Parameters
    ----------
    image_path  : path to H&E or IHC image
    tissue_mask : pre-computed tissue mask at ORIGINAL resolution [H, W] bool
    proc_width  : process at this width for speed (masks upscaled to original)

    Returns
    -------
    dict with:
        tc_mask, lc_mask, st_mask  : [H,W] bool at ORIGINAL resolution
        nuclei_mask                : [H,W] bool at original resolution
        tc_labels, lc_labels       : [H,W] int
        tc_count, lc_count         : int
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]

    # ---------------------------------------------------------------
    # Downsample for speed (full 1000x1000 watershed is very slow)
    # ---------------------------------------------------------------
    scale = proc_width / max(orig_w, 1)
    if scale < 1.0:
        proc_h = int(orig_h * scale)
        proc_w = proc_width
        image_small = cv2.resize(image_rgb, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        if tissue_mask is not None:
            tissue_small = cv2.resize(
                tissue_mask.astype(np.uint8), (proc_w, proc_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            tissue_small = None
    else:
        image_small  = image_rgb
        tissue_small = tissue_mask
        proc_h, proc_w = orig_h, orig_w

    # Use downsampled tissue mask or compute one
    if tissue_small is None:
        from kandus_method.stain_analysis import get_tissue_mask
        tissue_small = get_tissue_mask(image_small)

    # ---------------------------------------------------------------
    # Step 1: Extract Hematoxylin channel (on downsampled image)
    # ---------------------------------------------------------------
    try:
        from skimage.color import rgb2hed
        hed = rgb2hed(image_small)
        h_channel = np.clip(hed[:, :, 0], 0, None).astype(np.float32)
    except ImportError:
        gray = cv2.cvtColor(image_small, cv2.COLOR_RGB2GRAY)
        h_channel = (255 - gray).astype(np.float32) / 255.0

    h_norm = (h_channel / (h_channel.max() + 1e-8) * 255).astype(np.uint8)

    # ---------------------------------------------------------------
    # Step 2: Adaptive nuclei detection
    # ---------------------------------------------------------------
    h_tissue = h_norm.copy()
    h_tissue[~tissue_small] = 0

    h_blur = cv2.bilateralFilter(h_tissue, 9, 75, 75)

    nuclei_bin = cv2.adaptiveThreshold(
        h_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=-2
    )
    nuclei_bin[~tissue_small] = 0

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    nuclei_bin = cv2.morphologyEx(nuclei_bin, cv2.MORPH_OPEN, k3)

    # ---------------------------------------------------------------
    # Step 3: Distance transform + Watershed
    # ---------------------------------------------------------------
    dist = cv2.distanceTransform(nuclei_bin, cv2.DIST_L2, 5)

    _, sure_fg = cv2.threshold(dist, 0.30 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sure_bg = cv2.dilate(nuclei_bin, k7, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown > 0] = 0

    markers_ws = markers.copy()
    img_u8 = image_small.astype(np.uint8)    # use downsampled BGR for watershed
    try:
        cv2.watershed(img_u8, markers_ws)
    except cv2.error:
        markers_ws = markers

    nuclei_labels = markers_ws.copy()
    nuclei_labels[nuclei_labels <= 1] = 0

    # ---------------------------------------------------------------
    # Step 4: Classify nuclei by size (percentile-based)
    # ---------------------------------------------------------------
    max_label = int(nuclei_labels.max())
    if max_label == 0:
        return _no_nuclei_result(tissue_mask, orig_h, orig_w)

    areas = []
    for lbl in range(2, max_label + 1):
        a = int((nuclei_labels == lbl).sum())
        if a > 5:
            areas.append((lbl, a))

    if not areas:
        return _no_nuclei_result(tissue_mask, orig_h, orig_w)

    area_vals = np.array([a for _, a in areas])
    lc_threshold = float(np.percentile(area_vals, 35))
    tc_min = max(lc_threshold, 50.0)

    lc_nuclei_mask = np.zeros(nuclei_labels.shape, dtype=np.uint8)
    tc_nuclei_mask = np.zeros(nuclei_labels.shape, dtype=np.uint8)

    for lbl, area in areas:
        comp = (nuclei_labels == lbl).astype(np.uint8)
        if area <= lc_threshold:
            lc_nuclei_mask |= comp
        elif area > tc_min:
            tc_nuclei_mask |= comp

    # ---------------------------------------------------------------
    # Step 5: Build compartment masks at SMALL resolution
    # ---------------------------------------------------------------
    k4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    lc_region_s = cv2.dilate(lc_nuclei_mask, k4, iterations=2)

    k8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    tc_region_s = cv2.dilate(tc_nuclei_mask, k8, iterations=2)

    lc_mask_s = (lc_region_s > 0) & tissue_small
    tc_mask_s = (tc_region_s > 0) & tissue_small & ~lc_mask_s
    st_mask_s = tissue_small & ~tc_mask_s & ~lc_mask_s
    nuclei_s  = nuclei_labels > 0

    # ---------------------------------------------------------------
    # Step 6: Upsample masks back to ORIGINAL resolution
    # ---------------------------------------------------------------
    def _up(m):
        return cv2.resize(m.astype(np.uint8), (orig_w, orig_h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    tc_mask_orig    = _up(tc_mask_s)
    lc_mask_orig    = _up(lc_mask_s)
    st_mask_orig    = _up(st_mask_s)
    nuclei_orig     = _up(nuclei_s)
    tc_labels_orig  = cv2.resize(tc_nuclei_mask * (nuclei_labels > 0).astype(np.uint8),
                                  (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    lc_labels_orig  = cv2.resize(lc_nuclei_mask * (nuclei_labels > 0).astype(np.uint8),
                                  (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    tc_count = len([lbl for lbl, a in areas if a > tc_min])
    lc_count = len([lbl for lbl, a in areas if a <= lc_threshold])

    # ---------------------------------------------------------------
    # Step 7: Extract centroid coordinates (at original resolution)
    # ---------------------------------------------------------------
    scale_back = 1.0 / scale if scale < 1.0 else 1.0

    tc_coords = _extract_nuclei_coords(
        nuclei_labels, areas, lc_threshold, tc_min,
        cell_type="TC", scale=scale_back,
    )
    lc_coords = _extract_nuclei_coords(
        nuclei_labels, areas, lc_threshold, tc_min,
        cell_type="LC", scale=scale_back,
    )
    st_coords = _sample_stroma_coords(st_mask_s, scale=scale_back, step=20)

    return {
        "tc_mask":      tc_mask_orig,
        "lc_mask":      lc_mask_orig,
        "st_mask":      st_mask_orig,
        "nuclei_mask":  nuclei_orig,
        "tc_labels":    tc_labels_orig,
        "lc_labels":    lc_labels_orig,
        "tc_count":     tc_count,
        "lc_count":     lc_count,
        "all_areas":    area_vals,
        # Per-cell coordinates at ORIGINAL image resolution
        # Each list item: {"x": int, "y": int, "area_px": int, "cell_type": str}
        "tc_coords":    tc_coords,
        "lc_coords":    lc_coords,
        "st_coords":    st_coords,
    }


def _no_nuclei_result(tissue_mask: np.ndarray, orig_h: int = 0, orig_w: int = 0) -> dict:
    """Return all-stroma result when no nuclei are detected."""
    h = orig_h or tissue_mask.shape[0]
    w = orig_w or tissue_mask.shape[1]
    if tissue_mask.shape[0] != h or tissue_mask.shape[1] != w:
        tm = cv2.resize(tissue_mask.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        tm = tissue_mask
    empty = np.zeros((h, w), dtype=bool)
    return {
        "tc_mask":     empty,
        "lc_mask":     empty,
        "st_mask":     tm,
        "nuclei_mask": empty,
        "tc_labels":   np.zeros((h, w), dtype=np.int32),
        "lc_labels":   np.zeros((h, w), dtype=np.int32),
        "tc_count":    0,
        "lc_count":    0,
        "all_areas":   np.array([]),
        "tc_coords":   [],
        "lc_coords":   [],
        "st_coords":   _sample_stroma_coords(tm, scale=1.0, step=20),
    }


# ---------------------------------------------------------------------------
# Coordinate extraction helpers
# ---------------------------------------------------------------------------

def _extract_nuclei_coords(
    nuclei_labels: np.ndarray,
    areas: list[tuple[int, int]],
    lc_threshold: float,
    tc_min: float,
    cell_type: str,  # "TC" or "LC"
    scale: float = 1.0,
) -> list[dict]:
    """
    Extract centroid coordinates from watershed nucleus labels.

    Parameters
    ----------
    nuclei_labels : [H, W] int array (watershed output at SMALL resolution)
    areas         : list of (label, area) pairs
    lc_threshold  : size cutoff below which nuclei are LC
    tc_min        : minimum size to be TC
    cell_type     : 'TC' or 'LC'
    scale         : multiply coords by this to get ORIGINAL resolution coords

    Returns
    -------
    list of dicts: [{x, y, area_px, cell_type}, ...]
    """
    try:
        from skimage.measure import regionprops
    except ImportError:
        # Fallback: use connected components centroids via cv2
        coords = []
        for lbl, area in areas:
            if cell_type == "LC" and area > lc_threshold:
                continue
            if cell_type == "TC" and area <= tc_min:
                continue
            mask = (nuclei_labels == lbl).astype(np.uint8)
            M = cv2.moments(mask)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"] * scale)
                cy = int(M["m01"] / M["m00"] * scale)
                coords.append({"x": cx, "y": cy,
                               "area_px": int(area * scale * scale),
                               "cell_type": cell_type})
        return coords

    # Build a label image containing only the requested cell type
    filtered = np.zeros_like(nuclei_labels, dtype=np.int32)
    valid_labels = set()
    for lbl, area in areas:
        if cell_type == "LC" and area <= lc_threshold:
            filtered[nuclei_labels == lbl] = lbl
            valid_labels.add(lbl)
        elif cell_type == "TC" and area > tc_min:
            filtered[nuclei_labels == lbl] = lbl
            valid_labels.add(lbl)

    if filtered.max() == 0:
        return []

    props = regionprops(filtered)
    coords = []
    for prop in props:
        if prop.label not in valid_labels:
            continue
        # regionprops centroid: (row, col) = (y, x)
        cy, cx = prop.centroid
        cx_orig = int(cx * scale)
        cy_orig = int(cy * scale)
        area_orig = int(prop.area * scale * scale)
        coords.append({
            "x":         cx_orig,
            "y":         cy_orig,
            "area_px":   area_orig,
            "cell_type": cell_type,
        })
    return coords


def _sample_stroma_coords(
    st_mask: np.ndarray,
    scale: float = 1.0,
    step: int = 20,
) -> list[dict]:
    """
    Sample representative stroma positions from the stroma mask.
    Since stroma has no distinct nuclei, we sample at a regular grid.

    Parameters
    ----------
    st_mask : [H, W] bool mask at SMALL resolution
    scale   : multiply coords by this to get ORIGINAL resolution coords
    step    : grid spacing in small-resolution pixels

    Returns
    -------
    list of dicts: [{x, y, cell_type='ST'}, ...]
    """
    ys, xs = np.where(st_mask)
    if len(ys) == 0:
        return []
    # Subsample: take every N-th point from the set
    # (using argmin over a grid is expensive; use modular skip instead)
    idx = np.arange(0, len(ys), step)
    coords = []
    for i in idx:
        coords.append({
            "x":         int(xs[i] * scale),
            "y":         int(ys[i] * scale),
            "cell_type": "ST",
        })
    return coords


def extract_cell_coordinates(
    segmentation_result: dict,
) -> dict[str, list[dict]]:
    """
    Extract pre-computed coordinates from a segmentation result dict.

    Use this when you already have the result of segment_tissue() and
    just want the coordinates in a clean format.

    Returns
    -------
    dict with keys 'TC', 'LC', 'ST' each containing a list of
    coordinate dicts: [{x, y, area_px (TC/LC only), cell_type}, ...]
    """
    return {
        "TC": segmentation_result.get("tc_coords", []),
        "LC": segmentation_result.get("lc_coords", []),
        "ST": segmentation_result.get("st_coords", []),
    }
