"""
visualize_coords.py  — Draw TC / LC / ST cell positions on H&E image.

Color scheme (OpenCV BGR):
  TC  Tumor cells    → Red      (0, 0, 220)
  LC  Lymphocytes    → Green    (0, 200, 50)
  ST  Stroma sample  → Blue     (200, 130, 0)   (small dots, semi-opaque)

Usage:
  python visualize_coords.py [core_id]

  e.g.  python visualize_coords.py 005_r1c5
"""
from __future__ import annotations
import sys
import csv
from pathlib import Path
import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
CELL_COLORS = {
    "TC": (0,   0, 220),    # Red   (BGR)
    "LC": (0, 200,  50),    # Green
    "ST": (180, 100,  0),   # Dark-blue / brown tint
}
CELL_RADIUS = {
    "TC": 6,
    "LC": 4,
    "ST": 2,
}
CELL_THICKNESS = {
    "TC": -1,   # filled
    "LC": -1,   # filled
    "ST": -1,   # filled
}
ALPHA = {
    "TC": 0.90,
    "LC": 0.90,
    "ST": 0.45,  # stroma points are more transparent (there are thousands)
}

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW   = Path("data_raw")
HE_DIR     = next((d for d in DATA_RAW.iterdir() if "HE" in d.name), None)
COORDS_DIR = Path("results/coords")
OUT_DIR    = Path("results/overlays")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def find_he_image(core_id: str) -> Path | None:
    if HE_DIR is None:
        return None
    for f in HE_DIR.iterdir():
        if core_id in f.name:
            return f
    return None


def load_coords(csv_path: Path) -> dict[str, list[tuple[int, int]]]:
    coords: dict[str, list] = {"TC": [], "LC": [], "ST": []}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ct = row["cell_type"]
            if ct in coords:
                coords[ct].append((int(row["x"]), int(row["y"])))
    return coords


def draw_overlay(image_bgr: np.ndarray, coords: dict) -> np.ndarray:
    """Draw colored dots on a copy of the image, cell-type by cell-type."""
    result = image_bgr.copy()

    for cell_type in ["ST", "LC", "TC"]:   # draw in order: ST behind, TC on top
        color  = CELL_COLORS[cell_type]
        radius = CELL_RADIUS[cell_type]
        thick  = CELL_THICKNESS[cell_type]
        alpha  = ALPHA[cell_type]
        pts    = coords.get(cell_type, [])

        if not pts:
            continue

        # Draw on a transparent layer then blend
        layer = result.copy()
        for (x, y) in pts:
            cv2.circle(layer, (x, y), radius, color, thick, lineType=cv2.LINE_AA)

        cv2.addWeighted(layer, alpha, result, 1 - alpha, 0, result)

    return result


def add_legend(img: np.ndarray, counts: dict) -> np.ndarray:
    """Add a simple legend to the top-left corner."""
    legend = [
        ("TC  Tumor cells", CELL_COLORS["TC"], counts["TC"]),
        ("LC  Lymphocytes", CELL_COLORS["LC"], counts["LC"]),
        ("ST  Stroma pts",  CELL_COLORS["ST"], counts["ST"]),
    ]
    pad   = 12
    lh    = 28
    lw    = 270
    box_h = pad + lh * len(legend) + pad
    # Overlay a semi-transparent dark box
    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (8 + lw, 8 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    for i, (label, color, count) in enumerate(legend):
        y = 8 + pad + i * lh + 16
        cv2.circle(img, (20, y - 4), 7, color, -1, lineType=cv2.LINE_AA)
        text = f"{label}  ({count})"
        cv2.putText(img, text, (34, y), cv2.FONT_HERSHEY_DUPLEX,
                    0.48, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def visualize_core(core_id: str):
    csv_path = COORDS_DIR / f"{core_id}_cells.csv"
    if not csv_path.exists():
        print(f"[ERROR] No coordinate CSV found for {core_id}: {csv_path}")
        print("  Run the pipeline first:  python -m kandus_method.run_pipeline --mode batch ...")
        return

    he_path = find_he_image(core_id)
    if he_path is None:
        print(f"[ERROR] Can't find HE image for {core_id}")
        return

    print(f"Loading H&E: {he_path}")
    img = cv2.imread(str(he_path))
    if img is None:
        print(f"[ERROR] Failed to read image: {he_path}")
        return

    print(f"Loading coords: {csv_path}")
    coords = load_coords(csv_path)
    counts = {k: len(v) for k, v in coords.items()}
    print(f"  TC={counts['TC']}  LC={counts['LC']}  ST={counts['ST']}")

    print("Drawing overlay...")
    out = draw_overlay(img, coords)
    out = add_legend(out, counts)

    out_path = OUT_DIR / f"{core_id}_overlay.png"
    cv2.imwrite(str(out_path), out)
    print(f"Saved → {out_path}")
    return out_path


def visualize_all():
    """Process every CSV in results/coords/."""
    csvs = sorted(COORDS_DIR.glob("*_cells.csv"))
    print(f"Found {len(csvs)} core CSVs")
    for csv_path in csvs:
        core_id = csv_path.stem.replace("_cells", "")
        print(f"\n--- {core_id} ---")
        visualize_core(core_id)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize_core(sys.argv[1])
    else:
        print("Usage: python visualize_coords.py <core_id>")
        print("       python visualize_coords.py all")
        print("       e.g. python visualize_coords.py 005_r1c5")
