"""
run_pipeline.py - Kandu's Method: End-to-End Pipeline Runner
=============================================================
Runs the complete Kandu's Method pipeline on a single TMA core or
an entire data_raw/ directory, producing all 8 required PRD outputs.

Pipeline flow per core
----------------------
  1. Load PDL1 image + HE image
  2. Stain Analysis (stain_analysis.py)
     -> PDL1_percent, TC_PDL1, LC_PDL1, ST_PDL1
  3. CNN Morphology Model (cnn_model.py via infer_cnn.py)
     -> pdl1_prob_he (from H&E tiles)
  4. Scoring (scoring.py)
     -> CPS, CPS++, spatial metrics, full feature vector
  5. Patient-level aggregation across all cores

Required outputs (PRD Section 11)
----------------------------------
  tumor_percent    immune_percent    PDL1_percent
  TC_PDL1          LC_PDL1           ST_PDL1
  CPS              CPS++

Usage - single core
-------------------
  python -m kandus_method.run_pipeline
      --mode      single
      --pdl1_img  ./data_raw/02-008_PDL1.../image.jpeg
      --he_img    ./data_raw/02-008_HE.../image.jpeg
      --checkpoint ./kandus_method/checkpoints/best_model_resnet101.pt

Usage - full data_raw directory
--------------------------------
  python -m kandus_method.run_pipeline
      --mode     batch
      --data_raw ./data_raw
      --checkpoint ./kandus_method/checkpoints/best_model_resnet101.pt
      --output   ./results/pipeline_results.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from kandus_method.stain_analysis import analyze_pdl1_image, analyze_pd1_image
from kandus_method.scoring import compute_scores, aggregate_patient
from kandus_method.data_raw_adapter import DataRawAdapter


# ---------------------------------------------------------------------------
# Single-core pipeline
# ---------------------------------------------------------------------------

def run_single_core(
    pdl1_img_path:  str | Path,
    he_img_path:    Optional[str | Path],
    pd1_img_path:   Optional[str | Path],
    checkpoint:     Optional[str | Path],
    core_id:        str = "unknown",
    alpha:          float = 0.7,
    device:         str = "cpu",
    debug_dir:      Optional[str | Path] = None,
) -> dict:
    """
    Run the full pipeline on a single TMA core.
    Returns the complete feature dict with all PRD outputs + PD1 metrics.
    """
    print(f"\n[Pipeline] Core: {core_id}")

    # ------------------------------------------------------------------
    # Step 1: Tissue Segmentation (from H&E for best accuracy)
    # ------------------------------------------------------------------
    from kandus_method.tissue_segmentation import segment_tissue
    from kandus_method.stain_analysis import get_tissue_mask
    import cv2

    seg_source = he_img_path if he_img_path else pdl1_img_path
    img_bgr = cv2.imread(str(pdl1_img_path))
    pdl1_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tissue_mask = get_tissue_mask(pdl1_rgb)

    compartments = segment_tissue(seg_source, tissue_mask=tissue_mask)
    tc_count = compartments.get("tc_count", 1)
    lc_count = compartments.get("lc_count", 0)
    tc_mask  = compartments.get("tc_mask")
    lc_mask  = compartments.get("lc_mask")

    # ------------------------------------------------------------------
    # Step 2: PD-L1 Stain Analysis
    # ------------------------------------------------------------------
    print(f"  [1/4] PDL1 stain analysis...")
    stain_result = analyze_pdl1_image(
        pdl1_image_path = pdl1_img_path,
        he_image_path   = he_img_path,
        debug           = (debug_dir is not None),
    )
    # Inject cell counts and compartment masks into stain result for CPS
    stain_result["_tc_count"]     = max(tc_count, 1)
    stain_result["_lc_count"]     = lc_count
    stain_result["_compartments"] = compartments
    stain_result["tc_area"] = int(tc_mask.sum()) if tc_mask is not None else stain_result.get("tc_area", 0)
    stain_result["lc_area"] = int(lc_mask.sum()) if lc_mask is not None else stain_result.get("lc_area", 0)
    stain_result["st_area"] = int(compartments.get("st_mask", np.array([])).sum()) if compartments.get("st_mask") is not None else stain_result.get("st_area", 0)

    print(f"        PDL1%={stain_result['PDL1_percent']:.3f}  "
          f"TC_pos={stain_result['TC_PDL1']:.3f}  "
          f"LC_pos={stain_result['LC_PDL1']:.3f}  "
          f"ST_pos={stain_result['ST_PDL1']:.3f}")

    # ------------------------------------------------------------------
    # Step 2.5: PD1 Stain Analysis (exhausted TILs)
    # ------------------------------------------------------------------
    pd1_result = None
    if pd1_img_path is not None and Path(str(pd1_img_path)).exists():
        print(f"  [2/4] PD1 stain analysis (TIL exhaustion)...")
        try:
            pd1_result = analyze_pd1_image(
                pd1_image_path = pd1_img_path,
                he_image_path  = he_img_path,
                compartments   = compartments,  # reuse — no re-segmentation
            )
            print(f"        PD1%={pd1_result['PD1_percent']:.3f}  "
                  f"PD1_LC={pd1_result['PD1_LC']:.3f}  "
                  f"TIL_density={pd1_result['TIL_density']:.4f}  "
                  f"exhaustion={pd1_result['exhaustion_score']:.3f}")
        except Exception as e:
            print(f"        [warn] PD1 analysis failed: {e}")
    else:
        print(f"  [2/4] PD1 skipped (no image found)")

    # ------------------------------------------------------------------
    # Step 3: CNN Morphology (H&E tiles)
    # ------------------------------------------------------------------
    cnn_prob = 0.0
    if he_img_path is not None and checkpoint is not None:
        print(f"  [3/4] CNN inference (H&E tiles)...")
        try:
            cnn_prob = _run_cnn_inference(he_img_path, checkpoint, device)
            print(f"        pdl1_prob_he={cnn_prob:.4f}")
        except Exception as e:
            print(f"        [warn] CNN inference skipped: {e}")
    else:
        print(f"  [3/4] CNN skipped (no checkpoint or H&E image)")

    # ------------------------------------------------------------------
    # Step 4: Scoring
    # ------------------------------------------------------------------
    print(f"  [4/4] Computing CPS and CPS++...")
    scores = compute_scores(
        stain_result = stain_result,
        cnn_prob     = cnn_prob,
        lc_mask      = lc_mask,
        tc_mask      = tc_mask,
        alpha        = alpha,
    )
    print(f"        CPS={scores['CPS']:.2f}  "
          f"CPS++={scores['CPS_plus_plus']:.4f}  "
          f"tumor%={scores['tumor_percent']:.3f}  "
          f"immune%={scores['immune_percent']:.3f}  "
          f"TC_count={tc_count}  LC_count={lc_count}")

    # ------------------------------------------------------------------
    # Step 5: Debug overlays (optional)
    # ------------------------------------------------------------------
    if debug_dir is not None:
        try:
            from kandus_method.visualization_debug import save_debug_overlays
            stain_result["CPS"]          = scores["CPS"]
            stain_result["CPS_plus_plus"] = scores["CPS_plus_plus"]
            save_debug_overlays(
                image_path      = he_img_path or pdl1_img_path,
                stain_result    = stain_result,
                output_dir      = Path(debug_dir) / core_id,
                core_id         = core_id,
                pdl1_image_path = pdl1_img_path,
            )
        except Exception as e:
            print(f"        [warn] Debug overlays failed: {e}")

    # ------------------------------------------------------------------
    # Step 6: Save per-cell coordinate CSV
    # ------------------------------------------------------------------
    tc_coords = compartments.get("tc_coords", [])
    lc_coords = compartments.get("lc_coords", [])
    st_coords = compartments.get("st_coords", [])

    coords_dir = Path("results") / "coords"
    coords_dir.mkdir(parents=True, exist_ok=True)
    csv_path = coords_dir / f"{core_id}_cells.csv"

    import csv as _csv
    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=["x", "y", "area_px", "cell_type"])
        writer.writeheader()
        for c in tc_coords:
            writer.writerow({"x": c["x"], "y": c["y"],
                             "area_px": c.get("area_px", ""), "cell_type": "TC"})
        for c in lc_coords:
            writer.writerow({"x": c["x"], "y": c["y"],
                             "area_px": c.get("area_px", ""), "cell_type": "LC"})
        for c in st_coords:
            writer.writerow({"x": c["x"], "y": c["y"],
                             "area_px": "", "cell_type": "ST"})

    print(f"        Coords saved → {csv_path}  "
          f"(TC={len(tc_coords)}, LC={len(lc_coords)}, ST={len(st_coords)})")

    output = {
        "core_id":    core_id,
        "pdl1_image": str(pdl1_img_path),
        "pd1_image":  str(pd1_img_path) if pd1_img_path else None,
        "he_image":   str(he_img_path) if he_img_path else None,
        **scores,
        # PD1 metrics (if available)
        **({
            "PD1_percent":      pd1_result["PD1_percent"],
            "PD1_LC":           pd1_result["PD1_LC"],
            "PD1_TC":           pd1_result["PD1_TC"],
            "PD1_ST":           pd1_result["PD1_ST"],
            "TIL_density":      pd1_result["TIL_density"],
            "exhaustion_score": pd1_result["exhaustion_score"],
        } if pd1_result else {}),
    }
    return output


def _run_cnn_inference(
    he_path:    str | Path,
    checkpoint: str | Path,
    device:     str = "cpu",
    tile_size:  int = 512,
    stride:     int = 256,
    max_tiles:  int = 16,
    sub_batch:  int = 8,
) -> float:
    """
    Tile H&E image and run MIL forward pass to get pdl1_prob_he.
    Returns a float [0, 1].
    """
    import random
    from kandus_method.cnn_model import MILClassifier
    from kandus_method.dataset_kandu import HETileDataset

    device_obj = torch.device(
        device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    # Load model
    ckpt    = torch.load(checkpoint, map_location=device_obj, weights_only=False)
    config  = ckpt.get("config", {})
    model   = MILClassifier(
        backbone   = config.get("backbone",   "resnet101"),
        hidden_dim = config.get("hidden_dim", 256),
        dropout    = config.get("dropout",    0.0),   # eval mode
    ).to(device_obj)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Tile H&E image
    tile_ds  = HETileDataset(he_path, tile_size=tile_size, stride=stride)
    indices  = list(range(len(tile_ds)))
    if len(indices) > max_tiles:
        indices = random.sample(indices, max_tiles)

    tiles = torch.stack([tile_ds[i][0] for i in indices]).to(device_obj)

    with torch.no_grad():
        poi_prob, _, _ = model(tiles, sub_batch=sub_batch)

    return float(poi_prob.cpu().item())


# ---------------------------------------------------------------------------
# Batch pipeline over entire data_raw directory
# ---------------------------------------------------------------------------

def run_batch(
    data_raw:   str | Path,
    checkpoint: Optional[str | Path],
    output:     Optional[str | Path] = None,
    alpha:      float = 0.7,
    device:     str   = "cuda",
    debug_dir:  Optional[str | Path] = None,
) -> dict:
    """
    Run the full pipeline on all TMA cores in data_raw/.
    CNN model is loaded ONCE and reused across all cores for speed.
    """
    # Load CNN model once
    _cached_model = None
    _cached_device = None
    if checkpoint and Path(checkpoint).exists():
        try:
            from kandus_method.cnn_model import MILClassifier
            dev = torch.device(device if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
            cfg  = ckpt.get("config", {})
            _cached_model = MILClassifier(
                backbone   = cfg.get("backbone",   "resnet101"),
                hidden_dim = cfg.get("hidden_dim", 256),
                dropout    = 0.0,
            ).to(dev)
            _cached_model.load_state_dict(ckpt["model_state"])
            _cached_model.eval()
            _cached_device = dev
            print(f"[Pipeline] CNN model loaded on {dev}")
        except Exception as e:
            print(f"[Pipeline] CNN load failed: {e} — skipping CNN inference")

    adapter = DataRawAdapter(data_raw)
    print(f"[Pipeline] {adapter.summary()}")

    per_core_results = []

    for rec in adapter.records:
        try:
            result = run_single_core(
                pdl1_img_path = rec.pdl1_path,
                he_img_path   = rec.he_path,
                pd1_img_path  = rec.pd1_path,
                checkpoint    = checkpoint,
                core_id       = rec.core_id,
                alpha         = alpha,
                device        = device,
                debug_dir     = debug_dir,
            )
            # Add ground-truth label if available
            result["pdl1_label_gt"]  = rec.pdl1_label
            result["stain_score_gt"] = rec.stain_score
            per_core_results.append(result)
        except Exception as e:
            print(f"  [ERROR] Core {rec.core_id}: {e}")

    # Patient-level aggregation
    if per_core_results:
        patient_scores = aggregate_patient(per_core_results, method="mean")
    else:
        patient_scores = {}

    print("\n" + "="*60)
    print("[Pipeline] PATIENT-LEVEL RESULTS")
    print("="*60)
    for key in ["tumor_percent", "immune_percent", "PDL1_percent",
                "TC_PDL1", "LC_PDL1", "ST_PDL1", "CPS", "CPS_plus_plus"]:
        val = patient_scores.get(key, "N/A")
        print(f"  {key:30s}: {val}")
    print(f"  {'CPS_category':30s}: {patient_scores.get('CPS_category','N/A')}")
    print("="*60)

    full_result = {
        "per_core":  per_core_results,
        "patient":   patient_scores,
    }

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(full_result, f, indent=2, default=str)
        print(f"\n[Pipeline] Results saved to: {out_path}")

    return full_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Kandu's Method - Full Pipeline")
    p.add_argument("--mode",       choices=["single", "batch"], default="batch")
    p.add_argument("--data_raw",   default="./data_raw",
                   help="data_raw/ folder (batch mode)")
    p.add_argument("--pdl1_img",   default=None,
                   help="Path to single PDL1 image (single mode)")
    p.add_argument("--he_img",     default=None,
                   help="Path to matching HE image (single mode, optional)")
    p.add_argument("--checkpoint", default="./kandus_method/checkpoints/best_model_resnet101.pt",
                   help="Path to trained CNN checkpoint")
    p.add_argument("--output",     default="./results/pipeline_results.json",
                   help="Output JSON path (batch mode)")
    p.add_argument("--alpha",      type=float, default=0.7,
                   help="CPS++ weighting: alpha*CPS + (1-alpha)*spatial")
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.mode == "batch":
        run_batch(
            data_raw   = args.data_raw,
            checkpoint = args.checkpoint,
            output     = args.output,
            alpha      = args.alpha,
            device     = args.device,
        )
    else:
        if not args.pdl1_img:
            print("Error: --pdl1_img required for single mode")
        else:
            result = run_single_core(
                pdl1_img_path = args.pdl1_img,
                he_img_path   = args.he_img,
                checkpoint    = args.checkpoint,
                alpha         = args.alpha,
                device        = args.device,
            )
            print("\nResult:")
            print(json.dumps(result, indent=2, default=str))
