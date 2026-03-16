# Kandu's Method — Weakly Supervised PD-L1 Biomarker Prediction Pipeline

## Overview

Kandu's Method is a computational pathology pipeline for estimating PD-L1 biomarker status in Triple-Negative Breast Cancer (TNBC) from histopathology images. It integrates three independent information sources — H&E morphology, PD-L1 IHC stain analysis, and PD-1 stain analysis — and combines them into clinically meaningful biomarker scores (CPS and CPS++) without requiring pixel-level annotations.

The system is **weakly supervised**: it learns from image-level binary labels (PDL1 positive/negative per core) using Multiple Instance Learning (MIL). No manual segmentation masks are needed.

---

## Dataset

| Property | Value |
|----------|-------|
| Patient | `02-008` |
| Total cores (POIs) | **168** |
| Stain types per core | H&E, PD-L1 (SP142-Springbio), PD-1 (NAT105-CellMarque) |
| Image size | 2256 × 1440 pixels per core |
| Labels | Image-level binary (`PDL1_label`: 0/1, `PD1_label`: 0/1) |
| TMA grid | 14 rows × 12 columns (named `r1c1` → `r14c12`) |

Each core is treated as one Point of Interest (POI). The 168 cores all belong to one TNBC patient on a Tissue Micro-Array (TMA) slide.

---

## Project Structure

```
TNBC/
├── data_raw/
│   ├── 02-008_HE_A12_v2_s13/              168 H&E core images
│   ├── 02-008_PDL1(SP142)-Springbio.../   168 PD-L1 IHC images
│   ├── 02-008_PD1(NAT105)-CellMarque.../  168 PD-1 IHC images
│   ├── annotation_task_automated.xlsx      image-level labels per core
│   └── pd1_ground_truth.csv               PD-1 stain scores and binary labels
│
├── kandus_method/
│   ├── stain_analysis.py        PD-L1 and PD-1 DAB stain quantification
│   ├── tissue_segmentation.py   H&E tissue compartment detection
│   ├── scoring.py               CPS and CPS++ score computation
│   ├── cnn_model.py             MIL CNN model definition (ResNet101 + attention)
│   ├── dataset_kandu.py         Dataset loader and image tiling
│   ├── data_raw_adapter.py      Adapter to load the data_raw directory format
│   ├── train_cnn.py             CNN training script
│   ├── train_data_raw.py        Training entry point for data_raw format
│   ├── infer_cnn.py             CNN inference utilities
│   ├── run_pipeline.py          End-to-end pipeline runner
│   ├── visualization_debug.py   Debug overlay generation
│   └── checkpoints/
│       └── best_model_resnet101.pt   Trained CNN checkpoint (epoch 29, AUC=0.85)
│
├── results/
│   └── pipeline_results.json    Final results: 168 per-core + 1 patient summary
│
└── docs/
    └── Project Proposal SMRIST.pdf
```

---

## Pipeline Architecture

The pipeline runs in 4 sequential steps per core image:

```
H&E image  ──► [Step 1] Tissue Segmentation ──────────────────────────────────┐
               (watershed + nuclei classification)                             │
                     │                                                         │
                     ▼                                                         │
               tc_mask, lc_mask, st_mask                                      │
               tc_count, lc_count                                              │
                     │                                                         ▼
PD-L1 image ──► [Step 2] DAB Stain Analysis ──► PDL1_percent ──► [Step 4] Scoring
               (color deconvolution)              TC_PDL1          ──► CPS
                                                  LC_PDL1          ──► CPS++
                                                  ST_PDL1          ──► All 8 outputs

PD-1 image  ──► [Step 2.5] PD-1 Stain Analysis ─► TIL_density
               (same as PD-L1 pipeline)            exhaustion_score

H&E image  ──► [Step 3] CNN (MIL ResNet101) ──► pdl1_prob_he (supplementary)
               (tiles → attention aggregation)
```

---

## Step 1: Tissue Segmentation (`tissue_segmentation.py`)

**Input:** H&E image (2256×1440 px)

The image is first downsampled to 512px wide for speed, then processed:

1. **Hematoxylin channel extraction** — using `skimage.color.rgb2hed` (H-E-D color deconvolution). The H channel highlights all nuclei.

2. **Adaptive nuclei detection** — bilateral filtering to reduce noise, then OpenCV adaptive Gaussian thresholding to detect nuclei regardless of local staining intensity variation.

3. **Watershed segmentation** — distance transform + watershed algorithm separates touching nuclei into individual instances. Each nucleus gets a unique integer label.

4. **Classification by nucleus size (percentile-based):**
   - Bottom 35% of nuclei by area → **Lymphocytes** (small, round, ~8–20 µm)
   - Top 65% of nuclei by area → **Tumor cells** (large, pleomorphic, ~15–40 µm)

5. **Compartment region building:**
   - `lc_region` = lymphocyte nuclei dilated by 4px (captures cytoplasm)
   - `tc_region` = tumor nuclei dilated by 8px (captures surrounding cytoplasm)
   - `st_mask` = tissue − (tc_region ∪ lc_region) → stroma

6. **Upscale** all masks back to original resolution.

**Why percentile-based thresholds (not fixed pixel sizes)?**
Fixed thresholds (e.g., "< 80 pixels = lymphocyte") fail across different zoom levels and staining intensities. Percentile-based classification adapts to the actual cell size distribution within each image.

**Outputs:**
```
tc_mask    : [H, W] bool  — tumor cell region pixels
lc_mask    : [H, W] bool  — lymphocyte region pixels
st_mask    : [H, W] bool  — stroma region pixels
tc_count   : int          — number of tumor nuclei detected
lc_count   : int          — number of lymphocyte nuclei detected
```

---

## Step 2: PD-L1 Stain Analysis (`stain_analysis.py`)

**Input:** PD-L1 IHC image (brown DAB staining = PD-L1 positive)

### Background / Tissue Mask
LAB color space is used to separate tissue from white glass background:
- Pixels with L > 220 (very bright) are background
- Pixels with any A or B channel deviation > 3 are tissue
- Morphological closing (9×9) fills holes; opening (3×3) removes specks

### DAB Channel Extraction (H-E-D Color Deconvolution)
The Ruifrok & Johnston (2001) stain separation matrix is applied:

```
M = [Hematoxylin, Eosin, DAB] optical density matrix
DAB channel = M_inv · optical_density(pixel)
```

The DAB channel (D) is a float map where higher values = more brown staining.

### Two-Stage DAB Positive Mask

**Stage 1 — RGB brown pre-filter:**
Only pixels where `R > G` and `R > B` can be DAB positive.
This immediately eliminates hematoxylin (blue, B > R) and eosin (pink, uniform RGB) misclassification as DAB.

**Stage 2 — Adaptive Otsu thresholding:**
Otsu threshold computed within the tissue mask on the D channel. Pixels above threshold = DAB positive.

### Compartment-Specific PD-L1 Scores
Using the `tc_mask`, `lc_mask`, `st_mask` from Step 1:

```
TC_PDL1 = DAB+ pixels ∩ tc_mask  /  tc_mask pixels
LC_PDL1 = DAB+ pixels ∩ lc_mask  /  lc_mask pixels
ST_PDL1 = DAB+ pixels ∩ st_mask  /  st_mask pixels
PDL1_percent = all DAB+ pixels   /  tissue pixels
```

**Outputs:**
```
PDL1_percent : float [0,1] — fraction of total tissue area DAB+
TC_PDL1      : float [0,1] — fraction of tumor compartment DAB+
LC_PDL1      : float [0,1] — fraction of lymphocyte compartment DAB+
ST_PDL1      : float [0,1] — fraction of stroma compartment DAB+
tc_area      : int — tumor pixel area (for CPS)
lc_area      : int — lymphocyte pixel area
st_area      : int — stroma pixel area
tissue_area  : int — total tissue pixel area
```

---

## Step 2.5: PD-1 Stain Analysis

Same DAB detection pipeline applied to PD-1 IHC images. Outputs:
- `PD1_percent`, `PD1_LC`, `PD1_TC`, `PD1_ST`
- `TIL_density` — lymphocyte density (lc_area / tissue_area)
- `exhaustion_score` — combination of PD-1+ lymphocytes and spatial distribution, representing T-cell exhaustion in the tumor microenvironment

---

## Step 3: CNN Morphology Model (`cnn_model.py`)

**Architecture: Attention-based Multiple Instance Learning (MIL)**

Large H&E images cannot be fed whole into a CNN. Instead:

### Tiling (`dataset_kandu.py`)
```
Image size : 2256 × 1440 px
Tile size  : 512 × 512 px
Stride     : 256 px  (50% overlap)
→ ~40–50 tiles per image
```

Each tile is returned with its `(x, y)` top-left pixel coordinate.

### MIL Model (Ilse et al., 2018)

```
[N tiles, 3, 512, 512]
      │
      ▼
ResNet101 backbone (ImageNet pretrained, head removed)
      │
      ▼  [N, 2048] feature vectors
      │
      ▼
Attention network (2-layer MLP → softmax weights)
      │  [N, 1] attention weights
      ▼
Weighted sum → [1, 2048] POI-level embedding
      │
      ▼
Linear → Sigmoid → pdl1_prob_he   (scalar [0,1])
```

During inference, up to 16 tiles are randomly sampled per image for speed.

### Training Details
- **Backbone:** ResNet101 (timm library)
- **Hidden dim:** 256
- **Dropout:** 0.25
- **Loss:** Binary cross-entropy
- **Epochs trained:** 29
- **Best validation AUC:** **0.853**
- **Labels:** image-level PDL1_label (0/1) from `annotation_task_automated.xlsx`
- **Checkpoint:** `kandus_method/checkpoints/best_model_resnet101.pt` (169 MB)

The CNN output `pdl1_prob_he` is stored as a supplementary feature. It represents what the H&E morphology alone predicts about PD-L1 status, without seeing the actual PD-L1 stain.

---

## Step 4: Scoring (`scoring.py`)

### Tissue Composition
```
tumor_percent  = tc_area  / tissue_area
immune_percent = lc_area  / tissue_area
stroma_percent = st_area  / tissue_area
```

### CPS (Combined Positive Score)
FDA-approved formula for PD-L1 scoring in TNBC:

$$CPS = \frac{TC_{PDL1+} \times tc\_count + LC_{PDL1+} \times lc\_count}{tc\_count} \times 100$$

- Uses **cell counts** (number of nuclei from watershed), not pixel areas
- Clamped to [0, 100]

**Why cell counts instead of pixel areas?**  
Earlier versions used pixel area ratios. Since tumor cells are much larger than lymphocytes (~5× area), LC_PDL1 × lc_area could dominate the numerator even when few immune cells are positive, causing CPS to exceed 100 unrealistically. Cell counts (number of nuclei) are the biologically correct unit.

### Spatial Interaction Metrics

```
immune_density = lc_area / tissue_area

tumor_boundary_interaction:
  1. Dilate tc_mask by 15px → tc_dilated
  2. border = tc_dilated − tc_mask  (ring around tumor)
  3. tumor_boundary_interaction = (lc_mask ∩ border).sum() / lc_area
     (fraction of immune cells physically adjacent to tumor boundary)

spatial_modifier = 0.6 × min(immune_density × 5, 1.0)
                 + 0.4 × tumor_boundary_interaction
```

### CPS++ (Kandu's Method Extension)
Extends CPS with spatial tumor-immune interaction context:

$$CPS\text{++} = \alpha \times \frac{CPS}{100} + (1-\alpha) \times \text{spatial\_modifier}$$

Where `α = 0.7` (70% weighted toward clinical CPS score, 30% toward spatial context).

CPS++ is bounded [0, 1].

---

## Required Outputs Per Core (PRD Section 11)

| Output | Description | Source |
|--------|-------------|--------|
| `tumor_percent` | Fraction of tissue that is tumor | Tissue segmentation (H&E) |
| `immune_percent` | Fraction of tissue that is immune cells | Tissue segmentation (H&E) |
| `PDL1_percent` | Fraction of tissue DAB+ | Stain analysis (PD-L1 image) |
| `TC_PDL1` | Tumor cell PD-L1 positivity rate | Stain × segmentation overlap |
| `LC_PDL1` | Lymphocyte PD-L1 positivity rate | Stain × segmentation overlap |
| `ST_PDL1` | Stroma PD-L1 positivity rate | Stain × segmentation overlap |
| `CPS` | Combined Positive Score [0–100] | FDA formula, cell-count based |
| `CPS_plus_plus` | Spatially-extended CPS [0–1] | 0.7×CPS/100 + 0.3×spatial |

Additional outputs: `stroma_percent`, `immune_density`, `tumor_boundary_interaction`, `spatial_modifier`, `pdl1_prob_he` (CNN), `PD1_percent`, `PD1_LC`, `TIL_density`, `exhaustion_score`.

---

## Patient-Level Aggregation

After all 168 cores are processed, scores are aggregated by **mean pooling**:

```python
patient_score[feature] = mean([core[feature] for core in all_cores])
```

A clinical CPS category is assigned:
- `CPS < 1`  → **Negative** — Unlikely to respond to immunotherapy
- `1 ≤ CPS < 10` → **Low Positive**
- `CPS ≥ 10` → **High Positive** — Strong candidate for immunotherapy

---

## Actual Results for Patient 02-008

### Patient-Level Summary (mean across 168 cores)

| Feature | Value |
|---------|-------|
| `tumor_percent` | 0.302 (30.2%) |
| `immune_percent` | 0.134 (13.4%) |
| `stroma_percent` | 0.565 (56.5%) |
| `PDL1_percent` | 0.00088 (0.09%) |
| `TC_PDL1` | 0.00092 |
| `LC_PDL1` | 0.00049 |
| `ST_PDL1` | 0.00075 |
| `CPS` | **0.175** |
| `CPS_plus_plus` | **0.116** |
| `pdl1_prob_he` (CNN) | 0.145 |
| **CPS category** | **Negative (CPS < 1)** |

The patient is classified as **PD-L1 negative** — unlikely to respond to immunotherapy based on this TMA data.

---

## How to Run

### Full batch pipeline on all 168 cores:
```powershell
cd c:\Users\kandu\Downloads\TNBC
python -m kandus_method.run_pipeline `
    --mode batch `
    --data_raw ./data_raw `
    --checkpoint ./kandus_method/checkpoints/best_model_resnet101.pt `
    --output ./results/pipeline_results.json
```

### Single core:
```powershell
python -m kandus_method.run_pipeline `
    --mode single `
    --pdl1_img "./data_raw/02-008_PDL1(SP142)-Springbio_A12_v3_b3/02-008_PDL1(SP142)-Springbio_A12_v3_b3_001_r1c1.jpg.jpeg" `
    --he_img   "./data_raw/02-008_HE_A12_v2_s13/02-008_HE_A12_v2_s13_001_r1c1.jpg.jpeg" `
    --checkpoint ./kandus_method/checkpoints/best_model_resnet101.pt
```

### Train the CNN:
```powershell
python -m kandus_method.train_data_raw `
    --data_raw ./data_raw `
    --labels   ./data_raw/annotation_task_automated.xlsx `
    --backbone resnet101 `
    --epochs   30 `
    --output   ./kandus_method/checkpoints
```

---

## Dependencies

```
torch / torchvision
timm
opencv-python (cv2)
scikit-image (skimage)
numpy
pandas
openpyxl
albumentations
matplotlib
```

Install:
```powershell
python -m pip install torch torchvision timm opencv-python scikit-image numpy pandas openpyxl albumentations matplotlib
```

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Weakly supervised (image-level labels) | No pixel masks available in dataset |
| MIL with attention (not global pooling) | Allows identifying which tiles drive the prediction (interpretability) |
| Percentile-based nucleus classification | More robust than fixed-pixel thresholds across staining intensities |
| Cell counts for CPS (not pixel areas) | FDA definition uses cells, not areas; pixel ratios caused CPS > 100 |
| Two-stage DAB filter (RGB + OD) | RGB pre-filter eliminates hematoxylin misclassified as DAB |
| LAB-space tissue mask | More robust than HSV saturation threshold on IHC slides |
| α = 0.7 for CPS++ | Keeps CPS++ clinically anchored to standard CPS while adding spatial context |
