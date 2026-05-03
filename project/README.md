# A Hybrid 3D-CNN and Vision Transformer Framework for Pulmonary Nodule Detection and Classification

**CSD416 Project Phase II**  
Department of Computer Science and Engineering  
College of Engineering Attingal, Thiruvananthapuram — PIN 695 101  
APJ Abdul Kalam Technological University — April 2026

**Authors:** A. Anudeep (CEA22CS001), Abhishek S M (CEA22CS007), Aleena Krishnan (CEA22CS011), Ananthu S B (CEA22CS016)  
**Guide:** Dr Remya R S

---

## Project Overview

This project implements a two-stage AI-based computer-aided detection and diagnosis (CADe/CADx) system for pulmonary nodule analysis from Low-Dose CT (LDCT) scans.

- **Stage 1** — 3D U-Net for volumetric nodule detection (segmentation)
- **Stage 2** — 3D Vision Transformer (ViT) for malignancy classification
- **Explainability** — Grad-CAM heatmaps on the ViT for MALIGNANT predictions
- **Interface** — Gradio web UI with three-panel output

---

## Reported Performance (Table 7.1)

| Metric      | Score   | Clinical Significance                                 |
|-------------|---------|-------------------------------------------------------|
| Accuracy    | 89.12%  | Overall correctness of the diagnostic system          |
| Recall      | 90.00%  | Ability to correctly identify patients with cancer    |
| Precision   | 88.24%  | Probability that a Malignant prediction is correct    |
| Specificity | 88.00%  | Ability to correctly identify Benign cases            |

---

## Architecture

```
CT Scan (.mhd/.raw)
        ↓
  Preprocessing
  (resample to 1mm³, HU normalisation)
        ↓
 ┌──────────────────┐
 │  Stage 1         │
 │  3D U-Net        │  → Nodule probability mask + bounding boxes
 │  Detection       │
 └──────────────────┘
        ↓
   Nodules found?
   ↙           ↘
  NO            YES
  ↓              ↓
"No nodule    Crop 32³ ROI
 detected"    around nodule
                  ↓
          ┌──────────────┐
          │  Stage 2     │
          │  ViT         │  → BENIGN / MALIGNANT + score
          │  Classifier  │
          └──────────────┘
                  ↓
            MALIGNANT?
            ↙        ↘
          NO           YES
          ↓              ↓
      No Grad-CAM    Grad-CAM heatmap
                     on raw CT slice
                  ↓
            Gradio UI output
```

---

## Repository Structure

```
project/
├── preprocessing/
│   ├── __init__.py
│   ├── resample.py          HU windowing, isotropic resampling, world-to-voxel
│   ├── patch_extractor.py   Nodule mask generation, U-Net patch sampling, ViT ROI
│   └── dataset_builder.py   LUNA16 + LIDC-IDRI merge, UID maps, train/val split
│
├── models/
│   ├── __init__.py
│   ├── unet3d.py            3D U-Net (4-level encoder-decoder, ~4M params)
│   └── vit3d.py             3D Vision Transformer (~4.89M params)
│
├── training/
│   ├── __init__.py
│   ├── train_unet.py        Dice + Focal loss, Adam, CosineAnnealingLR
│   └── train_vit.py         Weighted CrossEntropy, AdamW, CosineAnnealingLR
│
├── explainability/
│   ├── __init__.py
│   └── gradcam.py           GradCAM3D + overlay builder
│
├── inference/
│   ├── __init__.py
│   └── pipeline.py          Sliding-window detection, classification, full pipeline
│
├── ui/
│   ├── __init__.py
│   └── app.py               Gradio Blocks UI (three-panel output)
│
└── utils/
    ├── __init__.py
    └── metrics.py           Accuracy, Recall, Precision, Specificity, F1
```

---

## Datasets

| Dataset   | Purpose                          | Source                          |
|-----------|----------------------------------|---------------------------------|
| LUNA16    | Nodule coordinates + raw scans   | luna16.grand-challenge.org      |
| LIDC-IDRI | Malignancy scores (1–5)          | wiki.cancerimagingarchive.net   |

Merge key: `seriesuid` (shared between both datasets)  
Label convention: Score 1–2 → Benign (0) | Score 3 → Excluded | Score 4–5 → Malignant (1)

---

## Requirements

```
Python       >= 3.8
PyTorch      >= 1.10  (with CUDA)
MONAI        >= 1.0   (3D UNet, DiceCELoss, DiceMetric, sliding_window_inference)
SimpleITK    >= 2.0
gradio       >= 3.50
numpy
scipy
pandas
matplotlib
Pillow
```

Install:
```bash
pip install monai SimpleITK gradio scipy pandas matplotlib Pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Running the UI

```bash
python ui/app.py
```

Set the weights directory:
```bash
export STAGE1_DIR=/path/to/weights
python ui/app.py
```

Upload both `.mhd` and `.raw` files from the same CT scan.

---

## Hardware Requirements (Section 4.9)

| Component | Minimum              | Recommended               |
|-----------|----------------------|---------------------------|
| GPU       | NVIDIA T4, 8GB VRAM  | NVIDIA P100/V100, 16GB    |
| RAM       | 16 GB                | 32 GB                     |
| Storage   | 256 GB SSD           | 1 TB SSD (for training)   |

---

*College of Engineering Attingal — Department of Computer Science and Engineering*
