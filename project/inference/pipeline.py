"""
inference/pipeline.py
=====================
A Hybrid 3D-CNN and Vision Transformer Framework for
Pulmonary Nodule Detection and Classification
College of Engineering Attingal — CSD416 Project Phase II

Module: Inference Pipeline (Algorithm 1 — Section 6, Chapter 7)
----------------------------------------------------------------
Implements the complete end-to-end inference pipeline as described
in Algorithm 1 of the project report (Chapter 6, page 39).

Pipeline stages:
    1. PreprocessScan       — load MHD/RAW, resample to 1mm isotropic,
                              normalise HU values
    2. RunDetection         — MONAI sliding_window_inference with 3D U-Net →
                              voxel-wise probability mask
    3. GenerateCandidates   — threshold + connected components →
                              nodule centroid coordinates
    4. ExtractPatches       — 32×32×32 ROI crops centred on candidates
    5. RunClassification    — 3D ViT forward pass → malignancy scores
    6. Grad-CAM             — applied only when score >= threshold

Output:
    - If no candidates: "No nodule detected"
    - If BENIGN: CT overview + ROI crop (no Grad-CAM)
    - If MALIGNANT: CT overview + ROI crop + Grad-CAM heatmap

Authors : A. Anudeep, Abhishek S M, Aleena Krishnan, Ananthu S B
Guide   : Dr Remya R S
"""

import os
import pickle
import json
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import label as scipy_label
from PIL import Image as PILImage

from preprocessing.resample import (
    load_and_resample,
    normalize_hu,
    world_to_voxel,
    display_lung_window,
)
from preprocessing.patch_extractor import extract_roi, ROI_SIZE
from explainability.gradcam import GradCAM3D, build_gradcam_overlay


# ---------------------------------------------------------------------------
# Inference configuration
# ---------------------------------------------------------------------------
UNET_THRESHOLD     : float = 0.5    # Heatmap threshold for nodule detection
MIN_NODULE_VOXELS  : int   = 10     # Minimum connected component size
MALIGNANCY_THRESHOLD: float = 0.5   # ViT score above which Grad-CAM is shown


# ---------------------------------------------------------------------------
# Sliding-window U-Net detection (Algorithm 1: RunDetection)
# ---------------------------------------------------------------------------

def run_detection(
    vol_norm: np.ndarray,
    unet: nn.Module,
    device: torch.device,
    patch_size: int = 64,
    stride: int = 32,
    threshold: float = UNET_THRESHOLD,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Perform sliding-window inference with the 3D U-Net using MONAI.

    Uses MONAI's sliding_window_inference utility as specified in
    Section 4.10.3 of the project report:

        "MONAI provides ... inference tools (sliding_window_inference)."

    sliding_window_inference slides a 64×64×64 window over the volume
    with the specified overlap, averages overlapping predictions using
    a Gaussian importance map, then applies a threshold to binarise
    the heatmap.  Connected components are extracted as candidate
    nodule locations (Algorithm 1: GenerateCandidates).

    Parameters
    ----------
    vol_norm : np.ndarray, shape (D, H, W)
        HU-normalised CT volume, values in [0, 1].
    unet : nn.Module
        Trained UNet3D model in eval() mode.
    device : torch.device
    patch_size : int
        U-Net input cube side length. Default 64.
    stride : int
        Sliding window step size. Default 32 (50% overlap).
    threshold : float
        Probability threshold for binarising the heatmap. Default 0.5.

    Returns
    -------
    nodules : list of dict
        Each dict contains:
            centre     (np.ndarray [vz, vy, vx]),
            bbox       (tuple z0,y0,x0,z1,y1,x1),
            confidence (float).
        Sorted by descending confidence.
    heatmap : np.ndarray, shape (D, H, W)
        Averaged probability heatmap from MONAI sliding_window_inference.
    """
    from monai.inferers import sliding_window_inference

    # Overlap ratio from stride: overlap = 1 - stride/patch_size
    overlap = 1.0 - (stride / patch_size)

    vol_t = torch.tensor(vol_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    # (1, 1, D, H, W)

    unet.eval()
    with torch.no_grad():
        # MONAI sliding_window_inference handles tiling, overlap averaging
        # roi_size matches the 64³ patch size used during training
        pred = sliding_window_inference(
            inputs=vol_t,
            roi_size=(patch_size, patch_size, patch_size),
            sw_batch_size=4,
            predictor=unet,
            overlap=overlap,
            mode="gaussian",   # Gaussian weighting for smooth boundary blending
        )

    heatmap = pred.squeeze().cpu().numpy()   # (D, H, W)

    print(f"  Sliding window complete | heatmap max: {heatmap.max():.4f} | threshold: {threshold}")

    # Connected components (Algorithm 1: GenerateCandidates)
    binary          = (heatmap > threshold).astype(np.int32)
    labeled, n_comp = scipy_label(binary)

    nodules = []
    for comp_id in range(1, n_comp + 1):
        comp_mask = labeled == comp_id
        if comp_mask.sum() < MIN_NODULE_VOXELS:
            continue
        coords     = np.argwhere(comp_mask)
        centre     = coords.mean(axis=0).astype(int)
        confidence = float(heatmap[comp_mask].mean())
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0)
        nodules.append({
            'centre':     centre,
            'bbox':       (z0, y0, x0, z1, y1, x1),
            'confidence': confidence,
        })

    nodules.sort(key=lambda n: n['confidence'], reverse=True)
    return nodules, heatmap


# ---------------------------------------------------------------------------
# ViT classification (Algorithm 1: RunClassification)
# ---------------------------------------------------------------------------

def run_classification(
    vol_norm: np.ndarray,
    nodule_centre: np.ndarray,
    vit: nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    """
    Extract a 32×32×32 ROI and run the ViT malignancy classifier.

    Parameters
    ----------
    vol_norm : np.ndarray, shape (D, H, W)
        HU-normalised CT volume.
    nodule_centre : np.ndarray, shape (3,) [vz, vy, vx]
        Candidate nodule centroid in voxel space.
    vit : nn.Module
        Trained ViT3D model in eval() mode.
    device : torch.device

    Returns
    -------
    roi_t : torch.Tensor, shape (1, 1, 32, 32, 32)
        ROI tensor on device (needed for Grad-CAM).
    prob : np.ndarray, shape (2,)
        Softmax probabilities [P(Benign), P(Malignant)].
    logits : torch.Tensor, shape (1, 2)
        Raw logits.
    """
    vz, vy, vx = int(nodule_centre[0]), int(nodule_centre[1]), int(nodule_centre[2])
    half = ROI_SIZE // 2

    vz_c = int(np.clip(vz, half, vol_norm.shape[0] - half))
    vy_c = int(np.clip(vy, half, vol_norm.shape[1] - half))
    vx_c = int(np.clip(vx, half, vol_norm.shape[2] - half))

    roi = extract_roi(
        # Convert normalised back to raw for extract_roi which renormalises
        vol_norm * 1400.0 - 1000.0,
        np.array([vx_c, vy_c, vz_c]),
        roi_size=ROI_SIZE,
        augment=False,
    )

    roi_t = torch.tensor(roi).unsqueeze(0).unsqueeze(0).float().to(device)

    vit.eval()
    with torch.no_grad():
        logits = vit(roi_t)
        prob   = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    return roi_t, prob, logits


# ---------------------------------------------------------------------------
# Full pipeline function
# ---------------------------------------------------------------------------

def run_inference(
    mhd_path: str,
    unet: nn.Module,
    vit: nn.Module,
    gradcam: GradCAM3D,
    device: torch.device,
    annotation_df=None,
) -> Dict:
    """
    End-to-end inference on a single CT scan.

    Implements the complete pipeline from Algorithm 1 of the project
    report.  Returns a result dictionary used by the Gradio UI.

    Parameters
    ----------
    mhd_path : str
        Path to the .mhd file.  Paired .raw must be in same directory.
    unet : nn.Module
        Trained UNet3D.
    vit : nn.Module
        Trained ViT3D.
    gradcam : GradCAM3D
        Initialised GradCAM3D instance.
    device : torch.device
    annotation_df : pd.DataFrame or None
        If provided, uses ground-truth nodule coordinates for the ROI
        (bypasses sliding-window for demo on training scans).

    Returns
    -------
    dict with keys:
        ct_overview  : np.ndarray (H, W, 3) — axial CT slice at nodule level
        roi_image    : np.ndarray (256, 256, 3) or None — extracted nodule patch
        gradcam_image: np.ndarray (256, 256, 3) or None — Grad-CAM overlay
        label        : str — 'MALIGNANT', 'BENIGN', or 'NO NODULE DETECTED'
        mal_score    : float — malignancy probability in [0, 1]
        nodule_count : int
    """
    print(f"\nInference: {os.path.basename(mhd_path)}")

    # ---- Stage 0: Load and preprocess ------------------------------------
    try:
        vol_raw, origin, spacing = load_and_resample(mhd_path)
    except Exception as e:
        return {
            'ct_overview': None, 'roi_image': None, 'gradcam_image': None,
            'label': f'Load error: {e}', 'mal_score': 0.0, 'nodule_count': 0,
        }

    vol_norm = normalize_hu(vol_raw)
    print(f"  Volume: {vol_raw.shape}")

    # ---- Stage 1: Detection ---------------------------------------------
    print("  Stage 1: U-Net detection...")
    nodules, heatmap = run_detection(vol_norm, unet, device)
    print(f"  Candidates: {len(nodules)}")

    if len(nodules) == 0:
        ct_rgb = display_lung_window(vol_raw)
        return {
            'ct_overview': ct_rgb, 'roi_image': None, 'gradcam_image': None,
            'label': 'NO NODULE DETECTED', 'mal_score': 0.0, 'nodule_count': 0,
        }

    # Use highest-confidence candidate
    best        = nodules[0]
    vz, vy, vx  = best['centre']
    ct_overview = display_lung_window(vol_raw, slice_idx=int(vz))

    # ---- Stage 2: Classification ----------------------------------------
    print("  Stage 2: ViT classification...")
    roi_t, prob, logits = run_classification(
        vol_norm, best['centre'], vit, device
    )
    vit_mal_prob = float(prob[1])
    print(f"  ViT: benign={prob[0]:.3f} malignant={prob[1]:.3f}")

    # ---- ROI display image -----------------------------------------------
    half_r  = ROI_SIZE // 2
    vz_c    = int(np.clip(vz, half_r, vol_raw.shape[0] - half_r))
    vy_c    = int(np.clip(vy, half_r, vol_raw.shape[1] - half_r))
    vx_c    = int(np.clip(vx, half_r, vol_raw.shape[2] - half_r))

    roi_raw = vol_raw[
        vz_c-half_r:vz_c+half_r,
        vy_c-half_r:vy_c+half_r,
        vx_c-half_r:vx_c+half_r,
    ]
    if roi_raw.shape != (ROI_SIZE,) * 3:
        roi_raw = np.zeros((ROI_SIZE,) * 3, dtype=np.float32)

    roi_mid  = roi_raw[half_r]
    lo, hi   = -1350.0, 150.0
    roi_disp = np.clip(roi_mid, lo, hi)
    roi_disp = ((roi_disp - lo) / (hi - lo) * 255).astype(np.uint8)
    roi_img  = np.array(
        PILImage.fromarray(np.stack([roi_disp]*3, axis=-1).astype(np.uint8))
        .resize((256, 256), PILImage.NEAREST)
    )

    # ---- Stage 3: Grad-CAM (malignant only) -----------------------------
    if vit_mal_prob >= MALIGNANCY_THRESHOLD:
        label     = "MALIGNANT"
        mal_score = vit_mal_prob
        print("  Stage 3: Grad-CAM...")
        try:
            cam_3d, _ = gradcam.generate(roi_t, target_class=1)
            gradcam_img = build_gradcam_overlay(cam_3d, roi_raw, 10.0, spacing)
        except Exception as e:
            print(f"  Grad-CAM failed: {e} — using fallback")
            gradcam_img = build_gradcam_overlay(
                np.zeros((ROI_SIZE,)*3), roi_raw, 10.0, spacing
            )
    else:
        label       = "BENIGN"
        mal_score   = vit_mal_prob
        gradcam_img = None
        print("  Benign — Grad-CAM skipped")

    return {
        'ct_overview':   ct_overview,
        'roi_image':     roi_img,
        'gradcam_image': gradcam_img,
        'label':         label,
        'mal_score':     mal_score,
        'nodule_count':  len(nodules),
    }
