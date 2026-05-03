"""
preprocessing/patch_extractor.py
=================================
A Hybrid 3D-CNN and Vision Transformer Framework for
Pulmonary Nodule Detection and Classification
College of Engineering Attingal — CSD416 Project Phase II

Module: ROI Localisation & Patch Extraction (Section 6.4)
----------------------------------------------------------
Implements Voxel Cube Generation and patch sampling strategies
for both the 3D U-Net detection stage and the ViT classification
stage.

Two patch sizes are used throughout the system:

  PATCH_SIZE = 64  →  64×64×64 voxel cubes fed to the 3D U-Net.
               Centred on or near each annotated nodule, with
               balanced positive (nodule present) and negative
               (no nodule) sampling.

  ROI_SIZE   = 32  →  32×32×32 voxel cubes fed to the 3D ViT.
               Tightly centred on the nodule centroid.  Serves
               as the high-resolution input for ViT (Section 6.4.2).

The nodule target mask is a binary sphere drawn at each nodule's
centroid with radius = diameter / (2 × spacing), used as the
ground-truth segmentation map for U-Net training.

Authors : A. Anudeep, Abhishek S M, Aleena Krishnan, Ananthu S B
Guide   : Dr Remya R S
"""

import numpy as np
import torch
from typing import List, Tuple, Dict

from preprocessing.resample import normalize_hu, world_to_voxel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PATCH_SIZE: int = 64     # U-Net input cube side length (voxels)
ROI_SIZE:   int = 32     # ViT input cube side length (voxels)


# ---------------------------------------------------------------------------
# Target mask generation (Section 6.3.1)
# ---------------------------------------------------------------------------

def build_nodule_mask(
    vol_shape: tuple,
    nodules: List[Dict],
    origin: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """
    Generate a binary segmentation mask with spherical regions at each
    annotated nodule centroid.

    Each sphere has radius = diameter_mm / (2 × spacing[0]) voxels.
    The mask serves as the ground-truth output for 3D U-Net training,
    establishing the supervised learning foundation described in
    Section 6.3.2.

    Parameters
    ----------
    vol_shape : tuple (D, H, W)
        Shape of the resampled CT volume.
    nodules : list of dict
        Each dict must contain keys: x, y, z (world mm), diameter (mm).
    origin : np.ndarray, shape (3,)
        World-space origin of the volume (x, y, z) in mm.
    spacing : np.ndarray, shape (3,)
        Voxel spacing of the volume (x, y, z) in mm.

    Returns
    -------
    mask : np.ndarray, shape (D, H, W), dtype float32
        Binary mask: 1.0 inside nodule spheres, 0.0 elsewhere.
    """
    mask = np.zeros(vol_shape, dtype=np.float32)

    for nod in nodules:
        vox = world_to_voxel(origin, spacing, [nod['x'], nod['y'], nod['z']])
        vz, vy, vx = int(vox[2]), int(vox[1]), int(vox[0])

        # Radius in voxels — minimum 2 to ensure a visible sphere
        r = max(2, int(nod['diameter'] / (2.0 * float(spacing[0]))))

        z1 = max(0, vz - r); z2 = min(vol_shape[0], vz + r + 1)
        y1 = max(0, vy - r); y2 = min(vol_shape[1], vy + r + 1)
        x1 = max(0, vx - r); x2 = min(vol_shape[2], vx + r + 1)

        for zi in range(z1, z2):
            for yi in range(y1, y2):
                for xi in range(x1, x2):
                    dist_sq = (zi-vz)**2 + (yi-vy)**2 + (xi-vx)**2
                    if dist_sq <= r**2:
                        mask[zi, yi, xi] = 1.0

    return mask


# ---------------------------------------------------------------------------
# Patch extraction for U-Net (Section 6.4.1)
# ---------------------------------------------------------------------------

def extract_patches(
    volume: np.ndarray,
    mask: np.ndarray,
    nodules: List[Dict],
    origin: np.ndarray,
    spacing: np.ndarray,
    patch_size: int = PATCH_SIZE,
    n_positive: int = 20,
    n_negative: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample 3D patches and corresponding mask patches for U-Net training.

    Positive patches are centred near each annotated nodule centroid
    with random jitter (±8 voxels) to encourage translation invariance.
    Negative patches are drawn from random locations in the volume that
    are sufficiently far from all annotated nodules.

    Parameters
    ----------
    volume : np.ndarray, shape (D, H, W)
        Raw HU CT volume (not yet normalised).
    mask : np.ndarray, shape (D, H, W)
        Binary nodule mask produced by build_nodule_mask.
    nodules : list of dict
        Nodule annotation dicts with keys x, y, z, diameter.
    origin : np.ndarray, shape (3,)
    spacing : np.ndarray, shape (3,)
    patch_size : int
        Side length of the cubic patch. Default 64.
    n_positive : int
        Number of nodule-centred patches to sample. Default 20.
    n_negative : int
        Number of background patches to sample. Default 20.

    Returns
    -------
    patches : torch.Tensor, shape (N, 1, P, P, P), float32
        Normalised intensity patches.
    masks   : torch.Tensor, shape (N, 1, P, P, P), float32
        Corresponding binary mask patches.
    """
    half       = patch_size // 2
    vol_norm   = normalize_hu(volume)
    patches: List[np.ndarray] = []
    mask_patches: List[np.ndarray] = []

    # ---- Positive patches (nodule-centred) --------------------------------
    pos_per_nodule = max(1, n_positive // max(len(nodules), 1))

    for nod in nodules:
        vox = world_to_voxel(origin, spacing, [nod['x'], nod['y'], nod['z']])
        vz, vy, vx = int(vox[2]), int(vox[1]), int(vox[0])

        for _ in range(pos_per_nodule):
            jz = int(np.clip(vz + np.random.randint(-8, 9), half, volume.shape[0]-half))
            jy = int(np.clip(vy + np.random.randint(-8, 9), half, volume.shape[1]-half))
            jx = int(np.clip(vx + np.random.randint(-8, 9), half, volume.shape[2]-half))

            p = vol_norm[jz-half:jz+half, jy-half:jy+half, jx-half:jx+half]
            m = mask[jz-half:jz+half,     jy-half:jy+half, jx-half:jx+half]

            if p.shape == (patch_size,) * 3:
                patches.append(p)
                mask_patches.append(m)

    # ---- Negative patches (random background) -----------------------------
    n_neg_needed = n_negative
    attempts     = 0

    while n_neg_needed > 0 and attempts < 300:
        attempts += 1
        rz = np.random.randint(half, volume.shape[0]-half)
        ry = np.random.randint(half, volume.shape[1]-half)
        rx = np.random.randint(half, volume.shape[2]-half)

        # Reject if too close to any annotated nodule
        too_close = False
        for nod in nodules:
            vox = world_to_voxel(origin, spacing, [nod['x'], nod['y'], nod['z']])
            if (abs(rz - vox[2]) < half and
                abs(ry - vox[1]) < half and
                abs(rx - vox[0]) < half):
                too_close = True
                break

        if too_close:
            continue

        p = vol_norm[rz-half:rz+half, ry-half:ry+half, rx-half:rx+half]
        m = mask[rz-half:rz+half,     ry-half:ry+half, rx-half:rx+half]

        if p.shape == (patch_size,) * 3:
            patches.append(p)
            mask_patches.append(m)
            n_neg_needed -= 1

    # ---- Stack and add channel dimension ----------------------------------
    X = torch.tensor(np.array(patches),      dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor(np.array(mask_patches), dtype=torch.float32).unsqueeze(1)
    return X, Y


# ---------------------------------------------------------------------------
# ROI extraction for ViT (Section 6.4.2)
# ---------------------------------------------------------------------------

def extract_roi(
    volume: np.ndarray,
    voxel_coord: np.ndarray,
    roi_size: int = ROI_SIZE,
    augment: bool = False,
) -> np.ndarray:
    """
    Extract a tightly centred 3D ROI patch for ViT classification.

    The extracted 32×32×32 voxel cube (ROI_SIZE) serves as the
    high-resolution input for the Vision Transformer as described in
    Section 6.4.2.  Optional augmentation applies random axis flips,
    intensity jitter, and Gaussian noise to improve generalisation.

    Parameters
    ----------
    volume : np.ndarray, shape (D, H, W)
        Raw HU CT volume.
    voxel_coord : np.ndarray, shape (3,)
        Nodule centroid in voxel space [vx, vy, vz].
    roi_size : int
        Side length of the cubic ROI. Default 32.
    augment : bool
        If True, apply random flips, intensity jitter, and noise.
        Should be True during training, False during inference.

    Returns
    -------
    roi : np.ndarray, shape (roi_size, roi_size, roi_size), float32
        HU-normalised ROI patch, values in [0, 1].
    """
    half = roi_size // 2
    vx, vy, vz = int(voxel_coord[0]), int(voxel_coord[1]), int(voxel_coord[2])

    # Clamp centroid so the cube stays within volume bounds
    vz_c = int(np.clip(vz, half, volume.shape[0] - half))
    vy_c = int(np.clip(vy, half, volume.shape[1] - half))
    vx_c = int(np.clip(vx, half, volume.shape[2] - half))

    roi = volume[
        vz_c - half : vz_c + half,
        vy_c - half : vy_c + half,
        vx_c - half : vx_c + half,
    ]

    # Safety check — zero-pad if crop is off-edge
    if roi.shape != (roi_size, roi_size, roi_size):
        roi = np.zeros((roi_size, roi_size, roi_size), dtype=np.float32)

    roi = normalize_hu(roi)

    if augment:
        # Random axis flips — preserve 3D geometry
        for axis in range(3):
            if np.random.rand() > 0.5:
                roi = np.flip(roi, axis=axis).copy()

        # Intensity jitter — simulate scanner variability
        roi = roi + np.random.uniform(-0.05, 0.05)

        # Gaussian noise — improve robustness
        roi = roi + np.random.normal(0.0, 0.02, roi.shape)

        roi = np.clip(roi, 0.0, 1.0)

    return roi.astype(np.float32)
