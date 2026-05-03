"""
preprocessing/resample.py
=========================
A Hybrid 3D-CNN and Vision Transformer Framework for
Pulmonary Nodule Detection and Classification
College of Engineering Attingal — CSD416 Project Phase II

Module: Data Preprocessing & Volumetric Standardization
---------------------------------------------------------
Implements HU Windowing, Contrast Normalization, and Isotropic
Resampling as described in Chapter 6.2 of the project report.

Raw CT scans acquired from LUNA16 are stored in MHD/RAW format.
Each scan may have different voxel spacings depending on the CT
scanner and acquisition protocol. This module standardises all
scans to a uniform isotropic resolution of 1 x 1 x 1 mm, and
normalises Hounsfield Unit (HU) values to the [0, 1] range by
clipping to the lung window [-1000, 400] HU.

Authors : A. Anudeep, Abhishek S M, Aleena Krishnan, Ananthu S B
Guide   : Dr Remya R S
"""

import numpy as np
import SimpleITK as sitk


# ---------------------------------------------------------------------------
# Constants — Lung Window (as described in Section 6.2.1)
# ---------------------------------------------------------------------------
HU_MIN: int = -1000   # Lower bound of lung HU window
HU_MAX: int = 400     # Upper bound of lung HU window


def load_and_resample(
    mhd_path: str,
    new_spacing: tuple = (1.0, 1.0, 1.0),
) -> tuple:
    """
    Load a CT scan from MHD/RAW format and resample it to isotropic
    voxel spacing.

    Isotropic resampling (Section 6.2.2) converts all voxel dimensions
    to a uniform 1 mm x 1 mm x 1 mm size.  This preserves the true
    physical proportions of anatomical structures and ensures consistent
    spatial interpretation across scans from different CT scanners.

    Parameters
    ----------
    mhd_path : str
        Absolute path to the .mhd file.  The paired .raw file must
        reside in the same directory with a matching filename.
    new_spacing : tuple of float, optional
        Target voxel spacing in mm for each axis (z, y, x).
        Default is (1.0, 1.0, 1.0) — isotropic 1 mm.

    Returns
    -------
    volume : np.ndarray, shape (D, H, W), dtype float32
        Resampled CT volume as a NumPy array.
    origin : np.ndarray, shape (3,), dtype float64
        World-space origin of the resampled image in mm.
    spacing : np.ndarray, shape (3,), dtype float64
        Voxel spacing of the resampled image in mm (always new_spacing).

    Raises
    ------
    RuntimeError
        If SimpleITK cannot read the MHD file or locate the paired RAW.

    Notes
    -----
    SimpleITK reads the ElementDataFile entry inside the .mhd header to
    locate the binary .raw file.  Both files must be in the same folder.
    Linear interpolation (sitk.sitkLinear) is used for resampling.
    Out-of-bounds voxels are filled with -1000 HU (air).
    """
    img = sitk.ReadImage(mhd_path)

    origin  = np.array(img.GetOrigin())      # (x, y, z) in mm
    spacing = np.array(img.GetSpacing())     # (x, y, z) in mm
    size    = np.array(img.GetSize())        # (x, y, z) in voxels

    # Compute output size that preserves physical extent
    new_size = np.round(size * spacing / np.array(new_spacing)).astype(int)

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetSize(new_size.tolist())
    resample_filter.SetOutputOrigin(img.GetOrigin())
    resample_filter.SetOutputDirection(img.GetDirection())
    resample_filter.SetInterpolator(sitk.sitkLinear)
    resample_filter.SetDefaultPixelValue(HU_MIN)

    resampled = resample_filter.Execute(img)

    volume = sitk.GetArrayFromImage(resampled).astype(np.float32)
    # SimpleITK returns (z, y, x) — matches (D, H, W) convention used
    # throughout this project.

    return (
        volume,
        np.array(resampled.GetOrigin()),
        np.array(new_spacing),
    )


def normalize_hu(volume: np.ndarray) -> np.ndarray:
    """
    Apply HU Windowing and Contrast Normalization (Section 6.2.1).

    Raw CT intensities are clipped to the lung window [HU_MIN, HU_MAX]
    and linearly scaled to [0, 1].  Non-lung structures such as bone
    (> 400 HU) and air outside the body (< -1000 HU) are suppressed,
    allowing the model to focus on lung parenchyma and soft-tissue
    nodules.

    Parameters
    ----------
    volume : np.ndarray
        CT volume with raw HU values.  Any shape is accepted.

    Returns
    -------
    np.ndarray
        Normalised volume with values in [0, 1], same shape and
        float32 dtype.
    """
    volume = np.clip(volume, HU_MIN, HU_MAX)
    volume = (volume - HU_MIN) / float(HU_MAX - HU_MIN)
    return volume.astype(np.float32)


def world_to_voxel(
    origin: np.ndarray,
    spacing: np.ndarray,
    coord_world: list,
) -> np.ndarray:
    """
    Convert a world-space coordinate (mm) to voxel indices.

    Used during Ground Truth Mapping (Section 6.3) to locate annotated
    nodule centroids within the resampled volume.

    Parameters
    ----------
    origin : np.ndarray, shape (3,)
        World-space origin of the CT volume (x, y, z) in mm.
    spacing : np.ndarray, shape (3,)
        Voxel spacing of the CT volume (x, y, z) in mm.
    coord_world : list or array-like of length 3
        World coordinates [x, y, z] in mm (LUNA16 annotation format).

    Returns
    -------
    np.ndarray, shape (3,), dtype int
        Voxel indices [vx, vy, vz] corresponding to the world coordinate.
    """
    return np.round(
        (np.array(coord_world) - origin) / spacing
    ).astype(int)


def display_lung_window(volume: np.ndarray, slice_idx: int = None) -> np.ndarray:
    """
    Prepare a CT slice for display using the standard lung window.

    Applies window centre -600 HU, window width 1500 HU (common
    radiological lung viewing setting) and returns an 8-bit RGB array
    suitable for rendering in Gradio or matplotlib.

    Parameters
    ----------
    volume : np.ndarray, shape (D, H, W)
        Raw CT volume in HU.
    slice_idx : int, optional
        Axial slice index to extract.  Defaults to the middle slice.

    Returns
    -------
    np.ndarray, shape (H, W, 3), dtype uint8
        RGB representation of the selected axial slice.
    """
    if slice_idx is None:
        slice_idx = volume.shape[0] // 2

    wc, ww   = -600, 1500
    lo, hi   = wc - ww / 2, wc + ww / 2

    sl       = np.clip(volume[slice_idx], lo, hi)
    sl       = ((sl - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.stack([sl, sl, sl], axis=-1)
