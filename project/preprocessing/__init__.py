"""
preprocessing package
=====================
Data preprocessing modules for the Hybrid 3D-CNN and Vision
Transformer Framework for Pulmonary Nodule Detection and Classification.

Modules
-------
resample         : HU windowing, isotropic resampling, world-to-voxel
dataset_builder  : Ground-truth CSV construction, UID maps, train/val split
patch_extractor  : Nodule mask generation, U-Net patch sampling, ViT ROI extraction
"""

from preprocessing.resample import (
    load_and_resample,
    normalize_hu,
    world_to_voxel,
    display_lung_window,
    HU_MIN,
    HU_MAX,
)

from preprocessing.patch_extractor import (
    build_nodule_mask,
    extract_patches,
    extract_roi,
    PATCH_SIZE,
    ROI_SIZE,
)

from preprocessing.dataset_builder import (
    get_malignancy_label,
    build_ground_truth_csv,
    build_uid_maps,
    split_train_val,
    save_train_uids,
)

__all__ = [
    "load_and_resample",
    "normalize_hu",
    "world_to_voxel",
    "display_lung_window",
    "HU_MIN",
    "HU_MAX",
    "build_nodule_mask",
    "extract_patches",
    "extract_roi",
    "PATCH_SIZE",
    "ROI_SIZE",
    "get_malignancy_label",
    "build_ground_truth_csv",
    "build_uid_maps",
    "split_train_val",
    "save_train_uids",
]
