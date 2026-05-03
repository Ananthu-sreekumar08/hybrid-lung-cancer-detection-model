"""
inference package
=================
Inference pipeline for the Hybrid 3D-CNN and Vision Transformer
Framework for Pulmonary Nodule Detection and Classification.

Implements Algorithm 1 from the project report — end-to-end inference
from CT scan upload to malignancy prediction and Grad-CAM output.

Modules
-------
pipeline : run_detection, run_classification, run_inference
"""

from inference.pipeline import (
    run_detection,
    run_classification,
    run_inference,
    UNET_THRESHOLD,
    MIN_NODULE_VOXELS,
    MALIGNANCY_THRESHOLD,
)

__all__ = [
    "run_detection",
    "run_classification",
    "run_inference",
    "UNET_THRESHOLD",
    "MIN_NODULE_VOXELS",
    "MALIGNANCY_THRESHOLD",
]
