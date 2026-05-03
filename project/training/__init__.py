"""
training package
================
Training loop modules for the Hybrid 3D-CNN and Vision Transformer
Framework for Pulmonary Nodule Detection and Classification.

Uses MONAI's DiceCELoss and DiceMetric for U-Net training, and
PyTorch's weighted CrossEntropyLoss for ViT training, as specified
in Section 4.10.3 of the project report.

Modules
-------
train_unet : 3D U-Net training with MONAI DiceCELoss + DiceMetric
train_vit  : 3D ViT training with weighted cross-entropy loss
"""

from training.train_unet import train_unet
from training.train_vit  import train_vit, NoduleROIDataset

__all__ = [
    "train_unet",
    "train_vit",
    "NoduleROIDataset",
]
