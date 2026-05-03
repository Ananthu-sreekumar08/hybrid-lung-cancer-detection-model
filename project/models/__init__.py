"""
models package
==============
Deep learning model definitions for the Hybrid 3D-CNN and Vision
Transformer Framework for Pulmonary Nodule Detection and Classification.

Both models are built using MONAI (Medical Open Network for AI) and
PyTorch as specified in Section 4.10.3 of the project report.

Modules
-------
unet3d  : 3D U-Net wrapping monai.networks.nets.UNet (Stage 1)
vit3d   : 3D Vision Transformer built with PyTorch nn.Transformer (Stage 2)
"""

from models.unet3d import UNet3D, build_unet3d
from models.vit3d  import ViT3D, PatchEmbed3D

__all__ = ["UNet3D", "build_unet3d", "ViT3D", "PatchEmbed3D"]
