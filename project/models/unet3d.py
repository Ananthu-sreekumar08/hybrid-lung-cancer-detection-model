"""
models/unet3d.py
================
A Hybrid 3D-CNN and Vision Transformer Framework for
Pulmonary Nodule Detection and Classification
College of Engineering Attingal — CSD416 Project Phase II

Module: 3D U-Net Detection Model (Section 5.3 & 6.5)
------------------------------------------------------
Implements the 3D U-Net architecture used in Stage 1 of the
proposed system for volumetric lung nodule segmentation.

Built using MONAI (Medical Open Network for AI) as specified in
Section 4.10.3 of the project report:

    "MONAI: The medical imaging AI framework (e.g., version 1.0+).
     Provides the 3D UNet/ViT building blocks, specialised 3D
     transforms, medical data loaders, loss functions (DiceCELoss),
     metrics (DiceMetric), and inference tools
     (sliding_window_inference)."

MONAI's UNet provides the encoder–decoder structure with skip
connections as described in Section 5.3 and Figure 5.2:
    - 4-level encoder: 3D conv → BatchNorm → PReLU × 2 per level
    - MaxPool downsampling between levels
    - Transposed conv upsampling in decoder
    - Skip connections concatenate encoder and decoder feature maps
    - Final 1×1×1 conv → sigmoid probability map

Architecture (channels = (16, 32, 64, 128, 256)):
    Input  : (B, 1, 64, 64, 64)
    Output : (B, 1, 64, 64, 64)  — voxel-wise nodule probability map

Total parameters ≈ 4 M.

Authors : A. Anudeep, Abhishek S M, Aleena Krishnan, Ananthu S B
Guide   : Dr Remya R S
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm


def build_unet3d() -> nn.Module:
    """
    Construct the 3D U-Net using MONAI's UNet building blocks.

    Uses MONAI's UNet (monai.networks.nets.UNet) with:
        - spatial_dims = 3  (volumetric 3D input)
        - in_channels  = 1  (single-channel CT)
        - out_channels = 1  (binary nodule probability map)
        - channels     = (16, 32, 64, 128, 256) — 4-level hierarchy
        - strides      = (2, 2, 2, 2)           — MaxPool factor
        - num_res_units = 2                      — residual units per block
        - norm          = Norm.BATCH             — BatchNorm3d

    As specified in Section 4.10.3 of the project report, MONAI
    provides the 3D U-Net building blocks used in Stage 1 of the
    proposed hybrid framework.

    Returns
    -------
    nn.Module
        MONAI UNet3D instance ready for training.

    Notes
    -----
    The output is raw logits.  Apply torch.sigmoid() during inference
    or use MONAI's DiceCELoss (which handles sigmoid internally via
    sigmoid=True) during training.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    return model


class UNet3D(nn.Module):
    """
    Wrapper around MONAI's UNet for the pulmonary nodule detection stage.

    Wraps monai.networks.nets.UNet to provide a consistent interface
    with the rest of the project codebase, including sigmoid output
    activation and a count_parameters utility method.

    Used in Stage 1 (Feature Extraction & Nodule Detection) of the
    proposed hybrid framework as described in Section 4.2.1 and
    Figure 5.2 of the project report.

    Parameters
    ----------
    base_ch : int
        Base feature channel count for the first encoder level.
        Default 16.  Channels double at each level:
        (base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16).

    Input
    -----
    x : torch.Tensor, shape (B, 1, D, H, W)
        Normalised CT patch, values in [0, 1].

    Output
    ------
    torch.Tensor, shape (B, 1, D, H, W)
        Nodule probability map with values in (0, 1) after sigmoid.
    """

    def __init__(self, base_ch: int = 16):
        super().__init__()
        b = base_ch
        # MONAI UNet — provides encoder-decoder + skip connections
        # as described in Section 5.3 and Figure 5.2
        self.unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(b, b*2, b*4, b*8, b*16),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MONAI UNet returns logits — apply sigmoid for probability map
        return torch.sigmoid(self.unet(x))

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
