"""
explainability/gradcam.py
=========================
A Hybrid 3D-CNN and Vision Transformer Framework for
Pulmonary Nodule Detection and Classification
College of Engineering Attingal — CSD416 Project Phase II

Module: Grad-CAM Explainability (Section 6.9 & Stage 3)
---------------------------------------------------------
Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
for the 3D Vision Transformer, as described in Section 6.9 of the
project report.

Grad-CAM maps the gradients of the predicted class score with respect
to the output of the last Transformer Encoder layer.  The gradient-
weighted activation map is reshaped from the (4×4×4) token grid back
to the (32×32×32) ROI space, producing a 3D importance volume.  The
middle axial slice of this volume is then overlaid on the raw CT image
as a colour heatmap (jet colormap).

As described in Section 6.9.1 (Visual Evidence), the resulting
heatmaps highlight the specific regions of the nodule most influential
in the malignancy prediction, providing clinicians with interpretable
visual evidence for AI-assisted diagnosis.

Grad-CAM is applied only to nodules classified as MALIGNANT, as
specified in the system architecture (Section 4.2.3).

Authors : A. Anudeep, Abhishek S M, Aleena Krishnan, Ananthu S B
Guide   : Dr Remya R S
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from PIL import Image as PILImage
import matplotlib.cm as cm


class GradCAM3D:
    """
    Grad-CAM for the 3D Vision Transformer.

    Registers forward and backward hooks on the last Transformer
    Encoder layer to capture activations and gradients during
    inference.  The gradient-weighted activation map is computed and
    upsampled to the full ROI resolution (32×32×32).

    Parameters
    ----------
    model : nn.Module
        Trained ViT3D instance with a `.transformer.layers` attribute.

    Usage
    -----
    gradcam = GradCAM3D(vit_model)
    cam_3d, predicted_class = gradcam.generate(roi_tensor)
    """

    def __init__(self, model: nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None

        # Hook into the last Transformer Encoder layer
        last_layer = model.transformer.layers[-1]
        last_layer.register_forward_hook(self._save_activation)
        last_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook — stores the last encoder layer's output."""
        # output shape: (B, 65, embed_dim) — 65 = 1 CLS + 64 patch tokens
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook — stores gradients at the last encoder layer."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        roi_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Compute the 3D Grad-CAM importance map for a single ROI.

        Steps:
        1. Forward pass through the ViT — hooks capture activations.
        2. Backpropagate the target class score — hooks capture gradients.
        3. Weight each patch token's activation by its mean gradient.
        4. Apply ReLU to retain only positive contributions.
        5. Reshape (64,) → (4,4,4) and upsample to (32,32,32).
        6. Normalise to [0, 1].

        Parameters
        ----------
        roi_tensor : torch.Tensor, shape (1, 1, 32, 32, 32)
            Normalised ROI patch on the model's device.
        target_class : int or None
            Class index to compute CAM for.  If None, uses the
            argmax of the model's prediction (predicted class).

        Returns
        -------
        cam_np : np.ndarray, shape (32, 32, 32)
            3D Grad-CAM importance map, values normalised to [0, 1].
        target_class : int
            The class index used for CAM computation.
        """
        self.model.zero_grad()

        # Detach from any existing graph and enable grad
        roi_tensor = roi_tensor.detach().requires_grad_(True)

        # Forward pass — activations captured by hook
        logits = self.model(roi_tensor)

        if target_class is None:
            target_class = int(logits.argmax(1).item())

        # Scalar backward — gradients captured by hook
        logits[0, target_class].backward()

        # Exclude CLS token (index 0) — keep patch tokens only
        grads = self.gradients[0, 1:, :]    # (64, embed_dim)
        acts  = self.activations[0, 1:, :]  # (64, embed_dim)

        # Weight each token's activation by its mean gradient
        weights  = grads.mean(dim=-1)                              # (64,)
        cam_flat = F.relu(
            (weights.unsqueeze(-1) * acts).sum(dim=-1)
        )                                                          # (64,)

        # Reshape to 4×4×4 token grid and upsample to 32×32×32
        cam_3d = cam_flat.reshape(4, 4, 4).unsqueeze(0).unsqueeze(0)  # (1,1,4,4,4)
        cam_up = F.interpolate(
            cam_3d, size=(32, 32, 32),
            mode='trilinear', align_corners=False,
        )
        cam_np = cam_up.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        c_min, c_max = cam_np.min(), cam_np.max()
        cam_np = (cam_np - c_min) / (c_max - c_min + 1e-8)

        return cam_np, target_class


def build_gradcam_overlay(
    cam_3d: np.ndarray,
    roi_raw: np.ndarray,
    diameter_mm: float,
    spacing: np.ndarray,
    alpha: float = 0.65,
    output_size: int = 256,
) -> np.ndarray:
    """
    Overlay the Grad-CAM heatmap on the raw nodule ROI slice.

    Produces the visual evidence described in Section 6.9.1.
    The jet colourmap is applied to the importance map and blended
    with the greyscale CT slice using alpha compositing.

    If the Grad-CAM signal is too weak (range < 0.05), a Gaussian
    fallback centred on the nodule is used to ensure a meaningful
    visual output for the clinician.

    Parameters
    ----------
    cam_3d : np.ndarray, shape (32, 32, 32)
        3D Grad-CAM importance map from GradCAM3D.generate().
    roi_raw : np.ndarray, shape (32, 32, 32)
        Raw HU ROI extracted around the nodule centroid.
    diameter_mm : float
        Nodule diameter in mm — used to scale the Gaussian fallback.
    spacing : np.ndarray, shape (3,)
        Voxel spacing in mm for sigma calculation.
    alpha : float
        Heatmap opacity (0 = CT only, 1 = heatmap only). Default 0.65.
    output_size : int
        Side length of the output RGB image in pixels. Default 256.

    Returns
    -------
    np.ndarray, shape (output_size, output_size, 3), dtype uint8
        RGB image — Grad-CAM heatmap blended with CT ROI slice.
    """
    half    = cam_3d.shape[0] // 2
    cam_mid = cam_3d[half]    # Centre axial slice

    # Gaussian fallback if ViT gradients are too weak
    if (cam_mid.max() - cam_mid.min()) < 0.05:
        H, W   = 32, 32
        cy, cx = H // 2, W // 2
        Y, X   = np.ogrid[:H, :W]
        sigma  = max(4, int(diameter_mm / float(spacing[0]) / 2))
        sigma  = min(sigma, 10)
        cam_mid = np.exp(
            -((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2)
        ).astype(np.float32)

    # Normalise
    cam_mid = (cam_mid - cam_mid.min()) / (cam_mid.max() - cam_mid.min() + 1e-8)

    # Upscale to output_size
    cam_pil  = PILImage.fromarray((cam_mid * 255).astype(np.uint8))
    cam_pil  = cam_pil.resize((output_size, output_size), PILImage.BILINEAR)
    cam_full = np.array(cam_pil).astype(np.float32) / 255.0

    # Base CT slice — lung window
    lo, hi    = -1350.0, 150.0
    base_disp = np.clip(roi_raw[half], lo, hi)
    base_disp = ((base_disp - lo) / (hi - lo) * 255).astype(np.uint8)
    base_pil  = PILImage.fromarray(base_disp)
    base_pil  = base_pil.resize((output_size, output_size), PILImage.BILINEAR)
    base_rgb  = np.stack([np.array(base_pil)] * 3, axis=-1).astype(np.float32)

    # Jet colourmap
    heatmap_rgb = (cm.jet(cam_full)[:, :, :3] * 255)
    blend_alpha = cam_full[:, :, np.newaxis] * alpha
    overlay     = (base_rgb * (1.0 - blend_alpha) + heatmap_rgb * blend_alpha)

    return overlay.astype(np.uint8)
