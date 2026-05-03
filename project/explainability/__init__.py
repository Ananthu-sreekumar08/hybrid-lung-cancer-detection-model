"""
explainability package
======================
Explainable AI module for the Hybrid 3D-CNN and Vision Transformer
Framework for Pulmonary Nodule Detection and Classification.

Implements Grad-CAM visualisation (Section 6.9) applied to the
3D Vision Transformer's last Transformer Encoder layer output.

Modules
-------
gradcam : GradCAM3D class and overlay builder
"""

from explainability.gradcam import GradCAM3D, build_gradcam_overlay

__all__ = ["GradCAM3D", "build_gradcam_overlay"]
