"""
ui package
==========
Gradio web interface for the Hybrid 3D-CNN and Vision Transformer
Framework for Pulmonary Nodule Detection and Classification.

Modules
-------
app : Gradio Blocks application — three-panel output UI (Figure 7.1)
"""

from ui.app import build_app, analyse_scan

__all__ = ["build_app", "analyse_scan"]
