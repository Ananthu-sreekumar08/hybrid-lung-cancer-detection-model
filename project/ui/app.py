"""
ui/app.py
=========
A Hybrid 3D-CNN and Vision Transformer Framework for
Pulmonary Nodule Detection and Classification
College of Engineering Attingal — CSD416 Project Phase II

Module: Gradio User Interface (Section 4.4.1, 4.5.1, Chapter 7)
-----------------------------------------------------------------
Implements the web-based user interface described in Section 4.5.1
of the project report.

The interface is built using the Gradio framework and presents
three visual output panels as described in Chapter 7 (Figure 7.1):

    Panel 1 — CT Scan Overview (Standardization)
        Axial CT slice at the nodule level, HU-windowed for lung
        display.  Shows the nodule location in full anatomical context.

    Panel 2 — Extracted Nodule (ROI / Candidate Nodule)
        32×32×32 voxel patch centred on the detected nodule, displayed
        as a 2D slice.  Corresponds to Figure 7.3 of the report.

    Panel 3 — Grad-CAM Heatmap (Explainability)
        Jet-colourmap Grad-CAM overlay on the ROI slice, shown only
        for MALIGNANT predictions.  Corresponds to Figure 7.4 / 6.9.

Result panel shows:
    Prediction      : MALIGNANT / BENIGN / NO NODULE DETECTED
    Malignancy Score: Probability in % (capped at 89% per Section 4.2.2)

Authors : A. Anudeep, Abhishek S M, Aleena Krishnan, Ananthu S B
Guide   : Dr Remya R S
"""

import os
import pickle
import json
import glob
import numpy as np
import torch
import gradio as gr

from models.unet3d        import UNet3D
from models.vit3d         import ViT3D
from explainability.gradcam import GradCAM3D, build_gradcam_overlay
from preprocessing.resample import (
    load_and_resample, normalize_hu, world_to_voxel, display_lung_window
)
from preprocessing.patch_extractor import extract_roi, ROI_SIZE
from inference.pipeline import (
    run_detection, run_classification, MALIGNANCY_THRESHOLD
)


# ---------------------------------------------------------------------------
# Configuration — update STAGE1 to match your weights directory
# ---------------------------------------------------------------------------
STAGE1_DIR : str = os.getenv('STAGE1_DIR', './weights')
DEVICE     : torch.device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def load_models(stage1_dir: str, device: torch.device):
    """
    Load trained U-Net and ViT weights from disk.

    Parameters
    ----------
    stage1_dir : str
        Directory containing unet_best.pth, vit_best.pth,
        preprocessing_params.pkl, and train_uids.json.
    device : torch.device

    Returns
    -------
    unet    : UNet3D
    vit     : ViT3D
    gradcam : GradCAM3D
    pp      : dict  — preprocessing parameters
    train_uids : set — UIDs used during training
    """
    unet = UNet3D(base_ch=16).to(device)
    vit  = ViT3D().to(device)

    unet.load_state_dict(
        torch.load(
            os.path.join(stage1_dir, 'unet_best.pth'),
            map_location=device,
        )
    )
    vit.load_state_dict(
        torch.load(
            os.path.join(stage1_dir, 'vit_best.pth'),
            map_location=device,
        )
    )
    unet.eval()
    vit.eval()

    with open(os.path.join(stage1_dir, 'preprocessing_params.pkl'), 'rb') as f:
        pp = pickle.load(f)

    with open(os.path.join(stage1_dir, 'train_uids.json')) as f:
        train_uids = set(json.load(f))

    gradcam = GradCAM3D(vit)

    print(f"Models loaded from {stage1_dir}")
    print(f"U-Net params : {unet.count_parameters()/1e6:.2f}M")
    print(f"ViT params   : {vit.count_parameters()/1e6:.2f}M")
    return unet, vit, gradcam, pp, train_uids


# ---------------------------------------------------------------------------
# Analysis function
# ---------------------------------------------------------------------------

def analyse_scan(
    mhd_file,
    raw_file,
    unet,
    vit,
    gradcam,
    pp,
):
    """
    Main analysis callback for the Gradio interface.

    Accepts uploaded .mhd and .raw files, copies them to a temporary
    directory, runs the full inference pipeline, and returns the three
    display images and the result text.

    Parameters
    ----------
    mhd_file : Gradio file object (.mhd)
    raw_file : Gradio file object (.raw)
    unet, vit, gradcam, pp : loaded model objects

    Returns
    -------
    ct_overview   : np.ndarray or None
    roi_image     : np.ndarray or None
    gradcam_image : np.ndarray or None
    result_text   : str
    """
    import shutil, tempfile
    from PIL import Image as PILImage
    import matplotlib.cm as cm

    if mhd_file is None or raw_file is None:
        return None, None, None, "Please upload both .mhd and .raw files."

    # Copy both files into the same temporary directory
    tmp_dir = tempfile.mkdtemp()
    mhd_name = os.path.basename(mhd_file.name)

    # Read expected raw filename from MHD header
    with open(mhd_file.name, 'r') as f:
        contents = f.read()
    expected_raw = None
    for line in contents.splitlines():
        if 'ElementDataFile' in line:
            expected_raw = line.split('=')[-1].strip()
            break

    mhd_dst = os.path.join(tmp_dir, mhd_name)
    raw_dst = os.path.join(tmp_dir, expected_raw or os.path.basename(raw_file.name))
    shutil.copy(mhd_file.name, mhd_dst)
    shutil.copy(raw_file.name, raw_dst)

    try:
        vol_raw, origin, spacing = load_and_resample(mhd_dst)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, None, None, f"Failed to load scan: {e}"

    vol_norm = normalize_hu(vol_raw)

    # ---- U-Net detection ------------------------------------------------
    nodules, _ = run_detection(vol_norm, unet, DEVICE)

    if len(nodules) == 0:
        ct_rgb = display_lung_window(vol_raw)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        result = "Prediction      NO NODULE DETECTED\nMalignancy Score  0.0%"
        return ct_rgb, None, None, result

    # ---- Highest-confidence candidate -----------------------------------
    best       = nodules[0]
    vz, vy, vx = best['centre']
    ct_overview = display_lung_window(vol_raw, slice_idx=int(vz))

    # ---- ROI display image ----------------------------------------------
    half_r = ROI_SIZE // 2
    vz_c = int(np.clip(vz, half_r, vol_raw.shape[0]-half_r))
    vy_c = int(np.clip(vy, half_r, vol_raw.shape[1]-half_r))
    vx_c = int(np.clip(vx, half_r, vol_raw.shape[2]-half_r))

    roi_raw = vol_raw[vz_c-half_r:vz_c+half_r,
                      vy_c-half_r:vy_c+half_r,
                      vx_c-half_r:vx_c+half_r]
    if roi_raw.shape != (ROI_SIZE,)*3:
        roi_raw = np.zeros((ROI_SIZE,)*3, dtype=np.float32)

    lo, hi   = -1350.0, 150.0
    roi_mid  = roi_raw[half_r]
    roi_disp = np.clip(roi_mid, lo, hi)
    roi_disp = ((roi_disp - lo) / (hi - lo) * 255).astype(np.uint8)
    roi_img  = np.array(
        PILImage.fromarray(np.stack([roi_disp]*3, axis=-1).astype(np.uint8))
        .resize((256, 256), PILImage.NEAREST)
    )

    # ---- ViT classification ---------------------------------------------
    roi_t, prob, logits = run_classification(
        vol_norm, best['centre'], vit, DEVICE
    )
    vit_mal_prob = float(prob[1])

    # ---- Classification and Malignancy Score ----------------------------
    if vit_mal_prob >= MALIGNANCY_THRESHOLD:
        label     = "MALIGNANT"
        mal_score = min(vit_mal_prob, 0.89)   # Capped at 89% per Section 4.2.2
        cam_3d, _ = gradcam.generate(roi_t, target_class=1)
        gradcam_img = build_gradcam_overlay(cam_3d, roi_raw, 10.0, spacing)
    else:
        label       = "BENIGN"
        mal_score   = vit_mal_prob
        gradcam_img = None

    result = (
        f"Prediction      {label}\n"
        f"Malignancy Score  {mal_score*100:.1f}%"
    )

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return ct_overview, roi_img, gradcam_img, result


# ---------------------------------------------------------------------------
# Gradio application
# ---------------------------------------------------------------------------

def build_app(stage1_dir: str = STAGE1_DIR) -> gr.Blocks:
    """
    Construct and return the Gradio Blocks application.

    The UI layout matches Figure 7.1 of the project report:
    - Upload row with .mhd and .raw file inputs and Analyse button
    - Three-panel image output row
    - Analysis result text panel

    Parameters
    ----------
    stage1_dir : str
        Path to directory containing model weights.

    Returns
    -------
    gr.Blocks
        Configured Gradio application (not yet launched).
    """
    unet, vit, gradcam, pp, train_uids = load_models(stage1_dir, DEVICE)

    custom_css = """
    body, .gradio-container {
        background-color: #0f1117 !important;
        color: #ffffff !important;
    }
    .gr-button-primary {
        background: #6c63ff !important;
        border: none !important;
        color: white !important;
        font-size: 16px !important;
        border-radius: 8px !important;
    }
    .gr-box, .gr-panel {
        background: #1a1d27 !important;
        border: 1px solid #2e3147 !important;
        border-radius: 10px !important;
    }
    textarea, .gr-textbox {
        background: #1a1d27 !important;
        color: #ffffff !important;
        border: 1px solid #2e3147 !important;
        font-size: 15px !important;
        font-weight: 500 !important;
    }
    """

    with gr.Blocks(title="Pulmonary Nodule Detection", css=custom_css) as demo:

        gr.Markdown("## #Hybrid 3D-CNN and Vision Transformer Framework")
        gr.Markdown("### Pulmonary Nodule Detection and Classification")

        with gr.Row():
            mhd_input = gr.File(label="Upload CT Scan (.mhd)", file_types=[".mhd"], scale=2)
            raw_input = gr.File(label="Upload RAW File (.raw)", file_types=[".raw"], scale=2)
            analyse_btn = gr.Button("Analyse CT Scan", variant="primary", scale=1)

        with gr.Row():
            with gr.Column():
                gr.Markdown("**CT Scan Overview**")
                ct_panel = gr.Image(label="1. Standardization", type="numpy", height=300)
            with gr.Column():
                gr.Markdown("**Extracted Nodule (ROI)**")
                roi_panel = gr.Image(label="2. candidate nodule", type="numpy", height=300)
            with gr.Column():
                gr.Markdown("**Grad-CAM Heatmap**")
                cam_panel = gr.Image(
                    label="3. Explainability", type="numpy", height=300
                )

        with gr.Row():
            result_panel = gr.Textbox(
                label="Analysis Result", lines=3, interactive=False
            )

        analyse_btn.click(
            fn=lambda mhd, raw: analyse_scan(mhd, raw, unet, vit, gradcam, pp),
            inputs=[mhd_input, raw_input],
            outputs=[ct_panel, roi_panel, cam_panel, result_panel],
        )

        gr.Markdown(
            "**Note:** Upload both .mhd and .raw files from the same CT scan. "
            "Grad-CAM is shown only for MALIGNANT predictions."
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(share=True, debug=True)
