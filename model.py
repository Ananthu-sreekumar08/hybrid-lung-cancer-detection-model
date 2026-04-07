import torch
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from einops import rearrange

# 1. DEFINE ARCHITECTURES (Must match your training setup exactly)
class HybridInference(torch.nn.Module):
    def __init__(self, cnn_path, vit_path):
        super().__init__()
        # Backbone (CNN)
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, 3, padding=1), torch.nn.BatchNorm3d(32), torch.nn.ReLU(),
            torch.nn.MaxPool3d(2),
            torch.nn.Conv3d(32, 64, 3, padding=1), torch.nn.BatchNorm3d(64), torch.nn.ReLU(),
            torch.nn.MaxPool3d(2),
            torch.nn.Conv3d(64, 128, 3, padding=1), torch.nn.BatchNorm3d(128), torch.nn.ReLU()
        )
        self.cnn.load_state_dict(torch.load(cnn_path, map_location='cpu'), strict=False)
        
        # Transformer (ViT)
        self.proj = torch.nn.Linear(128, 256)
        enc_layer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.vit = torch.nn.TransformerEncoder(enc_layer, num_layers=4)
        self.classifier = torch.nn.Linear(256, 2)
        self.vit_head = torch.load(vit_path, map_location='cpu') # Assuming full state_dict
        # self.load_state_dict(...) # Add full vit loading logic here if needed

    def forward(self, x):
        features = self.cnn(x)
        # Flattening logic
        seq = rearrange(features, 'b c d h w -> b (d h w) c')
        embeddings = self.proj(seq)
        output = self.vit(embeddings)
        return self.classifier(output[:, 0]), features

# 2. THE GRAD-CAM EXPLAINER
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._save_act)
        self.target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o): self.activations = o
    def _save_grad(self, m, gi, go): self.gradients = go[0]

    def generate(self, x, class_idx):
        self.model.zero_grad()
        output, _ = self.model(x)
        output[0, class_idx].backward()
        
        # Weighted combination of activation maps
        weights = torch.mean(self.gradients, dim=[2, 3, 4], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0) # ReLU
        return cam / (np.max(cam) + 1e-8)

# 3. THE PIPELINE FUNCTION
def run_pipeline(mhd_path, cnn_weights, vit_weights):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridInference(cnn_weights, vit_weights).to(device).eval()
    
    # --- Step A: Load & Resample ---
    itk_img = sitk.ReadImage(mhd_path)
    img = sitk.GetArrayFromImage(itk_img).astype(np.float32)
    spacing = np.array(itk_img.GetSpacing()[::-1])
    
    # Resample to 1mm isotropic (Matches training preprocessing)
    img_res = scipy.ndimage.zoom(img, spacing / 1.0, order=1)
    img_norm = np.clip((img_res + 1000) / 1400, 0, 1) # HU Normalization
    
    # --- Step B: Prediction ---
    # Create patch (e.g., center 64x64x64)
    mid_z, mid_y, mid_x = [s//2 for s in img_norm.shape]
    patch = img_norm[mid_z-32:mid_z+32, mid_y-32:mid_y+32, mid_x-32:mid_x+32]
    input_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
    
    # --- Step C: Grad-CAM ---
    cam_engine = GradCAM3D(model, model.cnn[-1])
    heatmap = cam_engine.generate(input_t, class_idx=1) # 1 = Malignant
    
    # Get Final Confidence
    with torch.no_grad():
        output, _ = model(input_t)
        prob = torch.softmax(output, dim=1)[0, 1].item()
    
    # --- Step D: Visualization ---
    plt.figure(figsize=(18, 6))
    
    # 1. The "Flattened" Scan View
    plt.subplot(1, 3, 1)
    plt.imshow(patch[32], cmap='gray')
    plt.title("CT Input (Middle Slice)")
    plt.axis('off')
    
    # 2. The Model Diagnosis
    plt.subplot(1, 3, 2)
    diagnosis = "MALIGNANT" if prob > 0.5 else "BENIGN"
    plt.text(0.5, 0.5, f"Diagnosis:\n{diagnosis}\n\nConfidence:\n{prob:.2%}", 
             ha='center', va='center', fontsize=15, fontweight='bold', color='red' if prob > 0.5 else 'green')
    plt.axis('off')
    plt.title("Model Decision")
    
    # 3. The Grad-CAM Visualization
    plt.subplot(1, 3, 3)
    plt.imshow(patch[32], cmap='gray')
    # Resize heatmap to match patch dimensions for overlay
    heatmap_res = scipy.ndimage.zoom(heatmap, np.array(patch.shape) / np.array(heatmap.shape))
    plt.imshow(heatmap_res[32], cmap='jet', alpha=0.4)
    plt.title("Explainable AI (Heatmap)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# USAGE:
# run_pipeline("patient_001.mhd", "cnn_model.pth", "vit_model.pth")