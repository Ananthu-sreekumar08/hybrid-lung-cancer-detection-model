import os, torch, sitk, numpy as np, pandas as pd, scipy.ndimage
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA & PREPROCESSING ENGINE
# ==========================================
class DataEngine:
    @staticmethod
    def normalize_hu(image):
        # Standard Lung Windowing (-1000 to 400 HU)
        return np.clip((image + 1000) / 1400, 0, 1)

    @staticmethod
    def resample(image, spacing, new_spacing=[1,1,1]):
        resize_factor = spacing / new_spacing
        new_shape = np.round(image.shape * resize_factor).astype(int)
        real_resize_factor = new_shape / image.shape
        return scipy.ndimage.zoom(image, real_resize_factor, order=1)

class LunaDataset(Dataset):
    def __init__(self, patient_list, df_ann, root_dir, patch_size=(64, 64, 64)):
        self.patient_list = patient_list
        self.df_ann = df_ann
        self.patch_size = patch_size
        self.path_dict = {f.replace(".mhd", ""): os.path.join(r, f) 
                          for r, d, fs in os.walk(root_dir) for f in fs if f.endswith(".mhd")}

    def __len__(self): return len(self.patient_list)

    def __getitem__(self, idx):
        pid = self.patient_list[idx]
        itk_img = sitk.ReadImage(self.path_dict[pid])
        img = sitk.GetArrayFromImage(itk_img)
        spacing = np.array(itk_img.GetSpacing()[::-1])
        origin = np.array(itk_img.GetOrigin()[::-1])
        
        # Preprocess
        img_res = DataEngine.resample(img, spacing)
        img_clean = DataEngine.normalize_hu(img_res)
        
        # Coordinate Mapping for Nodule Center
        ann = self.df_ann[self.df_ann['seriesuid'] == pid].iloc[0]
        v_coord = np.abs(np.array([ann.coordZ, ann.coordY, ann.coordX]) - origin)
        z, y, x = v_coord.astype(int)
        
        # 3D Patch Extraction
        p = self.patch_size[0] // 2
        patch = img_clean[max(0,z-p):z+p, max(0,y-p):y+p, max(0,x-p):x+p]
        
        # Ensure fixed shape (padding if necessary)
        final_patch = np.zeros(self.patch_size)
        dz, dy, dx = patch.shape
        final_patch[:dz, :dy, :dx] = patch
        
        return torch.from_numpy(final_patch).float().unsqueeze(0), torch.tensor(1 if ann.diameter_mm > 5 else 0)

# ==========================================
# 2. HYBRID MODEL (3D-CNN + ViT)
# ==========================================
class HybridLungNet(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN: Local Feature Extraction
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d(2), # 32x32x32
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d(2)  # 16x16x16
        )
        
        # ViT: Global Contextual Analysis
        self.proj = nn.Linear(64, 128) 
        enc_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        self.vit = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        x = self.cnn(x) # [B, 64, 16, 16, 16]
        x = rearrange(x, 'b c d h w -> b (d h w) c') # Flatten 3D to Sequence
        x = self.proj(x)
        x = self.vit(x)
        return self.classifier(x[:, 0]) # Prediction based on CLS token

# ==========================================
# 3. EXPLAINABILITY ENGINE (GRAD-CAM 3D)
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._save_act)
        self.target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o): self.activations = o
    def _save_grad(self, m, gi, go): self.gradients = go[0]

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        output = self.model(x)
        output[:, class_idx].backward()
        weights = torch.mean(self.gradients, dim=[2, 3, 4], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        return np.maximum(cam, 0) / (np.max(cam) + 1e-7)

# ==========================================
# 4. UNIFIED EXECUTION & INFERENCE
# ==========================================
# Setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df_ann = pd.read_csv('/kaggle/input/luna16/annotations.csv')
model = HybridLungNet().to(DEVICE)

# Run Inference on Example Patient
test_pid = df_ann['seriesuid'].iloc[0]
dataset = LunaDataset([test_pid], df_ann, '/kaggle/input/luna16')
img_t, label = dataset[0]
img_t = img_t.unsqueeze(0).to(DEVICE)

# Generate Explainability Heatmap
cam_engine = GradCAM(model, model.cnn[4]) # Target last Conv layer
heatmap = cam_engine(img_t, class_idx=1)

# Visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1); plt.imshow(img_t.cpu()[0,0,32], cmap='gray'); plt.title("Input CT Voxel")
plt.subplot(1, 2, 2); plt.imshow(img_t.cpu()[0,0,32], cmap='gray')
plt.imshow(scipy.ndimage.zoom(heatmap, 4), cmap='jet', alpha=0.4); plt.title("AI Attention (Grad-CAM)")
plt.show()