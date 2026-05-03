"""
training/train_vit.py
=====================
A Hybrid 3D-CNN and Vision Transformer Framework for
Pulmonary Nodule Detection and Classification
College of Engineering Attingal — CSD416 Project Phase II

Module: 3D ViT Training Loop (Section 6.6–6.8)
------------------------------------------------
Trains the 3D Vision Transformer on 32×32×32 voxel ROI patches
extracted from annotated nodule centroids using ground-truth coordinates
from LUNA16 / LIDC-IDRI.

Loss function : Weighted Cross-Entropy (class weights inversely
                proportional to class frequency to address imbalance)
Optimiser     : AdamW (lr = 3e-4, weight_decay = 0.05)
Scheduler     : CosineAnnealingLR over VIT_EPOCHS
Precision     : Mixed precision (torch.amp) for faster training
Checkpoint    : Best model saved by maximum validation accuracy

Performance results (Table 7.1 of project report):
    Accuracy    : 89.12%   (Hybrid U-Net + ViT system)
    Recall      : 90.00%
    Precision   : 88.24%
    Specificity : 88.00%

Authors : A. Anudeep, Abhishek S M, Aleena Krishnan, Ananthu S B
Guide   : Dr Remya R S
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Optional

from preprocessing.resample import load_and_resample, world_to_voxel
from preprocessing.patch_extractor import extract_roi, ROI_SIZE
from preprocessing.dataset_builder import get_malignancy_label


# ---------------------------------------------------------------------------
# Hyperparameters (matching project report configuration)
# ---------------------------------------------------------------------------
VIT_EPOCHS    : int   = 30
BATCH_SIZE    : int   = 8
LEARNING_RATE : float = 3e-4
WEIGHT_DECAY  : float = 0.05
VAL_FRACTION  : float = 0.20


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NoduleROIDataset(Dataset):
    """
    PyTorch Dataset for ViT classification training.

    Loads CT volumes on-the-fly, extracts a 32×32×32 ROI centred on
    each annotated nodule, normalises intensities, and returns a
    (patch_tensor, label_tensor) pair.

    Parameters
    ----------
    uids : list of str
        Series UIDs to include in this dataset split.
    uid_to_nodules : dict {str: list of dict}
        Maps series UID → list of nodule annotation dicts.
    uid_to_raw : dict {str: str}
        Maps series UID → path to .mhd file.
    mal_df : pd.DataFrame or None
        LIDC-derived malignancy DataFrame, or None to use diameter proxy.
    augment : bool
        If True apply random flips, intensity jitter, noise. Default True.
    """

    def __init__(
        self,
        uids: List[str],
        uid_to_nodules: Dict[str, list],
        uid_to_raw: Dict[str, str],
        mal_df=None,
        augment: bool = True,
    ):
        self.samples = []    # (mhd_path, nodule_dict, label, uid)
        self.augment = augment

        for uid in uids:
            raw_path = uid_to_raw.get(uid)
            if raw_path is None:
                continue
            if uid not in uid_to_nodules:
                continue
            for nod in uid_to_nodules[uid]:
                label = get_malignancy_label(uid, nod['diameter'], mal_df)
                if label == -1:           # Exclude ambiguous (score=3)
                    continue
                self.samples.append((raw_path, nod, label, uid))

        print(f"NoduleROIDataset: {len(self.samples)} nodules loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        raw_path, nod, label, uid = self.samples[idx]

        try:
            vol, origin, spacing = load_and_resample(raw_path)
        except Exception as e:
            print(f"[WARN] Failed to load {raw_path}: {e}")
            return (
                torch.zeros(1, ROI_SIZE, ROI_SIZE, ROI_SIZE),
                torch.tensor(0, dtype=torch.long),
            )

        vox = world_to_voxel(origin, spacing, [nod['x'], nod['y'], nod['z']])

        roi = extract_roi(vol, vox, roi_size=ROI_SIZE, augment=self.augment)

        return (
            torch.tensor(roi).unsqueeze(0).float(),  # (1, 32, 32, 32)
            torch.tensor(label, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_vit(
    vit: nn.Module,
    uids: List[str],
    uid_to_nodules: Dict[str, list],
    uid_to_raw: Dict[str, str],
    output_dir: str,
    device: torch.device,
    mal_df=None,
    epochs: int = VIT_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
) -> nn.Module:
    """
    Train the 3D Vision Transformer for nodule malignancy classification.

    Constructs a NoduleROIDataset from the provided training UIDs,
    computes per-class weights to handle the benign / malignant imbalance,
    and runs the AdamW + CosineAnnealingLR training loop.  The best model
    by validation accuracy is saved as vit_best.pth.

    Parameters
    ----------
    vit : nn.Module
        Initialised ViT3D instance.
    uids : list of str
        Training series UIDs.
    uid_to_nodules : dict
        Maps series UID → list of nodule annotation dicts.
    uid_to_raw : dict
        Maps series UID → .mhd file path.
    output_dir : str
        Directory for saving checkpoints.
    device : torch.device
    mal_df : pd.DataFrame or None
        LIDC malignancy DataFrame.
    epochs, batch_size, lr : training hyperparameters.

    Returns
    -------
    nn.Module
        Trained ViT with best validation accuracy weights loaded.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- Build dataset and split ------------------------------------------
    full_ds = NoduleROIDataset(
        uids, uid_to_nodules, uid_to_raw, mal_df, augment=True
    )
    n_val   = max(1, int(len(full_ds) * VAL_FRACTION))
    n_train = len(full_ds) - n_val

    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Validation dataset should not use augmentation
    val_ds.dataset.augment = False

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ---- Class weights (inverse frequency) --------------------------------
    all_labels = [full_ds.samples[i][2] for i in range(len(full_ds))]
    n_mal   = sum(all_labels)
    n_ben   = len(all_labels) - n_mal
    n_total = len(all_labels)

    w_ben = n_total / (2.0 * max(n_ben, 1))
    w_mal = n_total / (2.0 * max(n_mal, 1))
    class_weights = torch.tensor([w_ben, w_mal], dtype=torch.float32).to(device)

    print(f"Train: {n_train} | Val: {n_val}")
    print(f"Benign: {n_ben} | Malignant: {n_mal}")
    print(f"Class weights → benign: {w_ben:.2f} | malignant: {w_mal:.2f}")

    # ---- Optimiser and scheduler -----------------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(vit.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler    = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_acc  = 0.0
    best_val_loss = float('inf')
    vit.to(device)

    # ---- Training loop ---------------------------------------------------
    for epoch in range(1, epochs + 1):

        # -- Train --
        vit.train()
        train_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = vit(xb)
                    loss   = criterion(logits, yb)
            else:
                logits = vit(xb)
                loss   = criterion(logits, yb)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vit.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vit.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += yb.size(0)

        train_loss /= len(train_loader)
        train_acc   = correct / total

        # -- Validate --
        vit.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits = vit(xb)
                else:
                    logits = vit(xb)
                val_loss    += criterion(logits.float(), yb).item()
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total   += yb.size(0)

        val_loss /= len(val_loader)
        val_acc   = val_correct / val_total
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"trn_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f} | "
            f"lr={current_lr:.6f}"
        )

        # Save best by accuracy, break ties with val_loss
        if val_acc > best_val_acc or (
            val_acc == best_val_acc and val_loss < best_val_loss
        ):
            best_val_acc  = val_acc
            best_val_loss = val_loss
            torch.save(vit.state_dict(), os.path.join(output_dir, 'vit_best.pth'))
            print(
                f"  ✓ Saved best ViT "
                f"(val_acc={val_acc:.3f} val_loss={val_loss:.4f})"
            )

    # Final checkpoint
    torch.save(vit.state_dict(), os.path.join(output_dir, 'vit_final.pth'))

    # Reload best weights
    vit.load_state_dict(
        torch.load(os.path.join(output_dir, 'vit_best.pth'), map_location=device)
    )
    print(f"ViT training complete. Best val acc: {best_val_acc:.3f}")
    return vit
