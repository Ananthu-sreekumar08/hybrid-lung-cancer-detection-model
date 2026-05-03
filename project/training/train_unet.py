"""
training/train_unet.py
======================
A Hybrid 3D-CNN and Vision Transformer Framework for
Pulmonary Nodule Detection and Classification
College of Engineering Attingal — CSD416 Project Phase II

Module: 3D U-Net Training Loop (Section 6.5)
--------------------------------------------
Trains the 3D U-Net segmentation model on 64×64×64 voxel patches
sampled from the LUNA16 dataset.

Uses MONAI's training utilities as specified in Section 4.10.3:

    "MONAI provides ... loss functions (DiceCELoss), metrics
     (DiceMetric), and inference tools (sliding_window_inference)."

Loss function : monai.losses.DiceCELoss
    Combined Dice + Cross-Entropy loss.  sigmoid=True applies sigmoid
    internally so the model can output raw logits.  lambda_dice=0.5
    and lambda_ce=0.5 give equal weighting to both components.

Metric        : monai.metrics.DiceMetric
    Computes mean Dice score on the validation set after thresholding
    predictions at 0.5.

Optimiser     : Adam (lr = 1e-3, weight_decay = 1e-5)
Scheduler     : CosineAnnealingLR over UNET_EPOCHS
Precision     : Mixed precision (torch.amp) for faster Kaggle GPU training
Checkpoint    : Best model saved by maximum validation Dice score

Performance results from the project report (Table 7.1):
    The hybrid system (U-Net + ViT) achieved overall accuracy of 89.12%,
    recall 90.00%, precision 88.24%, specificity 88.00%.

Authors : A. Anudeep, Abhishek S M, Aleena Krishnan, Ananthu S B
Guide   : Dr Remya R S
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from monai.losses  import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose


# ---------------------------------------------------------------------------
# Hyperparameters (matching project report configuration)
# ---------------------------------------------------------------------------
UNET_EPOCHS  : int   = 20
BATCH_SIZE   : int   = 4
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY : float = 1e-5
VAL_FRACTION : float = 0.15


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_unet(
    unet: nn.Module,
    all_patches: torch.Tensor,
    all_masks: torch.Tensor,
    output_dir: str,
    device: torch.device,
    epochs: int = UNET_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
) -> nn.Module:
    """
    Train the 3D U-Net segmentation model using MONAI losses and metrics.

    Uses MONAI's DiceCELoss (combined Dice + Cross-Entropy) and
    DiceMetric for validation as specified in Section 4.10.3 of the
    project report.

    Splits the patch dataset into training and validation sets, runs
    the training loop, and saves the best model checkpoint by maximum
    validation Dice score.

    Parameters
    ----------
    unet : nn.Module
        Initialised UNet3D instance (wraps MONAI UNet internally).
    all_patches : torch.Tensor, shape (N, 1, 64, 64, 64)
        All extracted CT patches, normalised to [0, 1].
    all_masks : torch.Tensor, shape (N, 1, 64, 64, 64)
        Corresponding binary nodule segmentation masks.
    output_dir : str
        Directory where checkpoints will be saved.
    device : torch.device
        Training device (cuda or cpu).
    epochs : int
        Number of training epochs. Default 20.
    batch_size : int
        Mini-batch size. Default 4.
    lr : float
        Initial learning rate for Adam. Default 1e-3.

    Returns
    -------
    nn.Module
        Trained U-Net with best validation Dice weights loaded.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- Dataset split ----------------------------------------------------
    dataset = TensorDataset(all_patches, all_masks)
    n_val   = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val

    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    print(f"Train patches: {n_train} | Val patches: {n_val}")

    # ---- MONAI loss and metric -------------------------------------------
    # DiceCELoss — sigmoid=True applies sigmoid internally on logits
    # lambda_dice=0.5, lambda_ce=0.5 — equal weighting (Section 4.10.3)
    loss_fn = DiceCELoss(
        sigmoid=True,
        lambda_dice=0.5,
        lambda_ce=0.5,
    )

    # DiceMetric — computes mean Dice on thresholded predictions
    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
    )

    # Post-processing: sigmoid → threshold at 0.5 for metric computation
    post_pred   = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label  = AsDiscrete(threshold=0.5)

    # ---- Optimiser and scheduler -----------------------------------------
    optimizer = torch.optim.Adam(
        unet.parameters(), lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler    = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_dice     = -1.0
    unet.to(device)

    # ---- Training loop ---------------------------------------------------
    for epoch in range(1, epochs + 1):

        # -- Train --
        unet.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    # UNet3D.forward applies sigmoid — pass raw logits
                    # to DiceCELoss by calling the internal MONAI unet
                    logits = unet.unet(xb)
                    loss   = loss_fn(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = unet.unet(xb)
                loss   = loss_fn(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -- Validate with MONAI DiceMetric --
        unet.eval()
        dice_metric.reset()

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits = unet.unet(xb)
                else:
                    logits = unet.unet(xb)

                # Apply sigmoid + threshold for metric
                pred_bin  = torch.stack([post_pred(lo)  for lo in logits])
                label_bin = torch.stack([post_label(lb) for lb in yb])
                dice_metric(y_pred=pred_bin, y=label_bin)

        mean_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"trn_loss={train_loss:.4f} | val_dice={mean_dice:.4f} | "
            f"lr={current_lr:.6f}"
        )

        # Save best checkpoint by Dice score
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(unet.state_dict(), os.path.join(output_dir, 'unet_best.pth'))
            print(f"  ✓ Saved best U-Net (val_dice={mean_dice:.4f})")

    # Final checkpoint
    torch.save(unet.state_dict(), os.path.join(output_dir, 'unet_final.pth'))

    # Reload best weights
    unet.load_state_dict(
        torch.load(os.path.join(output_dir, 'unet_best.pth'), map_location=device)
    )
    print(f"U-Net training complete. Best val Dice: {best_dice:.4f}")
    return unet
