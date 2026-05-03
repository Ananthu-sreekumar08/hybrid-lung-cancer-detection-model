"""
models/vit3d.py
===============
A Hybrid 3D-CNN and Vision Transformer Framework for
Pulmonary Nodule Detection and Classification
College of Engineering Attingal — CSD416 Project Phase II

Module: 3D Vision Transformer Classification Model (Section 5.4 & 6.6–6.8)
---------------------------------------------------------------------------
Implements the 3D Vision Transformer (ViT) used in Stage 2 of the
proposed hybrid system for malignancy classification of detected nodules.

The ViT tokenises a 32×32×32 voxel ROI into 8×8×8 non-overlapping
cubic patches (64 tokens), projects each to a 256-dimensional embedding,
prepends a learnable CLS token, adds positional embeddings, and passes
the sequence through a stack of Transformer Encoder layers.  The final
CLS token representation is passed to a two-class linear head producing
Benign (0) / Malignant (1) logits.

Architecture summary:
    Input       : (B, 1, 32, 32, 32)
    Patch embed : 4×4×4 grid → 64 tokens of dim 256
    + CLS token : sequence length = 65
    Transformer : depth=6, heads=8, mlp_ratio=4 (Pre-LN)
    Output      : (B, 2)  logits → softmax → malignancy probability

Total parameters ≈ 4.89 M.

The self-attention mechanism (Section 6.8.1) allows each token to attend
to every other token, capturing long-range spatial relationships within
the nodule — a key advantage over local CNN receptive fields.

Authors : A. Anudeep, Abhishek S M, Aleena Krishnan, Ananthu S B
Guide   : Dr Remya R S
"""

import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding — Linear Projection (Section 6.7).

    Divides a 32×32×32 voxel cube into non-overlapping 8×8×8 patches
    using a strided 3D convolution.  Each patch is projected into a
    256-dimensional embedding vector (feature space).

    This step corresponds to the tokenisation stage described in
    Section 6.6.1 (Patch Partitioning) of the project report.

    Parameters
    ----------
    vol_size : int
        Side length of the input cubic volume. Default 32.
    patch_size : int
        Side length of each cubic patch. Default 8.
        Produces (vol_size // patch_size)^3 = 64 tokens.
    in_ch : int
        Number of input channels. Default 1 (greyscale CT).
    embed_dim : int
        Dimensionality of the patch embedding. Default 256.
    """

    def __init__(
        self,
        vol_size:   int = 32,
        patch_size: int = 8,
        in_ch:      int = 1,
        embed_dim:  int = 256,
    ):
        super().__init__()
        self.n_patches = (vol_size // patch_size) ** 3
        self.proj = nn.Conv3d(
            in_ch, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, 32, 32, 32)

        Returns
        -------
        torch.Tensor, shape (B, n_patches, embed_dim)
            Sequence of patch token embeddings.
        """
        x = self.proj(x)                   # (B, embed_dim, 4, 4, 4)
        x = x.flatten(2)                   # (B, embed_dim, 64)
        x = x.transpose(1, 2)             # (B, 64, embed_dim)
        return x


class ViT3D(nn.Module):
    """
    3D Vision Transformer for pulmonary nodule malignancy classification.

    Used in Stage 2 (Sequence Modelling & Malignancy Prediction) of the
    proposed hybrid framework as described in Section 4.2.2 and Figure 5.3.

    The model processes the 32×32×32 ROI through the following steps:

    1. Patch embedding (Linear Projection, Section 6.7)
    2. CLS token prepended and positional embeddings added
       (Positional Tagging, Section 6.7)
    3. Transformer Encoder — multi-head self-attention + MLP blocks
       (Section 6.8.1)
    4. Layer normalisation on the output sequence
    5. CLS token → classification head → Benign / Malignant logits
       (Section 6.8.2)

    Parameters
    ----------
    vol_size : int
        Side length of input volume in voxels. Default 32.
    patch_size : int
        Side length of each cubic token patch. Default 8.
    embed_dim : int
        Embedding dimensionality. Default 256.
    depth : int
        Number of Transformer Encoder layers. Default 6.
    n_heads : int
        Number of self-attention heads. Default 8.
    mlp_ratio : float
        Ratio of MLP hidden dim to embed_dim. Default 4.0.
    n_classes : int
        Number of output classes. Default 2 (Benign / Malignant).

    Input
    -----
    x : torch.Tensor, shape (B, 1, 32, 32, 32)
        Normalised ROI patch, values in [0, 1].

    Output
    ------
    torch.Tensor, shape (B, 2)
        Raw logits.  Apply softmax to obtain class probabilities.
        Index 0 → Benign probability.
        Index 1 → Malignant probability.
    """

    def __init__(
        self,
        vol_size:   int   = 32,
        patch_size: int   = 8,
        embed_dim:  int   = 256,
        depth:      int   = 6,
        n_heads:    int   = 8,
        mlp_ratio:  float = 4.0,
        n_classes:  int   = 2,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed3D(vol_size, patch_size, 1, embed_dim)
        n_patches = (vol_size // patch_size) ** 3

        # Learnable CLS token — aggregates global image information
        # Used as the basis for malignancy prediction (Section 6.8.2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional embeddings — preserve spatial relationships
        # (Positional Tagging, Section 6.7)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        nn.init.trunc_normal_(self.cls_token,  std=0.02)
        nn.init.trunc_normal_(self.pos_embed,  std=0.02)

        # Transformer Encoder — Pre-LN (norm_first=True) for stable training
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            batch_first=True,
            norm_first=True,          # Pre-LayerNorm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Step 1 — Patch tokenisation (Section 6.6)
        tokens = self.patch_embed(x)                        # (B, 64, 256)

        # Step 2 — Prepend CLS token and add positional embeddings
        cls = self.cls_token.expand(B, -1, -1)             # (B,  1, 256)
        tokens = torch.cat([cls, tokens], dim=1)            # (B, 65, 256)
        tokens = tokens + self.pos_embed                    # (B, 65, 256)

        # Step 3 — Transformer Encoder (self-attention, Section 6.8.1)
        tokens = self.transformer(tokens)                   # (B, 65, 256)

        # Step 4 — Layer normalisation
        tokens = self.norm(tokens)                          # (B, 65, 256)

        # Step 5 — Classification from CLS token (index 0)
        return self.head(tokens[:, 0])                      # (B, 2)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
