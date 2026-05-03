"""
preprocessing/dataset_builder.py
=================================
A Hybrid 3D-CNN and Vision Transformer Framework for
Pulmonary Nodule Detection and Classification
College of Engineering Attingal — CSD416 Project Phase II

Module: Ground Truth Mapping & Annotation (Section 6.3)
--------------------------------------------------------
Merges LUNA16 annotation coordinates with LIDC-IDRI malignancy
scores to produce a unified ground-truth CSV that links each
nodule to its world-space centroid, diameter, and binary
malignancy label.

LUNA16 provides nodule coordinates and diameters but no
malignancy labels.  LIDC-IDRI provides per-nodule malignancy
scores (1–5) for the same CT series.  Both datasets share the
same seriesuid, enabling a direct join on that key.

Malignancy labelling convention (as used in training):
    Score 1–2  →  Benign  (label = 0)
    Score 3    →  Excluded (ambiguous)
    Score 4–5  →  Malignant (label = 1)

Authors : A. Anudeep, Abhishek S M, Aleena Krishnan, Ananthu S B
Guide   : Dr Remya R S
"""

import os
import json
import random
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Malignancy label helper
# ---------------------------------------------------------------------------

def get_malignancy_label(
    seriesuid: str,
    diameter_mm: float,
    mal_df: Optional[pd.DataFrame],
) -> int:
    """
    Return the binary malignancy label for a nodule.

    If a malignancy CSV derived from LIDC-IDRI is available and the
    series is present, the radiologist consensus score is used.
    Otherwise a diameter proxy (>= 8 mm treated as malignant) is
    applied as a fallback.

    Parameters
    ----------
    seriesuid : str
        LUNA16 series UID of the scan containing this nodule.
    diameter_mm : float
        Maximum diameter of the nodule in millimetres.
    mal_df : pd.DataFrame or None
        DataFrame with columns ['seriesuid', 'malignancy'] loaded
        from an LIDC-derived malignancy CSV, or None if unavailable.

    Returns
    -------
    int
        0  — Benign
        1  — Malignant
        -1 — Ambiguous (score == 3); caller should exclude this nodule.
    """
    if mal_df is not None and seriesuid in mal_df['seriesuid'].values:
        score = int(
            mal_df.loc[mal_df['seriesuid'] == seriesuid, 'malignancy'].values[0]
        )
        if score == 3:
            return -1          # Ambiguous — exclude from training
        return 1 if score >= 4 else 0

    # Diameter proxy fallback
    return 1 if diameter_mm >= 8.0 else 0


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_ground_truth_csv(
    annotations_csv: str,
    malignancy_csv: Optional[str],
    output_csv: str,
) -> pd.DataFrame:
    """
    Build and save the unified ground-truth CSV.

    Loads LUNA16 annotations, optionally merges LIDC-IDRI malignancy
    scores, assigns binary labels, removes ambiguous entries, and
    writes the result to *output_csv*.

    Parameters
    ----------
    annotations_csv : str
        Path to LUNA16 annotations.csv containing columns:
        seriesuid, coordX, coordY, coordZ, diameter_mm.
    malignancy_csv : str or None
        Path to an LIDC-derived CSV with columns:
        seriesuid, malignancy (score 1–5).
        Pass None to use diameter proxy only.
    output_csv : str
        Destination path for the merged ground-truth CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns:
        seriesuid, coordX, coordY, coordZ, diameter_mm, label.
    """
    ann = pd.read_csv(annotations_csv)

    mal_df = None
    if malignancy_csv is not None and os.path.exists(malignancy_csv):
        mal_df = pd.read_csv(malignancy_csv)
        print(f"Malignancy CSV loaded: {len(mal_df)} entries")
    else:
        print("Malignancy CSV not found — using diameter proxy (>=8 mm = malignant)")

    labels = []
    for _, row in ann.iterrows():
        label = get_malignancy_label(
            row['seriesuid'], row['diameter_mm'], mal_df
        )
        labels.append(label)

    ann['label'] = labels

    # Remove ambiguous entries (score == 3)
    before = len(ann)
    ann = ann[ann['label'] != -1].reset_index(drop=True)
    after  = len(ann)
    print(f"Removed {before - after} ambiguous nodules (score=3)")

    n_benign    = (ann['label'] == 0).sum()
    n_malignant = (ann['label'] == 1).sum()
    print(f"Ground truth CSV: {n_benign} benign | {n_malignant} malignant")

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    ann.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
    return ann


def build_uid_maps(
    raw_base_dirs: List[str],
    annotations_csv: str,
) -> Tuple[Dict[str, str], Dict[str, list]]:
    """
    Build lookup dictionaries for scan files and nodule annotations.

    Scans the subset directories for .mhd files and constructs a
    mapping from series UID to file path, and from series UID to
    a list of nodule annotation dicts.

    Parameters
    ----------
    raw_base_dirs : list of str
        Root directories that contain LUNA16 subset folders.
        Example: ['/kaggle/input/luna16']
    annotations_csv : str
        Path to LUNA16 annotations.csv.

    Returns
    -------
    uid_to_raw : dict {str: str}
        Maps series UID → absolute path to .mhd file.
    uid_to_nodules : dict {str: list of dict}
        Maps series UID → list of nodule dicts, each with keys:
        x, y, z, diameter.
    """
    # Find all .mhd files under the provided base directories
    raw_mhds: List[str] = []
    for base in raw_base_dirs:
        raw_mhds.extend(
            glob.glob(os.path.join(base, '**', '*.mhd'), recursive=True)
        )

    uid_to_raw: Dict[str, str] = {
        os.path.basename(p).replace('.mhd', ''): p
        for p in raw_mhds
    }
    print(f"Found {len(uid_to_raw)} .mhd scan files")

    # Build nodule annotation map
    ann = pd.read_csv(annotations_csv)
    uid_to_nodules: Dict[str, list] = {}
    for _, row in ann.iterrows():
        uid = row['seriesuid']
        if uid not in uid_to_nodules:
            uid_to_nodules[uid] = []
        uid_to_nodules[uid].append({
            'x':        float(row['coordX']),
            'y':        float(row['coordY']),
            'z':        float(row['coordZ']),
            'diameter': float(row['diameter_mm']),
        })

    print(f"Annotations cover {len(uid_to_nodules)} unique series")
    return uid_to_raw, uid_to_nodules


def split_train_val(
    valid_uids: List[str],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Perform a reproducible train / validation split on series UIDs.

    Parameters
    ----------
    valid_uids : list of str
        UIDs of scans that have both annotation and raw file.
    val_fraction : float
        Fraction of scans reserved for validation. Default 0.15.
    seed : int
        Random seed for reproducibility. Default 42.

    Returns
    -------
    train_uids : list of str
    val_uids   : list of str
    """
    random.seed(seed)
    shuffled = valid_uids.copy()
    random.shuffle(shuffled)

    n_val       = max(1, int(len(shuffled) * val_fraction))
    val_uids    = shuffled[:n_val]
    train_uids  = shuffled[n_val:]

    print(f"Split — train: {len(train_uids)} | val: {len(val_uids)}")
    return train_uids, val_uids


def save_train_uids(train_uids: List[str], output_path: str) -> None:
    """
    Persist the list of training series UIDs to a JSON file.

    This file is critical for preventing data leakage during inference:
    the inference pipeline reads this list and warns if an uploaded
    scan matches a training UID.

    Parameters
    ----------
    train_uids : list of str
        Series UIDs used for training.
    output_path : str
        Destination .json file path.
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(train_uids, f, indent=2)
    print(f"Training UIDs saved: {output_path}  ({len(train_uids)} entries)")
