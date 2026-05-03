"""
utils/metrics.py
================
A Hybrid 3D-CNN and Vision Transformer Framework for
Pulmonary Nodule Detection and Classification
College of Engineering Attingal — CSD416 Project Phase II

Module: Performance Metrics (Section 7 — Result and Analysis)
--------------------------------------------------------------
Computes standard medical imaging evaluation metrics used to assess
the performance of the proposed Hybrid Model (3D-CNN + Vision Transformer)
as reported in Table 7.1 of the project report.

Reported metrics for the Hybrid System (U-Net + ViT):
    Accuracy    : 89.12%  — Overall correctness of the diagnostic system
    Recall      : 90.00%  — Ability to correctly identify malignant cases
    Precision   : 88.24%  — Probability that a Malignant prediction is correct
    Specificity : 88.00%  — Ability to correctly identify Benign cases

Architecture comparison (Table 7.2):
    3D CNN alone          : 89.10%
    Vision Transformer    : 88.40%
    Proposed Hybrid Model : 89.12%

Authors : A. Anudeep, Abhishek S M, Aleena Krishnan, Ananthu S B
Guide   : Dr Remya R S
"""

import numpy as np
from typing import Dict, List


def compute_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, int]:
    """
    Compute binary confusion matrix components.

    Positive class = Malignant (1), Negative class = Benign (0).

    Parameters
    ----------
    y_true : list of int
        Ground-truth labels (0 = Benign, 1 = Malignant).
    y_pred : list of int
        Predicted labels (0 = Benign, 1 = Malignant).

    Returns
    -------
    dict with keys:
        TP : True Positives  — correctly predicted Malignant
        TN : True Negatives  — correctly predicted Benign
        FP : False Positives — Benign predicted as Malignant
        FN : False Negatives — Malignant predicted as Benign
    """
    TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    TN = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    FP = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    FN = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}


def compute_accuracy(cm: Dict[str, int]) -> float:
    """
    Overall diagnostic accuracy.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Reported value (Table 7.1): 89.12%
    Clinical significance: Overall correctness of the diagnostic system.

    Parameters
    ----------
    cm : dict
        Confusion matrix from compute_confusion_matrix.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    total = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']
    if total == 0:
        return 0.0
    return (cm['TP'] + cm['TN']) / total


def compute_recall(cm: Dict[str, int]) -> float:
    """
    Recall (Sensitivity / True Positive Rate).

    Recall = TP / (TP + FN)

    Reported value (Table 7.1): 90.00%
    Clinical significance: Ability to correctly identify patients with
    cancer.  High recall is critical in oncology screening — missing
    a malignant nodule (FN) is more harmful than a false alarm (FP).

    Parameters
    ----------
    cm : dict
        Confusion matrix from compute_confusion_matrix.

    Returns
    -------
    float
        Recall in [0, 1].
    """
    denom = cm['TP'] + cm['FN']
    if denom == 0:
        return 0.0
    return cm['TP'] / denom


def compute_precision(cm: Dict[str, int]) -> float:
    """
    Precision (Positive Predictive Value).

    Precision = TP / (TP + FP)

    Reported value (Table 7.1): 88.24%
    Clinical significance: Probability that a Malignant prediction is
    correct.  High precision reduces unnecessary follow-up procedures.

    Parameters
    ----------
    cm : dict
        Confusion matrix from compute_confusion_matrix.

    Returns
    -------
    float
        Precision in [0, 1].
    """
    denom = cm['TP'] + cm['FP']
    if denom == 0:
        return 0.0
    return cm['TP'] / denom


def compute_specificity(cm: Dict[str, int]) -> float:
    """
    Specificity (True Negative Rate).

    Specificity = TN / (TN + FP)

    Reported value (Table 7.1): 88.00%
    Clinical significance: Ability to correctly identify Benign cases,
    reducing false alarms that lead to unnecessary biopsies.

    Parameters
    ----------
    cm : dict
        Confusion matrix from compute_confusion_matrix.

    Returns
    -------
    float
        Specificity in [0, 1].
    """
    denom = cm['TN'] + cm['FP']
    if denom == 0:
        return 0.0
    return cm['TN'] / denom


def compute_f1(cm: Dict[str, int]) -> float:
    """
    F1 Score — harmonic mean of Precision and Recall.

    F1 = 2 × (Precision × Recall) / (Precision + Recall)

    Parameters
    ----------
    cm : dict
        Confusion matrix.

    Returns
    -------
    float
        F1 score in [0, 1].
    """
    prec = compute_precision(cm)
    rec  = compute_recall(cm)
    denom = prec + rec
    if denom == 0:
        return 0.0
    return 2.0 * prec * rec / denom


def evaluate_model(
    y_true: List[int],
    y_pred: List[int],
    print_report: bool = True,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics and optionally print a report.

    Parameters
    ----------
    y_true : list of int
        Ground-truth labels.
    y_pred : list of int
        Predicted labels.
    print_report : bool
        If True, prints a formatted metrics table. Default True.

    Returns
    -------
    dict with keys: accuracy, recall, precision, specificity, f1
        All values in [0, 1].
    """
    cm          = compute_confusion_matrix(y_true, y_pred)
    accuracy    = compute_accuracy(cm)
    recall      = compute_recall(cm)
    precision   = compute_precision(cm)
    specificity = compute_specificity(cm)
    f1          = compute_f1(cm)

    if print_report:
        print("=" * 55)
        print(f"{'Metric':<20} {'Score':>10}  {'(Report value)':>15}")
        print("=" * 55)
        print(f"{'Accuracy':<20} {accuracy*100:>9.2f}%  {'(89.12%)':>15}")
        print(f"{'Recall':<20} {recall*100:>9.2f}%  {'(90.00%)':>15}")
        print(f"{'Precision':<20} {precision*100:>9.2f}%  {'(88.24%)':>15}")
        print(f"{'Specificity':<20} {specificity*100:>9.2f}%  {'(88.00%)':>15}")
        print(f"{'F1 Score':<20} {f1*100:>9.2f}%  {'':>15}")
        print("=" * 55)
        print(f"Confusion Matrix:")
        print(f"  TP={cm['TP']}  TN={cm['TN']}  FP={cm['FP']}  FN={cm['FN']}")
        print("=" * 55)

    return {
        'accuracy':    accuracy,
        'recall':      recall,
        'precision':   precision,
        'specificity': specificity,
        'f1':          f1,
    }


# ---------------------------------------------------------------------------
# Architecture comparison table (Table 7.2)
# ---------------------------------------------------------------------------

ARCHITECTURE_COMPARISON = {
    'Proposed Hybrid Model (U-Net + ViT)': 89.12,
    '3D CNN alone':                        89.10,
    'Vision Transformer (ViT)':            88.40,
    '3D VGG-like Model':                   95.00,
    'DenseNet-121':                        95.60,
    '3D ResNet-50':                        89.10,
    'YOLOv8 (Detector)':                   90.32,
}


def print_architecture_comparison() -> None:
    """
    Print the architecture comparison table from Table 7.2.
    """
    print("=" * 55)
    print(f"{'Model Architecture':<40} {'Accuracy':>10}")
    print("=" * 55)
    for model, acc in ARCHITECTURE_COMPARISON.items():
        marker = " ◄" if 'Proposed' in model else ""
        print(f"{model:<40} {acc:>9.2f}%{marker}")
    print("=" * 55)
