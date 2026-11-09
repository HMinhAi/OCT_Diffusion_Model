"""Evaluation metrics according to Paper Section 4.2.2.

Metrics:
    - Area Under ROC Curve (AUC)
    - F1-Score: 2TP / (2TP + FN + FP)
    - Average Classification Accuracy (ACC): (TP + TN) / (TP + TN + FP + FN)
    - Sensitivity (SEN): TP / (TP + FN)
    - Specificity (SPE): TN / (TN + FP)

The optimal threshold is selected based on maximizing F1-score.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve


def calculate_anomaly_metrics(
    labels: Union[List[int], np.ndarray],
    scores: Union[List[float], np.ndarray],
    num_thresholds: int = 1000,
) -> Dict[str, float]:
    """Calculate AUC, best F1-score, ACC, SEN, and SPE based on best F1 threshold.
    
    Args:
        labels: 1D array of ground truth labels (0 = Normal, 1 = Anomaly/Disease)
        scores: 1D array of predicted anomaly scores (higher = more anomalous)
        num_thresholds: Number of threshold candidates to search for optimal F1
        
    Returns:
        Dictionary containing:
            - "auc": ROC AUC in percentage [0, 100]
            - "f1": Best F1 score in percentage [0, 100]
            - "acc": Classification accuracy in percentage [0, 100]
            - "sen": Sensitivity in percentage [0, 100]
            - "spe": Specificity in percentage [0, 100]
            - "best_threshold": Threshold giving maximum F1
    """
    y_true = np.asarray(labels, dtype=np.int32)
    y_scores = np.asarray(scores, dtype=np.float64)

    # Replace NaNs or Infs if any
    y_scores = np.nan_to_num(y_scores, nan=0.0, posinf=1.0, neginf=0.0)

    # Check if both classes are present
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        return {
            "auc": 50.0,
            "f1": 0.0,
            "acc": 0.0,
            "sen": 0.0,
            "spe": 0.0,
            "best_threshold": 0.5,
        }

    # 1. Compute ROC AUC
    try:
        auc = float(roc_auc_score(y_true, y_scores)) * 100.0
    except Exception:
        auc = 50.0

    # 2. Search for threshold that maximizes F1-score & Youden's J & SPE constraints
    thresholds = np.unique(
        np.percentile(y_scores, np.linspace(0.01, 99.99, num_thresholds))
    )

    best_f1 = -1.0
    best_acc = 0.0
    best_sen = 0.0
    best_spe = 0.0
    best_thresh = float(thresholds[0]) if len(thresholds) > 0 else 0.5

    best_j = -999.0
    youden_thresh = best_thresh
    youden_sen = 0.0
    youden_spe = 0.0
    youden_acc = 0.0

    thresh_spe95 = float(thresholds[-1]) if len(thresholds) > 0 else 0.5
    sen_spe95 = 0.0
    spe_spe95 = 0.0

    thresh_spe90 = float(thresholds[-1]) if len(thresholds) > 0 else 0.5
    sen_spe90 = 0.0
    spe_spe90 = 0.0

    thresh_spe85 = float(thresholds[-1]) if len(thresholds) > 0 else 0.5
    sen_spe85 = 0.0
    spe_spe85 = 0.0

    thresh_sen93 = float(thresholds[0]) if len(thresholds) > 0 else 0.5
    sen_sen93 = 0.0
    spe_sen93 = 0.0

    target_met = False
    target_thresh = None
    target_sen = 0.0
    target_spe = 0.0

    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    for thresh in thresholds:
        pred_pos = y_scores >= thresh

        tp = np.sum((y_true == 1) & pred_pos)
        fp = np.sum((y_true == 0) & pred_pos)
        fn = np.sum((y_true == 1) & (~pred_pos))
        tn = np.sum((y_true == 0) & (~pred_pos))

        total = tp + tn + fp + fn
        acc = float(tp + tn) / total if total > 0 else 0.0
        sen = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
        spe = float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0

        # F1-score: 2TP / (2TP + FN + FP)
        f1_denom = 2 * tp + fn + fp
        f1 = (2.0 * tp / f1_denom) if f1_denom > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
            best_acc = acc
            best_sen = sen
            best_spe = spe

        # Youden J statistic
        j_val = sen + spe - 1.0
        if j_val > best_j:
            best_j = j_val
            youden_thresh = float(thresh)
            youden_sen = sen
            youden_spe = spe
            youden_acc = acc

        # SPE >= 0.90
        if spe >= 0.90 and thresh <= thresh_spe90:
            thresh_spe90 = float(thresh)
            sen_spe90 = sen
            spe_spe90 = spe

        # SPE >= 0.95
        if spe >= 0.95 and thresh <= thresh_spe95:
            thresh_spe95 = float(thresh)
            sen_spe95 = sen
            spe_spe95 = spe

        # SPE >= 0.85
        if spe >= 0.85 and sen >= sen_spe85:
            thresh_spe85 = float(thresh)
            sen_spe85 = sen
            spe_spe85 = spe

        # SEN >= 0.93
        if sen >= 0.93 and spe >= spe_sen93:
            thresh_sen93 = float(thresh)
            sen_sen93 = sen
            spe_sen93 = spe

        # Joint Target (SEN >= 93% and SPE >= 85%)
        if sen >= 0.93 and spe >= 0.85:
            target_met = True
            if (sen + spe) > (target_sen + target_spe):
                target_thresh = float(thresh)
                target_sen = sen
                target_spe = spe

    return {
        "auc": round(auc, 2),
        "f1": round(best_f1 * 100.0, 2),
        "acc": round(best_acc * 100.0, 2),
        "sen": round(best_sen * 100.0, 2),
        "spe": round(best_spe * 100.0, 2),
        "best_threshold": best_thresh,
        "youden_sen": round(youden_sen * 100.0, 2),
        "youden_spe": round(youden_spe * 100.0, 2),
        "youden_acc": round(youden_acc * 100.0, 2),
        "youden_threshold": youden_thresh,
        "sen_at_spe95": round(sen_spe95 * 100.0, 2),
        "spe_at_spe95": round(spe_spe95 * 100.0, 2),
        "threshold_spe95": thresh_spe95,
        "sen_at_spe90": round(sen_spe90 * 100.0, 2),
        "spe_at_spe90": round(spe_spe90 * 100.0, 2),
        "threshold_spe90": thresh_spe90,
        "sen_at_spe85": round(sen_spe85 * 100.0, 2),
        "spe_at_spe85": round(spe_spe85 * 100.0, 2),
        "threshold_spe85": thresh_spe85,
        "sen_at_sen93": round(sen_sen93 * 100.0, 2),
        "spe_at_sen93": round(spe_sen93 * 100.0, 2),
        "threshold_sen93": thresh_sen93,
        "target_met": target_met,
        "target_thresh": target_thresh,
        "target_sen": round(target_sen * 100.0, 2) if target_met else 0.0,
        "target_spe": round(target_spe * 100.0, 2) if target_met else 0.0,
    }
