import numpy as np

def jaccard(tp, fp, fn):
    """Calculate Jaccard similarity (Intersection over Union)."""
    denominator = fp + tp + fn
    if denominator == 0:
        return 1
    return round(tp / denominator, 2)

def dice_score(tp, fp, fn):
    """Calculate Dice Score (F1 Score)."""
    denominator = fp + 2 * tp + fn
    if denominator == 0:
        return 1
    return round(2 * tp / denominator, 2)

def precision(tp, fp):
    """Calculate Precision."""
    denominator = tp + fp
    if denominator == 0:
        return 1
    return round(tp / denominator, 2)

def sensitivity(tp, fn):
    """Calculate Sensitivity (Recall)."""
    denominator = tp + fn
    if denominator == 0:
        return 1
    return round(tp / denominator, 2)

def accuracy(tp, total):
    """Calculate Accuracy."""
    total_tp = np.sum(tp)
    if total == 0:
        return 1
    return round(total_tp / total, 2)
