"""Shared confidence-based rejection so an unrecognized sign reports
UNKNOWN_LABEL instead of whichever trained class happens to score highest.
Used by both baselines wherever a live prediction turns into a robot
command or gets shown to a user.
"""
from common.config import UNKNOWN_LABEL


def top_prediction(probs, classes, threshold):
    """probs: per-class probabilities (softmax output / predict_proba row),
    in the same order as `classes`.

    Returns (label, confidence): the highest-probability class, or
    (UNKNOWN_LABEL, confidence) if that probability is below `threshold`.
    """
    idx = max(range(len(classes)), key=lambda i: probs[i])
    conf = float(probs[idx])
    label = classes[idx] if conf >= threshold else UNKNOWN_LABEL
    return label, conf
