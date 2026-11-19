"""Classification metrics + CLUSTER bootstrap confidence intervals.

Why cluster: the 3 frames of a burst are near-copies, so they are not 3 independent test
samples. Resampling single images would pretend we have ~3x more independent evidence than we
do and give intervals that are too narrow. We resample whole SEQUENCES (bursts) instead.
"""
import numpy as np


def conf_tensor(y, p, seq, K):
    """Per-sequence confusion matrices: shape (n_sequences, K, K), rows = true class, cols = predicted."""
    useq, inv = np.unique(seq, return_inverse=True)
    T = np.zeros((len(useq), K, K), dtype=np.int32)
    np.add.at(T, (inv, np.asarray(y), np.asarray(p)), 1)
    return T


def from_cm(cm, present=None):
    """accuracy, macro-F1, per-class recall / precision / F1 from one (K,K) confusion matrix.
    Macro-F1 averages over classes in `present` (classes that exist in the test set's ground truth)."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    sup, pred = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(sup > 0, tp / sup, np.nan)
        precision = np.where(pred > 0, tp / pred, 0.0)
        f1 = np.where(sup + pred > 0, 2 * tp / (sup + pred), 0.0)
    if present is None:
        present = sup > 0
    return dict(accuracy=tp.sum() / cm.sum(), macro_f1=float(f1[present].mean()),
                recall=recall, precision=precision, f1=f1, support=sup)


def draw_weights(n, rng):
    """Bootstrap multiplicities: how many times each of n clusters is drawn (sums to n)."""
    return np.bincount(rng.integers(0, n, n), minlength=n)


def bootstrap(T, present, n_boot, rng, weights=None):
    """Return arrays (accuracy, macro_f1) over n_boot resamples of the sequences in T."""
    n = T.shape[0]
    W = weights if weights is not None else np.stack([draw_weights(n, rng) for _ in range(n_boot)])
    acc, f1 = np.empty(len(W)), np.empty(len(W))
    flat = T.reshape(n, -1).astype(np.float64)
    for i, w in enumerate(W):
        cm = (w @ flat).reshape(T.shape[1:])
        m = from_cm(cm, present)
        acc[i], f1[i] = m["accuracy"], m["macro_f1"]
    return acc, f1


def ci(a, level=0.95):
    lo, hi = np.percentile(a, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)
