"""Reference loss catalog (domain-agnostic, total, NumPy-vectorized).

Every function here is a "total" loss: it is defined for every real ``ŷ``.
Where a formula contains ``log``/``sqrt``/division, ``ŷ`` is clipped inside the
function so that finite-difference sweeps never produce NaN.

Convention: ``loss(ŷ, y)`` where ``ŷ`` is the prediction array and ``y`` is a
scalar target. The output has the same shape as ``ŷ``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

LossFn = Callable[[np.ndarray, float], np.ndarray]

_EPS = 1e-7


def _clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(p, _EPS, 1.0 - _EPS)


# --------------------------------------------------------------------------- #
# Binary probability losses (ŷ = p ∈ (0,1), y ∈ {0,1})                        #
# --------------------------------------------------------------------------- #

def bce(ŷ: np.ndarray, y: float) -> np.ndarray:
    """Binary cross-entropy. Proper scoring rule; the default choice."""
    p = _clip_prob(ŷ)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def focal(ŷ: np.ndarray, y: float, gamma: float = 2.0) -> np.ndarray:
    """Focal loss: down-weights easy (correct) examples by ``(1-p_t)^gamma``.

    ``p_t = p`` when ``y=1`` else ``1-p``.
    """
    p = _clip_prob(ŷ)
    pt = p if y > 0.5 else 1.0 - p
    return -((1.0 - pt) ** gamma) * np.log(pt)


def label_smoothing(ŷ: np.ndarray, y: float, smoothing: float = 0.1) -> np.ndarray:
    """BCE with label smoothing. Target becomes ``y* = y(1-s) + s/2``.

    Removes the ``log(0)`` singularity at the correct extreme and penalizes
    overconfidence (L(ŷ→1) stays finite).
    """
    p = _clip_prob(ŷ)
    y_star = y * (1.0 - smoothing) + smoothing / 2.0
    return -(y_star * np.log(p) + (1.0 - y_star) * np.log(1.0 - p))


def bimodal(ŷ: np.ndarray, y: float) -> np.ndarray:
    """4p(1-p) "bimodal" regularizer. Target-independent.

    Pathological: gradient is 0 at p=0.5 (no ambiguity-escape force) and it
    rewards *both* p→0 and p→1 equally (no directionality).
    """
    p = _clip_prob(ŷ)
    return 4.0 * p * (1.0 - p)


def contrastive_margin(
    ŷ: np.ndarray,
    y: float,
    hi: float = 0.85,
    lo: float = 0.15,
    weight: float = 3.0,
) -> np.ndarray:
    """Oracle-gated margin regularizer: force p ≥ hi where y=1, p ≤ lo where y=0."""
    p = _clip_prob(ŷ)
    if y > 0.5:
        return weight * np.maximum(hi - p, 0.0)
    return weight * np.maximum(p - lo, 0.0)


# --------------------------------------------------------------------------- #
# Binary logit losses (ŷ = z ∈ R, y ∈ {0,1})                                   #
# --------------------------------------------------------------------------- #

def bce_with_logits(ŷ: np.ndarray, y: float, temperature: float = 1.0) -> np.ndarray:
    """BCE on ``σ(T·z)``. High temperature sharpens decisions but saturates
    the confidently-wrong region (vanishing gradient for |z| ≫ 1/T)."""
    z = ŷ * temperature
    # numerically stable: log-sigmoid
    logp = -np.logaddexp(0.0, -z)          # log σ(z)
    logq = -np.logaddexp(0.0, z)           # log σ(-z)
    return -(y * logp + (1.0 - y) * logq)


# --------------------------------------------------------------------------- #
# Magnitude losses (ŷ = ŝ ∈ [0,∞), y = S* > 0)                                 #
# --------------------------------------------------------------------------- #

def l1(ŷ: np.ndarray, y: float) -> np.ndarray:
    """Absolute error. Constant pull (±1) everywhere, kink at optimum."""
    return np.abs(ŷ - y)


def l2(ŷ: np.ndarray, y: float) -> np.ndarray:
    """Squared error. Linear pull, gradient → 0 as error → 0."""
    return (ŷ - y) ** 2


def log_magnitude_l1(ŷ: np.ndarray, y: float) -> np.ndarray:
    """Log-magnitude L1. Curvature explodes as ŷ→0 (silence instability)."""
    s = np.clip(ŷ, _EPS, None)
    t = max(y, _EPS)
    return np.abs(np.log(s) - np.log(t))


def spectral_convergence(ŷ: np.ndarray, y: float) -> np.ndarray:
    """Normalized magnitude error: |ŷ - y| / y."""
    return np.abs(ŷ - y) / max(y, _EPS)


# --------------------------------------------------------------------------- #
# Score/margin losses (ŷ = d ∈ R, y = 0 reference)                             #
# --------------------------------------------------------------------------- #

def rank_hinge(ŷ: np.ndarray, y: float, margin: float = 0.1) -> np.ndarray:
    """Hinge on score difference ``d + margin``. Dead zone (gradient 0) for
    every ``d < -margin``."""
    return np.maximum(ŷ + margin, 0.0)


def rank_log_softmax(ŷ: np.ndarray, y: float, tau: float = 0.1) -> np.ndarray:
    """Smooth log-softmax ranking. Gradient is nonzero everywhere (no dead zone)."""
    logits = np.stack([-ŷ, np.zeros_like(ŷ)], axis=-1) / tau
    logp = -_log_softmax(logits, axis=-1)[..., 0]
    return logp


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return x - m - np.log(np.sum(e, axis=axis, keepdims=True))


# --------------------------------------------------------------------------- #
# Aggregate losses (NOT 1-D scannable — see docs/loss_catalog.md)              #
# --------------------------------------------------------------------------- #

def soft_dice(ŷ: np.ndarray, y: float) -> float:
    """Soft Dice over a whole batch: ``1 - 2·Σ(p·y)/(Σp + Σy)``.

    Aggregate losses couple all samples; they must be diagnosed by their
    per-sample gradient formula rather than a 1-D output sweep."""
    p = _clip_prob(np.asarray(ŷ, dtype=float))
    yv = np.full_like(p, y)
    num = 2.0 * np.sum(p * yv) + _EPS
    den = np.sum(p) + np.sum(yv) + _EPS
    return float(1.0 - num / den)


def tversky(ŷ: np.ndarray, y: float, alpha: float = 0.3, beta: float = 0.7) -> float:
    """Tversky loss (generalized Dice). ``beta > alpha`` weights false
    negatives (missed targets / leakage) more heavily."""
    p = _clip_prob(np.asarray(ŷ, dtype=float))
    yv = np.full_like(p, y)
    tp = np.sum(p * yv)
    fn = np.sum((1.0 - p) * yv)
    fp = np.sum(p * (1.0 - yv))
    return float(1.0 - (tp + _EPS) / (tp + alpha * fn + beta * fp + _EPS))
