"""Output-space loss landscape scanner.

A loss is treated as a function ``L(ŷ, y)`` where ``ŷ`` is the model's scalar
prediction and ``y`` is the target. Loss-Scout sweeps ``ŷ`` over a user-defined
:class:`Domain` while holding ``y`` fixed, and estimates the output-space
gradient and curvature with finite differences. No model, no dataset, no
autograd framework is required — only NumPy.

The loss function must be **total** (defined for every real ``ŷ``): clip
``ŷ`` inside the function when it feeds into ``log``/``sqrt``/division.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

# A vectorized loss: (y_hat_array, target_scalar) -> loss_array (same shape).
LossFn = Callable[[np.ndarray, float], np.ndarray]

# A predicate: which y_hat positions count as "wrong" for a given target.
WrongRegion = Callable[[np.ndarray, float], np.ndarray]


@dataclass
class Domain:
    """The prediction space to sweep, plus per-target diagnostics hints.

    Parameters
    ----------
    lo, hi:
        Inclusive sweep bounds of the prediction ``ŷ``.
    n:
        Number of grid points.
    targets:
        Representative target values to hold fixed during the sweep.
    name:
        Human-readable description of the prediction space.
    ambiguity:
        The "most ambiguous" prediction (decision boundary). If ``None``,
        the ambiguity-escape diagnostic is skipped.
    is_wrong:
        ``is_wrong(ŷ, target) -> bool array`` marking the region where a
        prediction is considered wrong for a given target.
    trivial:
        Degenerate constant outputs to probe for the trivial-solution trap.
    """

    lo: float
    hi: float
    n: int = 2001
    targets: Sequence[float] = (0.0, 1.0)
    name: str = "prediction space"
    ambiguity: Optional[float] = None
    is_wrong: Optional[WrongRegion] = None
    trivial: Sequence[float] = field(default_factory=lambda: (0.0, 0.5, 1.0))

    def grid(self) -> np.ndarray:
        return np.linspace(self.lo, self.hi, self.n)


def binary_probability(n: int = 2001) -> Domain:
    """Binary decision: prediction is a probability ``p ∈ (0, 1)``."""
    return Domain(
        lo=1e-6,
        hi=1 - 1e-6,
        n=n,
        targets=(0.0, 1.0),
        name="binary probability p∈(0,1)",
        ambiguity=0.5,
        is_wrong=lambda yh, t: (yh < 0.5) if t > 0.5 else (yh > 0.5),
        trivial=(0.5, 1e-6, 1 - 1e-6),
    )


def binary_logit(lo: float = -8.0, hi: float = 8.0, n: int = 2001) -> Domain:
    """Binary decision: prediction is a logit ``z ∈ [lo, hi]``."""
    return Domain(
        lo=lo,
        hi=hi,
        n=n,
        targets=(0.0, 1.0),
        name=f"binary logit z∈[{lo},{hi}]",
        ambiguity=0.0,
        is_wrong=lambda yh, t: (yh < 0.0) if t > 0.5 else (yh > 0.0),
        trivial=(0.0, lo, hi),
    )


def magnitude(
    lo: float = 0.0,
    hi: float = 4.0,
    target: float = 1.0,
    n: int = 4001,
) -> Domain:
    """Regression: prediction is an estimated magnitude ``ŝ ∈ [lo, hi]``."""
    return Domain(
        lo=lo,
        hi=hi,
        n=n,
        targets=(target,),
        name=f"magnitude ŝ∈[{lo},{hi}] (S*={target})",
        ambiguity=None,
        is_wrong=lambda yh, t: np.abs(yh - t) > 0.5,
        trivial=(0.0, target, hi),
    )


def score(lo: float = -1.0, hi: float = 1.0, n: int = 2001) -> Domain:
    """Ranking: prediction is a signed score/margin ``d ∈ [lo, hi]``."""
    return Domain(
        lo=lo,
        hi=hi,
        n=n,
        targets=(0.0,),
        name=f"score/margin d∈[{lo},{hi}]",
        ambiguity=0.0,
        is_wrong=lambda yh, t: yh < 0.0,
        trivial=(0.0, lo, hi),
    )


@dataclass
class ScanResult:
    """Sweep output for one loss over one domain."""

    domain: Domain
    loss_name: str
    loss_fn: LossFn
    grid: np.ndarray
    losses: Dict[float, np.ndarray]
    grads: Dict[float, np.ndarray]
    curvs: Dict[float, np.ndarray]

    def loss_at(self, value: float, target: float) -> float:
        """Loss at a single (possibly off-grid) prediction value."""
        return float(np.asarray(self.loss_fn(np.array([value]), target), dtype=float)[0])


def _central_diff(
    loss_fn: LossFn,
    x: np.ndarray,
    target: float,
    order: int,
    eps: float,
) -> np.ndarray:
    """Finite-difference derivative of ``loss_fn(·, target)`` at ``x``.

    Uses central differences in the interior and one-sided differences at the
    two endpoints. Because ``loss_fn`` is assumed total, evaluating slightly
    outside the sweep bounds is safe.
    """
    out = np.empty_like(x, dtype=float)
    if order == 1:
        out[1:-1] = (loss_fn(x[1:-1] + eps, target) - loss_fn(x[1:-1] - eps, target)) / (2 * eps)
        out[0] = (loss_fn(x[0] + eps, target) - loss_fn(x[0], target)) / eps
        out[-1] = (loss_fn(x[-1], target) - loss_fn(x[-1] - eps, target)) / eps
    elif order == 2:
        f = lambda v: loss_fn(v, target)  # noqa: E731
        out[1:-1] = (f(x[1:-1] + eps) - 2 * f(x[1:-1]) + f(x[1:-1] - eps)) / (eps**2)
        out[0] = (f(x[0] + 2 * eps) - 2 * f(x[0] + eps) + f(x[0])) / (eps**2)
        out[-1] = (f(x[-1]) - 2 * f(x[-1] - eps) + f(x[-1] - 2 * eps)) / (eps**2)
    else:
        raise ValueError("order must be 1 or 2")
    return out


def scan(
    loss_fn: LossFn,
    domain: Domain,
    loss_name: str = "loss",
    eps: float = 1e-4,
) -> ScanResult:
    """Sweep ``loss_fn`` over ``domain`` and estimate gradient/curvature.

    Parameters
    ----------
    loss_fn:
        Vectorized loss ``(ŷ_array, target) -> loss_array``. Must be total.
    domain:
        The prediction space to sweep (see :func:`binary_probability`,
        :func:`binary_logit`, :func:`magnitude`, :func:`score`).
    loss_name:
        Label used in reports.
    eps:
        Finite-difference step size.
    """
    grid = domain.grid()
    losses: Dict[float, np.ndarray] = {}
    grads: Dict[float, np.ndarray] = {}
    curvs: Dict[float, np.ndarray] = {}
    for t in domain.targets:
        losses[t] = np.asarray(loss_fn(grid, t), dtype=float)
        grads[t] = _central_diff(loss_fn, grid, t, order=1, eps=eps)
        curvs[t] = _central_diff(loss_fn, grid, t, order=2, eps=eps)
    return ScanResult(
        domain=domain,
        loss_name=loss_name,
        loss_fn=loss_fn,
        grid=grid,
        losses=losses,
        grads=grads,
        curvs=curvs,
    )
