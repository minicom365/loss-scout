"""Domain-agnostic diagnostic metrics and verdict logic.

Every metric below is computed from the output-space sweep alone. Thresholds
are intentionally loose defaults; they are the *only* domain-specific knob and
are documented in ``docs/diagnostics.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .scanner import ScanResult

# --- thresholds (see docs/diagnostics.md for rationale) ---
VANISH_ABS = 1e-3          # |g| below this counts as "dead"
VANISH_DEAD_FRAC = 0.5     # flag if this fraction of the wrong region is dead
AMBIGUITY_ABS = 1e-2       # |g| at the decision boundary below this = no escape
TRIVIAL_TOL = 1e-2         # loss within this (relative) of the sweep minimum = trap
ZERO_SING_RATIO = 1e3      # |h| near ŷ→0 vs median |h|, above this = 0-boundary singularity


@dataclass
class Diagnosis:
    """Structured diagnostic verdict for one loss/domain pair."""

    loss_name: str
    domain_name: str
    wrong_region_grad: Dict[float, float] = field(default_factory=dict)
    wrong_region_peak: float = 0.0
    ambiguity_force: Optional[float] = None
    trivial_response: Dict[float, float] = field(default_factory=dict)
    min_loss: float = 0.0
    max_curvature: float = 0.0
    grad_peak: float = 0.0
    flags: List[str] = field(default_factory=list)
    verdict: str = "healthy"

    def has_flags(self) -> bool:
        return len(self.flags) > 0


def diagnose(result: ScanResult) -> Diagnosis:
    """Compute the four pathology metrics for a scan result."""
    dom = result.domain
    d = Diagnosis(loss_name=result.loss_name, domain_name=dom.name)

    # Gradient peak across every target.
    d.grad_peak = max(float(np.max(np.abs(g))) for g in result.grads.values())

    # 1. Vanishing gradient: fraction of the wrong region that is "dead".
    if dom.is_wrong is not None:
        for t in dom.targets:
            mask = dom.is_wrong(result.grid, t)
            if mask.any():
                g = np.abs(result.grads[t][mask])
                d.wrong_region_grad[t] = float(np.mean(g))
                dead_frac = float(np.mean(g < VANISH_ABS))
                if dead_frac > VANISH_DEAD_FRAC:
                    d.flags.append(
                        f"vanishing gradient in wrong region "
                        f"(target={t:g}, {dead_frac:.0%} dead)"
                    )
    if d.wrong_region_grad:
        d.wrong_region_peak = max(d.wrong_region_grad.values())

    # 2. Ambiguity-escape force at the decision boundary.
    if dom.ambiguity is not None:
        # Pick the target whose "wrong" side flips at the boundary (largest target).
        t_ref = max(dom.targets)
        idx = int(np.argmin(np.abs(result.grid - dom.ambiguity)))
        force = float(np.abs(result.grads[t_ref][idx]))
        d.ambiguity_force = force
        if force < AMBIGUITY_ABS:
            d.flags.append(f"zero ambiguity-escape force (|g|={force:.3g} at ŷ={dom.ambiguity:g})")

    # 3. Trivial-solution trap: a *constant* output that is near-minimal for
    #    EVERY target simultaneously. Meaningful only with ≥ 2 targets (with a
    #    single target, a constant near it is just the correct answer).
    t_ref = max(dom.targets)
    d.min_loss = float(np.min(result.losses[t_ref]))
    d.trivial_response = {tv: result.loss_at(tv, t_ref) for tv in dom.trivial}
    if len(dom.targets) >= 2:
        for tv in dom.trivial:
            near_min_for_all = True
            for t in dom.targets:
                losses_t = result.losses[t]
                mn = float(np.min(losses_t))
                span = float(np.max(losses_t) - mn)
                lv = result.loss_at(tv, t)
                if span > 0 and (lv - mn) / span >= TRIVIAL_TOL:
                    near_min_for_all = False
                    break
            if near_min_for_all:
                d.flags.append(
                    f"trivial-solution trap "
                    f"(constant ŷ={tv:g} near-minimal for all targets)"
                )

    # 4. Zero-boundary singularity (log-type blow-up at a legitimate
    #    prediction such as silence). Only checked when the sweep includes 0.
    h_all = np.concatenate([c.ravel() for c in result.curvs.values()])
    d.max_curvature = float(np.max(np.abs(h_all)))
    if not np.isfinite(d.max_curvature):
        d.flags.append("curvature explosion (non-finite second derivative)")
    elif dom.lo <= 0.0:
        n = len(result.grid)
        near0 = float(np.mean(np.abs(h_all[: max(2, n // 50)])))
        med = float(np.median(np.abs(h_all)))
        if med > 0 and near0 / med > ZERO_SING_RATIO:
            d.flags.append(
                f"singularity at zero boundary "
                f"(|h| near ŷ→0 = {near0:.3g} vs median {med:.3g})"
            )

    d.verdict = "pathological" if d.has_flags() else "healthy"
    return d
