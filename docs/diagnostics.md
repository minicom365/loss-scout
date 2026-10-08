# Diagnostics — thresholds, rationale, and reading the report

Each `Diagnosis` reports four metrics plus a verdict (`healthy` /
`pathological`). This page documents the thresholds and how to interpret them.

## Metric 1 — Wrong-region mean gradient

**What:** mean of $|\partial L/\partial \hat y|$ over the region where the
prediction is *wrong* for a given target.

**Thresholds** (both must fail to flag):

| constant | value | meaning |
|---|---|---|
| `VANISH_ABS` | `1e-3` | absolute gradient below this = no pull at all |
| `VANISH_REL` | `1e-3` | relative to the sweep's gradient peak |

**Why relative too:** a loss scaled by a tiny weight has small gradients
everywhere, but its *shape* may still be fine. Relative comparison isolates
*shape* pathologies from *scaling* ones.

**Example:** `bce_with_logits` at temperature 8 saturates the confidently-wrong
region (`|z|>3`): gradient → 0 there even though the gradient near 0 is large.

## Metric 2 — Ambiguity-escape force

**What:** $|\partial L/\partial \hat y|$ at the decision boundary (the "most
ambiguous" prediction, e.g. $p=0.5$ for binary, $z=0$ for logits).

**Threshold:** `AMBIGUITY_ABS = 1e-2`.

**Why:** a model initialized near the boundary (or pushed there by symmetric
targets) needs a first-order force to leave it. The `4p(1-p)` bimodal
regularizer has **exactly zero** force at $p=0.5$ — it cannot move a model
stuck at the uniform output.

## Metric 3 — Trivial-solution response

**What:** the loss at degenerate *constant* outputs (e.g. all-0, all-0.5,
all-1), compared to the sweep minimum.

**Threshold:** `TRIVIAL_TOL = 1e-2` (relative to the loss span).

**Why:** a loss that is minimized by "ignore the input and output a constant"
invites shortcuts. BCE penalizes $p\to0$ when the target is 1 with
$-\log p \to \infty$; the bimodal regularizer instead rewards **both** $p\to0$
and $p\to1$ equally, which is how an attenuator (all-zero mask) collapse slips
through.

## Metric 4 — Curvature explosion

**What:** peak of $|\partial^2 L/\partial \hat y^2|$ across the sweep.

**Threshold:** `CURVATURE_REL = 1e6` (curvature peak vs gradient peak).

**Why:** unbounded curvature means a small step in prediction space produces a
huge change in gradient → optimizer oscillation / NaN. `log`-magnitude losses
diverge as the prediction → 0 (the silence-frame instability).

## Reading the verdict

- `healthy` — no flag fired. The loss has a sane output-space shape; still
  verify it upper-bounds your real metric (framework step 5).
- `pathological` — at least one flag fired. Read the flag text: it names the
  metric and the value. Consider a repair (label smoothing, temperature
  reduction, switching hinge→log-softmax, adding eps, …).

## Tuning thresholds

If a domain legitimately needs looser/tighter thresholds, override the module
constants in `loss_scout/metrics.py` or pass a custom `Domain`. The constants are
the *only* domain-specific knob in the library.
