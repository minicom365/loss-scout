# Loss Catalog — families, prediction spaces, and known pathologies

The catalog below classifies loss families by their **prediction space** (the
$\hat y$ domain Loss-Scout sweeps) and records known pathologies. It is the
domain-agnostic counterpart of a project-specific loss audit.

## A. Binary probability — $\hat y = p \in (0,1)$, $y \in \{0,1\}$

| Loss | Proper? | Pathology | Verdict |
|---|---|---|---|
| `bce` | ✅ proper scoring rule | none (gold standard) | ✅ healthy |
| `label_smoothing` | ✅ | removes `log(0)` at the correct extreme; finite loss at $\hat y\to1$ | ✅ healthy |
| `focal` | ⚠️ (modulated) | down-weights easy examples; weaker boundary force | ✅/⚠️ for imbalance |
| `bimodal` ($4p(1-p)$) | ❌ | zero force at $p=0.5$; rewards $p\to0$ **and** $p\to1$ equally | ❌ pathological |
| `contrastive_margin` | ❌ (not a rule) | piecewise-constant gradient (kinks); needs oracle target | ⚠️ regularizer only |

## B. Binary logit — $\hat y = z \in \mathbb R$, $y \in \{0,1\}$

| Loss | Pathology |
|---|---|
| `bce_with_logits(T=1)` | linear pull, low curvature — optimal |
| `bce_with_logits(T≫1)` | keeps a constant pull (−T) in the wrong region; 8× curvature and bf16 output saturation |

**Guidance:** BCE-with-logits **does not** suffer vanishing gradient at any
temperature — its gradient in the wrong region is the constant `T·(σ(Tz) − y)`.
High temperature sharpens decisions but raises curvature (T²/4) and saturates
the output σ(Tz) to exactly 0/1 in low precision, so pair it with label
smoothing / clamping. Prefer T ≤ 4 unless you need very sharp decisions.

## C. Magnitude — $\hat y = \hat s \in [0,\infty)$, $y = S^* > 0$

| Loss | Gradient | Pathology |
|---|---|---|
| `l1` | constant $\pm1$ | kink at optimum (fine) |
| `l2` | linear $2(\hat s - S^*)$ | gradient→0 as error→0 (fine) |
| `log_magnitude_l1` | $1/\hat s$ | **curvature explosion as $\hat s\to0$** (silence frames) |
| `spectral_convergence` | constant (normalized) | kink only |

## D. Score / margin — $\hat y = d \in \mathbb R$

| Loss | Pathology |
|---|---|
| `rank_hinge` | dead zone (gradient 0) for all $d < -\text{margin}$ |
| `rank_log_softmax` | smooth, nonzero gradient everywhere — preferred |

## E. Scale / waveform — not 1-D scannable

**SI-SDR** is scale-invariant in the signal gain ($\gamma^2$ cancels in the
numerator and denominator), which neutralizes the "attenuation shortcut" — but
it is *phase-sensitive* (time-domain inner product) and has a $0/0$
discontinuity at an exactly-zero output. Diagnose it via a scale sweep
(`examples/scale_invariance.py`) rather than the standard 1-D scan.

## F. Aggregate losses — not 1-D scannable

**`soft_dice`** and **`tversky`** are *aggregate*: they sum over the whole
batch/minibatch, so a per-sample 1-D sweep is meaningless. Their per-sample
gradient is still computable in closed form:

$$
\frac{\partial \text{Dice}}{\partial p_i}
= -2\frac{y_i\, D - \text{num}}{D^2},
\qquad D = \sum p + \sum y,\; \text{num} = 2\sum p y
$$

Because they are bounded in $[0,1]$, contain no `log`, and align with F1-style
metrics, they are strong companions to BCE (per-bin precision + global
F1 alignment). Include them via the per-sample formula or evaluate them
aggregately — not with the 1-D scanner.

## Choosing a loss (decision heuristic)

1. **Binary decision** (mask, gate, presence): start with `bce`. Add
   `label_smoothing` to tame overconfidence and bf16 saturation. Add
   `soft_dice`/`tversky` as an aggregate F1-aligned companion.
2. **Regression** (magnitude, spectrum): `l1` or `l2`; avoid raw
   `log_magnitude_l1` on domains that reach zero.
3. **Ranking** (contrastive): prefer `rank_log_softmax` over `rank_hinge`.
4. **Never** use `bimodal` as a standalone anti-ambiguity term — it has zero
   force at the boundary and no directionality.
