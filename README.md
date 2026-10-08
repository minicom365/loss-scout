# Loss-Scout 🔍

**Model-free loss landscape diagnostics — scan any loss function *before* you train.**

Loss-Scout sweeps a loss formula's **output space** and reports four domain-agnostic
pathologies in ~0.1 s on CPU, with no model and no data:

| # | Pathology | What Loss-Scout detects |
|---|---|---|
| 1 | Vanishing gradient | $\|\partial L/\partial\hat y\| \to 0$ in the wrong-answer region |
| 2 | Ambiguity collapse | zero force at the decision boundary (model stuck at a uniform output) |
| 3 | Trivial-solution trap | a constant output nearly minimizes the loss |
| 4 | Curvature explosion | unbounded $\partial^2 L/\partial\hat y^2$ → NaN / instability |

> The chain rule splits the parameter gradient into
> $\nabla_\theta L = (\partial L/\partial\hat y)\cdot(\partial\hat y/\partial\theta)$.
> The first factor is **fully determined by the loss formula alone** — that is
> exactly what Loss-Scout scans.

## Why

Designing a loss by stacking terms and training for hours is expensive and
fragile. Loss-Scout moves the failure detection to *before* training: if a loss
has a dead zone or a shortcut in prediction space, no architecture can escape it.

## Install

```bash
git clone https://github.com/<your-user>/loss-scout.git
cd loss-scout
pip install -e .
```

Requires Python ≥ 3.9 and NumPy only (`matplotlib` optional, for `--plot`).

## Quickstart

```bash
loss-scout catalog                        # list losses & domains
loss-scout scan --loss bce --domain binary_probability
loss-scout scan --loss focal --domain binary_probability --json
loss-scout scan --loss bimodal --domain binary_probability --plot bimodal.png
```

```python
from loss_scout import scan, diagnose, format_report
from loss_scout import scanner as S
from loss_scout import losses as L

d = diagnose(scan(L.bce, S.binary_probability(), loss_name="bce"))
print(format_report(d))
```

Example output (abridged):

```text
### bce — binary probability p∈(0,1) — healthy
| wrong-region mean |g| | t=0:1668  t=1:1690 |
| ambiguity-escape force | 2.00 |
| max curvature |h|       | 1e12  (finite-diff at the log edge) |
```

## Visualization

`--plot PATH` renders the loss, gradient, and curvature curves together with the
verdict (requires `pip install "loss-scout[plot]"`):

```bash
loss-scout scan --loss log_magnitude_l1 --domain magnitude --plot logmag.png
```

```python
from loss_scout import scan, diagnose, plot_scan
from loss_scout import scanner as S
from loss_scout import losses as L

r = scan(L.bce, S.binary_probability(), loss_name="bce")
plot_scan(r, diagnose(r), path="bce.png")   # 2×2: loss / gradient / curvature / verdict
```

## Repository layout

```
loss-scout/
├── loss_scout/
│   ├── scanner.py    # output-space sweep (NumPy finite differences)
│   ├── metrics.py    # 4 diagnostics + thresholds + verdict
│   ├── losses.py     # reference loss catalog (total, vectorized)
│   ├── report.py     # text / JSON / Markdown formatting
│   ├── plot.py       # matplotlib landscape figure
│   └── cli.py        # command-line interface
├── examples/         # binary, magnitude, scale-invariance sweeps
├── docs/
│   ├── guidebook.md      # the 5-step loss engineering framework
│   ├── diagnostics.md    # thresholds & interpretation
│   └── loss_catalog.md   # family catalog + decision heuristic
└── tests/            # pytest
```

## Documentation

- **[Guidebook](docs/guidebook.md)** — the idea, the 5-step framework, quickstart.
- **[Diagnostics](docs/diagnostics.md)** — thresholds and how to read the report.
- **[Loss catalog](docs/loss_catalog.md)** — families, pathologies, and a
  decision heuristic for choosing a loss.

## Scope

Loss-Scout diagnoses the **output-space** shape of a loss. It is a pre-training
gate, not a replacement for post-hoc parameter-space loss-surface plots, and it
does not diagnose **aggregate** losses (Dice/Tversky) via 1-D sweeps — those are
handled with their per-sample gradient formula (see the catalog).

## License

[MIT](LICENSE)
