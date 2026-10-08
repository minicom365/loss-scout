# LossLens 🔍

**Model-free loss landscape diagnostics — scan any loss function *before* you train.**

LossLens sweeps a loss formula's **output space** and reports four domain-agnostic
pathologies in ~0.1 s on CPU, with no model and no data:

| # | Pathology | What LossLens detects |
|---|---|---|
| 1 | Vanishing gradient | $\|\partial L/\partial\hat y\| \to 0$ in the wrong-answer region |
| 2 | Ambiguity collapse | zero force at the decision boundary (model stuck at a uniform output) |
| 3 | Trivial-solution trap | a constant output nearly minimizes the loss |
| 4 | Curvature explosion | unbounded $\partial^2 L/\partial\hat y^2$ → NaN / instability |

> The chain rule splits the parameter gradient into
> $\nabla_\theta L = (\partial L/\partial\hat y)\cdot(\partial\hat y/\partial\theta)$.
> The first factor is **fully determined by the loss formula alone** — that is
> exactly what LossLens scans.

## Why

Designing a loss by stacking terms and training for hours is expensive and
fragile. LossLens moves the failure detection to *before* training: if a loss
has a dead zone or a shortcut in prediction space, no architecture can escape it.

## Install

```bash
git clone https://github.com/<your-user>/losslens.git
cd losslens
pip install -e .
```

Requires Python ≥ 3.9 and NumPy only.

## Quickstart

```bash
losslens catalog                        # list losses & domains
losslens scan --loss bce --domain binary_probability
losslens scan --loss focal --domain binary_probability --json
```

```python
from losslens import scan, diagnose, format_report
from losslens import scanner as S
from losslens import losses as L

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

## Repository layout

```
losslens/
├── losslens/
│   ├── scanner.py    # output-space sweep (NumPy finite differences)
│   ├── metrics.py    # 4 diagnostics + thresholds + verdict
│   ├── losses.py     # reference loss catalog (total, vectorized)
│   ├── report.py     # text / JSON / Markdown formatting
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

LossLens diagnoses the **output-space** shape of a loss. It is a pre-training
gate, not a replacement for post-hoc parameter-space loss-surface plots, and it
does not diagnose **aggregate** losses (Dice/Tversky) via 1-D sweeps — those are
handled with their per-sample gradient formula (see the catalog).

## License

[MIT](LICENSE)
