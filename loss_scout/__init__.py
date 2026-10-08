"""Loss-Scout — model-free loss landscape diagnostics.

Scan any loss function's output-space gradient (pull force) and curvature
(stability) *before* training a model, using only the loss formula itself.

The core insight: the chain rule splits a model's parameter gradient into

    dL/dθ = (dL/dŷ) · (dŷ/dθ)

The second factor belongs to the architecture, but the first factor
``dL/dŷ`` (and its derivative ``d²L/dŷ²``) is fully determined by the loss
formula alone. Loss-Scout sweeps the prediction space ``ŷ`` and reports four
domain-agnostic pathologies:

1. **Vanishing gradient** in the "wrong answer" region (model gets stuck).
2. **Zero ambiguity-escape force** at the decision boundary (model stays at a
   uniform / uninformative output).
3. **Trivial-solution trap** (a constant output nearly minimizes the loss).
4. **Curvature explosion** (unbounded second derivative → NaN / instability).
"""

from .scanner import Domain, ScanResult, scan
from .metrics import Diagnosis, diagnose
from .report import format_report, to_dict, to_json, to_markdown
from .plot import plot_scan

__all__ = [
    "Domain",
    "ScanResult",
    "scan",
    "Diagnosis",
    "diagnose",
    "format_report",
    "to_dict",
    "to_json",
    "to_markdown",
    "plot_scan",
]

__version__ = "0.1.0"
