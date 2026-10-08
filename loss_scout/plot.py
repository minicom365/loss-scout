"""Simple matplotlib visualization of a loss-landscape scan.

Produces a 2×2 figure:

- **top-left**: loss ``L(ŷ)`` per target
- **top-right**: gradient ``∂L/∂ŷ`` (the pull force) per target
- **bottom-left**: curvature ``∂²L/∂ŷ²`` per target
- **bottom-right**: textual verdict — flags, ambiguity force, trivial response

``matplotlib`` is an optional dependency: ``pip install "loss-scout[plot]"``.
"""

from __future__ import annotations

from typing import Optional

from .metrics import Diagnosis
from .scanner import ScanResult


def plot_scan(
    result: ScanResult,
    diagnosis: Optional[Diagnosis] = None,
    path: Optional[str] = None,
    show: bool = False,
    figsize: tuple = (11, 7),
    dpi: int = 150,
):
    """Plot a scan result (and optionally its diagnosis) and save/display it.

    Parameters
    ----------
    result:
        The :class:`~loss_scout.scanner.ScanResult` to plot.
    diagnosis:
        Optional :class:`~loss_scout.metrics.Diagnosis`; if given, the verdict
        and flags are rendered as text.
    path:
        If given, save the figure to this file (``.png``/``.svg``/``.pdf``).
    show:
        If True, open an interactive window (needs a display backend).
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_loss, ax_grad = axes[0]
    ax_curv, ax_text = axes[1]

    grid = result.grid

    for t, l in result.losses.items():
        ax_loss.plot(grid, l, label=f"target={t:g}")
    for t, g in result.grads.items():
        ax_grad.plot(grid, g, label=f"target={t:g}")
    for t, c in result.curvs.items():
        ax_curv.plot(grid, c, label=f"target={t:g}")

    ax_loss.set_title(f"loss L(ŷ) — {result.loss_name}")
    ax_loss.set_xlabel("prediction ŷ")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()

    ax_grad.set_title("gradient ∂L/∂ŷ (pull force)")
    ax_grad.set_xlabel("prediction ŷ")
    ax_grad.set_ylabel("∂L/∂ŷ")
    ax_grad.axhline(0, color="grey", linewidth=0.6)
    ax_grad.legend()

    ax_curv.set_title("curvature ∂²L/∂ŷ²")
    ax_curv.set_xlabel("prediction ŷ")
    ax_curv.set_ylabel("∂²L/∂ŷ²")
    ax_curv.axhline(0, color="grey", linewidth=0.6)
    ax_curv.legend()

    # Text panel: verdict.
    ax_text.axis("off")
    if diagnosis is not None:
        lines = [f"verdict: {diagnosis.verdict}", ""]
        if diagnosis.flags:
            lines.append("flags:")
            lines += [f"  ⚠ {f}" for f in diagnosis.flags]
        else:
            lines.append("flags: none (healthy)")
        if diagnosis.ambiguity_force is not None:
            lines.append(f"ambiguity-escape force: {diagnosis.ambiguity_force:.3g}")
        lines.append(f"gradient peak |g|: {diagnosis.grad_peak:.3g}")
        lines.append(f"max curvature |h|: {diagnosis.max_curvature:.3g}")
        if diagnosis.trivial_response:
            lines.append("trivial response: " + ", ".join(
                f"ŷ={k:g}→{v:.3g}" for k, v in diagnosis.trivial_response.items()
            ))
        ax_text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9,
                     family="monospace")
    else:
        ax_text.text(0.0, 1.0, result.loss_name, va="top", ha="left", fontsize=12)

    fig.suptitle(f"loss-scout scan — {result.domain.name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if path:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig
