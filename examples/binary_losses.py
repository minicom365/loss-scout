"""Example: scan common binary-decision losses.

Run:  python examples/binary_losses.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losslens import scan, diagnose, format_report, to_markdown
from losslens import scanner as S
from losslens import losses as L

if __name__ == "__main__":
    dom = S.binary_probability()
    diags = []
    for name, fn in [
        ("bce", L.bce),
        ("label_smoothing(0.1)", L.label_smoothing),
        ("focal(gamma=2)", L.focal),
        ("bimodal 4p(1-p)", L.bimodal),
        ("contrastive_margin", L.contrastive_margin),
    ]:
        diags.append(diagnose(scan(fn, dom, loss_name=name)))

    print(format_report(diags))

    # Logit-space temperature comparison.
    dom_z = S.binary_logit()
    print("--- logit space (temperature effect) ---")
    for name, fn in [
        ("bce_with_logits(T=1)", L.bce_with_logits),
        ("bce_with_logits(T=8)", lambda z, y: L.bce_with_logits(z, y, temperature=8.0)),
    ]:
        print(to_markdown(diagnose(scan(fn, dom_z, loss_name=name))))
