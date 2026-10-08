"""Example: scan regression/magnitude losses.

Run:  python examples/magnitude_losses.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loss_scout import scan, diagnose, format_report
from loss_scout import scanner as S
from loss_scout import losses as L

if __name__ == "__main__":
    dom = S.magnitude(lo=0.0, hi=4.0, target=1.0)
    diags = [
        diagnose(scan(L.l1, dom, loss_name="L1")),
        diagnose(scan(L.l2, dom, loss_name="L2/MSE")),
        diagnose(scan(L.log_magnitude_l1, dom, loss_name="log-magnitude L1")),
        diagnose(scan(L.spectral_convergence, dom, loss_name="spectral convergence")),
    ]
    print(format_report(diags))
