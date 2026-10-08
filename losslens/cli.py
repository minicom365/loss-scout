"""Command-line interface: ``python -m losslens`` / ``losslens``.

Examples
--------
    losslens catalog
    losslens scan --loss bce --domain binary_probability
    losslens scan --loss bce_with_logits --domain binary_logit --json
    losslens scan --loss focal --domain binary_probability --markdown
"""

from __future__ import annotations

import argparse
import sys

from . import losses as L
from . import scanner as S
from .metrics import diagnose
from .report import format_report, to_json, to_markdown

_LOSS_REGISTRY = {
    "bce": L.bce,
    "focal": L.focal,
    "label_smoothing": L.label_smoothing,
    "bimodal": L.bimodal,
    "contrastive_margin": L.contrastive_margin,
    "bce_with_logits": L.bce_with_logits,
    "l1": L.l1,
    "l2": L.l2,
    "log_magnitude_l1": L.log_magnitude_l1,
    "spectral_convergence": L.spectral_convergence,
    "rank_hinge": L.rank_hinge,
    "rank_log_softmax": L.rank_log_softmax,
}

_DOMAIN_REGISTRY = {
    "binary_probability": S.binary_probability,
    "binary_logit": S.binary_logit,
    "magnitude": S.magnitude,
    "score": S.score,
}


def _cmd_catalog(args: argparse.Namespace) -> int:
    print("Losses:")
    for name in _LOSS_REGISTRY:
        print(f"  {name}")
    print("\nDomains:")
    for name in _DOMAIN_REGISTRY:
        print(f"  {name}")
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    loss_fn = _LOSS_REGISTRY[args.loss]
    domain = _DOMAIN_REGISTRY[args.domain]()
    result = S.scan(loss_fn, domain, loss_name=args.loss)
    d = diagnose(result)
    if args.json:
        print(to_json(d))
    elif args.markdown:
        print(to_markdown(d))
    else:
        print(format_report(d))
    return 1 if d.has_flags() else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="losslens", description="Model-free loss landscape diagnostics.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("catalog", help="list available losses and domains").set_defaults(func=_cmd_catalog)

    sp = sub.add_parser("scan", help="scan a loss over a domain")
    sp.add_argument("--loss", required=True, choices=sorted(_LOSS_REGISTRY))
    sp.add_argument("--domain", required=True, choices=sorted(_DOMAIN_REGISTRY))
    sp.add_argument("--json", action="store_true", help="emit JSON")
    sp.add_argument("--markdown", action="store_true", help="emit a Markdown block")
    sp.set_defaults(func=_cmd_scan)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
