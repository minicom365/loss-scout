"""Minimal tests for the scanner and diagnostics."""

import numpy as np

from losslens import scanner as S
from losslens import losses as L
from losslens.metrics import diagnose


def test_bce_is_healthy():
    d = diagnose(S.scan(L.bce, S.binary_probability(), loss_name="bce"))
    assert d.verdict == "healthy"
    # BCE has a nonzero ambiguity-escape force at p=0.5.
    assert d.ambiguity_force is not None and d.ambiguity_force > 0.1
    # p→1 (correct extreme) should be near-minimal loss.
    assert d.trivial_response[1 - 1e-6] < 1e-4


def test_bimodal_is_pathological():
    d = diagnose(S.scan(L.bimodal, S.binary_probability(), loss_name="bimodal"))
    assert d.verdict == "pathological"
    assert any("ambiguity" in f or "trivial" in f for f in d.flags)


def test_log_magnitude_curvature_explodes():
    d = diagnose(S.scan(L.log_magnitude_l1, S.magnitude(), loss_name="logmag"))
    assert d.max_curvature > 1e6
    assert d.verdict == "pathological"
    assert any("zero boundary" in f for f in d.flags)


def test_l2_and_l1_are_healthy():
    for fn in (L.l1, L.l2):
        d = diagnose(S.scan(fn, S.magnitude(), loss_name="reg"))
        assert d.verdict == "healthy"


def test_temperature_scales_pull_not_vanishing():
    dom = S.binary_logit()
    d1 = diagnose(S.scan(L.bce_with_logits, dom, loss_name="T=1"))
    d8 = diagnose(S.scan(lambda z, y: L.bce_with_logits(z, y, temperature=8.0), dom, loss_name="T=8"))
    # BCE-with-logits keeps a *constant* (non-vanishing) pull in the wrong
    # region; a higher temperature scales that pull up, it does not kill it.
    assert d1.verdict == "healthy"
    assert d8.verdict == "healthy"
    assert d8.wrong_region_peak > d1.wrong_region_peak
    assert d8.grad_peak > d1.grad_peak


def test_rank_hinge_dead_zone():
    dom = S.score(lo=-1.0, hi=1.0)
    d = diagnose(S.scan(L.rank_hinge, dom, loss_name="hinge"))
    # The wrong region (d<0) contains the dead zone (d<-margin).
    assert d.wrong_region_grad[0.0] < 0.5
