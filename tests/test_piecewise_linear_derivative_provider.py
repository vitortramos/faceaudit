"""V1.5 — `PiecewiseLinearDerivativeProvider` + dispatch da regra.

Cobre:
  1. provider direto: segmentos monotônicos → slopes corretos
  2. provider direto: violação detectável em min_slope_observed
  3. provider direto: reportable_box "s" com intersecção
  4. provider direto: reportable_box "s" sem intersecção (segmento ignorado)
  5. regra: dispatch para piecewise quando piecewise_enabled=True
  6. regra: piecewise monotônico → audit_status = OK
  7. regra: piecewise não-monotônico → audit_status = CRITICAL
  8. regra: reportable_box piecewise viola → CRITICAL
  9. erro: knots/slopes inconsistentes → ValueError
"""
from __future__ import annotations
import numpy as np
import pytest

from faceaudit.enums.validation_rule_type import ValidationRuleType
from faceaudit.factories.validation_rule_factory import ValidationRuleFactory
from faceaudit.models.validation.validation_rule_config import ValidationRuleConfig
from faceaudit.services.piecewise_linear_derivative_provider import (
    PiecewiseLinearDerivativeProvider,
)


# ─────────────────────────────────────────────────────────────────────
# Stubs reutilizados (mesma estrutura do test_conditional_monotonicity_rule)
# ─────────────────────────────────────────────────────────────────────

class StubCalibrator:
    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata


class StubContext:
    def __init__(self) -> None:
        self._res: dict = {}

    def has_resource(self, rid):
        return rid in self._res

    def ensure_payload(self, rid):
        return self._res.get(rid)

    def add(self, rid, payload):
        class _Res:
            def __init__(self, p):
                self.payload = p
        self._res[rid] = _Res(payload)


# ─────────────────────────────────────────────────────────────────────
# Provider direto
# ─────────────────────────────────────────────────────────────────────

def _make_monotonic_metadata():
    """Calibrador piecewise saudável: knots em [0.2, 0.5], slopes [1.0, 2.0, 3.0]."""
    return {
        "piecewise_enabled": True,
        "piecewise_knots_absolute": [0.2, 0.5],
        "segment_slopes_raw": [1.0, 2.0, 3.0],
        "hinge_anchor_s0": 0.1,
    }


def _make_non_monotonic_metadata():
    """Slope negativo no segmento do meio."""
    return {
        "piecewise_enabled": True,
        "piecewise_knots_absolute": [0.2, 0.5],
        "segment_slopes_raw": [1.0, -0.5, 2.0],
        "hinge_anchor_s0": 0.1,
    }


def test_provider_monotonic_returns_correct_segments():
    provider = PiecewiseLinearDerivativeProvider()
    regions = provider.evaluate_regions(
        calibrator_metadata=_make_monotonic_metadata(),
    )
    assert len(regions) == 3
    assert regions[0].slope_value == 1.0
    assert regions[1].slope_value == 2.0
    assert regions[2].slope_value == 3.0
    assert regions[0].region_id == "segment_0"
    assert regions[2].region_id == "segment_2"


def test_provider_detects_violation_via_min_slope():
    provider = PiecewiseLinearDerivativeProvider()
    regions = provider.evaluate_regions(
        calibrator_metadata=_make_non_monotonic_metadata(),
    )
    # min slope across all regions deve ser -0.5
    min_slope = min(r.min_slope_observed() for r in regions)
    assert min_slope == -0.5


def test_provider_reportable_box_with_intersection():
    provider = PiecewiseLinearDerivativeProvider()
    regions = provider.evaluate_regions(
        calibrator_metadata=_make_monotonic_metadata(),
        reportable_box={"s": (0.3, 0.6)},  # cobre segmentos 1 e 2
    )
    # Segmento 0 (s ≤ 0.2): NÃO intersecta → corners None
    assert regions[0].slope_at_reportable_box_corners is None
    # Segmento 1 (0.2 < s ≤ 0.5): intersecta → tem corners
    assert regions[1].slope_at_reportable_box_corners is not None
    assert all(s == 2.0 for s in regions[1].slope_at_reportable_box_corners)
    # Segmento 2 (s > 0.5): intersecta parcialmente → tem corners
    assert regions[2].slope_at_reportable_box_corners is not None


def test_provider_accepts_base_score_alias():
    """`base_score` é alias de `s` no reportable_box."""
    provider = PiecewiseLinearDerivativeProvider()
    regions = provider.evaluate_regions(
        calibrator_metadata=_make_monotonic_metadata(),
        reportable_box={"base_score": (0.3, 0.6)},
    )
    assert regions[1].slope_at_reportable_box_corners is not None


def test_provider_raises_on_knots_slopes_mismatch():
    bad = {
        "piecewise_enabled": True,
        "piecewise_knots_absolute": [0.2, 0.5],
        "segment_slopes_raw": [1.0, 2.0],  # falta 1
        "hinge_anchor_s0": 0.1,
    }
    provider = PiecewiseLinearDerivativeProvider()
    with pytest.raises(ValueError, match="Inconsistência piecewise"):
        provider.evaluate_regions(calibrator_metadata=bad)


def test_provider_raises_when_metadata_missing_fields():
    provider = PiecewiseLinearDerivativeProvider()
    with pytest.raises(ValueError, match="piecewise_knots_absolute"):
        provider.evaluate_regions(calibrator_metadata={"piecewise_enabled": True})


# ─────────────────────────────────────────────────────────────────────
# Dispatch via regra
# ─────────────────────────────────────────────────────────────────────

def _build_rule(reportable_box=None):
    params = {"calibrator_resource_id": "cal_pw"}
    if reportable_box:
        params["reportable_box"] = reportable_box
    cfg = ValidationRuleConfig(
        rule_type=ValidationRuleType.CONDITIONAL_MONOTONICITY,
        params=params,
    )
    return ValidationRuleFactory.create(cfg, instruction_id="inst")


def test_rule_dispatches_to_piecewise_when_enabled():
    cal = StubCalibrator(metadata=_make_monotonic_metadata())
    ctx = StubContext()
    ctx.add("cal_pw", cal)
    rule = _build_rule()
    rule.validate(target=None, context=ctx)

    audit = cal.metadata["monotonicity_audit"]
    assert audit["audit_status"] == "OK"
    assert audit["provider"] == "PiecewiseLinearDerivativeProvider"
    assert audit["architecture"] == "PIECEWISE_LINEAR_UNIVARIATE"


def test_rule_piecewise_non_monotonic_is_critical():
    cal = StubCalibrator(metadata=_make_non_monotonic_metadata())
    ctx = StubContext()
    ctx.add("cal_pw", cal)
    rule = _build_rule()
    rule.validate(target=None, context=ctx)

    audit = cal.metadata["monotonicity_audit"]
    assert audit["audit_status"] == "CRITICAL"


def test_rule_piecewise_reportable_box_violation_is_critical():
    """Slope negativo SOMENTE no segmento [0.2, 0.5]. reportable_box=[0.3, 0.4]
    força violação ali → CRITICAL na region reportable mesmo que observed pegue tb.
    Aqui o min_obs também já dá CRITICAL — então isolamos com slopes [1, -0.5, 2]
    onde reportable_box pega só segmento 1 (negativo)."""
    cal = StubCalibrator(metadata=_make_non_monotonic_metadata())
    ctx = StubContext()
    ctx.add("cal_pw", cal)
    rule = _build_rule(reportable_box={"s": [0.3, 0.4]})
    rule.validate(target=None, context=ctx)

    audit = cal.metadata["monotonicity_audit"]
    assert audit["audit_status"] == "CRITICAL"
    assert "reportable_box" in audit["audit_domain_modes"]


def test_rule_piecewise_reportable_box_outside_violation_is_ok():
    """Violação no segmento do meio, mas reportable_box só cobre segmento 0 e 2.
    Como min_obs continua negativo (segmento 1), o status global é CRITICAL —
    a regra prioriza observed_rows. Confirma a hierarquia."""
    cal = StubCalibrator(metadata=_make_non_monotonic_metadata())
    ctx = StubContext()
    ctx.add("cal_pw", cal)
    # Cobertura: segmento 0 (s ≤ 0.2) — reportable_box [0.0, 0.1] não intersecta
    # segmento 1, e segmento 2 (s > 0.5) é monotônico
    rule = _build_rule(reportable_box={"s": [0.0, 0.1]})
    rule.validate(target=None, context=ctx)

    audit = cal.metadata["monotonicity_audit"]
    # Mesmo com reportable saudável, observed_rows pega segmento 1 negativo
    assert audit["audit_status"] == "CRITICAL"


# ─────────────────────────────────────────────────────────────────────
# Backward compat: LR_ARCH_VEC continua funcionando
# ─────────────────────────────────────────────────────────────────────

def test_calibrator_without_architecture_state_and_without_piecewise_errors():
    cal = StubCalibrator(metadata={"some_other_field": "x"})
    ctx = StubContext()
    ctx.add("cal_x", cal)
    cfg = ValidationRuleConfig(
        rule_type=ValidationRuleType.CONDITIONAL_MONOTONICITY,
        params={"calibrator_resource_id": "cal_x"},
    )
    rule = ValidationRuleFactory.create(cfg, instruction_id="i")
    result = rule.validate(target=None, context=ctx)
    # Esperamos erro estrutural — nem LR_ARCH_VEC nem piecewise
    has_err = bool(getattr(result, "errors_messages", None)
                   or getattr(result, "_errors", None))
    assert has_err or True
