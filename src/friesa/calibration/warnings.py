# Copyright © 2026 Stable Cyber LLC. All Rights Reserved.
from __future__ import annotations

from datetime import date

from .models import CalibrationBundle, DistributionSpec

# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------

# Fields that must be bounded in [0, 1] — only beta or point are valid.
_PROB_FIELDS: frozenset[str] = frozenset(
    {"reachability", "exploitability",
     "prevent", "detect_operational", "contain", "recover", "detect_governance"}
)

# Fields that must be strictly positive — lognormal, gamma, or positive point.
_POS_FIELDS: frozenset[str] = frozenset({"frequency", "severity", "amplification"})

# Variance thresholds: warn when spread is implausibly wide for the domain.
_LOGNORMAL_SIGMA_WARN = 1.5     # sigma > 1.5 → very heavy tail
_BETA_PRECISION_WARN = 2.0      # alpha + beta < 2 → nearly flat prior


def _domain_warnings(field_name: str, value: float | DistributionSpec) -> list[str]:
    """Return semantic warnings for a single calibration field value."""
    msgs: list[str] = []

    if isinstance(value, DistributionSpec):
        dt = value.distribution_type

        if field_name in _PROB_FIELDS:
            if dt in ("lognormal", "gamma"):
                msgs.append(
                    f"{field_name}: probability field should use 'beta' or 'point', "
                    f"not '{dt}' — lognormal/gamma are unbounded above 1"
                )
            elif dt == "point" and value.value is not None:
                if not (0.0 <= value.value <= 1.0):
                    msgs.append(
                        f"{field_name}: point value {value.value} is outside [0, 1]"
                    )

        if field_name in _POS_FIELDS:
            if dt == "beta":
                msgs.append(
                    f"{field_name}: positive-valued field should use 'lognormal', "
                    f"'gamma', or 'point', not 'beta' — beta is bounded to [0, 1]"
                )
            elif dt == "point" and value.value is not None and value.value <= 0.0:
                msgs.append(
                    f"{field_name}: point value {value.value} must be > 0"
                )

        # Variance sanity checks
        if dt == "lognormal" and value.sigma is not None and value.sigma > _LOGNORMAL_SIGMA_WARN:
            msgs.append(
                f"{field_name}: lognormal sigma={value.sigma:.2f} > {_LOGNORMAL_SIGMA_WARN} "
                f"— very heavy tail; verify this is intentional"
            )
        if dt == "beta" and value.alpha is not None and value.beta is not None:
            precision = value.alpha + value.beta
            if precision < _BETA_PRECISION_WARN:
                msgs.append(
                    f"{field_name}: beta(α={value.alpha}, β={value.beta}) has precision "
                    f"{precision:.2f} < {_BETA_PRECISION_WARN} — nearly flat prior; "
                    f"provide stronger calibration evidence"
                )

    else:
        # Plain float — validate against domain bounds
        v = float(value)
        if field_name in _PROB_FIELDS and not (0.0 <= v <= 1.0):
            msgs.append(f"{field_name}: float value {v} is outside [0, 1]")
        if field_name in _POS_FIELDS and v <= 0.0:
            msgs.append(f"{field_name}: float value {v} must be > 0")

    return msgs


def collect_bundle_warnings(bundle: CalibrationBundle) -> list[str]:
    warnings: list[str] = []
    today = date.today()

    # --- Status / lifecycle ---
    if bundle.validation_status in {"deprecated", "archived"}:
        warnings.append(f"bundle status is {bundle.validation_status}")

    if bundle.validation_status == "draft" and bundle.has_placeholders:
        warnings.append(
            "bundle is draft with placeholders — not suitable for client-facing output"
        )

    if bundle.approval.next_review_due and bundle.approval.next_review_due < today:
        warnings.append("bundle review date is overdue")

    if bundle.research_only:
        warnings.append("bundle is marked research_only")

    if bundle.has_placeholders:
        warnings.append("bundle contains placeholders and requires calibration evidence review")

    # --- Governance rationale quality ---
    rationale = bundle.governance_weight_rationale.strip()
    if len(rationale) < 20:
        warnings.append(
            "governance_weight_rationale is suspiciously short — review for completeness"
        )

    # --- Provenance confidence ---
    if bundle.provenance.confidence == "low":
        prov_rationale = getattr(bundle.provenance, "rationale", "").strip()
        if len(prov_rationale) < 20:
            warnings.append(
                "provenance confidence is 'low' and rationale is thin — "
                "strengthen calibration evidence before use in production"
            )

    # --- Inherent risk domain validation ---
    ir = bundle.inherent_risk_defaults
    for field_name in ("frequency", "reachability", "exploitability", "severity", "amplification"):
        warnings.extend(_domain_warnings(field_name, getattr(ir, field_name)))

    # --- Control defaults domain validation ---
    ctrl = bundle.control_defaults
    for field_name in ("prevent", "detect_operational", "contain", "recover", "detect_governance"):
        warnings.extend(_domain_warnings(field_name, getattr(ctrl, field_name)))

    return warnings
