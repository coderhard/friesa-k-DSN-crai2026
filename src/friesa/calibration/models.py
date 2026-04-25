# Copyright © 2026 Stable Cyber LLC. All Rights Reserved.
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# P1-A: Distribution specifications for calibration parameters
# ---------------------------------------------------------------------------

class DistributionSpec(BaseModel):
    """Declares the probability distribution for a calibration parameter.

    Use ``distribution_type="point"`` for legacy point-estimate bundles.
    Use ``lognormal`` for positive unbounded quantities (frequency, severity).
    Use ``beta`` for probabilities bounded in [0, 1] (reachability, exploitability).
    Use ``gamma`` for positive quantities with skew (rate parameters).

    For ``lognormal``: ``mean`` is the *arithmetic* mean of X (not the log-scale
    mean μ).  The sampler converts internally via μ = ln(mean) − σ²/2.
    For ``beta`` / ``gamma``: ``alpha`` and ``beta`` are the standard shape parameters.
    """

    distribution_type: Literal["point", "lognormal", "beta", "gamma"]
    # point
    value: float | None = None
    # lognormal  (arithmetic mean + log-scale std dev)
    mean: float | None = None
    sigma: float | None = None
    # beta / gamma  (standard shape parameters)
    alpha: float | None = None
    beta: float | None = None
    # metadata
    units: str = ""
    confidence: Literal["low", "medium", "high"] = "medium"
    source_type: str = ""

    @property
    def nominal(self) -> float:
        """Central point estimate: mean for lognormal, E[X] for beta/gamma."""
        if self.distribution_type == "point":
            assert self.value is not None, "point distribution requires value"
            return self.value
        if self.distribution_type == "lognormal":
            assert self.mean is not None, "lognormal requires mean"
            return self.mean
        if self.distribution_type in ("beta", "gamma"):
            assert self.alpha is not None and self.beta is not None
            if self.distribution_type == "beta":
                return self.alpha / (self.alpha + self.beta)
            # gamma: mean = shape / rate = alpha / beta
            return self.alpha / self.beta
        raise ValueError(f"Unknown distribution_type: {self.distribution_type}")


def _nominal(v: "float | DistributionSpec") -> float:
    """Return the scalar point estimate from a float or DistributionSpec."""
    if isinstance(v, DistributionSpec):
        return v.nominal
    return float(v)


class InherentRiskDefaults(BaseModel):
    # Each field accepts a plain float (legacy) or a DistributionSpec.
    # Numeric constraints (ge/le) are not enforced on DistributionSpec values;
    # the calibration author is responsible for choosing sensible parameters.
    frequency: float | DistributionSpec
    reachability: float | DistributionSpec
    exploitability: float | DistributionSpec
    severity: float | DistributionSpec
    amplification: float | DistributionSpec


class ControlDefaults(BaseModel):
    """Control effectiveness values.

    Each field accepts a plain float (legacy bundles) or a ``DistributionSpec``
    for distribution-aware uncertainty quantification.  Probability-valued
    controls should use ``beta`` distributions; point floats must be in [0, 1].
    Domain enforcement is done in ``collect_bundle_warnings`` rather than the
    model layer so semantically-suspect bundles remain loadable for research.
    """

    prevent: float | DistributionSpec
    detect_operational: float | DistributionSpec
    contain: float | DistributionSpec
    recover: float | DistributionSpec
    detect_governance: float | DistributionSpec


class RateParameters(BaseModel):
    lambda_base: float = Field(gt=0.0)
    mu_base: float = Field(gt=0.0)
    gamma_base: float = Field(gt=0.0)
    rho_base: float = Field(gt=0.0)
    kappa_base: float = Field(gt=0.0)
    beta_p: float = Field(ge=0.0)
    beta_d: float = Field(ge=0.0)
    beta_c: float = Field(ge=0.0)
    beta_r: float = Field(ge=0.0)
    beta_g: float = Field(ge=0.0)
    alpha_1: float = Field(gt=0.0)
    alpha_2: float = Field(gt=0.0)
    alpha_3: float = Field(gt=0.0)


class Provenance(BaseModel):
    source: str
    rationale: str
    confidence: Literal["low", "medium", "high"]
    last_reviewed: date
    owner: str


class Approval(BaseModel):
    owner: str
    last_reviewed: date
    next_review_due: date | None = None
    approved_by: str | None = None
    approval_date: date | None = None


class CalibrationBundle(BaseModel):
    schema_version: str = "1.0"
    scenario_id: str = Field(min_length=1)
    scenario_name: str = Field(min_length=1)
    scenario_family: str = "generic"
    description: str
    intended_use: str = "research_and_consulting"
    limitations: str = ""
    units: str = "dimensionless / scenario-defined"
    sector: str = "generic"
    criticality: str = "medium"
    safety_relevance: bool = True
    research_only: bool = False
    client_signoff_required: bool = False
    has_placeholders: bool = True
    validation_status: Literal["draft", "internal_review", "validated", "deprecated", "archived"] = "draft"
    tags: list[str] = Field(default_factory=list)
    time_horizon_default: float = Field(gt=0.0)
    governance_weight_default: float = Field(ge=0.0)
    governance_weight_rationale: str = Field(min_length=1)
    inherent_risk_defaults: InherentRiskDefaults
    rate_parameters: RateParameters
    control_defaults: ControlDefaults
    provenance: Provenance
    approval: Approval
    assumptions: list[str] = Field(default_factory=list)

    @field_validator("scenario_id")
    @classmethod
    def validate_scenario_id(cls, value: str) -> str:
        lowered = value.strip().lower().replace("-", "_").replace(" ", "_")
        if not lowered:
            raise ValueError("scenario_id cannot be empty")
        return lowered

    @model_validator(mode="after")
    def check_review_dates(self) -> "CalibrationBundle":
        if self.approval.next_review_due and self.approval.next_review_due < self.approval.last_reviewed:
            raise ValueError("next_review_due must not be earlier than last_reviewed")
        return self
