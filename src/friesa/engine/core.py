# Copyright © 2026 Stable Cyber LLC. All Rights Reserved.
from __future__ import annotations

import math
from dataclasses import dataclass

from friesa.calibration.models import CalibrationBundle, ControlDefaults, InherentRiskDefaults, RateParameters, _nominal

from .models import ALL_STATES, Observability, OperationalState, RiskResult, State


@dataclass(frozen=True)
class EffectiveInputs:
    controls: ControlDefaults
    inherent_risk: InherentRiskDefaults
    time_horizon: float
    governance_weight: float


def merge_inputs(
    bundle: CalibrationBundle,
    control_overrides: dict[str, float] | None = None,
    risk_overrides: dict[str, float] | None = None,
    time_horizon: float | None = None,
    governance_weight: float | None = None,
) -> EffectiveInputs:
    controls = bundle.control_defaults.model_copy(
        update={k: v for k, v in (control_overrides or {}).items() if v is not None}
    )
    inherent_risk = bundle.inherent_risk_defaults.model_copy(
        update={k: v for k, v in (risk_overrides or {}).items() if v is not None}
    )
    return EffectiveInputs(
        controls=controls,
        inherent_risk=inherent_risk,
        time_horizon=time_horizon or bundle.time_horizon_default,
        governance_weight=governance_weight if governance_weight is not None else bundle.governance_weight_default,
    )


def compute_rates(controls: ControlDefaults, parameters: RateParameters) -> dict[str, float]:
    # _nominal() resolves DistributionSpec → point estimate; no-op for plain floats.
    prevent = _nominal(controls.prevent)
    detect_op = _nominal(controls.detect_operational)
    contain = _nominal(controls.contain)
    recover = _nominal(controls.recover)
    detect_gov = _nominal(controls.detect_governance)
    return {
        "lambda": parameters.lambda_base * math.exp(-parameters.beta_p * prevent),
        "mu": parameters.mu_base * math.exp(parameters.beta_d * detect_op),
        "gamma": parameters.gamma_base * math.exp(parameters.beta_c * contain),
        "rho": parameters.rho_base * math.exp(parameters.beta_r * recover),
        "kappa_down": parameters.kappa_base * math.exp(-parameters.beta_g * detect_gov),
        "kappa_up": parameters.kappa_base * math.exp(parameters.beta_g * detect_gov),
        "alpha_1": parameters.alpha_1,
        "alpha_2": parameters.alpha_2,
        "alpha_3": parameters.alpha_3,
    }


def build_generator(controls: ControlDefaults, parameters: RateParameters) -> tuple[list[list[float]], list[State]]:
    rates = compute_rates(controls, parameters)
    states = ALL_STATES
    index = {state: idx for idx, state in enumerate(states)}
    size = len(states)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]

    def add(source: State, destination: State, rate: float) -> None:
        if rate <= 0:
            return
        matrix[index[source]][index[destination]] += rate

    for observability in Observability:
        add((OperationalState.safe, observability), (OperationalState.compromised, observability), rates["lambda"])
        add((OperationalState.compromised, observability), (OperationalState.detected, observability), rates["mu"])
        add((OperationalState.detected, observability), (OperationalState.contained, observability), rates["gamma"])
        add((OperationalState.contained, observability), (OperationalState.recovered, observability), rates["rho"])

    for operation, alpha_name in [
        (OperationalState.compromised, "alpha_1"),
        (OperationalState.detected, "alpha_2"),
        (OperationalState.contained, "alpha_3"),
    ]:
        for observability in Observability:
            add((operation, observability), (OperationalState.catastrophic, observability), rates[alpha_name])

    for operation in OperationalState:
        add((operation, Observability.observable), (operation, Observability.unobservable), rates["kappa_down"])
        add((operation, Observability.unobservable), (operation, Observability.observable), rates["kappa_up"])

    for row_idx in range(size):
        matrix[row_idx][row_idx] = -sum(matrix[row_idx])

    return matrix, states


def identity(size: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]


def matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    size = len(left)
    result = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for k in range(size):
            if left[i][k] == 0.0:
                continue
            for j in range(size):
                result[i][j] += left[i][k] * right[k][j]
    return result


def vecmul(vector: list[float], matrix: list[list[float]]) -> list[float]:
    size = len(vector)
    result = [0.0 for _ in range(size)]
    for j in range(size):
        result[j] = sum(vector[i] * matrix[i][j] for i in range(size))
    return result


def uniformized_transition(generator: list[list[float]], tau: float, epsilon: float = 1e-9) -> list[list[float]]:
    size = len(generator)
    nu = max(abs(generator[i][i]) for i in range(size))
    if nu == 0.0:
        return identity(size)

    step = identity(size)
    for i in range(size):
        for j in range(size):
            step[i][j] += generator[i][j] / nu

    result = [[0.0 for _ in range(size)] for _ in range(size)]
    term = identity(size)
    poisson_weight = math.exp(-nu * tau)
    for i in range(size):
        for j in range(size):
            result[i][j] += poisson_weight * term[i][j]

    k = 1
    cumulative = poisson_weight
    while 1.0 - cumulative > epsilon:
        term = matmul(term, step)
        poisson_weight *= (nu * tau) / k
        for i in range(size):
            for j in range(size):
                result[i][j] += poisson_weight * term[i][j]
        cumulative += poisson_weight
        k += 1
    return result


def catastrophe_probability(generator: list[list[float]], states: list[State], tau: float) -> float:
    probabilities = [0.0 for _ in states]
    probabilities[states.index((OperationalState.safe, Observability.observable))] = 1.0
    transition = uniformized_transition(generator, tau)
    terminal = vecmul(probabilities, transition)
    return (
        terminal[states.index((OperationalState.catastrophic, Observability.observable))]
        + terminal[states.index((OperationalState.catastrophic, Observability.unobservable))]
    )


def governance_risk(generator: list[list[float]], states: list[State], tau: float, steps: int = 200) -> float:
    delta = tau / steps
    total = 0.0
    for step_idx in range(steps + 1):
        t = step_idx * delta
        probabilities = [0.0 for _ in states]
        probabilities[states.index((OperationalState.safe, Observability.observable))] = 1.0
        transition = uniformized_transition(generator, t)
        current = vecmul(probabilities, transition)
        subtotal = 0.0
        for op in [OperationalState.compromised, OperationalState.detected, OperationalState.contained]:
            subtotal += current[states.index((op, Observability.unobservable))]
        weight = 0.5 if step_idx in {0, steps} else 1.0
        total += weight * subtotal
    return total * delta


def compute_residual_risk(
    bundle: CalibrationBundle,
    control_overrides: dict[str, float] | None = None,
    risk_overrides: dict[str, float] | None = None,
    time_horizon: float | None = None,
    governance_weight: float | None = None,
    normalize_gov: bool = False,
) -> RiskResult:
    """Compute R_res = (F·R_i·E_x·S·A) / K  +  w_gov · R_gov.

    Unit analysis
    -------------
    Main residual term  : [loss · events · year⁻¹]
                          (F [events·year⁻¹] × R_i [–] × E_x [–] × S [loss] × A [–]) / K [–]
    R_gov               : [seconds · probability]   — integral of P(unobservable) over τ
    w_gov               : to make addition dimensionally valid, w_gov must carry
                          [loss · events · year⁻¹ · seconds⁻¹ · probability⁻¹]
                          OR the caller sets normalize_gov=True to divide R_gov by τ first,
                          reducing R_gov to a dimensionless occupancy fraction.

    Parameters
    ----------
    normalize_gov : if True, divides R_gov by τ before applying w_gov.
                    This makes the governance penalty a dimensionless fraction
                    (mean unobservable occupancy) and ensures w_gov carries the
                    same units as the main residual term.
    """
    effective = merge_inputs(
        bundle=bundle,
        control_overrides=control_overrides,
        risk_overrides=risk_overrides,
        time_horizon=time_horizon,
        governance_weight=governance_weight,
    )
    baseline_controls = ControlDefaults(
        prevent=0.0,
        detect_operational=0.0,
        contain=0.0,
        recover=0.0,
        detect_governance=0.0,
    )

    baseline_q, baseline_states = build_generator(baseline_controls, bundle.rate_parameters)
    controlled_q, controlled_states = build_generator(effective.controls, bundle.rate_parameters)

    baseline_cat = catastrophe_probability(baseline_q, baseline_states, effective.time_horizon)
    controlled_cat = catastrophe_probability(controlled_q, controlled_states, effective.time_horizon)
    gov_risk = governance_risk(controlled_q, controlled_states, effective.time_horizon)

    # Resolve nominal values from DistributionSpec fields (no-op for plain floats).
    inherent = (
        _nominal(effective.inherent_risk.frequency)
        * _nominal(effective.inherent_risk.reachability)
        * _nominal(effective.inherent_risk.exploitability)
        * _nominal(effective.inherent_risk.severity)
        * _nominal(effective.inherent_risk.amplification)
    )
    k_value = baseline_cat / max(controlled_cat, 1e-12)

    gov_term = gov_risk / max(effective.time_horizon, 1e-12) if normalize_gov else gov_risk
    residual = inherent / max(k_value, 1e-12) + effective.governance_weight * gov_term

    return RiskResult(
        k=k_value,
        catastrophe_probability=controlled_cat,
        governance_risk=gov_risk,
        residual_risk=residual,
    )
