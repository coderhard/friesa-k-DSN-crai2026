# Copyright © 2026 Stable Cyber LLC. All Rights Reserved.
"""Monte Carlo uncertainty analysis for FRIESA-K calibration bundles.

Method selection
----------------
``n_ssa > 0``  (default: 200)
    Uses the Gillespie SSA to estimate catastrophe_probability and
    governance_risk per MC sample.  The zero-control baseline (K numerator)
    is also computed via SSA (4× more trajectories, computed once) so the
    full path never calls uniformization — making it tractable for arbitrarily
    long time horizons (e.g. the 12-hour oncology scenario).
    SSA is 10–100× faster than uniformization and is the recommended path
    for AWS production runs.

``n_ssa = 0``
    Falls back to the exact uniformization path (``compute_residual_risk``).
    Use this for debugging or when you need exact per-sample governance_risk
    (e.g. for small sample counts where SSA variance would dominate).
"""

from __future__ import annotations

import math

import numpy as np

from friesa.calibration.models import CalibrationBundle, ControlDefaults, _nominal
from friesa.engine.core import (
    build_generator,
    catastrophe_probability as _exact_cat_prob,
    compute_residual_risk,
    merge_inputs,
)
from friesa.engine.models import ALL_STATES
from friesa.engine.ssa import _ssa_estimate


def _sample(spec, rng: np.random.Generator) -> float:
    """Draw one sample from a DistributionSpec, or return the float directly."""
    from friesa.calibration.models import DistributionSpec
    if not isinstance(spec, DistributionSpec):
        return float(spec)
    if spec.distribution_type == "point":
        assert spec.value is not None
        return spec.value
    if spec.distribution_type == "lognormal":
        assert spec.mean is not None and spec.sigma is not None
        mu = math.log(spec.mean) - 0.5 * spec.sigma ** 2
        return float(rng.lognormal(mean=mu, sigma=spec.sigma))
    if spec.distribution_type == "beta":
        assert spec.alpha is not None and spec.beta is not None
        return float(rng.beta(spec.alpha, spec.beta))
    if spec.distribution_type == "gamma":
        assert spec.alpha is not None and spec.beta is not None
        return float(rng.gamma(shape=spec.alpha, scale=1.0 / spec.beta))
    raise ValueError(f"Unknown distribution_type: {spec.distribution_type}")


def _tail_stats(arr: np.ndarray) -> dict:
    """Descriptive + tail-risk statistics for a 1-D sample array."""
    p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99])
    tail_mask = arr >= p95
    cvar_95 = float(arr[tail_mask].mean()) if tail_mask.any() else float(p95)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "percentiles": {
            "p50": float(p50),
            "p90": float(p90),
            "p95": float(p95),
            "p99": float(p99),
        },
        "cvar_95": cvar_95,
    }


def run_uncertainty_analysis(
    bundle: CalibrationBundle,
    samples: int = 250,
    seed: int | None = None,
    exceedance_threshold: float | None = None,
    n_ssa: int = 200,
    normalize_gov: bool = False,
) -> dict:
    """Run Monte Carlo uncertainty analysis over a calibration bundle.

    Sampling behaviour
    ------------------
    Inherent risk fields and control defaults are sampled according to their
    declared ``DistributionSpec``:

    * ``lognormal`` / ``beta`` / ``gamma`` → draw from the declared distribution
    * ``point`` → always returns the declared value (no spread)
    * plain ``float`` → passed through unchanged (deterministic for that field)

    This means a bundle with all point / float values produces a *zero-variance*
    MC result — which is correct and expected.  Meaningful uncertainty bands
    require at least some fields to carry non-point distributions.

    Parameters
    ----------
    bundle    : loaded calibration bundle
    samples   : number of Monte Carlo draws
    seed      : numpy RNG seed — pass an int for reproducible runs
    exceedance_threshold :
        if provided, the output includes the fraction of samples where
        residual_risk exceeds this value
    n_ssa     : SSA trajectories per MC sample.  ``n_ssa > 0`` uses the
                Gillespie SSA (fast, recommended for AWS).  ``n_ssa = 0``
                falls back to exact uniformization (slow, exact).
    normalize_gov :
        if True, divide governance_risk by τ before weighting (dimensionless
        occupancy fraction — use when comparing bundles with different time
        horizons)

    Returns
    -------
    dict with keys: samples, seed, method, catastrophe_probability,
    governance_risk, k, residual_risk  (each a stats sub-dict with mean/min/
    max/percentiles/cvar_95), and optionally exceedance.
    """
    rng = np.random.default_rng(seed)
    tau = bundle.time_horizon_default
    w_gov = bundle.governance_weight_default
    use_ssa = n_ssa > 0

    # --- Precompute baseline (zero-control) catastrophe probability ---
    # The zero-control case never varies across MC samples, so we compute it once.
    # SSA path: use SSA for the baseline — uniformization is intractable for long
    # time horizons (e.g. τ=43 200 s oncology scenario) and is not needed here
    # since the baseline SSA estimate is divided out in the K ratio anyway.
    if use_ssa:
        baseline_controls = ControlDefaults(
            prevent=0.0, detect_operational=0.0,
            contain=0.0, recover=0.0, detect_governance=0.0,
        )
        baseline_q, baseline_states = build_generator(baseline_controls, bundle.rate_parameters)
        baseline_cat, _ = _ssa_estimate(baseline_q, baseline_states, tau, n_ssa * 4, rng)

    results_cat: list[float] = []
    results_gov: list[float] = []
    results_k: list[float] = []
    results_res: list[float] = []

    for _ in range(samples):
        # --- Sample controls from DistributionSpec; pass plain floats through unchanged ---
        ctrl = bundle.control_defaults
        sampled_controls = ControlDefaults(
            prevent=float(np.clip(_sample(ctrl.prevent, rng), 0.0, 1.0)),
            detect_operational=float(np.clip(_sample(ctrl.detect_operational, rng), 0.0, 1.0)),
            contain=float(np.clip(_sample(ctrl.contain, rng), 0.0, 1.0)),
            recover=float(np.clip(_sample(ctrl.recover, rng), 0.0, 1.0)),
            detect_governance=float(np.clip(_sample(ctrl.detect_governance, rng), 0.0, 1.0)),
        )

        # --- Sample inherent risk from declared distributions ---
        ir = bundle.inherent_risk_defaults
        freq = max(0.0, _sample(ir.frequency, rng))
        reach = float(np.clip(_sample(ir.reachability, rng), 0.0, 1.0))
        expl = float(np.clip(_sample(ir.exploitability, rng), 0.0, 1.0))
        sev = max(1e-6, _sample(ir.severity, rng))
        amp = max(1.0, _sample(ir.amplification, rng))
        inherent = freq * reach * expl * sev * amp

        if use_ssa:
            # Build controlled generator (rate_parameters never vary)
            ctrl_q, ctrl_states = build_generator(sampled_controls, bundle.rate_parameters)

            # SSA estimates controlled_cat and gov_risk in one trajectory batch.
            # _rng is shared so the full run is reproducible from a single seed.
            controlled_cat, gov_risk = _ssa_estimate(
                ctrl_q, ctrl_states, tau, n_ssa, rng
            )
            k_val = baseline_cat / max(controlled_cat, 1e-12)
            gov_term = gov_risk / max(tau, 1e-12) if normalize_gov else gov_risk
            residual = inherent / max(k_val, 1e-12) + w_gov * gov_term

        else:
            # Exact uniformization fallback
            control_overrides = {
                "prevent": sampled_controls.prevent,
                "detect_operational": sampled_controls.detect_operational,
                "contain": sampled_controls.contain,
                "recover": sampled_controls.recover,
                "detect_governance": sampled_controls.detect_governance,
            }
            risk_overrides = {
                "frequency": freq, "reachability": reach,
                "exploitability": expl, "severity": sev, "amplification": amp,
            }
            r = compute_residual_risk(
                bundle,
                control_overrides=control_overrides,
                risk_overrides=risk_overrides,
                normalize_gov=normalize_gov,
            )
            controlled_cat = r.catastrophe_probability
            gov_risk = r.governance_risk
            k_val = r.k
            residual = r.residual_risk

        results_cat.append(controlled_cat)
        results_gov.append(gov_risk)
        results_k.append(k_val)
        results_res.append(residual)

    cat_arr = np.array(results_cat)
    gov_arr = np.array(results_gov)
    k_arr = np.array(results_k)
    res_arr = np.array(results_res)

    output: dict = {
        "samples": samples,
        "seed": seed,
        "method": f"ssa(n={n_ssa})" if use_ssa else "uniformization",
        "catastrophe_probability": _tail_stats(cat_arr),
        "governance_risk": _tail_stats(gov_arr),
        "k": _tail_stats(k_arr),
        "residual_risk": _tail_stats(res_arr),
    }

    if exceedance_threshold is not None:
        output["exceedance"] = {
            "threshold": exceedance_threshold,
            "probability": float((res_arr > exceedance_threshold).mean()),
        }

    return output
