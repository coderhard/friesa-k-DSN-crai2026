# Copyright © 2026 Stable Cyber LLC. All Rights Reserved.
"""
Reproduce the fault-injection simulation results from:

  "From Agent Failure Paths to Quantified Residual Risk:
   A Compositional Framework for Resilient Agentic AI"
  CRAI Workshop @ DSN 2026

Usage:
  pip install -e .
  python reproduce_paper.py

Expected output matches Table I and §IV results in the paper.
"""

import pathlib
from friesa.calibration.loader import load_bundle_from_path as load_bundle
from friesa.engine.core import compute_residual_risk
from friesa.uncertainty.monte_carlo import run_uncertainty_analysis

BUNDLE_DIR = pathlib.Path(__file__).parent / "calibration" / "bundles"
SEED = 42
N_SAMPLES = 5000   # MC draws — matches "N=5,000" in the paper
N_SSA = 200        # SSA trajectories per sample — matches engine default

SCENARIOS = [
    {
        "name": "Case A — Warehouse Robot",
        "bundle": "crai_warehouse_robot",
        "normalize_gov": True,
    },
    {
        "name": "Case B — Banking Agent",
        "bundle": "crai_banking_agent",
        "normalize_gov": True,
    },
]


def run(scenario: dict) -> None:
    bundle = load_bundle(BUNDLE_DIR / f"{scenario['bundle']}.yaml")

    det = compute_residual_risk(
        bundle,
        normalize_gov=scenario["normalize_gov"],
    )

    mc = run_uncertainty_analysis(
        bundle,
        samples=N_SAMPLES,
        seed=SEED,
        normalize_gov=scenario["normalize_gov"],
        n_ssa=N_SSA,
    )

    print(f"\n{'=' * 60}")
    print(f"  {scenario['name']}")
    print(f"{'=' * 60}")
    print(f"  τ            = {bundle.time_horizon_default} s")
    print(f"  w_gov        = {bundle.governance_weight_default}")
    print(f"  K            = {det.k:.3f}")
    print(f"  π_F          = {det.catastrophe_probability:.3f}")
    print(f"  R_res (det)  = {det.residual_risk:,.0f}")
    print(f"  --- Fault-injection simulation (N={N_SAMPLES} samples × {N_SSA} SSA trajectories, seed={SEED}) ---")
    print(f"  mean R_res   = {mc['residual_risk']['mean']:,.0f}")
    print(f"  p95  R_res   = {mc['residual_risk']['percentiles']['p95']:,.0f}")


if __name__ == "__main__":
    print("\nFRIESA-K / CPSAINT — CRAI 2026 Paper Reproduction")
    print("Fault-injection via Gillespie SSA\n")
    for scenario in SCENARIOS:
        run(scenario)
    print("\nDone. Results should match §IV of the paper.")
