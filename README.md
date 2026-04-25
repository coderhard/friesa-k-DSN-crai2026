# friesa-k-DSN-crai2026

Research artifact for:

> **From Agent Failure Paths to Quantified Residual Risk: A Compositional Framework for Resilient Agentic AI**
> CRAI Workshop @ The 56th Annual IEEE/IFIP International Conference on Dependable Systems and Networks
 (DSN) 2026

This repository contains the minimal implementation needed to reproduce the 
fault-injection simulation results reported in the paper. It is scoped to the 
two case studies presented: the hard real-time warehouse robot (Case A) and 
the financial-services agent with governance-observability instrumentation 
(Case B).

---

## What This Is

**CPSAINT** decomposes integrity failure structurally across seven layers — 
Physical, Sensing, Data, Compute, Actuation, Environment, and Time — 
using a five-mode failure alphabet: 
corruption, delay, omission, replay, and coupling\_abuse.

**FRIESA-K** maps each valid failure path to a quantified residual-risk estimate:

```
R_res(π, τ, u) = (F · R_i · E · S · A) / K(π, τ, u)  +  w_gov · R_gov(π, τ, u)
```

where `K` is derived from a controlled absorbing CTMC — not assigned as an informal score — and `R_gov` is a governance-observability dwell-time penalty.

Fault injection is implemented via Gillespie SSA (Stochastic Simulation Algorithm), which injects failure conditions at the entry layer of each instantiated path and tracks catastrophic absorption probability over the evaluation horizon.

---

## Repository Structure

```
friesa-k-DSN-crai2026/
├── reproduce_paper.py          # Reproduces all §IV results
├── pyproject.toml
├── calibration/
│   └── bundles/
│       ├── crai_warehouse_robot.yaml   # Case A bundle
│       └── crai_banking_agent.yaml     # Case B bundle
└── src/
    └── friesa/
        ├── engine/             # Deterministic path + Gillespie SSA
        ├── uncertainty/        # Monte Carlo uncertainty analysis
        ├── calibration/        # Bundle loader and validation
        └── domain/             # Layer and failure-mode definitions
```

---

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/coderhard/friesa-k-DSN-crai2026.git
cd friesa-k-DSN-crai2026
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Reproducing Paper Results

```bash
python reproduce_paper.py
```

Expected output matches Table I and §IV of the paper:

| Scenario | τ (s) | K | π_F | R_res (det) | Mean (N=5000) | p95 |
|---|---|---|---|---|---|---|
| Warehouse Robot | 0.75 | 1.527 | 0.506 | 2,563,283 | 2,553,038 | 6,766,279 |
| Banking Agent | 0.75 | 1.339 | 0.223 | 966,412 | 954,231 | 2,348,066 |

All runs use `seed=42` and `normalize_gov=True`.

---

## Citation

If you use this artifact in your research, please cite:

```bibtex
@inproceedings{karim2026friesak,
  title     = {From Agent Failure Paths to Quantified Residual Risk:
               A Compositional Framework for Resilient Agentic AI},
  author    = {Karim, Hassan and Sitharaman, Sai and Gupta, Deepti and Rawat, Danda B.},
  booktitle = {CRAI Workshop @ The 56th Annual IEEE/IFIP International Conference
               on Dependable Systems and Networks (DSN)},
  year      = {2026}
}
```

---

## Full Framework

This repository is scoped to the CRAI 2026 paper artifact. The full FRIESA-K / CPSAINT will be released separately.

---

## License

This software is licensed under the **Business Source License 1.1 (BUSL-1.1)**.

**Licensor:** Stable Cyber, LLC

**Licensed Work:** From Agent Failure Paths to Quantified Residual Risk: A Compositional Framework for Resilient Agentic AI

**Additional Use Grant:** Use for academic research, education, and non-commercial experimentation is permitted, provided the accompanying paper is cited.

**Change Date:** 2031-05-30

**Change License:** MIT

On the Change Date, this software will automatically convert to the MIT License. Until then, commercial use, production deployment, and derivative products require a separate commercial license from Stable Cyber LLC.

For licensing inquiries: https://orcid.org/0000-0002-5441-049X
