# Sodium-Ion Multiscale ML

**A multi-fidelity machine-learning pipeline for sodium-ion battery design — from molecular electrolyte simulation to cell-level performance prediction.**

> 🚧 **Status:** active research. Feasibility fully validated (June 2026); pipeline under construction.
> Target: arXiv preprint (Dec 2026) → *Digital Discovery* (RSC).

---

## What this is

Sodium-ion batteries are the low-cost, earth-abundant alternative to lithium-ion — but the modeling community lacks open data and reusable tools for them. This project builds that missing infrastructure: an **open, end-to-end pipeline** that connects **molecular electrolyte chemistry** to **cell-level battery performance**, with a fast, uncertainty-aware ML surrogate at its core.

## The pipeline

```
 Molecular simulation          Cell simulation             Machine learning
 ────────────────────          ───────────────             ────────────────
 MACE-OFF / GFN2-xTB  ──►  PyBaMM Na-ion DFN  ──►  Multi-fidelity GP surrogate
 (MD → conductivity κₑ)      (~5000 sims)            + SHAP  (interpretability)
 PySCF (HOMO/LUMO)           COMSOL P2D+thermal      + Bayesian optimization
                             (high-fidelity anchor)   + Conformal uncertainty
```

1. **Molecular** — MD (MACE-OFF / GFN2-xTB) → ion transport (κₑ, Dₑ) via Nernst–Einstein; DFT (PySCF) → HOMO/LUMO stability screening
2. **QSPR** — structure→property model: molecular descriptors → electrolyte conductivity
3. **Cell** — PyBaMM sodium-ion DFN (Chayambuka 2022 parameters), Latin-Hypercube parameter sweep
4. **High-fidelity anchor** — COMSOL thermally-coupled DFN (the isothermal↔thermal fidelity gap)
5. **Surrogate** — multi-fidelity Gaussian Process; emulates the simulators in milliseconds
6. **Analysis** — SHAP interpretability, Bayesian optimization for design, conformal prediction intervals
7. **Validation** — against Chayambuka 2022 experimental discharge curves

## Validated feasibility (not assumed — actually run)

| Component | Result |
|---|---|
| PyBaMM Na-ion DFN | ✅ solves in **0.45 s**; **100 % solver success** over an 80-sample LHS sweep |
| Rate capability (C/10 → 5C) | ✅ all solve; capacity falls physically |
| Electronic properties (PySCF) | ✅ propylene carbonate: HOMO **−7.95 eV**, dipole **5.45 D** (exp ≈ 4.9 D) |
| ML spine (multi-fidelity GP + SHAP + BO) | ✅ runs end-to-end on real PyBaMM data |
| MD compute cost (GFN2-xTB) | 📏 measured **0.80 s/step** on CPU → needs a GPU (16 GB sufficient) |

Reproduce these with the scripts in [`validation_scripts/`](validation_scripts/).

## Tech stack

**Molecular:** MACE-OFF · GFN2-xTB (tblite) · PySCF · ASE · RDKit
**Battery physics:** PyBaMM · COMSOL
**ML:** GPyTorch · BoTorch · SHAP · XGBoost · scikit-learn · SALib

## Repository layout

```
├── NaIon_Project_Proposal.md       # readable project summary
├── NaIon_MultiScale_ML_Outline.md  # full technical plan (validated)
├── References_and_Links.md         # all papers, datasets, tools
├── validation_scripts/             # feasibility proofs (runnable)
├── src/                            # pipeline source
├── notebooks/                      # analysis notebooks
├── data/                           # raw + processed
└── results/                        # figures, tables
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install pybamm torch gpytorch botorch shap xgboost SALib scikit-learn pandas numpy
pip install ase rdkit pyscf tblite huggingface_hub   # molecular tools
pip install mace-torch                                # MACE-OFF (GPU recommended)
```

All open-source · Python 3.12 · no conda, no HPC required.

## Key references

- **Chayambuka et al. (2022)**, *Physics-based modeling of sodium-ion batteries part II*, Electrochimica Acta 404, 139764 — the source of the PyBaMM Na-ion model and our validation data
- **PyBaMM** — [docs.pybamm.org](https://docs.pybamm.org)
- **MACE-OFF** — Kovács et al., *JACS* (2024)

Full list in [`References_and_Links.md`](References_and_Links.md).

## License

Code: MIT. Data and figures: CC BY 4.0.
