# Validation Scripts — Na-ion Multi-Fidelity ML Project

These are the **feasibility-validation scripts** run in June 2026 to prove every load-bearing
part of the pipeline works *before* committing to the project. They are throwaway proofs —
the seed of the real codebase, not the final code.

## What each script proves

| Script | Proves | Key result |
|--------|--------|-----------|
| `test_pybamm_naion.py` | PyBaMM Na-ion model exists, solves, params settable | Loads, 0.45 s/sim |
| `test_electronic_props.py` | HOMO/LUMO/dipole computable for a solvent | PySCF: HOMO −7.95 eV, dipole 5.45 D (matches exp) |
| `test_pybamm_robustness.py` | The 5000-sim sweep won't crash; rate capability works | 100% solver success (80/80) |
| `test_ml_spine.py` | Multi-fidelity GP + importance + Bayesian optimization | All 4 run end-to-end on real PyBaMM data |
| `test_md_killtest.py` | The molecular-dynamics compute cost (the one real risk) | 0.80 s/step CPU → needs a GPU (16 GB ok) |

## How to run

Two virtual environments were used (because PyTorch is too big for the small `/tmp`):

```bash
# Env 1 — full stack incl. PyTorch (on the Storage drive)
source /run/media/dcode/Storage/.naion_venv/bin/activate
python test_pybamm_naion.py
python test_pybamm_robustness.py
python test_ml_spine.py

# Env 2 — xTB/PySCF/RDKit (electronic props + MD kill-test)
# (these need: pip install pybamm ase rdkit pyscf tblite huggingface_hub)
python test_electronic_props.py
python test_md_killtest.py
```

If you set up fresh on another machine, install with:

```bash
python3 -m venv venv && source venv/bin/activate
pip install pybamm torch gpytorch botorch shap xgboost SALib scikit-learn pandas numpy
pip install ase rdkit pyscf tblite huggingface_hub        # molecular tools
```

All installed via plain `pip` on Python 3.12 — no conda, no HPC.
