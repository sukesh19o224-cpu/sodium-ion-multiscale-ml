# References & Links — Na-ion Multi-Fidelity ML Project

*Every paper, dataset, tool, and model link shared during planning (June–July 2026). Grouped by purpose. ⭐ = read first.*

---

## 1. Foundational — your model's source (READ FIRST)
- ⭐ **Chayambuka et al. 2022, Part II** — *Physics-based modeling of sodium-ion batteries part II. Model and validation*, Electrochimica Acta 404 (2022) 139764. **Open access (CC BY).** The source of PyBaMM's Na-ion model + your validation data.
  https://www.sciencedirect.com/science/article/pii/S0013468621020478 · DOI: 10.1016/j.electacta.2021.139764
  *(PDF already in your Downloads: `1_s2.0_S0013468621020478_main.pdf`)*
- **PyBaMM sodium-ion model docs** — https://docs.pybamm.org/en/latest/source/examples/notebooks/models/sodium-ion.html
- **COMSOL "1D Isothermal Sodium-Ion Battery" model example** (source of PyBaMM's parameter values) — https://www.comsol.com/model/1d-isothermal-sodium-ion-battery-117341
- **Ionworks blog** — Na-ion model now in PyBaMM — https://www.ionworks.com/blog/sodium-ion-battery-model-now-available-in-pybamm

## 2. ⚠️ Closest competitors — the novelty check (READ BEFORE BUILDING)
- ⭐ **Chandel et al. 2025** — *Multiscale MD and electrochemical modelling of sodium-ion batteries: impact of electrolyte composition* — **the biggest overlap with your molecular→cell pillar.**
  https://www.sciencedirect.com/science/article/abs/pii/S2352152X25014628 · (SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5042651)
- **ML-enhanced multiscale modeling of high-rate Na-ion (electrochemical + thermal)** — https://www.sciencedirect.com/science/article/abs/pii/S2352152X25021589
- **Modeling & simulation of Na-ion: electrochemical mechanism + ML** — https://www.sciencedirect.com/science/article/pii/S2773153725000490
- **Sun et al. — PINN high-fidelity Na-ion SPM, multiscale optimization (BMS)** — SSRN abstract 5648362
- **Bayesian optimization for NASICON Na-ion cathode design** — arXiv:2411.01117

## 3. Tools & software (all open-source)
- **MACE-OFF** (organic ML force field) — Kovács et al., JACS 2024 — https://pubs.acs.org/doi/10.1021/jacs.4c07099 · arXiv:2312.15211
- **MACE foundation models** (MACE-MP, MACE-OFF) — https://github.com/ACEsuit/mace-foundations
- **MACE-MP-0** foundation model — Batatia et al. 2023 (for crystalline electrodes)
- **GFN2-xTB validated for Na/Li carbonate electrolyte MD** — PMC10537190
- **PySCF, Quantum ESPRESSO, GPAW, Psi4, LAMMPS** — open DFT/MD engines (see outline §7)

## 4. Open data / reusable assets
- **BatteryLife** (Na-ion real cycling data, MIT license) — SIGKDD 2025
  GitHub: https://github.com/Ruifeng-Tan/BatteryLife · HuggingFace: https://huggingface.co/datasets/Battery-Life/BatteryLife_Processed · arXiv:2502.18807
- **Electrolyte Genome** (LBNL/Persson, ~4830 molecules HOMO/LUMO/IP/EA) — via Materials Project molecules API
- **Materials Project** — DFT crystal database (electrode descriptors)

## 5. Methods you build on
- **Multi-fidelity hierarchies in battery models (PINN, SPM)** — arXiv:2312.17329
- **Parameter-embedded Neural Operators (PE-FNO), mesh-agnostic Li-ion** — arXiv:2508.08087
- **Multi-fidelity Gaussian-process surrogates for physics regression** — arXiv:2404.11965
- **Molecular→conductivity QSPR + SHAP (interpretable electrolyte ML)** — MGEA 2025 — https://onlinelibrary.wiley.com/doi/10.1002/mgea.70032
- **COSMO-RS QSPR + boosting ML for ionic conductivity** — https://pubs.acs.org/doi/10.1021/acssuschemeng.4c00307
- **Cross-domain ionic conductivity ML (Na & Li)** — arXiv:2003.04922

## 6. Accuracy-improvement / v2 directions (residual learning, ageing)
- **Mechanistically-guided residual learning for battery state monitoring** — Nature Communications 2026 — https://www.nature.com/articles/s41467-025-67565-z
- **Augmented physics-based Li-ion model + ensemble sparse learning + conformal prediction** — arXiv:2507.00353
- **Hybrid physics-based residual learning for EV battery discharge** — arXiv:2603.01587
- **Physics-based modelling of ageing mechanisms in sodium-ion batteries (2025)** — https://www.sciencedirect.com/science/article/abs/pii/S037877532500672X

## 7. Electrode materials (Friend B, Option B2)
- **MACE-MP NEB migration-barrier benchmark** — Digital Discovery 2026 (your target journal)
- **ML-driven high-throughput screening for NASICON Na-ion cathodes** — https://pubs.acs.org/doi/10.1021/acsami.3c18448
- **Literature-derived migration-barrier dataset (Li/Na/K)** — Nature Scientific Data — https://www.nature.com/articles/s41597-025-06196-x
- **Oxyfluoride frameworks as Na-ion cathodes** — arXiv:2405.07614

## 8. Context / background (from your original proposal)
- Sulzer et al. 2021 — PyBaMM software paper
- Severson et al. 2019 (Nature Energy) — battery lifetime ML benchmark
- Attia et al. 2020 (Nature) — closed-loop battery optimization
- Abolhasani & Kumacheva 2023 (Nature Reviews) — self-driving labs
- van Ekeren — Na-ion electrolyte chemistry (phosphate/carbonate systems)

---

*Note: a few links (SSRN abstracts, Digital Discovery MACE-NEB) were seen via search summaries — verify the DOI/final citation before adding to your reference list.*
