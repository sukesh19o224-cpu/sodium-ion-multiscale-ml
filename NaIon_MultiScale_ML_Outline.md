# Multi-Fidelity Machine Learning Framework for Na-ion Battery Performance Prediction
## From Molecular Electrolyte Simulation to Thermal-Electrochemical Digital Twins

**Project Type:** Research Paper (arXiv preprint + Journal submission)  
**Target Journal:** Digital Discovery (RSC) — first choice | npj Computational Materials — second  
**Target arXiv Submission:** December 2026  
**Team:**
- **Person 1 (You)** — ML, molecular simulation (MACE-OFF/xTB/PySCF), surrogate modeling, SHAP, BO, paper lead
- **Person 2 (Friend A)** — PyBaMM electrochemical simulation, synthetic data generation
- **Person 3 (Friend B)** — COMSOL thermal-electrochemical simulation

---

## 0. The Core Problem Being Solved

Na-ion batteries are a promising low-cost alternative to Li-ion batteries, but the ML/modeling community faces a fundamental bottleneck: **there is almost no open, structured Na-ion simulation or experimental data available for training machine learning models.** Unlike Li-ion, which has decades of parameter sets, open datasets (MIT-Stanford, NASA, CALCE), and mature simulation tools, Na-ion has:

- Only 1-2 validated DFN parameter sets in PyBaMM (Chayambuka 2022)
- No open cycling datasets comparable to Li-ion repositories
- No systematic multi-scale simulation framework
- No multi-fidelity surrogate model

This work builds that missing infrastructure — a **complete multi-scale physics-to-ML pipeline** for Na-ion batteries using three simulation fidelity levels and a unified machine learning framework on top.

---

## 0.5 Feasibility Validation, Reusable Assets & Final Plan
### *(Hands-on validated June 2026 — every load-bearing claim below was actually run, not assumed)*

> **Verdict: GREENLIGHT.** The core engine works and is *faster* than the proposal assumed. Three conceptual errors in the original draft were found and fixed (see ❌ items). Several top-lab open assets let us stand on existing work instead of rebuilding it.

### A. What was tested live (with proofs)

| Claim | Test run | Result | Status |
|-------|----------|--------|--------|
| `pybamm.sodium_ion.BasicDFN()` exists (Chayambuka 2022) | Installed PyBaMM 26.6, instantiated + solved | ✅ Loads, 51 params | **VERIFIED** |
| ~2 s per PyBaMM sim | Timed default + 1C discharge solve | ✅ **0.45 s** (4× faster) → 5000-sim sweep ≈ 40 min single-core | **VERIFIED — better than claimed** |
| Parameter perturb → re-solve (LHS sweep premise) | Perturbed `Negative particle diffusivity [m2.s-1]`, re-solved | ✅ Works | **VERIFIED** |
| HOMO / LUMO / dipole computable for a Na-ion solvent | Built propylene carbonate (RDKit) → **PySCF B3LYP/def2-SVP** | ✅ HOMO = **−7.95 eV**, dipole = **5.45 D** (exp ≈ 4.9 D), 19 s/molecule → ~5 min for 15 solvents | **VERIFIED — matches experiment** |
| Fast electronic screening | Same molecule → **GFN2-xTB (tblite+ASE)** | ✅ Runs in **0.01 s**, gives dipole + partial charges | **VERIFIED** |
| **Sweep won't choke** (5000-sim risk) | 80-sample LHS over *full* proposal param ranges, 1C discharge | ✅ **100% solver success (80/80)**, capacity CV=10% (learnable spread), 0.40 s/sim → 5000 ≈ **34 min** | **VERIFIED — biggest hidden risk cleared** |
| **Rate capability robustness** | C/10, C/5, 1C, 2C, 5C on Na model | ✅ **All solve**, capacity falls physically 0.0027→0.0005 Ah | **VERIFIED** |
| **Open fidelity pair for ML development** | Coarse-mesh (8 pts) vs fine-mesh (30 pts) DFN | ✅ 0.34 s vs 0.78 s, **3.5% capacity gap** → lets the multi-fidelity GP be built immediately, in parallel with the COMSOL thermal anchor | **VERIFIED** |
| **ML spine end-to-end** (MF-GP + importance + BO) | Built 2-fidelity Na-ion dataset (55 low + 22 high), ran AR1 multi-fidelity GP, RandomForest importance, EI Bayesian optimization | ✅ **All 4 ran on real PyBaMM data.** MF-GP trains & matches/beats single-fi; importance ranking is *physically sensible* (neg. particle radius 53% + neg. diffusivity 31% dominate capacity = anode-limited, correct Na physics); BO loop proposes→evaluates→updates cleanly | **VERIFIED** |
| **MD compute kill-test** (the one real unknown) | Built 106-atom Na-salt-in-solvent cluster, ran real GFN2-xTB molecular dynamics, measured speed | ✅ Pipeline runs (Na⁺ moves); **0.80 s/step CPU → ~22 h/100 ps single-core.** Verdict: MD needs a **GPU (MACE-OFF)** or parallelization; **1× 16 GB NVIDIA GPU is sufficient**; Electrolyte Genome is the no-MD fallback | **MEASURED — risk quantified, mitigations in hand** |

*Reproducible test scripts: `test_pybamm_naion.py`, `test_electronic_props.py`, `test_pybamm_robustness.py`, `test_ml_spine.py` (validation folder). All tools installed via plain `pip` on Python 3.12 — no conda, no HPC. Full stack (PyBaMM, torch, GPyTorch, BoTorch, SHAP, XGBoost, PySCF, tblite/xTB, RDKit) installed and the ML spine ran end-to-end on real Na-ion data. Note: the MF-GP "+99%" and BO "+0%" figures are tiny-validation-set artifacts (near-saturated toy objective), not production numbers — the point proven is that every component **runs cleanly end-to-end on real data**, so production scale-up is execution, not research risk.*

### B. Three corrected errors (the original draft was wrong here)

1. **❌ MACE cannot output HOMO/LUMO/dipole.** MACE-MP/MACE-OFF are *interatomic potentials* — they give energy & forces only, no electronic structure. **✅ FIX:** compute HOMO/LUMO/dipole with **PySCF DFT** (validated above) or pull them pre-computed from the **Electrolyte Genome** (see assets). Use **GFN2-xTB** for fast dipole/charge screening.
2. **❌ MACE-MP is trained on inorganic crystals** — wrong distribution for organic liquid electrolytes. **✅ FIX:** use **MACE-OFF23/24** (organic-trained, H/C/N/O/P/S/F… — covers all our solvents) for the electrolyte MD, with **GFN2-xTB** as a literature-validated alternative (published for Na⁺/Li⁺ carbonate MD).
3. **❌ Solvent self-diffusion ≠ electrolyte conductivity κ_e.** Feeding solvent D into PyBaMM's `kappa_e` is physically invalid. **✅ FIX:** simulate the *salt-in-solvent* system, extract **ion** MSDs, apply **Nernst–Einstein** (state ideal-dissociation caveat).
4. **Tooling:** prefer **GPyTorch** over `emukit/GPy` (latter semi-abandoned, breaks on Py 3.12). Remap Sobol parameter names to real PyBaMM keys (`Negative particle diffusivity [m2.s-1]`, `Negative electrode exchange-current density [A.m-2]`, `… porosity`, `… particle radius [m]`, `… conductivity [S.m-1]`).

### C. Reusable open assets (validated reachable — stand on these, don't rebuild)

| Asset | Source | License | Use in our pipeline |
|-------|--------|---------|---------------------|
| **BatteryLife — Na-ion subset** | HuggingFace `Battery-Life/BatteryLife_Processed` (`NA-ion/` + `NA-ion_labels.json`) — SIGKDD 2025 | **MIT** | **Layer 9 real-data validation.** 31 real Na-ion cells, 12 protocols, 2.0–4.0 V, 25 °C. ✅ confirmed reachable via `huggingface_hub` |
| **Electrolyte Genome** | LBNL/Persson via Materials Project molecules API | Open/free (API key) | **Layer 1 electronic properties.** ~4830 electrolyte molecules with HOMO/LUMO/IP/EA at B3LYP/6-31+G(d)+PCM → pre-computed values + benchmark for our PySCF calcs |
| **MACE-OFF23 / MACE-OFF24** | HuggingFace `mace-foundations` + GitHub `ACEsuit/mace-off` | Academic (ASL) | **Layer 1 electrolyte MD** (`from mace.calculators import mace_off`) |
| **PyBaMM Na-ion DFN** | Built-in `pybamm.sodium_ion.BasicDFN()` + Chayambuka 2022 set | BSD-3 | **Layer 2** — already validated running |
| **MACE-MP-0** | HuggingFace `mace-foundations` | Academic | Optional — only for crystalline electrode surfaces, not liquids |

### D. Honest timeline (AI accelerates *code*, not the compute, the thermal model, or the writing)

- **Stage-1 preprint** (PySCF+xTB on ~6 solvents → PyBaMM sweep → GP + SHAP + BO): **6–10 weeks** → Sept–Oct 2026. A complete paper on its own.
- **Full preprint** (adds the COMSOL P2D+thermal tier + PINN + all 11 figures): **4–5 months** → **Dec 2026 target is realistic.**
- **Journal (Digital Discovery)**: + **3–6 months** peer review → mid-2027.
- The work that takes real calendar time is the molecular MD compute, the COMSOL thermal build, and writing the science carefully — all planned for, none of it ML-code risk.

### D2. The 3-tier multi-fidelity strategy — COMSOL is the high-fidelity anchor

The surrogate spans three physics-fidelity levels, and the **COMSOL thermally-coupled tier is the high-fidelity anchor of the whole multi-fidelity story.** It adds the temperature physics (heat generation, thermal gradients) that an *isothermal* model fundamentally cannot capture — and that **isothermal ↔ thermal** jump is what makes "multi-fidelity" a *real scientific claim* rather than a numerical exercise. **Note (dimensionality):** the essential fidelity gap is *thermal*, not geometric — so the anchor is a **P2D + thermal** model (thermally-coupled DFN), which is what PyBaMM *can't* do. True 2D geometry is an **optional stretch** (adds spatial effects), not required for the multi-fidelity claim. Because the open PyBaMM tiers are already validated, the ML pipeline and the COMSOL thermal model are built **in parallel** and converge into the final surrogate. (Literature precedent for in-model fidelity hierarchies: arXiv 2312.17329, PE-FNO.)

| Tier | Fidelity | Engine | Cost | Role in the surrogate |
|------|----------|--------|------|------|
| **Low** | coarse-mesh isothermal P2D (DFN) | PyBaMM (open) | ~0.34 s | broad, cheap coverage of the parameter space |
| **High** | fine-mesh isothermal P2D (DFN) | PyBaMM (open) | ~0.78 s | ✅ validated; sharpens the cheap tier |
| **Anchor** | **P2D + thermal** (thermally-coupled DFN; true 2D optional) | **COMSOL** | ~8 min | **the high-fidelity centerpiece — temperature physics + the key differentiator vs other Na-ion ML work** |

*The open PyBaMM tiers let ML development start on day one; the COMSOL thermal tier is the scientific centerpiece the surrogate is built to reproduce cheaply.*

### E. Final plan — ship in two stages

**Stage 1 (every tool validated, 100% PyBaMM sweep success):**
Electrolyte Genome / PySCF electronic props (~6 solvents) → xTB MD for κ_e via Nernst–Einstein → PyBaMM 5000-sim LHS sweep (~34 min) → Sobol → GPyTorch multi-fidelity GP (open coarse/fine DFN pair for development) + XGBoost baseline → SHAP → BoTorch BO → conformal UQ → validate against **BatteryLife Na-ion**. A complete, submittable paper on its own.

**Stage 2 (the high-fidelity tier + extensions — where the paper becomes distinctive):**
The **COMSOL P2D + thermal tier** (thermally-coupled DFN; true 2D optional) is added as the high-fidelity anchor (temperature maps, thermal gradients — the centerpiece that sets this apart from other Na-ion ML work) → MACE-OFF MD cross-check → PINN comparison.

---

### F. Honest Scientific Critique & Reframe (the pipeline runs — does the *science* survive a tough reviewer?)

> The tooling is fully validated. But "the code runs" ≠ "the science is sound." A hard look found **2 real flaws + 2 improvements**. None require abandoning anything — fixing them makes the paper *more honest AND more novel* (one shaky claim → three clean ones). All solutions below have published precedent.

**🔴 FIX 1 — Molecular descriptors (HOMO/LUMO/dipole) physically CANNOT drive PyBaMM capacity. (Proven by inspection.)**
The Na-ion DFN ingests the electrolyte through *only*: `Electrolyte conductivity`, `Electrolyte diffusivity`, `Cation transference number`, `Initial concentration`. Electronic-structure inputs (HOMO/LUMO/stability/SEI): **none exist in the model.** So a SHAP claim like *"solvation energy explains 34% of capacity variance"* would be **spurious** — all solvent descriptors co-vary (computed from the same molecules), so the ML just attributes capacity to whatever correlates with κ_e. A reviewer will catch this instantly (no causal pathway).
- **✅ Solution — split into two honest models** (established QSPR methodology, e.g. MGEA 2025 LASSO+SHAP electrolyte, COSMO-RS+ML ionic-conductivity):
  - **(a) Molecular → transport QSPR:** predict κ_e / D_e from molecular descriptors (MACE-OFF/xTB MD + HOMO/dipole/solvation). *Genuinely novel for Na-ion.* SHAP here = "which molecular feature predicts conductivity" → legitimate.
  - **(b) Transport → cell:** PyBaMM maps κ_e/D_e + electrode params → capacity. SHAP here = "which cell parameter drives capacity" → legitimate.
  - **HOMO/LUMO** used *separately* for electrochemical-stability-window screening — never as a capacity driver.

**🔴 FIX 2 — "Validation against BatteryLife real data" is overstated.**
BatteryLife Na-ion = 31 *commercial* cells, measuring *cycle-life fade*, *unknown* DFN params. Our model = *single discharge, no degradation*. You can't quantitatively validate fade you don't model against cells whose parameters you don't know.
- **✅ Solution:** Quantitative validation → vs **Chayambuka 2022 experimental curves** (same system, known params — already Friend A's task). BatteryLife/literature → **qualitative realism bounds only** ("are predicted capacities/rate-fade in the right ballpark?"). Do **not** report a fabricated MAE against it.

**🟡 IMPROVE 1 — Mesh-only multi-fidelity is scientifically thin.** Coarse vs fine mesh = *same physics*, just numerical convergence (measured 3.5% gap, GP ρ≈1.0 → nearly identical). A multi-fidelity result there just learns discretization error.
- **✅ Solution:** The scientifically meaningful fidelity gap is across *different physics*: **isothermal ↔ thermal**. **This is the real scientific justification for COMSOL** (or a PyBaMM thermal model) — it's what makes "multi-fidelity" a legitimate claim, not a bonus. Keep the mesh pair only as a labeled numerical de-risk fallback; lead with the physics-level gap.

**🟡 IMPROVE 2 — BO "optimal design" finds trivial, possibly unrealizable extremes** (maximizing capacity just pushes parameters to box corners).
- **✅ Solution:** Constrain BO to *physically achievable* ranges — exactly what **electrode DFT (Friend B, Option 1 below)** provides (migration barriers → realistic diffusivity). Ties the team together and makes BO meaningful.

**✅ Already sound (no fix):** PyBaMM sweep + Sobol (valid — state it's a property of the Chayambuka model, not all Na-ion reality); the multi-fidelity *methods* question (answerable, standard accepted framing = efficiency tool, not new physics); the **open-pipeline / open-data infrastructure** contribution (genuinely valuable — Na-ion lacks this); molecular property tables as *screening*.

**Reframed contributions (what now survives review):**
1. First **open multi-scale Na-ion simulation→ML pipeline** + released datasets (infrastructure).
2. **Molecular→transport QSPR** for Na-ion electrolytes (structure-property, with SHAP).
3. **Physics-level multi-fidelity** cell surrogate (isothermal↔thermal), honestly framed as an efficiency emulator of the simulator.
4. **DFT-grounded design optimization** (BO constrained to realizable electrode descriptors).
5. Separate **electrochemical-stability screening** (HOMO/LUMO) of Na-ion solvents.

---

### G. Friend B's role — three options (she has a COMSOL license; pick one)

Friend B owns a **core pillar** — and there are three strong ways to do it. The thermal tier is scientifically load-bearing here (per Improve 1, it's what gives the multi-fidelity claim its teeth), so the COMSOL route is genuinely central to the paper.

| | **Option B1 — COMSOL thermal as CORE** | **Option B2 — Electrode Materials (MACE-MP + NEB)** | **Option B3 — BOTH (strongest)** |
|---|---|---|---|
| **What** | **P2D + thermal** Na-ion model (thermally-coupled DFN; true 2D optional); the *high-fidelity physics tier* | Na⁺ migration barriers (NEB), intercalation voltage, formation energy, volume change for cathodes (NVPF, NaMnO₂, Prussian blue, NaFePO₄) | Electrode DFT as the core pillar **+** COMSOL thermal tier |
| **Why core** | Provides the **real physics fidelity gap** (isothermal↔thermal) that makes multi-fidelity legitimate | **Feeds surrogate inputs** (electrode descriptors) + **grounds BO** in realizable ranges → 3-pillar story: electrolyte (you) + electrode (B) + cell (A) | Both: legit fidelity gap *and* realizable design constraints |
| **Tools / license** | COMSOL (she has it) | MACE-MP + ASE NEB + pymatgen (open, GPU-friendly) | COMSOL + MACE-MP |
| **She learns** | Multiphysics FEM, thermal modeling | DFT, NEB, ML interatomic potentials, materials screening | Both skill sets |
| **Citable precedent** | COMSOL Li-ion thermal (adapt to Na) | **MACE-MP NEB benchmark — Digital Discovery 2026** (your target journal!), NASICON DFT+ML screening | Both |
| **Effort** | A substantial multi-week model build — plan time for it; the thermal data is scientifically essential | Moderate — open tools, validated approach | Most work, but the highest-impact paper |

**Recommendation:** **B3 (both)** is scientifically strongest — electrode DFT fixes Improve 2 (realizable BO) *and* COMSOL fixes Improve 1 (real fidelity gap). If time-limited, **B1** alone is excellent (thermal is the legitimate fidelity gap) or **B2** alone (cleanest open 3-pillar story). Friend B picks based on appetite for COMSOL vs atomistic modeling.

---

## 1. Research Objectives

### Primary Objective
Build an **open multi-scale Na-ion modeling→ML pipeline** with three honest, separable links: (a) a **molecular→transport QSPR** that predicts electrolyte conductivity/diffusivity from solvent descriptors; (b) a **physics-level multi-fidelity cell surrogate** (isothermal↔thermal) that emulates the electrochemical model at a fraction of the cost; and (c) **DFT-grounded design optimization** constrained to physically realizable parameters.

> **⚠️ Reframed June 2026 (see Section 0.5-F):** the original "predict capacity directly from molecular descriptors" claim is *physically invalid* — the DFN model has no electronic-structure inputs, so HOMO/LUMO/dipole have no causal pathway to capacity. The split above is the corrected, defensible framing.

### Secondary Objectives
1. Compute electrolyte properties for Na-ion solvents — **transport** (κ_e, D_e via MACE-OFF/xTB MD) feeds the cell model; **electronic** (HOMO/LUMO via PySCF) feeds a *separate* electrochemical-stability screening
2. Perform global sensitivity analysis (Sobol) of Na-ion DFN parameters — *scoped as a property of the Chayambuka 2022 model, not all Na-ion reality*
3. Demonstrate Bayesian optimization of Na-ion design, **constrained to realizable ranges** (from electrode DFT)
4. Quantify the Pareto tradeoff between simulation fidelity and computational cost (honest efficiency claim, not new physics)
5. Release all simulation data and surrogate model code as open-source tools

### The Central Scientific Question
*"Can molecular electrolyte descriptors predict Na-ion transport properties (QSPR), and can a physics-level multi-fidelity surrogate (isothermal↔thermal) reproduce high-fidelity cell behavior at near-zero cost — validated quantitatively against the source model and qualitatively against real Na-ion cells?"*

---

## 2. Team Responsibilities & Work Split

### Person 1 — You (ML + Molecular Simulation)
**Tools:** MACE-OFF, GFN2-xTB (tblite), PySCF, ASE, RDKit, Materials Project API, GPyTorch, scikit-learn, BoTorch, SHAP, SALib, DeepXDE (PINN)

| Task | Description | Timeline |
|------|-------------|----------|
| MACE-OFF / xTB / PySCF setup | Install, test on PC molecule, benchmark vs experiment (validated: HOMO −7.95 eV, dipole 5.45 D) | Week 1-2 |
| Materials Project featurization | Query crystal descriptors for Na-ion electrode materials via mp-api | Week 2-3 |
| Molecular screening (2 engines) | MACE-OFF/xTB MD → κ_e/D_e (transport, feeds cell); PySCF → HOMO/LUMO (separate stability screen) for 6-15 solvents | Week 3-6 |
| Molecular→transport QSPR | Train + SHAP: molecular descriptors → κ_e/D_e (the corrected, valid link) | Week 6-7 |
| Sobol sensitivity analysis | Identify most important DFN parameters from PyBaMM outputs | Week 7-8 |
| Multi-fidelity GP surrogate | Build/train GP on isothermal↔thermal fidelity pair (open coarse/fine fallback ready) | Week 9-12 |
| PINN model | Train physics-informed neural network as comparison model | Week 10-13 |
| BO optimization loop | Use surrogate to discover optimal Na-ion design | Week 12-14 |
| Uncertainty quantification | Conformal prediction intervals on surrogate outputs | Week 13-15 |
| SHAP analysis | Feature importance across all fidelity levels | Week 14-16 |
| Paper writing lead | Write all sections, coordinate co-author inputs | Week 16-20 |

### Person 2 — Friend A (PyBaMM)
**Tools:** PyBaMM, Python, SALib (for sampling), pandas

| Task | Description | Timeline |
|------|-------------|----------|
| Na-ion DFN setup | Implement Chayambuka2022 parameter set in PyBaMM | Week 1-3 |
| Literature parameter extraction | Collect additional Na-ion DFN parameters from 20+ papers | Week 2-5 |
| Latin Hypercube sampling | Design parameter space sampling scheme | Week 4-5 |
| Synthetic data generation | Run 5000-10000 GCD simulations across parameter space | Week 5-8 |
| Rate capability simulations | Simulate C/10, C/5, 1C, 2C, 5C for each parameter set | Week 7-9 |
| Literature validation | Compare PyBaMM outputs against published experimental data | Week 9-11 |
| Data formatting | Structure outputs into clean ML-ready format (CSV/HDF5) | Week 10-11 |
| Paper contribution | Write Section 3.2 (PyBaMM methodology and data) | Week 16-18 |

### Person 3 — Friend B — *core pillar; pick a role from Section 0.5-G*
> **Friend B chooses ONE of three core roles (see Section 0.5-G for the full comparison).** She has a COMSOL license, so all three are open to her. Recommended: **B3 (both)** if she has appetite, else **B1** or **B2**.

**Option B1 — COMSOL thermal as the high-fidelity physics tier** *(Tools: COMSOL Batteries module, MATLAB LiveLink)*

| Task | Description | Timeline |
|------|-------------|----------|
| Na-ion thermal model | **Start from the COMSOL 1D Na-ion example, add thermal coupling (Electrochemical Heating node) → P2D + thermal.** True 2D geometry optional if time allows | Week 1-5 |
| Parameter calibration | Validate COMSOL model against Chayambuka 2022 data | Week 4-7 |
| Thermal-electrochemical sweep | Run ~200 high-fidelity sims; **fix at 25 °C for the apples-to-apples multi-fidelity GP**, sweep temperature separately | Week 7-12 |
| Temperature effects | Simulate 10/25/40/60 °C (the isothermal↔thermal gap = the *real* fidelity difference) | Week 11-13 |
| Data export + write | Export to ML pipeline; write Section 3.4 (thermal results) | Week 13-18 |

**Option B2 — Electrode Materials (atomistic, open-source)** *(Tools: MACE-MP, ASE NEB, pymatgen, optional DFT)*

| Task | Description | Timeline |
|------|-------------|----------|
| Cathode set + relaxation | NVPF, NaMnO₂, Prussian blue, NaFePO₄ — relax with MACE-MP (validated for crystals) | Week 1-4 |
| Na⁺ migration barriers | CI-NEB via ASE + MACE-MP → ion mobility (→ rate capability, realistic D ranges) | Week 4-9 |
| Voltage + stability | Intercalation voltage, formation energy, volume change on sodiation | Week 7-11 |
| Descriptor export | Electrode descriptor table → surrogate inputs **+ realizable bounds for BO** | Week 11-13 |
| Validate + write | Benchmark a subset vs DFT-NEB literature; write electrode-materials section | Week 13-18 |

**Option B3 — Both:** B2 as the core pillar (electrode descriptors + realizable BO bounds) **+** B1 COMSOL thermal tier. The strongest, most complete paper.

> **Why Friend B's role is central:** the **thermal tier is what makes the multi-fidelity claim scientifically real** (isothermal↔thermal is a genuine physics gap), and the **electrode DFT is what makes the design optimization meaningful** (realizable constraints). Both are load-bearing contributions the paper is built around — Friend B owns a defining part of the science.

---

## 3. Technical Architecture — Full Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 0: FEATURIZATION                       │
│  Materials Project API → crystal descriptors for electrodes     │
│  (ionic radius, lattice params, formation energy, band gap)     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                 LAYER 1: MOLECULAR FIDELITY                     │
│  TWO ENGINES (corrected — see 0.5-F):                           │
│  TRANSPORT path -> feeds the cell model:                        │
│    MACE-OFF / GFN2-xTB MD -> ion MSD -> Nernst-Einstein         │
│    -> kappa_e, D_e ; + solvation energy, coordination number    │
│  ELECTRONIC path -> SEPARATE stability screen (NOT capacity):   │
│    PySCF DFT / Electrolyte Genome -> HOMO, LUMO, dipole         │
│  Solvents: PC, EC, DMC, DME, DOL, TEP, TMP, FEC, VC            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│            LAYER 2: ELECTROCHEMICAL FIDELITY (1D)               │
│  PyBaMM Na-ion DFN → synthetic GCD curves                       │
│  • 5000-10000 simulations via Latin Hypercube Sampling          │
│  • Parameters varied: D_s_neg, D_s_pos, k_neg, k_pos,          │
│    ε_neg, ε_pos, R_neg, R_pos, σ_neg, σ_pos,                   │
│    kappa_e / D_e (from Layer-1 QSPR), L_neg, L_pos             │
│  • Outputs: capacity, ICE, voltage profile, rate capability     │
│  • Speed: ~0.45s per sim (validated); 100% sweep success        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│   LAYER 3: THERMAL-ELECTROCHEMICAL FIDELITY (P2D+THERMAL, ANCHOR)│
│  COMSOL → high-fidelity thermally-coupled DFN (true 2D optional) │
│  • ~200 high-fidelity simulations (~8 min each)                 │
│  • Adds: heat generation, temperature gradients (the real gap)  │
│  • Temperature range: 10°C to 60°C                              │
│  • Outputs: capacity + thermal profiles, temperature maps       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│               LAYER 4: SENSITIVITY ANALYSIS                     │
│  SALib (Sobol indices) → which parameters matter most           │
│  • First-order and total-order Sobol indices                    │
│  • Applied to PyBaMM outputs                                    │
│  • Reduces input dimensionality for surrogate training          │
│  • Scientific result: identifiable vs unidentifiable params     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│            LAYER 5: MULTI-FIDELITY SURROGATE                    │
│  Two models trained and compared:                               │
│                                                                 │
│  A) Multi-Fidelity Gaussian Process (GPyTorch)                  │
│     Fidelity gap = DIFFERENT PHYSICS (real), corrected:         │
│     • Low: isothermal DFN, PyBaMM (5000 pts)                    │
│     • High: thermal-coupled (COMSOL/PyBaMM-thermal, ~200)       │
│     • (coarse/fine mesh pair = numerical de-risk fallback)      │
│     • Outputs prediction + uncertainty interval                 │
│                                                                 │
│  B) Physics-Informed Neural Network (DeepXDE)                   │
│     • DFN equations embedded as loss function constraints       │
│     • Trained on same data as GP                                │
│     • Comparison model — does physics constraint help?          │
│                                                                 │
│  C) XGBoost baseline                                            │
│     • No physics, pure data-driven                              │
│     • Lower bound on accuracy                                   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│          LAYER 6: UNCERTAINTY QUANTIFICATION                    │
│  Conformal Prediction (crepes library)                          │
│  • Calibrated prediction intervals on surrogate outputs         │
│  • "280 ± 15 mAh/g at 95% confidence"                          │
│  • Flags high-uncertainty regions → trigger new simulation      │
│  • Makes surrogate trustworthy for real use                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│              LAYER 7: BAYESIAN OPTIMIZATION                     │
│  BoTorch → find optimal Na-ion electrode/electrolyte design     │
│  • Surrogate as objective function                              │
│  • Maximize: capacity + rate capability                         │
│  • Constrain to PHYSICALLY REALIZABLE ranges from electrode     │
│    DFT (Friend B NEB→achievable D), not just box bounds         │
│  • Validate top-3 BO suggestions with PyBaMM                    │
│  • Result: best achievable (not just box-corner) design         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                  LAYER 8: SHAP ANALYSIS                         │
│  SHAP (TreeSHAP / KernelSHAP) — on TWO models separately:       │
│  • (a) molecular descriptor → which predicts kappa_e/D_e?       │
│  • (b) cell/electrode parameter → which drives capacity?        │
│    (NEVER molecular descriptor → capacity: no causal path)      │
│  • Compare Na-ion sensitivity vs Li-ion (from literature)       │
│  • Two valid findings along two real causal pathways            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│              LAYER 9: VALIDATION                                │
│  Real literature data (never used in training)                  │
│  • 15-20 Na-ion papers with reported capacity and rate data     │
│  • BatteryLife dataset (Na-ion subset)                          │
│  • Report: MAE, RMSE, R² for each model at each fidelity       │
│  • Pareto plot: accuracy vs computational cost                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Methodology — Your Work (ML Sections)

### 4.1 Molecular Simulation (MACE-OFF + xTB + PySCF)

**Which tool for what (corrected — see Section 0.5-F):**
- **MACE-OFF23/24** = ML interatomic potential trained on *organic molecules* (the correct choice for liquid electrolytes; MACE-MP is for inorganic crystals → use it only for Friend B's electrode crystals). Plugs into ASE. Used for energetics + MD.
- **GFN2-xTB** (tblite) = fast semi-empirical QM for dipole/charges + the salt-in-solvent MD (literature-validated for Na⁺ carbonates).
- **PySCF** (DFT) = HOMO/LUMO/dipole (validated: PC HOMO −7.95 eV, dipole 5.45 D).

**Installation:**
```bash
pip install mace-torch ase rdkit      # MACE-OFF (use mace_off, not mace_mp, for electrolytes)
pip install tblite pyscf              # xTB + DFT for electronic properties
```

**What you compute for each solvent:**

> **⚠️ CORRECTED (validated June 2026):** The original draft attributed HOMO/LUMO/dipole to MACE — this is impossible (MACE is an interatomic potential: energy & forces only, no electronic structure). The corrected engine assignment below was validated on propylene carbonate (see Section 0.5). **Two engines:** quantum chemistry (PySCF/xTB or Electrolyte Genome) for electronic properties; ML potential (MACE-OFF/xTB MD) for energetics & dynamics.

| Property | Physical meaning | How to compute (CORRECTED) | Why it matters for Na-ion |
|----------|-----------------|----------------|--------------------------|
| Na⁺ solvation free energy | How strongly Na⁺ binds to solvent | **MACE-OFF or xTB MD**: thermodynamic integration / energy difference, Na⁺ in vacuum vs solvent box | Determines ion dissociation and conductivity |
| Ion self-diffusion → κ_e | How fast ions move in electrolyte | **MACE-OFF/xTB MD** of *salt-in-solvent*: ion MSD → **Nernst–Einstein** (not solvent self-diffusion) | Feeds PyBaMM `kappa_e` (with dissociation caveat) |
| HOMO energy | Oxidative stability of solvent | **PySCF DFT** (B3LYP/def2-SVP) or **Electrolyte Genome** lookup — *not MACE* | Predicts electrochemical stability window |
| LUMO energy | Reductive stability (SEI formation) | **PySCF DFT** or **Electrolyte Genome** lookup — *not MACE* | Predicts SEI formation potential |
| Dipole moment | Polarity of solvent molecule | **PySCF** (validated 5.45 D for PC vs exp ~4.9) or **GFN2-xTB** (0.01 s, fast screen) — *not MACE* | Correlates with dielectric constant |
| Coordination number | How many solvent molecules surround Na⁺ | Radial distribution function from **MACE-OFF/xTB MD** trajectory | Indicates solvation structure |

**Solvents to screen:**
- Carbonates: PC (propylene carbonate), EC (ethylene carbonate), DMC (dimethyl carbonate)
- Ethers: DME (dimethoxyethane), DOL (1,3-dioxolane), TEGDME
- Phosphates: TEP (triethyl phosphate), TMP (trimethyl phosphate) — van Ekeren's systems
- Additives: FEC (fluoroethylene carbonate), VC (vinylene carbonate)

**Workflow:**
```python
from mace.calculators import mace_mp
from ase.io import read
from ase.build import molecule
from ase.md.langevin import Langevin
from ase import units
import numpy as np

# Load universal potential
calc = mace_mp(model="medium", dispersion=True, device="cpu")

# For each solvent:
# 1. Build molecule from SMILES using RDKit
# 2. Attach MACE-OFF calculator (mace_off, not mace_mp, for organic solvents)
# 3. Geometry optimize
# 4. Compute single-point energy + dipole
# 5. Build solvation box (Na+ + 50 solvent molecules)
# 6. Run 100ps NVT MD at 298K
# 7. Extract MSD → diffusion coefficient
# 8. Extract RDF → coordination number
```

**Output:** A table of 10-15 solvents × 6 properties. This becomes your electrolyte feature matrix fed into PyBaMM and the surrogate.

**Validation:** Compute κ_e via **Nernst–Einstein from ion MSDs** (MACE-OFF or GFN2-xTB MD of the salt-in-solvent system) and compare against experimental conductivity in van Ekeren's papers (NaFSI/TEP, NaPF₆/EC:DMC). Report % error. Cross-check HOMO/LUMO/dipole against the **Electrolyte Genome** (B3LYP/6-31+G(d)+PCM) where the solvent is present. This is your molecular-layer benchmarking result. *(Note: GFN2-xTB MD is literature-validated for Na⁺/Li⁺ carbonate electrolytes; it overestimates ion–solvent binding, so benchmark per system.)*

---

### 4.2 Materials Project API Featurization

**What you get:**
Crystal structure descriptors for Na-ion electrode materials — cathodes (NVPF, NaMnO₂, NaFePO₄, Prussian blue analogues) and anode (hard carbon approximated as disordered graphite).

```python
from mp_api.client import MPRester

with MPRester("YOUR_API_KEY") as mpr:
    # Query NVPF
    docs = mpr.materials.search(
        formula="Na3V2(PO4)2F3",
        fields=["material_id", "formation_energy_per_atom",
                "band_gap", "volume", "density",
                "elements", "symmetry"]
    )
```

**Features extracted:**
- Formation energy per atom (thermodynamic stability)
- Band gap (electronic conductivity proxy)
- Volume per formula unit (relates to Na⁺ intercalation space)
- Crystal system / space group (structural descriptor)
- Average ionic radius of transition metal (activity descriptor)

These become the **electrode descriptors** fed into the surrogate alongside the MACE-OFF/xTB electrolyte transport properties.

---

### 4.3 Sobol Global Sensitivity Analysis

**Why this is important:**
The Na-ion DFN model has 20+ parameters. Training a surrogate on all 20 simultaneously is inefficient and overfits. Sobol analysis tells you which 8-10 parameters actually drive performance variation — the rest can be fixed to their literature values.

**Tool:** SALib (pip install SALib)

```python
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
import pybamm
import numpy as np

# Define parameter space
problem = {
    'num_vars': 10,
    'names': ['D_s_neg', 'D_s_pos', 'k_neg', 'k_pos',
              'eps_neg', 'eps_pos', 'R_neg', 'R_pos',
              'sigma_neg', 'kappa_e'],
    'bounds': [
        [1e-15, 1e-12],   # D_s_neg: solid diffusivity anode
        [1e-14, 1e-11],   # D_s_pos: solid diffusivity cathode
        [1e-12, 1e-9],    # k_neg: reaction rate constant anode
        [1e-12, 1e-9],    # k_pos: reaction rate constant cathode
        [0.2, 0.6],       # eps_neg: porosity anode
        [0.2, 0.6],       # eps_pos: porosity cathode
        [1e-7, 1e-5],     # R_neg: particle radius anode
        [1e-7, 1e-5],     # R_pos: particle radius cathode
        [10, 1000],       # sigma_neg: electronic conductivity
        [0.1, 2.0],       # kappa_e: electrolyte conductivity (from MACE-OFF/xTB MD QSPR)
    ]
}

# Generate Sobol sample
param_values = sobol_sample.sample(problem, 1024)

# Run PyBaMM for each sample (coordinate with Friend A)
outputs = run_pybamm_batch(param_values)  # Friend A provides this function

# Analyze
Si = sobol_analyze.analyze(problem, outputs['capacity'])
print(Si['S1'])   # First-order indices
print(Si['ST'])   # Total-order indices
```

**Expected result:** 3-5 parameters dominate (likely D_s_neg, kappa_e, eps_neg). The others have near-zero Sobol indices. This is your first scientific finding.

---

### 4.4 Multi-Fidelity Gaussian Process Surrogate

**Concept:**
A multi-fidelity GP learns from data at two fidelity levels simultaneously — treating low-fidelity (PyBaMM) and high-fidelity (COMSOL) as correlated information sources. It's more accurate than training on either alone because it learns the systematic difference (bias) between fidelity levels.

**Tool:** emukit (pip install emukit) or GPyTorch

```python
import emukit
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import (
    convert_x_list_to_array,
    convert_xy_lists_to_arrays
)

# X_low: PyBaMM parameter inputs (5000 points)
# Y_low: PyBaMM capacity outputs
# X_high: COMSOL parameter inputs (200 points, subset of above)
# Y_high: COMSOL capacity outputs

X_train, Y_train = convert_xy_lists_to_arrays(
    [X_low, X_high], [Y_low, Y_high]
)

# Build linear multi-fidelity model (AR1 model)
kernels = [GPy.kern.RBF(input_dim), GPy.kern.RBF(input_dim)]
model = GPyLinearMultiFidelityModel(X_train, Y_train, kernels, n_fidelities=2)
model.optimize()

# Predict at new point with uncertainty
X_test_high = np.hstack([X_new, np.ones((len(X_new), 1)) * 1])  # fidelity=1
mean, variance = model.predict(X_test_high)
```

**What you report:**
- Training error vs test error at each fidelity level
- Improvement from low→high fidelity data
- Prediction time: milliseconds vs minutes

---

### 4.5 PINN Comparison Model

**Concept:**
A Physics-Informed Neural Network embeds the DFN governing equations as soft constraints in the loss function. The network can't make predictions that violate battery physics.

**Tool:** DeepXDE (pip install deepxde)

**What you implement:**
- Simplified SPM (Single Particle Model) equations as PDE constraints
- Neural network trained to satisfy boundary conditions from DFN
- Same input/output as GP surrogate for fair comparison

**Why include it:**
- Provides meaningful comparison: data-driven GP vs physics-constrained PINN
- PINN should perform better in low-data regime
- GP should be faster and more flexible
- This comparison is the kind of rigorous analysis reviewers appreciate

---

### 4.6 Bayesian Optimization Loop

**Concept:**
Use your trained surrogate as the objective function for BO. Find the Na-ion DFN parameter set that maximizes predicted capacity subject to physically realistic constraints.

**Tool:** BoTorch (pip install botorch)

```python
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

# Your trained surrogate as the objective
def surrogate_objective(X):
    return surrogate_model.predict(X)

# Define bounds from literature parameter ranges
bounds = torch.tensor([
    [lower_bounds],
    [upper_bounds]
])

# Run BO loop
for iteration in range(20):
    # Fit GP to current observations
    gp = SingleTaskGP(train_X, train_Y)
    
    # Maximize Expected Improvement
    EI = ExpectedImprovement(gp, best_f=train_Y.max())
    candidate, _ = optimize_acqf(EI, bounds=bounds, q=1, num_restarts=5)
    
    # Evaluate candidate with PyBaMM (validation step)
    new_Y = run_pybamm(candidate)
    
    # Update observations
    train_X = torch.cat([train_X, candidate])
    train_Y = torch.cat([train_Y, new_Y])

# Report: top-3 optimal parameter sets and their predicted capacity
```

**What you report:**
- BO convergence curve (best capacity found vs iteration number)
- Top-3 optimal parameter sets with physical interpretation
- Validation: run these 3 sets through PyBaMM directly to confirm surrogate accuracy

---

### 4.7 Uncertainty Quantification

**Tool:** crepes (pip install crepes) for conformal prediction

**What you implement:**
- Calibrate prediction intervals using held-out calibration set
- Output: "prediction ± uncertainty" for every surrogate query
- Define threshold: if uncertainty > X%, flag for new simulation
- Active learning loop: high-uncertainty predictions trigger PyBaMM re-evaluation

**Why this matters:**
Without uncertainty quantification, a surrogate is a black box. With it, you know when to trust the prediction and when to run a real simulation. This makes the framework practically useful, not just academically interesting.

---

### 4.8 SHAP Analysis

**Tool:** shap (pip install shap)

> **⚠️ CORRECTED (see Section 0.5-F):** Run SHAP on the **two separate models**, never mixing molecular descriptors into the capacity prediction (they have no causal pathway — proven).

**What you compute (two clean SHAP analyses):**
- **(a) On the molecular→transport QSPR:** SHAP of molecular descriptors → **electrolyte conductivity κ_e / diffusivity D_e**. *"Which solvent feature predicts transport?"* — legitimate.
- **(b) On the transport→cell surrogate:** SHAP of cell/electrode parameters (+ κ_e/D_e) → **capacity, rate capability**. *"Which cell parameter drives performance?"* — legitimate.
- Summary, dependence, and interaction plots for each.

**The scientific finding you're looking for (correctly framed):**
Something like: *(a)* *"Na⁺ solvation free energy is the dominant molecular predictor of electrolyte conductivity (X% of variance in the QSPR)"* and, separately, *(b)* *"anode solid-phase diffusivity and particle radius dominate capacity in the cell surrogate (validated in test runs: ~31% + ~53%)."* Two interpretable results along two valid causal pathways — **not** a single (spurious) "molecular descriptor → capacity" claim.

---

## 5. Paper Structure — Full Outline

### Title
**"Multi-fidelity machine learning for Na-ion battery performance prediction: integrating molecular simulation, electrochemical modeling, and thermal-electrochemical digital twins"**

### Abstract (to be written last)
~250 words covering: problem, approach, key results (Pareto numbers, SHAP finding, BO result), conclusion.

---

### Section 1: Introduction (~800 words)

**1.1 Na-ion batteries — context and motivation**
- Why Na-ion matters (cost, abundance, post-2030 grid storage)
- Current state: commercialization in China (CATL, HiNa), Faradion in UK
- Performance gap vs Li-ion and what needs solving

**1.2 The data scarcity problem**
- ML has transformed Li-ion research (cite: Severson Nature Energy 2019, Attia Nature 2020)
- Na-ion has almost no open datasets, few validated parameter sets
- Existing ML for Na-ion is limited to hard carbon property prediction (cite 2025 papers)
- Nobody has built a full multi-scale ML framework for Na-ion

**1.3 Multi-fidelity ML — existing work and gap**
- Multi-fidelity methods in materials science (brief review)
- PINN surrogates for Li-ion (cite: Hassanaly 2024) — Li-ion only
- Transfer learning Li→Na (cite: EES Batteries 2026) — different problem
- Gap: no multi-fidelity ML specifically for Na-ion electrochemical + thermal prediction

**1.4 This work — contributions** *(corrected framing — see Section 0.5-F)*
Clearly numbered list:
1. First open multi-scale Na-ion simulation→ML pipeline + released datasets (infrastructure)
2. **Molecular→transport QSPR** for Na-ion electrolytes (MACE-OFF/xTB descriptors → κ_e/D_e, with SHAP) — the valid molecular link
3. First global sensitivity analysis (Sobol) of Na-ion DFN parameters (scoped to the Chayambuka 2022 model)
4. **Physics-level multi-fidelity** GP surrogate (isothermal↔thermal), framed as an efficiency emulator
5. **DFT-grounded** Bayesian optimization (constrained to realizable electrode descriptors)
6. Separate electrochemical-stability screening (HOMO/LUMO via PySCF) of Na-ion solvents
7. (Extensions) PINN comparison; open-source release of all data and code

**1.5 Paper organization**
One paragraph describing structure.

---

### Section 2: Methodology (~2000 words)

**2.1 Overall framework architecture**
- Pipeline figure (Figure 1) — the full stack from molecular to surrogate
- Brief description of each layer
- How layers connect and data flows between them

**2.2 Molecular-scale simulation (MACE-OFF + xTB + PySCF)**
*(Your work)*
- MACE-OFF (organic), GFN2-xTB, PySCF descriptions, training data, accuracy benchmarks
- ASE integration and workflow
- Solvent screening protocol (list of solvents, properties computed)
- MD simulation details (timestep, temperature, ensemble, duration)
- Validation against experimental conductivity data

**2.3 Materials Project featurization**
*(Your work)*
- API query protocol
- Features extracted and physical justification for each
- Electrode materials covered

**2.4 Electrochemical simulation — PyBaMM**
*(Friend A's work)*
- Na-ion DFN model equations (brief, cite Chayambuka 2022)
- Parameter space definition and literature sources
- Latin Hypercube Sampling scheme
- Simulation protocol (discharge rates, voltage windows)
- Dataset statistics (number of simulations, parameter ranges)

**2.5 Thermal-electrochemical simulation — COMSOL**
*(Friend B's work)*
- **P2D + thermal** (thermally-coupled DFN) — the essential fidelity gap is *thermal*, not geometric; true 2D geometry is an optional extension
- Started from the COMSOL 1D isothermal Na-ion example + added the Electrochemical Heating coupling
- Thermal coupling equations (heat generation, temperature dependence)
- Parameter sweep design (subset of PyBaMM space + temperature)
- Computational cost discussion

**2.6 Global sensitivity analysis**
*(Your work)*
- Sobol index theory (brief)
- SALib implementation
- How results informed surrogate input dimensionality

**2.7 Multi-fidelity surrogate model**
*(Your work)*
- Linear multi-fidelity GP theory (AR1 model)
- emukit implementation
- Training/validation/test split strategy
- Hyperparameter optimization

**2.8 PINN comparison model**
*(Your work)*
- DFN equations embedded as loss terms
- DeepXDE implementation
- Training protocol

**2.9 Bayesian optimization**
*(Your work)*
- BoTorch implementation
- Acquisition function choice (Expected Improvement)
- Constraint handling
- Validation protocol

**2.10 Uncertainty quantification**
*(Your work)*
- Conformal prediction framework
- Calibration protocol
- Active learning trigger criteria

**2.11 SHAP analysis**
*(Your work)*
- TreeSHAP vs KernelSHAP choice justification
- Feature interaction analysis

---

### Section 3: Results (~2500 words)

**3.1 MACE-OFF/xTB electrolyte screening results**
*(Your work — Figure 2)*
- Table: all solvents × all properties
- Figure: Na⁺ solvation energy vs experimental conductivity (validation)
- Key finding: which solvent gives best Na⁺ transport properties?
- Comparison: phosphate (TEP/TMP) vs ether (DME/DOL) vs carbonate (PC/EC)

**3.2 Global sensitivity analysis of Na-ion DFN parameters**
*(Your work — Figure 3)*
- Bar chart: Sobol first-order and total-order indices for each parameter
- Key finding: top 5 most influential parameters
- Comparison with Li-ion sensitivity from literature (if available)
- Discussion: which parameters are identifiable from GCD data?

**3.3 PyBaMM synthetic dataset characteristics**
*(Friend A's work — Figure 4)*
- Distribution plots of input parameters (were they sampled well?)
- Distribution of output capacity values
- Representative GCD curves at different parameter settings
- Validation against Chayambuka 2022 experimental data

**3.4 COMSOL thermal-electrochemical results**
*(Friend B's work — Figure 5)*
- Temperature maps at different C-rates
- How thermal gradients affect capacity
- Temperature dependence of capacity (Arrhenius behavior?)
- Systematic difference between PyBaMM (isothermal) and COMSOL (thermal)

**3.5 Multi-fidelity surrogate performance**
*(Your work — Figure 6)*
- Parity plot: predicted vs actual capacity for each model (GP, PINN, XGBoost)
- Table: MAE, RMSE, R² for each model at each fidelity level
- Learning curve: how does accuracy improve with more training data?
- Prediction time comparison (MACE-OFF/xTB: ms–s, PyBaMM: s, COMSOL: min, Surrogate: ms)

**3.6 PINN vs GP comparison**
*(Your work — Figure 7)*
- Accuracy comparison in low-data regime (10, 50, 100, 500, 5000 training points)
- When does physics constraint help? When does it hurt?
- Generalization outside training distribution

**3.7 Bayesian optimization results**
*(Your work — Figure 8)*
- BO convergence curve
- Top-3 discovered parameter sets with physical interpretation
- PyBaMM validation of BO-suggested designs
- How far are optimal parameters from Chayambuka 2022 baseline?

**3.8 Uncertainty quantification**
*(Your work — Figure 9)*
- Coverage plot: do 95% intervals contain 95% of true values?
- Uncertainty distribution across parameter space
- Which regions are well-characterized vs uncertain?

**3.9 SHAP feature importance**
*(Your work — Figure 10)*
- SHAP summary plot: all features ranked by importance
- Key finding stated clearly: "X is the dominant descriptor for capacity, Y for rate capability"
- Dependence plots for top-3 features
- Interaction effects

**3.10 Pareto analysis: fidelity vs computational cost**
*(All contributors — Figure 11)*
- X-axis: computational cost (log scale: ms → s → min → hours)
- Y-axis: prediction error (MAE on validation set)
- Points: MACE-OFF/xTB alone, PyBaMM alone, COMSOL alone, Multi-fidelity surrogate
- KEY RESULT: multi-fidelity achieves near-COMSOL accuracy at surrogate speed

---

### Section 4: Discussion (~800 words)

**4.1 Physical interpretation of SHAP results**
- What does it mean physically that parameter X dominates?
- How does this compare to Li-ion (where solid diffusivity usually dominates)?
- Implications for Na-ion electrode design priorities

**4.2 Where the framework succeeds and where it fails**
- Which Na-ion systems are well-predicted? Which are poorly predicted?
- MACE-OFF chosen over MACE-MP for liquids (MACE-MP is crystal-trained); remaining MACE-OFF/xTB accuracy limits reported honestly
- PyBaMM limitations (no SEI model, hard carbon plateau not perfectly captured)
- COMSOL assumptions (simplified geometry, no microstructure)

**4.3 Computational cost analysis**
- Full pipeline cost for one new prediction
- Break-even point: when is multi-fidelity surrogate more efficient than direct simulation?

**4.4 Comparison with existing Li-ion surrogate work**
- How does your framework compare to PINN surrogates for Li-ion (Hassanaly 2024)?
- How does Na-ion sensitivity differ from Li-ion?

**4.5 Path to experimental validation**
- What experimental data would most improve the surrogate?
- Which predicted optimal designs are most worth synthesizing?
- How could this framework integrate with autonomous labs (DTU AMPERE-2, Chueh group SDL)?

---

### Section 5: Conclusion (~300 words)

- Restate the problem
- Summarize the 6 contributions
- State the key quantitative results (Pareto numbers, top SHAP feature, BO improvement)
- State broader impact: framework is general, extendable to other battery chemistries
- Call to action: all code and data released openly at [GitHub link]

---

### Acknowledgements
- Computational resources used
- Any funding (MESC scholarship, DTU if relevant)
- arXiv endorser (Vegge)

---

### Data Availability
All simulation data released on Zenodo: [DOI placeholder]
All code released on GitHub: [link placeholder]

---

### References (~60-80 references)

Key references to read and cite:
- Chayambuka et al. 2022 — Na-ion DFN baseline (PyBaMM parameter set)
- Sulzer et al. 2021 — PyBaMM paper
- Batatia et al. 2023 — MACE-MP paper (foundation model; for crystalline electrodes)
- Kovács et al. 2023 — MACE-OFF paper (organic force field; for liquid electrolytes)
- Hassanaly et al. 2024 — PINN surrogate for Li-ion (compare against)
- He et al. 2026 (EES Batteries) — Li→Na transfer learning (differentiate from)
- Abolhasani & Kumacheva 2023 — SDL review
- van Ekeren PhD thesis + papers — electrolyte chemistry context
- Fisker-Bødker, Chang, Vegge 2025 (Digital Discovery) — AMPERE-2
- Attia et al. 2020 (Nature) — closed-loop battery optimization (Chueh group)
- Severson et al. 2019 (Nature Energy) — battery lifetime ML benchmark

---

## 6. Figures List

| Figure | Content | Who makes it | Tool |
|--------|---------|--------------|------|
| 1 | Full pipeline architecture schematic | You | Draw.io / matplotlib |
| 2 | MACE-OFF/xTB electrolyte screening results | You | matplotlib/seaborn |
| 3 | Sobol sensitivity analysis | You | SALib + matplotlib |
| 4 | PyBaMM synthetic dataset overview | Friend A | matplotlib |
| 5 | COMSOL thermal maps | Friend B | COMSOL + matplotlib |
| 6 | Multi-fidelity surrogate parity plots | You | matplotlib |
| 7 | PINN vs GP low-data comparison | You | matplotlib |
| 8 | BO convergence + optimal designs | You | BoTorch + matplotlib |
| 9 | Uncertainty quantification coverage | You | matplotlib |
| 10 | SHAP summary + dependence plots | You | shap library |
| 11 | Pareto: accuracy vs cost | All | matplotlib |

**You own 8 of 11 figures. Clear first-author contribution.**

---

## 7. Software Stack — Your Complete Environment

### Installation (all pip installable — VALIDATED on Python 3.12, no conda)
```bash
# Molecular simulation — ML potential + quantum chemistry (two engines)
pip install mace-torch ase rdkit          # MACE-OFF MD + geometry
pip install pyscf                          # DFT: HOMO/LUMO/dipole (VALIDATED)
pip install tblite                         # GFN2-xTB fast screen + MD (VALIDATED)

# Datasets / databases
pip install mp-api huggingface_hub         # Materials Project + BatteryLife/Electrolyte Genome

# Sensitivity analysis
pip install SALib

# Multi-fidelity ML  (prefer GPyTorch — emukit/GPy are semi-abandoned, break on Py3.12)
pip install gpytorch                        # use this for the multi-fidelity GP
# pip install emukit GPy                    # (optional/legacy — only if you must)

# PINN
pip install deepxde

# Bayesian optimization
pip install botorch

# Uncertainty quantification
pip install crepes

# Explainability
pip install shap

# General ML
pip install scikit-learn xgboost pandas numpy matplotlib seaborn

# Paper quality figures
pip install matplotlib seaborn plotly
```

### Compute requirements *(measured June 2026 — the MD is the one real cost)*
> **Kill-test result:** a GFN2-xTB MD of a 106-atom Na-salt-in-solvent cluster ran at **0.80 s/step on CPU** → ~**22 h per 100 ps single-core**. So CPU-only MD is impractical; **the MD needs a GPU (via MACE-OFF) or parallelization.**
> **GPU verdict:** **1× 16 GB NVIDIA GPU is sufficient for the MVP.** VRAM is a non-issue (electrolyte boxes are 200–500 atoms ≈ 1–4 GB). On a 16 GB GPU, MACE-OFF MD runs ~10–100× faster than xTB-on-CPU → a 100 ps trajectory drops to **~tens of min to a couple hours**. HPC is optional (only for huge boxes / many replicas / 50+ solvents).

- **Your laptop (CPU):** PySCF/xTB electronic props (fast), ML training, SHAP, BO — no GPU needed
- **16 GB NVIDIA GPU (Friend's):** **MACE-OFF MD on GPU** (the molecular-transport layer); also PINN training. *This closes the MD compute risk.*
- **Friend A's machine:** PyBaMM parameter sweeps (CPU — validated ~34 min for 5000 sims)
- **Friend B's machine:** COMSOL (license) *or* MACE-MP electrode NEB (open, also GPU-friendly) — per chosen role
- **HPC (optional):** only if scaling beyond the MVP
- **No HPC cluster needed**

---

## 8. Data Management Plan

### Folder structure (GitHub repo)
```
naion-multifidelity/
│
├── data/
│   ├── raw/
│   │   ├── pybamm_sweep_5000.csv          # Friend A generates
│   │   ├── comsol_thermal_200.csv         # Friend B generates
│   │   └── literature_validation_15.csv   # You collect
│   ├── processed/
│   │   ├── mace_mp_electrolyte_props.csv  # You generate
│   │   ├── materials_project_features.csv # You generate
│   │   └── sobol_indices.csv              # You generate
│   └── zenodo/                            # Upload to Zenodo for DOI
│
├── models/
│   ├── mf_gp_surrogate.pkl               # Trained surrogate
│   ├── pinn_model.pt                     # Trained PINN
│   └── xgboost_baseline.pkl              # Baseline model
│
├── notebooks/
│   ├── 01_mace_mp_screening.ipynb        # YOU
│   ├── 02_materials_project.ipynb        # YOU
│   ├── 03_sobol_analysis.ipynb           # YOU
│   ├── 04_pybamm_validation.ipynb        # Friend A
│   ├── 05_comsol_postprocess.ipynb       # Friend B
│   ├── 06_surrogate_training.ipynb       # YOU
│   ├── 07_pinn_comparison.ipynb          # YOU
│   ├── 08_bayesian_optimization.ipynb    # YOU
│   ├── 09_uncertainty_quant.ipynb        # YOU
│   └── 10_shap_analysis.ipynb            # YOU
│
├── paper/
│   ├── main.tex                          # LaTeX manuscript
│   ├── figures/                          # All paper figures
│   └── supplementary.tex                # SI material
│
└── README.md                             # How to reproduce everything
```

---

## 9. Timeline — Week by Week

### Phase 1: Setup & Learning (Weeks 1-4, July 1 - July 28)
**You:**
- Week 1: Install everything, run PyBaMM Na-ion notebook on Colab, get Materials Project API key
- Week 2: Run first PySCF + MACE-OFF calculation on PC molecule, verify results (validated: HOMO −7.95 eV, dipole 5.45 D)
- Week 3: Materials Project API — query NVPF and NaMnO₂ descriptors
- Week 4: Begin MACE-OFF/xTB screening (first 5 solvents)

**Friend A:**
- Week 1-2: Set up PyBaMM Na-ion DFN, reproduce Chayambuka 2022
- Week 3-4: Design Latin Hypercube sampling, test with 100 simulations

**Friend B:**
- Week 1-3: Build COMSOL Na-ion P2D+thermal model (from the 1D example + Electrochemical Heating), validate isothermal case against PyBaMM
- Week 4: Begin parameter sweep design

### Phase 2: Data Generation (Weeks 5-10, July 29 - September 8)
**You:**
- Week 5-6: Complete MACE-OFF/xTB screening (all 6-15 solvents)
- Week 7-8: Sobol sensitivity analysis (using Friend A's first batch of PyBaMM data)
- Week 9-10: Begin multi-fidelity GP implementation

**Friend A:**
- Week 5-8: Run full 5000-point PyBaMM parameter sweep
- Week 9-10: Literature validation, data formatting, hand data to you

**Friend B:**
- Week 5-10: Run COMSOL thermal simulations (~200 high-fidelity points)

### Phase 3: ML Development (Weeks 11-16, September 9 - October 20)
**You:**
- Week 11-12: Train multi-fidelity GP surrogate on PyBaMM data
- Week 12-13: Add COMSOL data to multi-fidelity model (when ready)
- Week 13: Train PINN comparison model
- Week 14: Run Bayesian optimization loop
- Week 15: Uncertainty quantification
- Week 16: SHAP analysis

**Friend A:** Literature search for additional validation data, help with figures

**Friend B:** Finalize COMSOL results, make thermal figures

### Phase 4: Writing (Weeks 17-22, October 21 - December 1)
- Week 17: Write Results sections (all)
- Week 18: Write Methodology sections
- Week 19: Write Introduction + Conclusion
- Week 20: Internal review (all three read everything)
- Week 21: Revisions based on feedback
- Week 22: Final check, submit to arXiv + Stanford application

### Phase 5: Journal Submission (January 2027)
- Polish paper based on arXiv community feedback
- Submit to Digital Discovery
- Peer review (2-4 months typical)

---

## 10. What To Study Before July 1

### Priority 1 — Run these immediately (this week)
```python
# 1. PyBaMM Na-ion — run this on Colab TODAY
!pip install pybamm
import pybamm
model = pybamm.sodium_ion.BasicDFN()
sim = pybamm.Simulation(model)
sim.solve([0, 3600])
sim.plot()

# 2. MACE-OFF — install and compute a solvent molecule energy (use mace_off for organics)
!pip install mace-torch ase
from mace.calculators import mace_mp
from ase.build import molecule
calc = mace_mp(model="small", dispersion=False)
mol = molecule("H2O")
mol.calc = calc
print(mol.get_potential_energy())

# 3. Materials Project — get API key from materialsproject.org
!pip install mp-api
from mp_api.client import MPRester
with MPRester("YOUR_KEY") as mpr:
    docs = mpr.materials.search(formula="NaCoO2", fields=["formation_energy_per_atom"])
    print(docs[0].formation_energy_per_atom)
```

### Priority 2 — Papers to read before July 1
1. **Chayambuka et al. 2022** (Electrochimica Acta) — your PyBaMM Na-ion baseline
2. **Batatia et al. 2023** (arXiv) — MACE-MP paper, understand what it is
3. **Abolhasani & Kumacheva 2023** (Nature Reviews) — SDL overview, context
4. **Hassanaly et al. 2024** (Journal of Energy Storage) — PINN for Li-ion, what you're building upon
5. **van Ekeren 2024** (ACS AMI) — the electrolyte chemistry you're screening

### Priority 3 — Concepts to understand
- Gaussian Process regression (3Blue1Brown video + scikit-learn docs)
- Multi-fidelity methods (GPyTorch / BoTorch multi-fidelity GP examples)
- Sobol sensitivity indices (SALib tutorial on GitHub)
- SHAP values (official SHAP documentation + 1 YouTube video)
- Bayesian optimization (BoTorch tutorial: "fitting a simple model")

---

## 11. Risk Register & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| COMSOL thermal model takes longer than planned | Medium | Low-Medium | ML pipeline is built in parallel on the open PyBaMM tiers, so the thermal anchor slots in when ready without holding up the rest. |
| MACE-OFF/xTB inaccurate for liquid electrolytes | Low-Medium | Medium | MACE-OFF is organic-trained; xTB lit-validated for Na carbonates. Report benchmark honestly — inaccuracy is itself a finding. |
| PyBaMM synthetic data doesn't match real data | Low-Medium | High | Real data is validation only. Gap analysis is publishable. |
| Three-person coordination fails | Medium | High | Weekly sync meeting. Clear data handoff deadlines. |
| Scope too large | Low (with team) | High | MVP = MACE-OFF/xTB + PyBaMM + GP + SHAP. COMSOL/PINN/BO = extensions. |
| Scooped by another paper | Low | High | Your exact combination (MACE-OFF/xTB electrolyte QSPR + PyBaMM + physics multi-fidelity GP for Na-ion) hasn't been done. |
| Not enough validation data | Medium | Medium | BatteryLife dataset + 15-20 literature points is sufficient for proof-of-concept. |

---

## 12. Success Metrics

The paper is successful if it achieves:

- [ ] Molecular properties (κ_e/D_e, HOMO/LUMO) validated within ~20% of experiment / Electrolyte Genome
- [ ] **Molecular→transport QSPR** achieves meaningful R² predicting κ_e/D_e (the scientific link)
- [ ] Surrogate emulates the cell model with R² > 0.90 *(efficiency metric — emulator fidelity, not a discovery claim)*
- [ ] **Physics-level** multi-fidelity GP (isothermal↔thermal) matches high-fidelity at a fraction of the cost
- [ ] BO finds a **realizable** (DFT-constrained) design improving on the Chayambuka baseline
- [ ] SHAP gives interpretable findings along **both** valid pathways (molecular→transport, cell→capacity)
- [ ] Quantitative validation vs Chayambuka 2022; qualitative realism vs BatteryLife/literature
- [ ] All code and data released openly on GitHub + Zenodo
- [ ] arXiv preprint posted by December 1, 2026; submitted to Digital Discovery by January 2027

---

*Last updated: June 2026*  
*Status: Outline complete — ready for execution*
