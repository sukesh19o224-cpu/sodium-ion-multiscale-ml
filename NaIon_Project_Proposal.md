
# Multi-Fidelity Machine Learning for Na-ion Batteries — Project Proposal

**A clean summary for the team. Full technical detail is in `NaIon_MultiScale_ML_Outline.md`.**


|  |  |
|---|---|
| **Output** | Research paper (arXiv preprint → journal) |
| **Target journal** | Digital Discovery (RSC) — 1st choice · npj Computational Materials — 2nd |
| **arXiv target** | December 2026 |
| **Team** | 3 people (see roles below) |
| **Status** | ✅ Feasibility hands-on validated (June 2026) — **greenlight** |

---

## 1. The problem we're solving

Na-ion batteries are the cheap, abundant alternative to Li-ion — but the ML/modeling community is stuck: **there's almost no open, structured Na-ion data to train models on.** Li-ion has decades of datasets and tools; Na-ion has only 1–2 validated simulation parameter sets and basically no open multi-scale pipeline.

**We build that missing infrastructure** — an open, end-to-end pipeline from molecular simulation → cell simulation → machine-learning surrogate, all for Na-ion, with the code and data released openly.

---

## 2. What we're actually building (3 honest pillars)

> ⚠️ Important: the model only "sees" the electrolyte through its **transport properties** (conductivity/diffusivity). So we do **not** claim "molecular structure → capacity" directly (that has no physical pathway). Instead we split into clean, defensible pieces:

1. **Molecular → transport (QSPR — Quantitative Structure–Property Relationship, a model that predicts a property from a molecule's structure):** predict electrolyte conductivity/diffusivity from solvent molecular descriptors — using **MACE-OFF** (Machine-learning Atomic Cluster Expansion–Organic Force Field; an AI that simulates organic molecules fast) and **GFN2-xTB** (a fast approximate quantum-chemistry method) to run **molecular dynamics** (MD — simulating the atoms moving), plus **DFT** (Density Functional Theory; accurate quantum chemistry). *Genuinely new for Na-ion.*
2. **Transport → cell:** feed those properties into **PyBaMM** (Python Battery Mathematical Modelling — our free battery-cell simulator) to get capacity, voltage, rate capability across thousands of simulated cells.
3. **ML surrogate + design:** a **multi-fidelity** (combining cheap-rough + expensive-accurate data) **Gaussian-Process (GP)** surrogate — an ML model that predicts a value *and* its uncertainty — that emulates the expensive simulators cheaply, plus **Bayesian Optimization (BO)** — a smart search for the best design in few tries — plus **SHAP** (SHapley Additive exPlanations; shows which inputs matter most) to explain what matters.
4. **Electronic stability (separate):** **HOMO/LUMO** (a molecule's key electron energy levels — they set the electrochemical stability window) screening of solvents, computed with **PySCF** (Python-based Simulations of Chemistry Framework; a DFT engine) — reported on its own, not mixed into capacity.

**The headline science:** a structure→property model for Na-ion electrolytes + a physics-level multi-fidelity surrogate that hits high-fidelity accuracy at a fraction of the cost.

### What's at the core (and how to describe it)

The **technical heart** of the paper is the **surrogate model** — the fast ML predictor that learns from all the simulations and replaces hours of computation with milliseconds. Everything feeds into it; it's the engine.

**But a surrogate *alone* would be incremental** ("a fast copy of a simulator" — reviewers have seen many). So the paper is the surrogate **plus** the three things that make it meaningful: *what it connects* (molecular chemistry → cell performance), *what it enables* (cheap design optimization + interpretability), and *that it's open* for Na-ion.

So when describing this project:
- ❌ **Weak:** *"a surrogate model for a battery simulator."*
- ✅ **Strong:** *"the first open ML pipeline linking Na-ion electrolyte chemistry to cell performance, with a fast multi-fidelity surrogate at its core."*

Same work — the second framing is the one that gets published and noticed.

---

## 3. Why we're confident it works (already validated, not assumed)

Every load-bearing piece was **actually run** in June 2026, not just planned:

| Checked | Result |
|---|---|
| PyBaMM Na-ion model exists & solves | ✅ 0.45 s/sim (4× faster than assumed) |
| 5000-sim parameter sweep won't crash | ✅ **100% solver success** over the full range |
| Electronic properties (HOMO/LUMO/dipole) | ✅ Match experiment (validated on propylene carbonate) |
| Multi-fidelity GP + SHAP + Bayesian optimization | ✅ All ran end-to-end on real Na-ion data |
| Open datasets to validate against | ✅ BatteryLife (Na-ion), Electrolyte Genome — confirmed reachable |

**Bottom line:** the remaining work is *execution*, not research risk.

---

## 4. Team roles

### Dcode (ML + Molecular + Paper lead)
Molecular simulation (MACE-OFF / xTB / PySCF), the molecular→transport QSPR, the multi-fidelity GP surrogate, SHAP, Bayesian optimization, uncertainty quantification — and writing the paper. 

### pallak  — PyBaMM (the data engine)
Set up the Na-ion cell model (Chayambuka 2022), design the parameter sampling, run the 5000-simulation sweep + rate-capability runs, validate against published experimental curves, and hand over clean ML-ready data. *Writes the PyBaMM methods section.*

### Shalini — pick ONE core role (you have a **COMSOL** license — a commercial multiphysics simulation software)
This used to feel like an "add-on." It's now a **core pillar** — choose what you'd enjoy most:

| Option | What you'd do | You'd learn |
|---|---|---|
| **B1 — COMSOL thermal** | Build a **P2D + thermal** Na-ion model (a thermally-coupled DFN — start from COMSOL's 1D Na-ion example and add heat; true 2D optional); provides the high-fidelity tier that makes "multi-fidelity" scientifically real (isothermal vs thermal) | Multiphysics **FEM** (Finite Element Method), thermal modeling |
| **B2 — Electrode materials** | Compute Na⁺ migration barriers, voltages & stability of cathode crystals (NVPF, NaMnO₂, etc.) using **MACE-MP** (the MACE AI model trained on **M**aterials **P**roject crystals — the right one for electrodes) + DFT, via **NEB** (Nudged Elastic Band; computes how hard it is for Na⁺ to hop through a crystal) — feeds the surrogate and grounds the optimization in *realizable* designs | DFT, NEB, ML potentials, materials screening |
| **B3 — Both** | B2 as the core + B1 COMSOL as the high-fidelity tier (strongest paper) | Both skill sets |

*All three are open to you. B3 is strongest if you have the appetite; B1 or B2 alone are both excellent.*

---


---

## 6. What we reuse instead of rebuilding (open assets)

- **PyBaMM** built-in Na-ion model (Chayambuka 2022)
- **MACE-OFF / GFN2-xTB / PySCF** — molecular simulation (all free, pip-installable)
- **BatteryLife** Na-ion dataset (HuggingFace, MIT license) — real cells for sanity-checking
- **Electrolyte Genome** (Materials Project) — pre-computed solvent electronic properties
- **MACE-MP** — for Friend B's electrode crystals (if B2/B3)

---

## 7. Why this won't be wasted effort (honestly risk-managed)

We're being upfront: not *every* part is 100% certain (that's true of any research). What matters is that **no single failure can collapse the project** — it's modular by design:

| If this fails… | …we still have a paper because |
|---|---|
| Molecular→transport QSPR is weak | The cell simulation + multi-fidelity surrogate + sensitivity analysis is already a complete paper. We also have Electrolyte Genome lookups as a fallback. |
| MACE-OFF molecular dynamics is too slow | Swap to faster GFN2-xTB, smaller boxes, or pre-computed Electrolyte Genome data. |
| COMSOL stalls (Friend B) | The MVP never depends on it — it's an upgrade, not a requirement. |
| One person gets stuck | The other two pillars stand alone; work is split so handoffs are minimal. |


**Compute (measured, not guessed):** a test molecular-dynamics run clocked **0.80 s/step on CPU** (~22 h per 100 ps single-core) — so CPU-only is too slow, but **a single 16 GB NVIDIA GPU is enough - Gputham GPU from his amma laptop** for the MVP (MACE-OFF MD on GPU runs ~10–100× faster → ~tens of min to a couple hours per run; VRAM is a non-issue for 200–500-atom boxes). Electronic properties (PySCF/xTB) run fine on a laptop CPU. **HPC is optional**, only for scaling beyond the MVP. → *The one real compute risk is covered by Friend's 16 GB GPU.*

**Honest quality bar:** this is a solid, genuinely publishable, skill-building paper and a strong fit for Digital Discovery — *not* a field-changing breakthrough, but real, citable computational research. Worth doing. ✅

---

## 8. What we'd love you to check / decide

- **Friend A:** Does the PyBaMM scope look right? Any extra outputs we should record from the sweep?
- **Friend B:** **Which role (B1 / B2 / B3) do you want?** Anything you'd add?
- **Both:** Anything missing, unclear, or worth adding before we start? Any papers we should cite?

> The detailed, fully-validated technical plan (with code, proofs, and section-by-section methodology) is in **`NaIon_MultiScale_ML_Outline.md`** — read that for the deep version.

---

## Appendix: Tools & terms explained (plain language)

New to some of these? Here's what each one is and why we use it. **You don't need to know all of them** — focus on the ones for your role.

### 🔋 Battery cell simulation
| Tool / term | Full form | What it is & why we use it |
|---|---|---|
| **PyBaMM** | Python Battery Mathematical Modelling | Free Python tool that simulates a battery cell's charge/discharge. **Our main cell simulator** (Friend A). |
| **DFN** | Doyle–Fuller–Newman model | The standard physics equations describing how a battery cell works inside. PyBaMM's Na-ion model uses this. |
| **COMSOL** | (brand name) | Commercial engineering software for multiphysics (heat + electrochemistry in 2D/3D). Friend B's **thermal** option. |
| **C-rate** | — | How fast you charge/discharge (1C = full discharge in 1 hour, 5C = in 12 min). |

### 🧪 Molecular simulation (the electrolyte/electrode chemistry)
| Tool / term | Full form | What it is & why we use it |
|---|---|---|
| **MACE-OFF** | Machine-learning Atomic Cluster Expansion – **O**rganic **F**orce **F**ield | An AI model that simulates **organic molecules** (our liquid electrolyte solvents) at near-quantum accuracy but much faster. Used for the **molecular dynamics** (atoms moving) to get conductivity. |
| **MACE-MP** | MACE – **M**aterials **P**roject | Same AI family but trained on **crystals** — so it's the right one for **electrode materials** (Friend B, Option B2). |
| **GFN2-xTB** | Geometry-, Frequency-, Noncovalent-eXtended Tight-Binding | A fast, approximate quantum-chemistry method. We use it for quick screening + an alternative way to run the molecular dynamics. |
| **PySCF** | Python-based Simulations of Chemistry Framework | A proper quantum-chemistry (DFT) engine. We use it to get a solvent's **HOMO/LUMO and dipole** (electronic properties). |
| **DFT** | Density Functional Theory | The accurate (but slower) quantum method behind PySCF. The "gold standard" for molecular electronic properties. |
| **MD** | Molecular Dynamics | Simulating atoms moving over time — how we measure how fast ions diffuse (→ conductivity). |
| **NEB** | Nudged Elastic Band | A method to compute how hard it is for a Na⁺ ion to hop through a crystal (migration barrier). Friend B, Option B2. |
| **ASE / RDKit** | Atomic Simulation Environment / — | Helper libraries: ASE runs the simulations; RDKit builds 3D molecules from chemical formulas. |

### 🤖 Machine learning
| Tool / term | Full form | What it is & why we use it |
|---|---|---|
| **QSPR** | Quantitative Structure–Property Relationship | A model that predicts a **property** (e.g., conductivity) from a molecule's **structure**. Our key "molecular → transport" link. |
| **GP** | Gaussian Process | An ML model that predicts a value **and** its uncertainty. The core of our **surrogate** (a fast stand-in for the slow simulators). |
| **Multi-fidelity** | — | Combining cheap-but-rough data with expensive-but-accurate data into one smarter model. |
| **SHAP** | SHapley Additive exPlanations | Explains **which inputs matter most** in an ML model (so results are interpretable, not a black box). |
| **BO** | Bayesian Optimization | A smart search that finds the **best design** in few tries. We use it (via BoTorch) to suggest optimal battery parameters. |
| **Sobol / SALib** | — / Sensitivity Analysis Library | Tells us **which parameters actually influence** performance, so we can ignore the rest. |
| **PINN** | Physics-Informed Neural Network | A neural network that obeys physics equations. An optional **comparison** model (stretch goal). |
| **GPyTorch / BoTorch** | — | The Python libraries we use for the GP surrogate and Bayesian optimization. |

### 📚 Data sources (open datasets we reuse)
| Source | What it gives us |
|---|---|
| **BatteryLife** | Real Na-ion battery cycling data (HuggingFace, free) — for sanity-checking against reality. |
| **Electrolyte Genome** | ~4,830 solvent molecules with pre-computed electronic properties — a backup if our own calculations are too slow. |
| **Materials Project** | Database of crystal materials (for electrode descriptors). |

---

*Status: validated, greenlight, ready to start. Last updated June 2026.*
