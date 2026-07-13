# Data Sources & Provenance

*Living record of where every input value comes from, so nothing is a silent
assumption at write-up. Update this whenever a new input is introduced.*
*Legend: ✅ solid / cite as-is · ⚠️ verify before publishing.*

---

## 1. Solvent electronic-property screen
`src/01_solvent_screening.py` → `data/processed/solvent_electronic_properties.csv`

### 1a. Method (all citable — for the Methods section)
| Component | What | Citation to use |
|---|---|---|
| Geometry embedding | RDKit ETKDG + MMFF94 optimization | Riniker & Landrum, *JCIM* 2015 (ETKDG); Halgren, *JCC* 1996 (MMFF94) |
| DFT functional | **B3LYP** | Becke, *JCP* 1993; Lee, Yang, Parr, *PRB* 1988 |
| Basis set | **def2-SVP** *(upgrade to def2-TZVP for final)* | Weigend & Ahlrichs, *PCCP* 2005 |
| Implicit solvent | **ddCOSMO** (domain-decomposition COSMO) | Lipparini et al., *JCTC* 2013; Cancès et al. |
| QC engine | **PySCF** | Sun et al., *WIREs Comput. Mol. Sci.* 2018; *JCP* 2020 |

### 1b. Molecular identities (SMILES) — ⚠️ verify against PubChem
| Solvent | SMILES | Formula | Verify |
|---|---|---|---|
| PC  | `CC1COC(=O)O1`       | C4H6O3   | ⚠️ PubChem "propylene carbonate" |
| EC  | `C1COC(=O)O1`        | C3H4O3   | ⚠️ PubChem "ethylene carbonate" |
| DMC | `COC(=O)OC`          | C3H6O3   | ⚠️ PubChem "dimethyl carbonate" |
| DME | `COCCOC`             | C4H10O2  | ⚠️ PubChem "1,2-dimethoxyethane" |
| TEP | `CCOP(=O)(OCC)OCC`   | C6H15O4P | ⚠️ PubChem "triethyl phosphate" |
| FEC | `O=C1OCC(F)O1`       | C3H3FO3  | ⚠️ PubChem "fluoroethylene carbonate" |

*These are standard, well-known molecules — SMILES are trivially checkable but
should be confirmed against a database once for the record.*

### 1c. Dielectric constants (ε) — ALL standardized to 25 °C
**Primary source:** Xu, *Chem. Rev.* **104**, 4303 (2004) — canonical battery-solvent table.
**Decision:** report all ε at **25 °C** for internal consistency.

| Solvent | ε (25 °C) | Status | Note |
|---|---|---|---|
| PC  | 64.9  | ✅ | measured, 25 °C |
| EC  | 90.0  | ⚠️ | **extrapolated to 25 °C** — measured value 89.78 is at 40 °C (EC is solid below ~36 °C). Footnote this. |
| DMC | 3.11  | ✅ | measured, 25 °C |
| DME | 7.20  | ✅ | measured, 25 °C |
| TEP | 13.0  | ⚠️ | source spread ~10.8–13.0 — cite a 2nd reference |
| FEC | 107.0 | ⚠️ | source spread — cite a 2nd reference |

**Robustness note (state in paper):** ddCOSMO's continuum response *saturates*
above ε ≈ 40, and low-ε shifts are tiny — so EC/PC/FEC results are insensitive to
their exact ε, and DMC barely shifts. The flagged ε uncertainties (EC/TEP/FEC) do
**not** materially affect the computed properties. Verified empirically: EC gave
identical condensed-phase HOMO/dipole for ε = 89.78 vs 90.0.

### 1d. Validation anchor
- **PC gas-phase:** computed HOMO −7.95 eV, dipole 5.45 D.
  Experimental gas-phase dipole of PC ≈ 4.9 D — ⚠️ cite a source (e.g., solvent
  property handbook) for the experimental comparison.

### 1e. Known limitations to disclose
- Implicit solvent captures **bulk polarity only** — not H-bonding or explicit
  Na⁺ coordination.
- DFT HOMO/LUMO are **not** rigorous ionization potentials / electron affinities;
  report trends/ordering, not absolute stability windows. (Planned upgrade: ΔSCF.)
- Single conformer per molecule (lowest MMFF); flexible solvents (DME, TEP) may
  warrant conformer averaging for final numbers.

---

## 2. (Placeholder) PyBaMM cell parameters
Source: `pybamm.sodium_ion` `Chayambuka2022` set → Chayambuka et al., *Electrochim.
Acta* 404, 139764 (2022); parameter values via the COMSOL "1D Isothermal Na-ion"
example. See `References_and_Links.md`. ⚠️ verify COMSOL↔Chayambuka value match.

## 3. (Placeholder) Electrode / Materials Project descriptors
To be added when Layer 0 runs. Source: Materials Project (cite Jain et al. 2013 +
the specific `mp-id`s used).
