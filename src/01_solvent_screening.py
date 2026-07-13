"""
01_solvent_screening.py
=======================
Electronic-property screen of Na-ion electrolyte solvents, in BOTH phases.

Runs ONE solvent per invocation by default (laptop-friendly) and APPENDS to the
CSV, so a crash or interrupt never loses completed work.

For each solvent:
  1. Build a 3D structure from SMILES (RDKit), optimize with MMFF.
  2. Fast semi-empirical pass with GFN2-xTB (dipole).
  3. DFT (PySCF, B3LYP/def2-SVP):
       a) GAS PHASE       -> HOMO, LUMO, gap, dipole
       b) CONDENSED PHASE -> same, with implicit solvent (ddCOSMO) using each
          solvent's own experimental dielectric constant.

Why both: gas-phase values are not what a real electrolyte experiences. The
implicit-solvent continuum is the standard cheap correction and matches the
Electrolyte Genome methodology (B3LYP + PCM).

These feed the ELECTRONIC-STABILITY screen only. They are NOT used to predict
cell capacity -- the DFN cell model has no electronic-structure inputs
(see outline Section 0.5-F).

Honest limitations:
  * Implicit solvent captures bulk polarity, NOT specific interactions
    (H-bonding, explicit Na+ coordination).
  * DFT HOMO/LUMO are not rigorous ionization potentials / electron affinities.
    Trends and ordering are meaningful; absolute values are not "the" stability
    window. For that, use dSCF (IP/EA) -- a planned upgrade.

Validation anchor: PC gas phase -> HOMO ~ -7.95 eV, dipole ~ 5.45 D.

Usage
-----
    python src/01_solvent_screening.py PC          # one solvent (recommended)
    python src/01_solvent_screening.py --list      # show solvents + status
    python src/01_solvent_screening.py --next      # run the next unfinished one
    python src/01_solvent_screening.py --all       # run everything (heavy!)
    python src/01_solvent_screening.py PC --threads 2 --memory 1500

Output (appended, deduplicated):
    data/processed/solvent_electronic_properties.csv
"""

from __future__ import annotations

import argparse
import os
import sys

# Thread caps MUST be set before numpy/pyscf import, or BLAS grabs every core
# and the laptop becomes unusable.
def _cap_threads(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=2)
_known, _ = _pre.parse_known_args()
_cap_threads(_known.threads)

import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")

EBOHR_TO_DEBYE = 2.5417464
HARTREE_TO_EV = 27.211386

# `eps` = experimental STATIC dielectric constant of the neat LIQUID solvent,
#         ALL REFERENCED TO 25 C for internal consistency (this is a bulk-liquid
#         property -- NOT in Materials Project, which stores crystal DFPT values).
# Primary source: Xu, Chem. Rev. 104, 4303 (2004) -- canonical battery-solvent table.
#
# CONSISTENCY / CAVEATS (state these in the paper):
#   * EC melts at ~36 C, so it is SOLID at 25 C. Its standard value (89.78) is
#     measured at 40 C; we use eps = 90.0 as the 25 C extrapolation (dielectric
#     rises weakly as T falls). Because the ddCOSMO continuum response SATURATES
#     above eps ~ 40, EC/PC/FEC results are insensitive to this -> no meaningful
#     error, only a self-consistent 25 C table.
#   * TEP and FEC show ~10-20% source-to-source spread; the low-/high-eps
#     saturation means the computed properties are robust to this. Cite a 2nd
#     reference for these two at write-up.
# Chayambuka 2022's cell uses NaPF6 in EC:PC (50:50 w/w) -> EC and PC are priority.
SOLVENTS: dict[str, dict] = {
    "PC":  dict(smiles="CC1COC(=O)O1",     name="propylene carbonate",      family="carbonate", eps=64.9),   # 25 C, measured
    "EC":  dict(smiles="C1COC(=O)O1",      name="ethylene carbonate",       family="carbonate", eps=90.0),   # 25 C, extrapolated (89.78 @ 40 C)
    "DMC": dict(smiles="COC(=O)OC",        name="dimethyl carbonate",       family="carbonate", eps=3.11),   # 25 C, measured
    "DME": dict(smiles="COCCOC",           name="1,2-dimethoxyethane",      family="ether",     eps=7.20),   # 25 C, measured
    "TEP": dict(smiles="CCOP(=O)(OCC)OCC", name="triethyl phosphate",       family="phosphate", eps=13.0),   # 25 C (+/-, verify)
    "FEC": dict(smiles="O=C1OCC(F)O1",     name="fluoroethylene carbonate", family="additive",  eps=107.0),  # 25 C (+/-, verify)
}

BASIS = "def2-svp"   # bump to def2-TZVP for publication-grade numbers
XC = "b3lyp"

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "data" / "processed" / "solvent_electronic_properties.csv"


# --------------------------------------------------------------------------- #
def build_geometry(smiles: str, seed: int = 42):
    """SMILES -> MMFF-optimized 3D geometry. Returns (symbols, coords[Angstrom])."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise RuntimeError(f"Embedding failed for {smiles}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return symbols, coords


def xtb_dipole(symbols, coords) -> dict:
    """Fast GFN2-xTB gas-phase dipole. Returns {} if tblite is unavailable."""
    try:
        from ase import Atoms
        from tblite.ase import TBLite
    except ImportError:
        return {}
    atoms = Atoms(symbols=symbols, positions=coords)
    atoms.calc = TBLite(method="GFN2-xTB", verbosity=0)
    atoms.get_potential_energy()
    res = atoms.calc.results
    if "dipole" in res:
        return {"xtb_dipole_D": float(np.linalg.norm(res["dipole"]) * EBOHR_TO_DEBYE)}
    return {}


def dft_properties(symbols, coords, eps: float | None, memory_mb: int) -> dict:
    """
    DFT single point -> HOMO, LUMO, gap, dipole.

    eps=None  -> gas phase
    eps=float -> implicit solvent (ddCOSMO continuum) with that dielectric constant
    """
    from pyscf import dft, gto

    atom_str = "\n".join(
        f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in zip(symbols, coords)
    )
    mol = gto.M(atom=atom_str, basis=BASIS, verbose=0)
    mol.max_memory = memory_mb          # keep PySCF inside a memory budget

    mf = dft.RKS(mol)
    mf.xc = XC
    if eps is not None:
        mf = mf.ddCOSMO()
        mf.with_solvent.eps = eps

    t0 = time.time()
    mf.kernel()
    dt = time.time() - t0
    if not mf.converged:
        raise RuntimeError(f"SCF did not converge (eps={eps})")

    mo_e = mf.mo_energy * HARTREE_TO_EV
    occ = mf.mo_occ
    homo = float(mo_e[occ > 0].max())
    lumo = float(mo_e[occ == 0].min())
    dipole = float(np.linalg.norm(mf.dip_moment(unit="Debye", verbose=0)))

    return {"HOMO_eV": homo, "LUMO_eV": lumo, "gap_eV": lumo - homo,
            "dipole_D": dipole, "time_s": round(dt, 1)}


# --------------------------------------------------------------------------- #
def load_done() -> set[str]:
    if CSV.exists():
        try:
            return set(pd.read_csv(CSV)["solvent"].astype(str))
        except Exception:
            return set()
    return set()


def append_row(row: dict) -> None:
    """Append one result, deduplicating on `solvent` and keeping the newest."""
    CSV.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if CSV.exists():
        df = pd.concat([pd.read_csv(CSV), df_new], ignore_index=True)
        df = df.drop_duplicates(subset="solvent", keep="last")
    else:
        df = df_new
    if "sol_HOMO_eV" in df.columns:
        df = df.sort_values("sol_HOMO_eV")
    df.to_csv(CSV, index=False)


def run_one(key: str, memory_mb: int) -> dict:
    meta = SOLVENTS[key]
    print(f"[{key}] {meta['name']}  (eps={meta['eps']})")

    symbols, coords = build_geometry(meta["smiles"])
    print(f"  geometry: {len(symbols)} atoms")

    row = {"solvent": key, "name": meta["name"], "family": meta["family"],
           "smiles": meta["smiles"], "n_atoms": len(symbols), "eps": meta["eps"]}
    row.update(xtb_dipole(symbols, coords))

    print("  DFT gas phase ...", end=" ", flush=True)
    gas = dft_properties(symbols, coords, eps=None, memory_mb=memory_mb)
    print(f"HOMO {gas['HOMO_eV']:.2f} eV | LUMO {gas['LUMO_eV']:.2f} eV | "
          f"dipole {gas['dipole_D']:.2f} D  ({gas['time_s']:.0f}s)")

    print("  DFT condensed (ddCOSMO) ...", end=" ", flush=True)
    sol = dft_properties(symbols, coords, eps=meta["eps"], memory_mb=memory_mb)
    print(f"HOMO {sol['HOMO_eV']:.2f} eV | LUMO {sol['LUMO_eV']:.2f} eV | "
          f"dipole {sol['dipole_D']:.2f} D  ({sol['time_s']:.0f}s)")

    for k, v in gas.items():
        row[f"gas_{k}"] = v
    for k, v in sol.items():
        row[f"sol_{k}"] = v
    row["dHOMO_eV"] = sol["HOMO_eV"] - gas["HOMO_eV"]
    row["ddipole_D"] = sol["dipole_D"] - gas["dipole_D"]

    print(f"  solvent shift: dHOMO {row['dHOMO_eV']:+.2f} eV | "
          f"ddipole {row['ddipole_D']:+.2f} D")

    if key == "PC":
        ok = abs(gas["HOMO_eV"] + 7.95) < 0.5 and abs(gas["dipole_D"] - 5.45) < 0.5
        print(f"  PC gas anchor -> {'PASS' if ok else 'CHECK'}")

    append_row(row)
    print(f"  saved -> {CSV.relative_to(REPO)}")
    return row


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("solvent", nargs="?", help=f"one of: {', '.join(SOLVENTS)}")
    p.add_argument("--all", action="store_true", help="run every solvent (heavy)")
    p.add_argument("--next", action="store_true", help="run the next unfinished solvent")
    p.add_argument("--list", action="store_true", help="show solvents and status")
    p.add_argument("--threads", type=int, default=2, help="BLAS threads (default 2)")
    p.add_argument("--memory", type=int, default=2000, help="PySCF max memory MB (default 2000)")
    args = p.parse_args()

    done = load_done()

    if args.list:
        print(f"{'solvent':8s} {'atoms':>6s} {'eps':>7s}  status")
        for k, m in SOLVENTS.items():
            n = len(m["smiles"])
            print(f"{k:8s} {'~'+str(n):>6s} {m['eps']:7.1f}  "
                  f"{'done' if k in done else 'pending'}")
        print(f"\n{len(done)}/{len(SOLVENTS)} complete -> {CSV}")
        return

    todo: list[str]
    if args.all:
        todo = [k for k in SOLVENTS if k not in done]
    elif args.next:
        remaining = [k for k in SOLVENTS if k not in done]
        if not remaining:
            print("All solvents already computed.")
            return
        todo = [remaining[0]]
    elif args.solvent:
        key = args.solvent.upper()
        if key not in SOLVENTS:
            sys.exit(f"Unknown solvent '{args.solvent}'. Choose from: {', '.join(SOLVENTS)}")
        todo = [key]
    else:
        p.print_help()
        return

    print(f"DFT {XC.upper()}/{BASIS} | threads={args.threads} | memory={args.memory} MB\n")
    for key in todo:
        t0 = time.time()
        try:
            run_one(key, args.memory)
        except Exception as exc:
            print(f"  FAILED: {exc}")
        print(f"  ({time.time() - t0:.0f}s)\n")

    done = load_done()
    print(f"{len(done)}/{len(SOLVENTS)} solvents complete.")
    remaining = [k for k in SOLVENTS if k not in done]
    if remaining:
        print(f"Next: python src/01_solvent_screening.py {remaining[0]}")


if __name__ == "__main__":
    main()
