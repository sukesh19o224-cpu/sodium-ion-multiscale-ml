"""
Validate the proposed FIX for proposal Problem 1:
MACE cannot give HOMO/LUMO/dipole -> use GFN2-xTB (fast screen) or DFT (PySCF).
Target molecule: propylene carbonate (PC), a core Na-ion solvent.
SMILES: CC1COC(=O)O1
We compute HOMO, LUMO, gap, dipole and compare to literature.
Literature anchors for PC: experimental dipole ~4.9 D; carbonate HOMO ~ -7 to -8 eV (DFT-dependent).
"""
import time, numpy as np

# ---------- 1. Build 3D geometry from SMILES (RDKit) ----------
from rdkit import Chem
from rdkit.Chem import AllChem
smiles = "CC1COC(=O)O1"  # propylene carbonate
mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol)
conf = mol.GetConformer()
symbols = [a.GetSymbol() for a in mol.GetAtoms()]
coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
print(f"Built PC: {len(symbols)} atoms ({''.join(sorted(set(symbols)))})")

# ---------- 2. FAST screening route: GFN2-xTB via tblite + ASE ----------
xtb_ok = False
try:
    from ase import Atoms
    from tblite.ase import TBLite
    atoms = Atoms(symbols=symbols, positions=coords)
    atoms.calc = TBLite(method="GFN2-xTB", verbosity=0)
    t0 = time.time()
    e = atoms.get_potential_energy()
    res = atoms.calc.results
    dt = time.time() - t0
    # tblite exposes orbital energies + dipole
    homo = lumo = dip = None
    if "orbital_energies" in res and "orbital_occupations" in res:
        oe = np.array(res["orbital_energies"]) * 27.2114  # Hartree->eV
        occ = np.array(res["orbital_occupations"])
        occupied = oe[occ > 0.5]
        virtual = oe[occ <= 0.5]
        homo, lumo = occupied.max(), virtual.min()
    if "dipole" in res:
        dip = np.linalg.norm(res["dipole"]) * 2.5417464  # e*Bohr -> Debye
    print(f"\n[GFN2-xTB]  time={dt:.2f}s")
    if homo is not None:
        print(f"  HOMO = {homo:.2f} eV | LUMO = {lumo:.2f} eV | gap = {lumo-homo:.2f} eV")
    if dip is not None:
        print(f"  dipole = {dip:.2f} D  (exp PC ~4.9 D)")
    xtb_ok = True
except Exception as ex:
    print("\n[GFN2-xTB] FAILED:", repr(ex)[:200])

# ---------- 3. DFT route: PySCF B3LYP/def2-SVP (publication-grade) ----------
try:
    from pyscf import gto, dft
    atom_str = "\n".join(f"{s} {x:.6f} {y:.6f} {z:.6f}"
                         for s, (x, y, z) in zip(symbols, coords))
    m = gto.M(atom=atom_str, basis="def2-svp", verbose=0)
    mf = dft.RKS(m); mf.xc = "b3lyp"
    t0 = time.time(); mf.kernel(); dt = time.time() - t0
    mo_e = mf.mo_energy * 27.2114
    occ = mf.mo_occ
    homo = mo_e[occ > 0].max(); lumo = mo_e[occ == 0].min()
    d = mf.dip_moment(unit="Debye", verbose=0)
    dip = float(np.linalg.norm(d))
    print(f"\n[PySCF B3LYP/def2-SVP]  time={dt:.2f}s")
    print(f"  HOMO = {homo:.2f} eV | LUMO = {lumo:.2f} eV | gap = {lumo-homo:.2f} eV")
    print(f"  dipole = {dip:.2f} D  (exp PC ~4.9 D)")
except Exception as ex:
    print("\n[PySCF] FAILED:", repr(ex)[:200])

print("\n=== Electronic-property validation complete ===")
