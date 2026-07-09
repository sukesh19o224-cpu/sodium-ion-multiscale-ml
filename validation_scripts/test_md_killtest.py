"""
PARTIAL KILL TEST (compute-feasibility half):
Build a small Na-salt-in-solvent cluster, run REAL GFN2-xTB molecular dynamics,
measure actual seconds/step on THIS hardware (CPU), and extrapolate to a full
100 ps production run. Proves: (a) the MD pipeline runs end-to-end,
(b) realistic compute cost. Does NOT give a converged conductivity (too short).
"""
import time, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.md.langevin import Langevin
from ase import units
from tblite.ase import TBLite

def make_pc():
    m = Chem.AddHs(Chem.MolFromSmiles("CC1COC(=O)O1"))
    AllChem.EmbedMolecule(m, randomSeed=1); AllChem.MMFFOptimizeMolecule(m)
    c = m.GetConformer()
    sym = [a.GetSymbol() for a in m.GetAtoms()]
    xyz = np.array([list(c.GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
    return sym, xyz - xyz.mean(0)

# Build a neutral cluster: N_pc propylene carbonates on a grid + Na + Cl
N_pc = 8
pc_sym, pc_xyz = make_pc()
symbols, positions = [], []
grid = [(x, y, z) for x in (0, 7) for y in (0, 7) for z in (0, 7)]  # 8 sites, 7 A apart
for site in grid[:N_pc]:
    symbols += pc_sym
    positions.append(pc_xyz + np.array(site))
positions = np.vstack(positions)
# add Na+ and Cl- in gaps (neutral overall)
symbols += ["Na", "Cl"]
positions = np.vstack([positions, np.array([[3.5, 3.5, 3.5], [3.5, 3.5, 10.5]])])

atoms = Atoms(symbols=symbols, positions=positions)
n_atoms = len(atoms)
print(f"System: {N_pc} PC + Na + Cl = {n_atoms} atoms (neutral cluster, no PBC)")

atoms.calc = TBLite(method="GFN2-xTB", verbosity=0)
na_index = symbols.index("Na")

# warm up (1 force eval) + a few MD steps to reach steady-state timing
dyn = Langevin(atoms, timestep=1.0*units.fs, temperature_K=300, friction=0.02)
print("Warming up (first SCF can be slow)...")
t0 = time.time(); atoms.get_potential_energy(); print(f"  first SCF: {time.time()-t0:.1f}s")

na0 = atoms.positions[na_index].copy()
N_STEPS = 60
t0 = time.time()
dyn.run(N_STEPS)
dt = time.time() - t0
sec_per_step = dt / N_STEPS
na_disp = np.linalg.norm(atoms.positions[na_index] - na0)

print(f"\nRan {N_STEPS} MD steps in {dt:.1f}s")
print(f"  -> {sec_per_step:.2f} s/step on this CPU ({n_atoms} atoms)")
print(f"  Na+ moved {na_disp:.2f} A in {N_STEPS} fs (sanity: it IS moving = MD works)")

# extrapolate to a real production run
for ps in (50, 100):
    steps = ps * 1000
    hrs = steps * sec_per_step / 3600
    print(f"  extrapolated {ps} ps run ({steps} steps): ~{hrs:.1f} h single-core CPU")
print("\nNotes:")
print("  * Real boxes are ~200-400 atoms (denser); xTB SCF scales ~O(N^2-3),")
print("    so production is slower per step -> GPU/MACE-OFF or Electrolyte Genome fallback.")
print("  * This proves the PIPELINE RUNS + gives a real compute-cost number (the risky half).")
print("=== partial kill test complete ===")
