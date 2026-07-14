"""
04_electrolyte_md.py
====================
Solution-phase MD of a real Na-ion electrolyte: Na+ + PF6- + EC:PC solvent mix.
Engine: MACE-MH-1 (spice_wB97M head) on GPU -- has Na AND high-level organic accuracy.

This is Layer 1b: the MD that gives Na+ SOLVATION STRUCTURE and (with long runs)
the diffusion coefficient -> conductivity via Nernst-Einstein.

Designed for Google Colab (free T4 GPU, ~13 GB RAM). Do NOT run on a low-RAM
laptop -- loading torch + CUDA + MACE spikes system RAM.

Pipeline (matches the physics you learned):
  1. Build molecules (RDKit) and PACK into a periodic box (bulk liquid).
  2. Load MACE-MH-1 -> gives energy + forces (the "force engine").
  3. Set velocities to 300 K (temperature = atom jiggling).
  4. EQUILIBRATE (Langevin NVT thermostat) -- let it settle, throw away.
  5. PRODUCE (Langevin NVT) -- collect the trajectory.
  6. ANALYZE: Na+ MSD -> diffusion; Na-O RDF -> solvation shell.

Honest scope for a FIRST run: modest box + short trajectory = proof-of-concept
(see solvation form, rough diffusion). A converged conductivity needs a bigger
box + longer run + replicas + finite-size (Yeh-Hummer) correction.

Usage (Colab):
    !pip install mace-torch ase rdkit
    !python src/04_electrolyte_md.py --n_solvent 16 --prod_steps 5000
"""

from __future__ import annotations
import argparse, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

# ---- solvent chemistry (EC:PC mix, matching Chayambuka's real cell) ----
SMILES = {
    "EC":  "C1COC(=O)O1",
    "PC":  "CC1COC(=O)O1",
    "PF6": "F[P-](F)(F)(F)(F)F",
}
MACE_REPO = "mace-foundations/mace-mh-1"
MACE_FILE = "mace-mh-1.model"
MACE_HEAD = "spice_wB97M"   # high-level organic head (has Na)


# --------------------------------------------------------------------------- #
def build_mol(smiles, seed=1):
    """SMILES -> (symbols, centered coords) via RDKit + MMFF."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=int(seed))
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    except Exception:
        pass
    c = m.GetConformer()
    sym = [a.GetSymbol() for a in m.GetAtoms()]
    xyz = np.array([list(c.GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
    return sym, xyz - xyz.mean(0)


def random_rotation(xyz, rng):
    """Apply a random 3D rotation (so molecules aren't all aligned)."""
    a, b, c = rng.uniform(0, 2*np.pi, 3)
    Rz = np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    Ry = np.array([[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]])
    Rx = np.array([[1,0,0],[0,np.cos(c),-np.sin(c)],[0,np.sin(c),np.cos(c)]])
    return xyz @ (Rz @ Ry @ Rx).T


def build_box(n_solvent, ratio_ec=0.5, seed=0):
    """
    Pack Na+ + PF6- + n_solvent (EC:PC) onto a grid inside a cubic PBC box.
    Grid placement guarantees no atom overlap (one molecule per cell).
    """
    from ase import Atoms
    rng = np.random.default_rng(seed)

    # species list: Na, PF6, then solvents (EC or PC by ratio)
    n_ec = int(round(n_solvent * ratio_ec))
    species = ["Na", "PF6"] + ["EC"]*n_ec + ["PC"]*(n_solvent - n_ec)
    n_sites = len(species)

    # cubic grid big enough to hold all sites; cell spacing avoids overlap
    ncell = int(np.ceil(n_sites ** (1/3)))
    spacing = 7.5                      # A between molecule centers (no overlap)
    L = ncell * spacing               # box length
    cells = [(i, j, k) for i in range(ncell) for j in range(ncell)
             for k in range(ncell)][:n_sites]
    rng.shuffle(cells)

    symbols, positions = [], []
    for sp, (i, j, k) in zip(species, cells):
        center = (np.array([i, j, k]) + 0.5) * spacing
        if sp == "Na":
            symbols.append("Na"); positions.append(center[None, :])
        else:
            s, x = build_mol(SMILES[sp if sp in SMILES else sp], seed=rng.integers(1e6))
            symbols += s
            positions.append(random_rotation(x, rng) + center)
    positions = np.vstack(positions)

    atoms = Atoms(symbols=symbols, positions=positions,
                  cell=[L, L, L], pbc=True)
    print(f"  box: {n_sites} species -> {len(atoms)} atoms, "
          f"cubic L={L:.1f} A (Na+ + PF6- + {n_ec} EC + {n_solvent-n_ec} PC)")
    return atoms


def load_mace(device):
    from huggingface_hub import hf_hub_download
    from mace.calculators.foundations_models import mace_mp
    path = hf_hub_download(MACE_REPO, MACE_FILE)
    return mace_mp(model=path, device=device, default_dtype="float32", head=MACE_HEAD)


# --------------------------------------------------------------------------- #
def run(args):
    import torch
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.io import Trajectory

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | torch {torch.__version__}")
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    print("Building electrolyte box ...")
    atoms = build_box(args.n_solvent, seed=args.seed)

    print("Loading MACE-MH-1 (spice head) ...")
    atoms.calc = load_mace(device)

    # temperature = atom jiggling: draw initial velocities at target T
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)

    dt = args.timestep * units.fs           # ~1 fs, set by fastest vibration
    dyn = Langevin(atoms, timestep=dt, temperature_K=args.temperature,
                   friction=args.friction)  # Langevin = thermostat (heat bath)

    # ---- equilibration (settle; discard) ----
    print(f"Equilibrating {args.equil_steps} steps ...")
    t0 = time.time()
    dyn.run(args.equil_steps)
    print(f"  equilibration done ({time.time()-t0:.0f}s, "
          f"{(time.time()-t0)/max(args.equil_steps,1)*1000:.0f} ms/step)")

    # ---- production (collect) ----
    na_index = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Na"][0]
    traj_path = out / "production.traj"
    traj = Trajectory(str(traj_path), "w", atoms)
    na_pos, times = [], []
    print(f"Producing {args.prod_steps} steps ...")
    t0 = time.time()
    for step in range(args.prod_steps):
        dyn.run(1)
        if step % args.sample_every == 0:
            traj.write()
            na_pos.append(atoms.positions[na_index].copy())
            times.append(step * args.timestep)     # fs
            if step % (args.sample_every*20) == 0:
                T = atoms.get_temperature()
                print(f"  step {step:6d}  T={T:6.1f} K  "
                      f"{(time.time()-t0)/max(step,1)*1000:.0f} ms/step")
    traj.close()
    prod_time = time.time() - t0
    print(f"  production done ({prod_time:.0f}s, "
          f"{prod_time/args.prod_steps*1000:.0f} ms/step)")

    # ---- analysis: Na+ MSD -> rough diffusion (Einstein relation) ----
    na_pos = np.array(na_pos); times = np.array(times)
    msd = ((na_pos - na_pos[0])**2).sum(axis=1)     # A^2
    np.savetxt(out / "na_msd.csv",
               np.column_stack([times, msd]),
               header="time_fs,msd_A2", delimiter=",", comments="")
    # Einstein: MSD = 6 D t  ->  D = slope/6  (rough from a short run!)
    if len(times) > 5:
        slope = np.polyfit(times[len(times)//2:], msd[len(times)//2:], 1)[0]  # A^2/fs
        D = slope / 6.0 * 1e-5   # A^2/fs -> cm^2/s  (1 A^2/fs = 1e-1 cm^2/s ... see note)
        # unit note: 1 A^2/fs = 1e-16 m^2 / 1e-15 s = 0.1 m^2/s = 1e3 cm^2/s
        D = slope / 6.0 * 1e3    # cm^2/s
        print(f"\n  Na+ MSD final: {msd[-1]:.2f} A^2 over {times[-1]/1000:.1f} ps")
        print(f"  rough D(Na+) ~ {D:.2e} cm^2/s   (SHORT run -- proof of concept only)")

    print(f"\nSaved: {traj_path.name}, na_msd.csv  ->  {out}")
    print("=== electrolyte MD complete ===")
    print("NOTE: this is a first proof-of-concept. For a real conductivity: bigger box,")
    print("      longer run, several ions, replicas, Yeh-Hummer correction, xTB cross-check.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_solvent", type=int, default=16, help="number of solvent molecules")
    p.add_argument("--temperature", type=float, default=300.0, help="target T (K)")
    p.add_argument("--timestep", type=float, default=1.0, help="dt in fs")
    p.add_argument("--friction", type=float, default=0.01, help="Langevin friction")
    p.add_argument("--equil_steps", type=int, default=1000, help="equilibration steps")
    p.add_argument("--prod_steps", type=int, default=5000, help="production steps")
    p.add_argument("--sample_every", type=int, default=10, help="sample interval")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="results/md_electrolyte")
    run(p.parse_args())


if __name__ == "__main__":
    main()
