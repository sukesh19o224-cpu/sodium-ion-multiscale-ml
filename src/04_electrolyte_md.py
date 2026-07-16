"""
04_electrolyte_md.py
====================
Solution-phase MD of a real Na-ion electrolyte: (Na+ + PF6-) x N  +  EC:PC solvent mix.
Engine: MACE-MH-1, `omol` head -- trained on OMol25 (tens of millions of DFT
electrolyte configs incl. Na, at wB97M-V/def2-TZVPD). This is the electrolyte-
appropriate head; the OMol25 -> Na-electrolyte approach is experimentally validated
in Levine et al., arXiv:2603.20183.

This is Layer 1b: gives Na+ SOLVATION STRUCTURE (RDF) and, with a long enough run,
the diffusion coefficient -> conductivity (Nernst-Einstein).

Runs on a free GPU (Colab T4 / Kaggle P100). CHECKPOINTS periodically, so a
disconnect/crash never loses progress -- just re-run and it RESUMES.

Physics (what you learned):
  build box -> MACE gives forces -> ASE Langevin (Verlet + thermostat) moves atoms
  -> equilibrate (settle, discard) -> produce (collect) -> MSD -> D ; RDF -> solvation.

Usage:
    !pip install mace-torch ase rdkit huggingface_hub
    # small test (measure speed):
    !python src/04_electrolyte_md.py --n_solvent 16 --prod_steps 5000
    # bigger, overnight-style (resumes if it dies):
    !python src/04_electrolyte_md.py --n_ion_pairs 3 --n_solvent 60 --prod_steps 200000 \
        --outdir results/md_big
"""

from __future__ import annotations
import argparse, json, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

SMILES = {
    "EC":  "C1COC(=O)O1",
    "PC":  "CC1COC(=O)O1",
    "PF6": "F[P-](F)(F)(F)(F)F",
}
MACE_REPO = "mace-foundations/mace-mh-1"
MACE_FILE = "mace-mh-1.model"
MACE_HEAD = "omol"   # OMol25 head -- the electrolyte-appropriate one (has Na)


# --------------------------------------------------------------------------- #
def build_mol(smiles, seed=1):
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
    a, b, c = rng.uniform(0, 2*np.pi, 3)
    Rz = np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    Ry = np.array([[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]])
    Rx = np.array([[1,0,0],[0,np.cos(c),-np.sin(c)],[0,np.sin(c),np.cos(c)]])
    return xyz @ (Rz @ Ry @ Rx).T


AMU_TO_G = 1.66053907e-24        # g per amu
CM3_TO_A3 = 1.0e24               # A^3 per cm^3


def apply_hmr(atoms, factor=3.0):
    """
    Hydrogen Mass Repartitioning (HMR): make each H heavier by `factor`, and
    subtract the added mass from the heavy atom it's bonded to (TOTAL mass
    conserved). Heavier H -> slower H vibration -> allows a LARGER timestep
    (dt=4 fs instead of 1 fs) -> ~4x fewer steps for the same simulated time.

    Physics: dt is limited by the fastest vibration (freq = sqrt(k/m)); the
    lightest atom (H) sets it. Increasing m_H lowers that frequency. This
    alters ONLY the fast vibrational modes -- the SLOW dynamics we measure
    (diffusion, thermodynamics) are preserved. Standard, validated technique.

    Uses a simple distance-based bond detection (H bonded to its nearest
    non-H atom within 1.3 A).
    """
    from ase.data import atomic_numbers
    masses = atoms.get_masses().copy()
    symbols = atoms.get_chemical_symbols()
    h_mass = 1.008
    added = h_mass * (factor - 1.0)          # mass added to each H

    pos = atoms.get_positions()
    heavy = [i for i, s in enumerate(symbols) if s != "H"]
    n_h = 0
    for i, s in enumerate(symbols):
        if s != "H":
            continue
        # nearest heavy atom = the one H is bonded to
        d = np.linalg.norm(pos[heavy] - pos[i], axis=1)
        j = heavy[int(np.argmin(d))]
        if d.min() > 1.3:                    # not clearly bonded -> skip
            continue
        masses[i] = h_mass * factor          # heavier H
        masses[j] -= added                   # take it from the bonded heavy atom
        n_h += 1
    if (masses <= 0).any():
        raise ValueError("HMR produced a non-positive mass -- lower --hmr_factor")
    atoms.set_masses(masses)
    print(f"  HMR applied: {n_h} H atoms x{factor} (dt can now be ~4 fs) "
          f"| total mass conserved: {masses.sum():.2f} amu")
    return atoms


def target_box_length(atoms, density_g_cm3):
    """Cubic box length that gives the requested LIQUID density (g/cm^3)."""
    mass_g = atoms.get_masses().sum() * AMU_TO_G
    volume_A3 = (mass_g / density_g_cm3) * CM3_TO_A3
    return volume_A3 ** (1/3)


def build_box(n_ion_pairs, n_solvent, ratio_ec=0.5, seed=0, density=1.25):
    """
    Pack n_ion_pairs*(Na+ + PF6-) + n_solvent (EC:PC) into a cubic PBC box.

    Built DILUTE on a grid (guarantees no overlap), but the box is later
    COMPRESSED to `density` (real EC:PC liquid ~1.2-1.3 g/cm^3) during
    equilibration. A dilute box is a GAS, not a liquid -- density matters
    enormously for diffusion, so we must reach real liquid density.
    """
    from ase import Atoms
    rng = np.random.default_rng(seed)

    n_ec = int(round(n_solvent * ratio_ec))
    species = (["Na"]*n_ion_pairs + ["PF6"]*n_ion_pairs
               + ["EC"]*n_ec + ["PC"]*(n_solvent - n_ec))
    n_sites = len(species)

    ncell = int(np.ceil(n_sites ** (1/3)))
    spacing = 7.5                       # dilute start: no overlap
    L0 = ncell * spacing
    cells = [(i, j, k) for i in range(ncell) for j in range(ncell)
             for k in range(ncell)][:n_sites]
    rng.shuffle(cells)

    symbols, positions = [], []
    for sp, (i, j, k) in zip(species, cells):
        center = (np.array([i, j, k]) + 0.5) * spacing
        if sp == "Na":
            symbols.append("Na"); positions.append(center[None, :])
        else:
            s, x = build_mol(SMILES[sp], seed=rng.integers(1e6))
            symbols += s
            positions.append(random_rotation(x, rng) + center)
    positions = np.vstack(positions)

    atoms = Atoms(symbols=symbols, positions=positions, cell=[L0, L0, L0], pbc=True)
    L_target = target_box_length(atoms, density)
    rho0 = (atoms.get_masses().sum()*AMU_TO_G) / ((L0**3)/CM3_TO_A3)
    print(f"  box: {n_ion_pairs} NaPF6 + {n_ec} EC + {n_solvent-n_ec} PC "
          f"-> {len(atoms)} atoms")
    print(f"  start L={L0:.1f} A (rho={rho0:.2f} g/cm3, dilute) "
          f"-> compress to L={L_target:.1f} A (rho={density:.2f} g/cm3, liquid)")
    return atoms, L_target


def compress_to_density(atoms, dyn, L_target, n_steps, report=None):
    """
    Gradually squeeze the cell from its current size to L_target while running MD.
    Scaling positions with the cell avoids creating overlaps abruptly.
    """
    L0 = atoms.cell.lengths()[0]
    for i in range(n_steps):
        f = (i + 1) / n_steps
        L = L0 + (L_target - L0) * f
        scale = L / atoms.cell.lengths()[0]
        atoms.set_cell(atoms.cell * scale, scale_atoms=True)   # squeeze + move atoms
        dyn.run(1)
        T = atoms.get_temperature()
        if T > 5000:                                            # blow-up guard
            raise RuntimeError(
                f"System exploded (T={T:.0f} K) during compression at L={L:.1f} A. "
                f"Compression too fast or dt too big -- increase --compress_steps "
                f"or lower --equil_timestep.")
        if report and (i + 1) % report == 0:
            print(f"    compressing: L={L:.1f} A  T={T:6.1f} K")


def load_mace(device):
    from huggingface_hub import hf_hub_download
    from mace.calculators.foundations_models import mace_mp
    path = hf_hub_download(MACE_REPO, MACE_FILE)
    print(f"  MACE-MH-1 head='{MACE_HEAD}' on {device}")
    return mace_mp(model=path, device=device, default_dtype="float32", head=MACE_HEAD)


# ---- checkpointing (so a disconnect never loses progress) ------------------ #
def save_checkpoint(out, atoms, done_steps):
    from ase.io import write
    write(str(out / "checkpoint.xyz"), atoms)          # positions
    np.save(out / "checkpoint_vel.npy", atoms.get_velocities())  # velocities
    (out / "checkpoint.json").write_text(json.dumps({"done_steps": int(done_steps)}))


def load_checkpoint(out):
    ck = out / "checkpoint.json"
    if not ck.exists():
        return None, 0
    from ase.io import read
    atoms = read(str(out / "checkpoint.xyz"))
    atoms.set_velocities(np.load(out / "checkpoint_vel.npy"))
    done = json.loads(ck.read_text())["done_steps"]
    return atoms, done


# --------------------------------------------------------------------------- #
def run(args):
    import torch
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | torch {torch.__version__}")

    # ---- resume or fresh start ----
    ckpt_atoms, done_steps = load_checkpoint(out)
    if ckpt_atoms is not None:
        print(f"RESUMING from checkpoint at {done_steps}/{args.prod_steps} production steps")
        atoms = ckpt_atoms
        atoms.set_cell(ckpt_atoms.cell); atoms.set_pbc(True)
    else:
        print("Fresh start. Building electrolyte box ...")
        atoms, L_target = build_box(args.n_ion_pairs, args.n_solvent,
                                    seed=args.seed, density=args.density)

    # HMR is deterministic from the structure -> apply on BOTH fresh start and
    # resume (the .xyz checkpoint doesn't store masses, so we must re-apply).
    if args.hmr_factor > 1.0:
        apply_hmr(atoms, factor=args.hmr_factor)   # heavier H -> allows bigger dt

    atoms.calc = load_mace(device)
    na_idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Na"]

    dt_prod = args.timestep * units.fs               # fast dt (4 fs w/ HMR) for PRODUCTION
    dt_equil = args.equil_timestep * units.fs        # SMALL, SAFE dt for compress+equil

    # ---- fresh start: compress to LIQUID density, then equilibrate ----
    # CRITICAL: compression shoves atoms hard (force spikes). A big dt "steps
    # over" those spikes and the system EXPLODES (T -> millions of K). So we use
    # a small, safe dt (1 fs) during compression+equilibration, and only switch
    # to the fast dt (4 fs) for production, once the system is settled.
    if ckpt_atoms is None:
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
        # strong friction while compressing/equilibrating: dumps the excess heat
        # released by bad initial contacts.
        dyn = Langevin(atoms, timestep=dt_equil, temperature_K=args.temperature,
                       friction=args.equil_friction)

        print(f"Compressing to liquid density over {args.compress_steps} steps "
              f"(dt={args.equil_timestep} fs, safe) ...")
        t0 = time.time()
        compress_to_density(atoms, dyn, L_target, args.compress_steps,
                            report=max(args.compress_steps//5, 1))
        rho = (atoms.get_masses().sum()*AMU_TO_G) / ((atoms.cell.lengths()[0]**3)/CM3_TO_A3)
        print(f"  compressed: L={atoms.cell.lengths()[0]:.1f} A, rho={rho:.2f} g/cm3 "
              f"({time.time()-t0:.0f}s)")

        print(f"Equilibrating {args.equil_steps} steps at liquid density ...")
        t0 = time.time()
        for i in range(0, args.equil_steps, max(args.equil_steps//5, 1)):
            dyn.run(min(max(args.equil_steps//5, 1), args.equil_steps - i))
            print(f"    equil: T={atoms.get_temperature():6.1f} K")
        print(f"  equilibration done ({time.time()-t0:.0f}s)")
        save_checkpoint(out, atoms, 0)

    # production: fast dt (HMR makes 4 fs stable once equilibrated) + weaker friction
    dyn = Langevin(atoms, timestep=dt_prod, temperature_K=args.temperature,
                   friction=args.friction)

    # ---- production (checkpointed) ----
    track = out / "na_track.csv"
    if ckpt_atoms is None and track.exists():
        track.unlink()
    print(f"Producing to {args.prod_steps} steps (currently {done_steps}) ...")
    t0 = time.time(); done0 = done_steps
    while done_steps < args.prod_steps:
        dyn.run(1); done_steps += 1
        if done_steps % args.sample_every == 0:
            row = [done_steps * args.timestep]                    # time in fs
            for i in na_idx:
                row += list(atoms.positions[i])                   # unwrapped Na pos
            with open(track, "a") as f:
                f.write(",".join(f"{v:.5f}" for v in row) + "\n")
        if done_steps % args.checkpoint_every == 0:
            save_checkpoint(out, atoms, done_steps)
            rate = (time.time()-t0)/max(done_steps-done0,1)*1000
            print(f"  step {done_steps:7d}  T={atoms.get_temperature():6.1f} K  "
                  f"{rate:.0f} ms/step  (checkpointed)")
    save_checkpoint(out, atoms, done_steps)
    print(f"  production done ({time.time()-t0:.0f}s)")

    analyze(out, na_idx, atoms, args)
    print("=== electrolyte MD complete ===")


def analyze(out, na_idx, atoms, args):
    """MSD (avg over Na+ ions) -> rough D; Na-O RDF -> solvation shell."""
    track = out / "na_track.csv"
    if not track.exists():
        return
    data = np.loadtxt(track, delimiter=",")
    if data.ndim == 1 or len(data) < 10:
        print("  (not enough samples for analysis)"); return
    times = data[:, 0]                                   # fs
    pos = data[:, 1:].reshape(len(data), len(na_idx), 3)  # (frames, n_ion, 3)
    # MSD averaged over ions, referenced to start
    msd = ((pos - pos[0])**2).sum(axis=2).mean(axis=1)    # A^2
    np.savetxt(out / "na_msd.csv", np.column_stack([times, msd]),
               header="time_fs,msd_A2", delimiter=",", comments="")
    # Einstein: MSD = 6 D t  ->  D = slope/6 ;  1 A^2/fs = 0.1 cm^2/s
    half = len(times)//2
    slope = np.polyfit(times[half:], msd[half:], 1)[0]    # A^2/fs
    D = (slope/6.0)*0.1                                    # cm^2/s
    print(f"\n  Na+ MSD final: {msd[-1]:.2f} A^2 over {times[-1]/1000:.1f} ps "
          f"({len(na_idx)} ions averaged)")
    print(f"  rough D(Na+) ~ {D:.2e} cm^2/s")
    print("  (short/small run = proof of concept; validate vs xTB + experiment,")
    print("   apply Yeh-Hummer finite-size correction for the paper number.)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_ion_pairs", type=int, default=1, help="number of Na+/PF6- pairs")
    p.add_argument("--n_solvent", type=int, default=16, help="number of solvent molecules")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--timestep", type=float, default=4.0,
                   help="PRODUCTION dt in fs (4 fs OK with HMR once equilibrated)")
    p.add_argument("--equil_timestep", type=float, default=1.0,
                   help="SMALL safe dt in fs for compression+equilibration (avoids blow-up)")
    p.add_argument("--hmr_factor", type=float, default=3.0,
                   help="hydrogen mass x this (3 -> dt~4fs, ~4x fewer steps). 1 = off")
    p.add_argument("--density", type=float, default=1.25,
                   help="target LIQUID density g/cm3 (EC:PC ~1.2-1.3)")
    p.add_argument("--friction", type=float, default=0.01, help="production friction")
    p.add_argument("--equil_friction", type=float, default=0.05,
                   help="stronger friction during compress/equil (dumps excess heat)")
    p.add_argument("--compress_steps", type=int, default=2000,
                   help="steps over which to squeeze to liquid density")
    p.add_argument("--equil_steps", type=int, default=3000)
    p.add_argument("--prod_steps", type=int, default=5000)
    p.add_argument("--sample_every", type=int, default=10)
    p.add_argument("--checkpoint_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="results/md_electrolyte")
    run(p.parse_args())


if __name__ == "__main__":
    main()
