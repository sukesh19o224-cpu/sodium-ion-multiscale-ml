"""
DE-RISK the PyBaMM layer before committing to the 5000-sim sweep.
Tests:
 1. What models live in pybamm.sodium_ion (is there an SPM for a low-fidelity tier)?
 2. Fidelity pair: coarse-mesh vs fine-mesh DFN -> speed + capacity gap (multi-fidelity WITHOUT COMSOL)
 3. Rate capability robustness: C/10, C/5, 1C, 2C, 5C -> does the Na model survive high C?
 4. Mini LHS sweep (80 samples) over proposal parameter ranges -> SOLVER SUCCESS RATE + capacity spread.
    This is the single most important number: if 40% of the param space fails to solve,
    the 5000-sweep plan needs adjustment NOW, not in week 8.
"""
import time, warnings, numpy as np
warnings.filterwarnings("ignore")
import pybamm

print("pybamm", pybamm.__version__)

# ---- 1. what's in sodium_ion ----
attrs = [a for a in dir(pybamm.sodium_ion) if not a.startswith("_")]
print("pybamm.sodium_ion exposes:", attrs)

base_model = pybamm.sodium_ion.BasicDFN()
pv0 = base_model.default_parameter_values

def run_one(pv, crate="1C", npts=None):
    """Solve a controlled discharge; return (capacity_Ah, solve_time, ok)."""
    model = pybamm.sodium_ion.BasicDFN()
    var_pts = None
    if npts is not None:
        var_pts = {"x_n": npts, "x_s": npts, "x_p": npts, "r_n": npts, "r_p": npts}
    try:
        t0 = time.time()
        sim = pybamm.Simulation(
            model, parameter_values=pv,
            experiment=pybamm.Experiment([f"Discharge at {crate} until 2.0 V"]),
            var_pts=var_pts,
        )
        sol = sim.solve()
        dt = time.time() - t0
        Q = float(sol["Discharge capacity [A.h]"].entries[-1])
        return Q, dt, True
    except Exception as e:
        return None, None, False

# ---- 2. coarse vs fine mesh as fidelity pair ----
print("\n--- Fidelity pair (mesh resolution), open-source multi-fidelity ---")
Qc, tc, okc = run_one(pv0, npts=8)    # coarse = low fidelity
Qf, tf, okf = run_one(pv0, npts=30)   # fine   = high fidelity
if okc and okf:
    print(f"  coarse(8 pts):  Q={Qc:.4f} Ah  t={tc:.2f}s")
    print(f"  fine(30 pts):   Q={Qf:.4f} Ah  t={tf:.2f}s")
    print(f"  speedup low->high: {tf/tc:.1f}x slower at high fidelity;  capacity gap={abs(Qf-Qc)/Qf*100:.2f}%")
    print("  => coarse/fine DFN is a VALID low/high fidelity pair (no COMSOL needed)")
else:
    print("  coarse ok:", okc, "fine ok:", okf)

# ---- 3. rate capability robustness ----
print("\n--- Rate capability robustness (high C-rate is where Na models crash) ---")
for cr in ["C/10", "C/5", "1C", "2C", "5C"]:
    Q, dt, ok = run_one(pv0, crate=cr)
    print(f"  {cr:>5}: {'OK ' if ok else 'FAIL'}  Q={Q if Q is None else round(Q,4)}  t={None if dt is None else round(dt,2)}")

# ---- 4. mini LHS sweep: SOLVER SUCCESS RATE over the proposal param space ----
print("\n--- Mini LHS sweep (80 samples) over proposal parameter ranges ---")
# proposal bounds (scalar, directly settable keys)
specs = {
    "Negative particle diffusivity [m2.s-1]": (1e-15, 1e-12),
    "Positive particle diffusivity [m2.s-1]": (1e-14, 1e-11),
    "Negative electrode porosity": (0.2, 0.6),
    "Positive electrode porosity": (0.2, 0.6),
    "Negative particle radius [m]": (1e-7, 1e-5),
    "Positive particle radius [m]": (1e-7, 1e-5),
    "Negative electrode conductivity [S.m-1]": (10, 1000),
}
keys = list(specs)
present = [k for k in keys if k in pv0]
print("  settable proposal params present:", len(present), "/", len(keys))
rng = np.random.default_rng(0)
N = 80
lows = np.array([specs[k][0] for k in present])
highs = np.array([specs[k][1] for k in present])
# log-uniform for the ones spanning orders of magnitude, uniform for porosity
logmask = np.array([("diffusivity" in k or "conductivity" in k or "radius" in k) for k in present])
U = rng.random((N, len(present)))
samples = np.where(logmask, 10**(np.log10(lows) + U*(np.log10(highs)-np.log10(lows))),
                   lows + U*(highs-lows))
ok_count, caps, t0 = 0, [], time.time()
for i in range(N):
    pv = pv0.copy()
    for j, k in enumerate(present):
        pv[k] = float(samples[i, j])
    Q, dt, ok = run_one(pv, crate="1C")
    if ok and Q is not None and Q > 0:
        ok_count += 1; caps.append(Q)
elapsed = time.time() - t0
caps = np.array(caps)
print(f"  SUCCESS RATE: {ok_count}/{N} = {100*ok_count/N:.0f}%  (avg {elapsed/N:.2f}s/sim)")
if len(caps):
    print(f"  capacity spread: min={caps.min():.4f}  max={caps.max():.4f}  "
          f"CV={caps.std()/caps.mean()*100:.0f}%  (need spread for ML to learn)")
print(f"  -> extrapolated 5000-sim sweep wall-clock: ~{elapsed/N*5000/60:.0f} min single-core")
print("\n=== PyBaMM robustness validation complete ===")
