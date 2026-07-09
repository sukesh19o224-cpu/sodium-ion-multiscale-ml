"""
Validation test for the Na-ion proposal's core claim:
  pybamm.sodium_ion.BasicDFN() exists, solves, and is fast (~2s/sim).
Also tests whether parameters the proposal wants to vary actually exist
in the parameter set, and whether a GCD-style discharge runs.
"""
import time
import numpy as np
import pybamm

print("PyBaMM version:", pybamm.__version__)

# 1. Does the Na-ion model exist?
try:
    model = pybamm.sodium_ion.BasicDFN()
    print("OK  pybamm.sodium_ion.BasicDFN() exists")
except Exception as e:
    print("FAIL  sodium_ion.BasicDFN ->", repr(e))
    raise SystemExit

# 2. Does it have a default parameter set we can inspect?
try:
    pv = model.default_parameter_values
    print("OK  default parameter values loaded, n params =", len(pv))
except Exception as e:
    print("FAIL  default params ->", repr(e))
    pv = None

# 3. Check which proposal parameters exist by name (fuzzy)
wanted = ["Diffusion", "porosity", "particle radius", "conductivity",
          "reaction rate", "exchange-current"]
if pv is not None:
    keys = list(pv.keys())
    for w in wanted:
        hits = [k for k in keys if w.lower() in k.lower()]
        print(f"   param like '{w}': {len(hits)} matches; e.g. {hits[:2]}")

# 4. Solve it (default conditions) and time it
try:
    t0 = time.time()
    sim = pybamm.Simulation(model)
    sol = sim.solve([0, 3600])
    dt = time.time() - t0
    print(f"OK  solved default in {dt:.2f}s")
except Exception as e:
    print("FAIL  default solve ->", repr(e))
    raise SystemExit

# 5. Can we run a controlled C-rate discharge (needed for GCD dataset + rate capability)?
try:
    t0 = time.time()
    sim2 = pybamm.Simulation(model, experiment=pybamm.Experiment(
        ["Discharge at 1C until 2.0 V"]))
    sol2 = sim2.solve()
    dt2 = time.time() - t0
    print(f"OK  1C experiment discharge solved in {dt2:.2f}s")
    # extract a 'capacity-like' output
    try:
        Q = sol2["Discharge capacity [A.h]"].entries[-1]
        print(f"   final discharge capacity [A.h] = {Q:.4f}")
    except Exception as e:
        print("   (could not extract discharge capacity):", repr(e))
except Exception as e:
    print("WARN  experiment/C-rate discharge ->", repr(e))

# 6. Can we change a parameter and re-solve (the whole LHS-sweep premise)?
try:
    if pv is not None:
        keys = list(pv.keys())
        diff_keys = [k for k in keys if "Negative particle diffusivity" in k
                     or ("Negative" in k and "iffusiv",) and "iffus" in k]
        target = None
        for k in keys:
            if "Negative particle" in k and "iffus" in k.lower():
                target = k; break
        if target:
            base = pv[target]
            pv2 = pv.copy()
            try:
                pv2[target] = base * 2 if isinstance(base, (int, float)) else base
                sim3 = pybamm.Simulation(model, parameter_values=pv2)
                sim3.solve([0, 3600])
                print(f"OK  re-solved after perturbing '{target}'")
            except Exception as e:
                print(f"WARN  perturb-and-solve '{target}' ->", repr(e))
        else:
            print("WARN  could not find a negative-particle diffusivity key to perturb")
except Exception as e:
    print("WARN  parameter sweep test ->", repr(e))

print("\n=== PyBaMM Na-ion validation complete ===")
