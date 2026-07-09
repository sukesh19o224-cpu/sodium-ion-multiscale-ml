"""
Validate the full Person-1 ML SPINE end-to-end on REAL PyBaMM data, sklearn-only
(no torch needed -> avoids the 2GB install; production will use GPyTorch/BoTorch/SHAP,
which are standard drop-ins. This proves the CONCEPT works on real Na-ion data).

Fidelity pair (validated earlier): coarse-mesh DFN = LOW fidelity, fine-mesh DFN = HIGH fidelity.
Tests:
 1. Generate 2-fidelity Na-ion dataset from PyBaMM.
 2. Multi-fidelity GP (Kennedy-O'Hagan AR1) vs single-fidelity GP -> does MF help?
 3. Feature importance (RandomForest; SHAP stand-in) -> interpretable ranking.
 4. Bayesian optimization (hand-coded Expected Improvement) -> finds high-capacity design.
"""
import time, warnings, numpy as np
warnings.filterwarnings("ignore")
import pybamm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

rng = np.random.default_rng(7)
PARAMS = {
    "Negative particle diffusivity [m2.s-1]": (1e-15, 1e-12, True),
    "Positive particle diffusivity [m2.s-1]": (1e-14, 1e-11, True),
    "Negative electrode porosity": (0.2, 0.6, False),
    "Positive electrode porosity": (0.2, 0.6, False),
    "Negative particle radius [m]": (1e-7, 1e-5, True),
    "Positive particle radius [m]": (1e-7, 1e-5, True),
    "Negative electrode conductivity [S.m-1]": (10, 1000, True),
}
keys = list(PARAMS)
pv0 = pybamm.sodium_ion.BasicDFN().default_parameter_values

def sample(n):
    cols = []
    for k in keys:
        lo, hi, islog = PARAMS[k]
        u = rng.random(n)
        cols.append(10**(np.log10(lo)+u*(np.log10(hi)-np.log10(lo))) if islog else lo+u*(hi-lo))
    return np.column_stack(cols)

def solve(X, npts):
    """Run PyBaMM at given mesh resolution; return capacities (NaN on failure)."""
    out = []
    vp = {"x_n":npts,"x_s":npts,"x_p":npts,"r_n":npts,"r_p":npts}
    for row in X:
        pv = pv0.copy()
        for k, v in zip(keys, row):
            pv[k] = float(v)
        try:
            sim = pybamm.Simulation(pybamm.sodium_ion.BasicDFN(), parameter_values=pv,
                experiment=pybamm.Experiment(["Discharge at 1C until 2.0 V"]), var_pts=vp)
            s = sim.solve()
            out.append(float(s["Discharge capacity [A.h]"].entries[-1]))
        except Exception:
            out.append(np.nan)
    return np.array(out)

# featurize: log-transform the log params for nicer GP inputs
def feat(X):
    Z = X.copy()
    for j, k in enumerate(keys):
        if PARAMS[k][2]:
            Z[:, j] = np.log10(Z[:, j])
    return Z

print("Generating 2-fidelity Na-ion dataset from PyBaMM...")
t0 = time.time()
X_low = sample(55); y_low = solve(X_low, 8)              # LOW fidelity (coarse)
m = ~np.isnan(y_low); X_low, y_low = X_low[m], y_low[m]
X_high = X_low[:22].copy(); y_high = solve(X_high, 26)   # HIGH fidelity (fine), subset
m2 = ~np.isnan(y_high); X_high, y_high = X_high[m2], y_high[m2]
print(f"  low-fi: {len(y_low)} pts | high-fi: {len(y_high)} pts | gen time {time.time()-t0:.1f}s")

# split high-fi into train/test
ntr = len(y_high) - 7
Xh_tr, yh_tr = X_high[:ntr], y_high[:ntr]
Xh_te, yh_te = X_high[ntr:], y_high[ntr:]

sc = StandardScaler().fit(feat(X_low))
def K(): return ConstantKernel(1.0)*RBF([1.0]*len(keys)) + WhiteKernel(1e-6)

# ---- 2a. SINGLE-fidelity GP: high-fi train only ----
gp_sf = GaussianProcessRegressor(kernel=K(), normalize_y=True, n_restarts_optimizer=3)
gp_sf.fit(sc.transform(feat(Xh_tr)), yh_tr)
pred_sf = gp_sf.predict(sc.transform(feat(Xh_te)))
rmse_sf = np.sqrt(np.mean((pred_sf - yh_te)**2))

# ---- 2b. MULTI-fidelity GP (AR1): low-fi GP + discrepancy GP ----
gp_lo = GaussianProcessRegressor(kernel=K(), normalize_y=True, n_restarts_optimizer=3)
gp_lo.fit(sc.transform(feat(X_low)), y_low)
lo_at_high = gp_lo.predict(sc.transform(feat(Xh_tr)))
rho = float(np.dot(lo_at_high, yh_tr)/np.dot(lo_at_high, lo_at_high))  # least-squares scale
disc = yh_tr - rho*lo_at_high
gp_d = GaussianProcessRegressor(kernel=K(), normalize_y=True, n_restarts_optimizer=3)
gp_d.fit(sc.transform(feat(Xh_tr)), disc)
pred_mf = rho*gp_lo.predict(sc.transform(feat(Xh_te))) + gp_d.predict(sc.transform(feat(Xh_te)))
rmse_mf = np.sqrt(np.mean((pred_mf - yh_te)**2))

print("\n[Multi-fidelity GP validation]")
print(f"  single-fidelity GP (high-fi only) RMSE = {rmse_sf:.5f} Ah")
print(f"  multi-fidelity  GP (AR1, rho={rho:.2f})  RMSE = {rmse_mf:.5f} Ah")
better = (rmse_sf-rmse_mf)/rmse_sf*100
print(f"  => multi-fidelity is {better:+.0f}% vs single-fidelity  "
      f"({'MF helps' if better>0 else 'MF not better on this tiny set'})")

# ---- 3. feature importance (SHAP stand-in) ----
rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(feat(X_low), y_low)
imp = rf.feature_importances_
order = np.argsort(imp)[::-1]
print("\n[Feature importance — capacity drivers (RandomForest; SHAP in production)]")
for i in order:
    print(f"   {imp[i]*100:5.1f}%  {keys[i]}")

# ---- 4. Bayesian optimization (Expected Improvement) ----
print("\n[Bayesian optimization — maximize capacity via Expected Improvement]")
gp_bo = gp_lo  # use the low-fi surrogate as cheap objective
best = y_low.max()
hist = [best]
for it in range(8):
    cand = sample(400)
    mu, sd = gp_bo.predict(sc.transform(feat(cand)), return_std=True)
    z = (mu - best)/(sd+1e-9)
    ei = (mu-best)*norm.cdf(z) + sd*norm.pdf(z)
    xstar = cand[np.argmax(ei)][None, :]
    ystar = solve(xstar, 8)[0]
    if not np.isnan(ystar) and ystar > best:
        best = ystar
    hist.append(best)
print(f"  start best={hist[0]:.4f} Ah -> after 8 BO iters best={best:.4f} Ah "
      f"({(best-hist[0])/hist[0]*100:+.1f}% improvement)")
print("  BO convergence:", [round(h,4) for h in hist])
print("\n=== ML SPINE validation complete ===")
