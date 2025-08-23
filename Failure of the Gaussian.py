# %% [markdown]
# Colab-ready: Gaussian proxy vs empirical K (self-contained)
import os, math, csv
import numpy as np
import pandas as pd

# -------------------- Config --------------------
SEED        = 123
M           = 6000            # grid size
U_MIN, U_MAX= -50.0, 50.0
N_ZEROS     = 1200            # synthetic "zeros"
GAP         = 1.0             # mean spacing of zeros
JITTER      = 0.06            # jitter for zeros (non-Gaussianity driver)
SIGMA       = 0.9             # arctan kernel width for phase
TAU_GAIN    = 2.5             # scales tau amplitude (mesoscopic structure)
A           = 1.1             # contrast in N = exp{ a * tau }
LOGN_CLIP   = 40.0            # numerical safety
WIN         = 400             # sliding window size
STEP        = 100             # stride
SAVE_DIR    = "/content"      # <- Colab-safe

np.random.seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------- Build synthetic phase/tau --------------------
# Synthetic "zeros" (quasi-lattice + jitter)
u = np.linspace(U_MIN, U_MAX, M)
gam = np.arange(N_ZEROS)*GAP + np.random.normal(0, JITTER, N_ZEROS)

# Phase as sum of arctan (integral of Lorentzian lattice), then center & standardize
UU = u[:, None]
GG = gam[None, :]
phi = np.arctan2(UU - GG, SIGMA).sum(axis=1)

# center & standardize phi, then scale to tau
phi -= (np.trapz(phi, u) / (u[-1]-u[0]))
phi_std = np.std(phi) + 1e-18
tau = TAU_GAIN * (phi / phi_std)

# Construct N = exp{ a * tau } and enforce <N>=1 (mean-gauge like in paper)
logN_raw = np.clip(A * tau, -LOGN_CLIP, LOGN_CLIP)
N_raw = np.exp(logN_raw)
meanN = np.trapz(N_raw, x=u)/(u[-1]-u[0])
N = N_raw / (meanN + 1e-18)

# -------------------- Sliding-window statistics --------------------
def windows(arr, w, step):
    for i in range(0, len(arr)-w+1, step):
        yield i, arr[i:i+w]

rows = []
for i, (idx, segN) in enumerate(windows(N, WIN, STEP)):
    segTau = tau[idx:idx+WIN]
    # empirical K over N (mean/var)
    mean_segN = float(np.mean(segN))
    var_segN  = float(np.var(segN, ddof=1))
    K_emp = mean_segN / (var_segN + 1e-18)

    # window variance of tau
    var_tau = float(np.var(segTau, ddof=1))

    # strict lognormal prediction under <N>=1 renorm: K = 1/(e^{a^2 Var(tau)} - 1)
    v = (A**2) * var_tau
    K_gauss_renorm = 1.0 / (math.exp(v) - 1.0 + 1e-300)

    # naive exponential proxy (often misused): K = e^{a^2 Var(tau)}
    K_exp = math.exp(v)

    rows.append({
        "win_start": idx,
        "mean_N": mean_segN,
        "var_N": var_segN,
        "K_emp": K_emp,
        "var_tau": var_tau,
        "K_gauss_renorm": K_gauss_renorm,
        "K_exp_naive": K_exp,
        "ratio_naive_over_emp": (K_exp / (K_emp+1e-300)),
        "ratio_renorm_over_emp": (K_gauss_renorm / (K_emp+1e-300)),
        "v=a2VarTau": v
    })

df = pd.DataFrame(rows)

# -------------------- Summaries & save --------------------
def q(x, p): 
    return float(np.quantile(x, p))

summary = {
    "wins": len(df),
    "A": A, "TAU_GAIN": TAU_GAIN, "SIGMA": SIGMA, "JITTER": JITTER,
    "K_emp_median": q(df["K_emp"], 0.5),
    "K_emp_p95":    q(df["K_emp"], 0.95),
    "K_gauss_renorm_median": q(df["K_gauss_renorm"], 0.5),
    "K_exp_naive_median":    q(df["K_exp_naive"],    0.5),
    "ratio_naive_over_emp_med": q(df["ratio_naive_over_emp"], 0.5),
    "ratio_naive_over_emp_p95": q(df["ratio_naive_over_emp"], 0.95),
    "ratio_renorm_over_emp_med": q(df["ratio_renorm_over_emp"], 0.5),
    "ratio_renorm_over_emp_p95": q(df["ratio_renorm_over_emp"], 0.95),
    "v_median": q(df["v=a2VarTau"], 0.5),
    "v_p95":    q(df["v=a2VarTau"], 0.95),
}

print("=== Gaussian proxy vs empirical K (sliding windows) ===")
for k, v in summary.items():
    if isinstance(v, float):
        # print large/small in scientific notation
        if abs(v) >= 1e4 or (abs(v) > 0 and abs(v) < 1e-3):
            print(f"{k:>30s}: {v:.3e}")
        else:
            print(f"{k:>30s}: {v:.6f}")
    else:
        print(f"{k:>30s}: {v}")

# Save CSVs
win_csv = os.path.join(SAVE_DIR, "gaussian_proxy_windows.csv")
sum_csv = os.path.join(SAVE_DIR, "gaussian_proxy_summary.csv")
df.to_csv(win_csv, index=False)
pd.DataFrame([summary]).to_csv(sum_csv, index=False)
print(f"\nSaved per-window metrics -> {win_csv}")
print(f"Saved summary -> {sum_csv}")

# (Optional) quick sanity check line for paper text
print("\nSanity check:")
print("Median(K_emp) vs Median(K_exp_naive): ",
      f"{summary['K_emp_median']:.3e} vs {summary['K_exp_naive_median']:.3e}")
print("Naive/Emp median ratio (orders of magnitude): ",
      f"{summary['ratio_naive_over_emp_med']:.3e}")
