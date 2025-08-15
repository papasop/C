# Colab-ready, self-contained (no file I/O)
# Forward (Riemann-phase → optics) with T_rel normalization by trapezoid mean,
# then Inverse reconstruction (from T_int or N_meas). Prints metrics only.
# ------------------------------------------------------------------------------

import numpy as np
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid, trapezoid

# -----------------------------
# 0) Utilities
# -----------------------------
def rmse(x, y):
    return float(np.sqrt(np.mean((x - y) ** 2)))

def corr(x, y):
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.linalg.norm(xm) * np.linalg.norm(ym)
    return float(np.dot(xm, ym) / denom) if denom != 0 else np.nan

# -----------------------------
# 1) Parameters (Scenario A baseline)
# -----------------------------
np.random.seed(42)

# ~30 imaginary parts γ_n of nontrivial zeros (ρ = 1/2 + iγ_n)
gamma = np.array([
    14.13472514, 21.02203964, 25.01085758, 30.42487613, 32.93506159,
    37.58617816, 40.91871901, 43.32707328, 48.00515088, 49.77383248,
    52.97032148, 56.44624770, 59.34704400, 60.83177852, 65.11254405,
    67.07981053, 69.54640171, 72.06715767, 75.70469070, 77.14484007,
    79.33737502, 82.91038085, 84.73549298, 87.42527461, 88.80911121,
    92.49189927, 94.65134404, 95.87063423, 98.83119422, 101.3178510
])

# Domain for u (increase M for tighter agreement)
u_min, u_max, M = 0.0, 120.0, 4000
u = np.linspace(u_min, u_max, M)
L = u[-1] - u[0]

# Phase → index parameters
a = 0.25       # coupling from phase to log-index
b = 1.0        # phase scaling
sigma = 2.0    # arctan smoothing
c0 = 299_792_458  # m/s

# Target relative dilation (dimensionless): Scenario A
target_T_rel  = 0.025216
target_mean_N = 1.0 + target_T_rel

# Measurement model
use_T_for_inverse = True   # True: reconstruct from T_int (harder); False: from N_meas
plot_quicklooks   = False  # set True to render quick plots

# Weak-modulation regime ⇒ tiny T_int span ⇒ reduce T-noise
noise_T_std = 2e-14    # seconds
noise_N_std = 2e-3     # relative multiplicative noise on N

# Savitzky–Golay smoothing for T-derivative path
savgol_win  = 151      # odd window; larger → stronger smoothing
savgol_poly = 3

# -----------------------------
# 2) Forward: γ → φ → τ → N (normalized by trapezoid mean) → c_eff → T_int
# -----------------------------
U = u[:, None]
G = gamma[None, :]
phi = np.arctan((U - G) / sigma).sum(axis=1)    # φ(u)
tau_true = b * (phi - phi.mean())               # τ(u), zero-mean

# Raw index and normalization using trapezoid (integral) mean
N0 = np.exp(a * tau_true)
Z_int = trapezoid(N0, u) / L                     # integral mean of N0
scale = (1.0 + target_T_rel) / Z_int             # desired mean / current mean
N_true = N0 * scale                              # enforce mean(N_true) = 1 + T_rel (integral sense)

# Derived observables
c_eff_true = c0 / N_true
T_int_true = cumulative_trapezoid(N_true, u, initial=0) / c0

# T_rel consistency checks
T_rel_from_mean_rect  = N_true.mean() - 1.0
T_rel_from_mean_trapz = (trapezoid(N_true, u) / L) - 1.0
T_rel_from_T          = (T_int_true[-1] - T_int_true[0]) / (L / c0) - 1.0

# -----------------------------
# 3) Measurements (add noise)
# -----------------------------
T_meas = T_int_true + np.random.normal(0.0, noise_T_std, size=T_int_true.shape)
N_meas = N_true * (1.0 + np.random.normal(0.0, noise_N_std, size=N_true.shape))

# -----------------------------
# 4) Inverse reconstruction
# -----------------------------
if use_T_for_inverse:
    # dT_int/du = N/c0  ⇒ N ≈ c0 * dT_meas/du (needs smoothing)
    dT_du = np.gradient(T_meas, u)
    N_hat_raw = c0 * dT_du
    w = savgol_win
    if w % 2 == 0: w -= 1
    # Keep window valid and odd
    w = max(5, min(w, len(u) - 1 if (len(u)-1) % 2 == 1 else len(u) - 2))
    if w % 2 == 0: w -= 1
    N_hat = savgol_filter(N_hat_raw, window_length=w, polyorder=savgol_poly, mode='interp')
else:
    N_hat = N_meas.copy()

# Guard and map back to τ̂, with centering
N_hat = np.clip(N_hat, 1e-12, None)
tau_hat = (1.0 / a) * np.log(N_hat)
tau_hat = tau_hat - tau_hat.mean()

# -----------------------------
# 5) Optional visuals (each in its own figure if enabled)
# -----------------------------
if plot_quicklooks:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(u, tau_true, label="τ (true)")
    plt.plot(u, tau_hat, "--", label="τ̂ (recon)")
    plt.xlabel("u"); plt.ylabel("phase τ"); plt.title("τ vs τ̂"); plt.legend(); plt.show()

    plt.figure()
    plt.plot(u, N_true, label="N (true)")
    plt.plot(u, N_hat, "--", label="N̂ (recon)")
    plt.xlabel("u"); plt.ylabel("index N"); plt.title("N vs N̂"); plt.legend(); plt.show()

    plt.figure()
    plt.plot(u, T_int_true, label="T_int (true)")
    plt.plot(u, T_meas, "--", label="T_meas (noisy)")
    plt.xlabel("u"); plt.ylabel("T_int(u) [s]"); plt.title("Internal time"); plt.legend(); plt.show()

# -----------------------------
# 6) Metrics (print-only)
# -----------------------------
metrics = {
    "use_T_for_inverse": bool(use_T_for_inverse),
    "a": float(a), "b": float(b), "sigma": float(sigma),
    "noise_T_std": float(noise_T_std), "noise_N_std": float(noise_N_std),
    "savgol_win": int(savgol_win), "savgol_poly": int(savgol_poly),
    "corr_tau_tauhat": corr(tau_true, tau_hat),
    "rmse_tau_tauhat": rmse(tau_true, tau_hat),
    "corr_tau_logceff": corr(tau_true, np.log(c_eff_true)),  # ≈ -1
    "N_mean_rect": float(N_true.mean()),
    "N_mean_trapz": float(trapezoid(N_true, u) / L),
    "Tint_span_s": float(T_int_true[-1] - T_int_true[0]),
    "T_rel_from_mean_rect": float(T_rel_from_mean_rect),
    "T_rel_from_mean_trapz": float(T_rel_from_mean_trapz),
    "T_rel_from_T": float(T_rel_from_T),
}

print("==== METRICS ====")
for k, v in metrics.items():
    print(f"{k}: {v}")

