# Self-contained Colab script: Forward (Riemann-phase → optics) + Inverse (reconstruct τ)
# Ends by printing metrics only. No files, no paths.
# --------------------------------------------------

import numpy as np
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

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
# 1) Parameters (edit here)
# -----------------------------
np.random.seed(42)

# ~30 imaginary parts γ_n of nontrivial zeros (ρ = 1/2 + i γ_n)
gamma = np.array([
    14.13472514, 21.02203964, 25.01085758, 30.42487613, 32.93506159,
    37.58617816, 40.91871901, 43.32707328, 48.00515088, 49.77383248,
    52.97032148, 56.44624770, 59.34704400, 60.83177852, 65.11254405,
    67.07981053, 69.54640171, 72.06715767, 75.70469070, 77.14484007,
    79.33737502, 82.91038085, 84.73549298, 87.42527461, 88.80911121,
    92.49189927, 94.65134404, 95.87063423, 98.83119422, 101.3178510
])

# Domain for u
u_min, u_max, M = 0.0, 120.0, 4000
u = np.linspace(u_min, u_max, M)

# Model parameters (Scenario A by default: a=0.25, sigma=2)
a = 0.25         # coupling from phase to log-index
b = 1.0          # phase scaling
sigma = 2.0      # arctan smoothing
c0 = 299_792_458 # m/s

# Measurement noise
noise_T_std = 2e-12   # seconds, noise on T_int
noise_N_std = 0.002   # relative multiplicative noise on N

# Inverse options
use_T_for_inverse = True  # True: reconstruct from T_meas; False: from N_meas
savgol_win = 101          # Savitzky–Golay window (odd)
savgol_poly = 3           # polynomial order

# -----------------------------
# 2) Forward mapping: γ → φ → τ → N → c_eff → T_int
# -----------------------------
U = u[:, None]
G = gamma[None, :]
phi = np.arctan((U - G) / sigma).sum(axis=1)     # φ(u)
tau_true = b * (phi - phi.mean())                # τ(u) centered
N_true = np.exp(a * tau_true)                    # N(u) = exp(a τ)
c_eff_true = c0 / N_true                         # c_eff
T_int_true = cumulative_trapezoid(N_true, u, initial=0) / c0  # T_int(u)

# -----------------------------
# 3) Generate measurements from forward (no external inputs)
# -----------------------------
T_meas = T_int_true + np.random.normal(0.0, noise_T_std, size=T_int_true.shape)
N_meas = N_true * (1.0 + np.random.normal(0.0, noise_N_std, size=N_true.shape))

# -----------------------------
# 4) Inverse reconstruction: T_meas→N̂ or N_meas→N̂, then τ̂=(1/a)log N̂
# -----------------------------
if use_T_for_inverse:
    # dT_int/du = N/c0  ⇒ N ≈ c0 * dT_meas/du
    dT_du = np.gradient(T_meas, u)
    N_hat_raw = c0 * dT_du
    w = savgol_win
    if w % 2 == 0: w -= 1
    w = max(5, min(w, len(u) - 1 if (len(u)-1)%2==1 else len(u)-2))
    if w % 2 == 0: w -= 1
    N_hat = savgol_filter(N_hat_raw, window_length=w, polyorder=savgol_poly, mode='interp')
else:
    N_hat = N_meas.copy()

N_hat = np.clip(N_hat, 1e-12, None)
tau_hat = (1.0 / a) * np.log(N_hat)
tau_hat = tau_hat - tau_hat.mean()  # center like τ construction

# -----------------------------
# 5) Optional quick looks (comment out if you want zero plots)
# -----------------------------
plt.figure(); plt.plot(u, tau_true, label="τ (true)"); plt.plot(u, tau_hat, "--", label="τ̂ (recon)")
plt.xlabel("u"); plt.ylabel("phase τ"); plt.title("τ vs τ̂"); plt.legend(); plt.show()

plt.figure(); plt.plot(u, N_true, label="N (true)"); plt.plot(u, N_hat, "--", label="N̂ (recon)")
plt.xlabel("u"); plt.ylabel("index N"); plt.title("N vs N̂"); plt.legend(); plt.show()

plt.figure(); plt.plot(u, T_int_true, label="T_int (true)"); plt.plot(u, T_meas, "--", label="T_meas (noisy)")
plt.xlabel("u"); plt.ylabel("T_int(u) [s]"); plt.title("Internal time"); plt.legend(); plt.show()

# -----------------------------
# 6) Metrics (print-only ending)
# -----------------------------
metrics = {
    "use_T_for_inverse": bool(use_T_for_inverse),
    "a": float(a), "b": float(b), "sigma": float(sigma),
    "noise_T_std": float(noise_T_std), "noise_N_std": float(noise_N_std),
    "savgol_win": int(savgol_win), "savgol_poly": int(savgol_poly),
    "corr_tau_tauhat": corr(tau_true, tau_hat),
    "rmse_tau_tauhat": rmse(tau_true, tau_hat),
    "corr_tau_logceff": corr(tau_true, np.log(c_eff_true)),  # ≈ -1 by construction
    "N_mean": float(N_true.mean()),
    "Tint_span_s": float(T_int_true[-1] - T_int_true[0]),
}

print("==== METRICS ====")
for k, v in metrics.items():
    print(f"{k}: {v}")
