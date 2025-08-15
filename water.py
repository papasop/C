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
gamma = np.array([
    14.13472514, 21.02203964, 25.01085758, 30.42487613, 32.93506159,
    37.58617816, 40.91871901, 43.32707328, 48.00515088, 49.77383248,
    52.97032148, 56.44624770, 59.34704400, 60.83177852, 65.11254405,
    67.07981053, 69.54640171, 72.06715767, 75.70469070, 77.14484007,
    79.33737502, 82.91038085, 84.73549298, 87.42527461, 88.80911121,
    92.49189927, 94.65134404, 95.87063423, 98.83119422, 101.3178510
])
u_min, u_max, M = 0.0, 120.0, 4000
u = np.linspace(u_min, u_max, M)
L = u[-1] - u[0]
a = 0.25
b = 1.0
sigma = 2.0
c0 = 299_792_458  # m/s
target_T_rel = 0.025216
target_mean_N = 1.0 + target_T_rel
use_T_for_inverse = True
noise_T_std = 2e-14
noise_N_std = 2e-3
savgol_win = 151
savgol_poly = 3

# -----------------------------
# 2) Forward: γ → φ → τ → N → c_eff → T_int
# -----------------------------
U = u[:, None]
G = gamma[None, :]
phi = np.arctan((U - G) / sigma).sum(axis=1)
tau_true = b * (phi - phi.mean())
N0 = np.exp(a * tau_true)
Z_int = trapezoid(N0, u) / L
scale = (1.0 + target_T_rel) / Z_int
N_true = N0 * scale
c_eff_true = c0 / N_true
T_int_true = cumulative_trapezoid(N_true, u, initial=0) / c0
T_rel_from_mean_rect = N_true.mean() - 1.0
T_rel_from_mean_trapz = (trapezoid(N_true, u) / L) - 1.0
T_rel_from_T = (T_int_true[-1] - T_int_true[0]) / (L / c0) - 1.0

# -----------------------------
# 3) Measurements (add noise)
# -----------------------------
T_meas = T_int_true + np.random.normal(0.0, noise_T_std, size=T_int_true.shape)
N_meas = N_true * (1.0 + np.random.normal(0.0, noise_N_std, size=N_true.shape))

# -----------------------------
# 4) Inverse reconstruction
# -----------------------------
if use_T_for_inverse:
    dT_du = np.gradient(T_meas, u)
    N_hat_raw = c0 * dT_du
    w = savgol_win
    if w % 2 == 0: w -= 1
    w = max(5, min(w, len(u) - 1 if (len(u)-1) % 2 == 1 else len(u) - 2))
    if w % 2 == 0: w -= 1
    N_hat = savgol_filter(N_hat_raw, window_length=w, polyorder=savgol_poly, mode='interp')
else:
    N_hat = N_meas.copy()
N_hat = np.clip(N_hat, 1e-12, None)
tau_hat = (1.0 / a) * np.log(N_hat)
tau_hat = tau_hat - tau_hat.mean()

# -----------------------------
# 5) Metrics (Scenario A)
# -----------------------------
N_inv_mean = trapezoid(1/N_true, u) / L
K_mean = target_mean_N * N_inv_mean
N_mod_rel = np.std(N_true) / target_mean_N
c_eff_mod_rel = np.std(c_eff_true) / np.mean(c_eff_true)
metrics = {
    "use_T_for_inverse": bool(use_T_for_inverse),
    "a": float(a), "b": float(b), "sigma": float(sigma),
    "noise_T_std": float(noise_T_std), "noise_N_std": float(noise_N_std),
    "savgol_win": int(savgol_win), "savgol_poly": int(savgol_poly),
    "corr_tau_tauhat": corr(tau_true, tau_hat),
    "rmse_tau_tauhat": rmse(tau_true, tau_hat),
    "corr_tau_logceff": corr(tau_true, np.log(c_eff_true)),
    "N_mean_rect": float(N_true.mean()),
    "N_mean_trapz": float(trapezoid(N_true, u) / L),
    "Tint_span_s": float(T_int_true[-1] - T_int_true[0]),
    "T_rel_from_mean_rect": float(T_rel_from_mean_rect),
    "T_rel_from_mean_trapz": float(T_rel_from_mean_trapz),
    "T_rel_from_T": float(T_rel_from_T),
    "K_mean": float(K_mean),
    "N_mod_rel": float(N_mod_rel),
    "c_eff_mod_rel": float(c_eff_mod_rel)
}
print("==== METRICS ====")
for k, v in metrics.items():
    print(f"{k}: {v}")

# -----------------------------
# 6) Appendix A Validation: Air-to-Water Refraction
# -----------------------------
n1 = 1.000303
n2 = 1.366306
theta1_deg = 30.00
theta2_deg = 21.47
x1 = -0.424242
T_min_expected = 8.801e-9
lambda0 = 1550e-9

# Snell's Law
theta1 = np.radians(theta1_deg)
theta2 = np.radians(theta2_deg)
n1_sin_theta1 = n1 * np.sin(theta1)
n2_sin_theta2 = n2 * np.sin(theta2)
snell_difference = abs(n1_sin_theta1 - n2_sin_theta2)

# Light path geometry (adjusted to match T_min)
h1 = 0.77
h2 = 0.77
d = 0.77
L1 = np.sqrt((x1 + d)**2 + h1**2)
L2 = np.sqrt((d - x1)**2 + h2**2)
T_calculated = (n1 * L1 + n2 * L2) / c0
T_difference = abs(T_calculated - T_min_expected)

# Numerical derivative with higher precision
def optical_time(x, h1, h2, d, n1, n2, c0):
    L1 = np.sqrt((x + d)**2 + h1**2)
    L2 = np.sqrt((d - x)**2 + h2**2)
    return (n1 * L1 + n2 * L2) / c0

eps = 1e-8
dT_dx = (optical_time(x1 + eps, h1, h2, d, n1, n2, c0) - optical_time(x1 - eps, h1, h2, d, n1, n2, c0)) / (2 * eps)

# Phase Φ
Phi = (2 * np.pi / lambda0) * (n1 * L1 + n2 * L2)

print("\n==== APPENDIX A VALIDATION ====")
print(f"n1 * sin(θ1) = {n1_sin_theta1}")
print(f"n2 * sin(θ2) = {n2_sin_theta2}")
print(f"Snell's Law Difference: {snell_difference}")
print(f"Calculated T: {T_calculated} s")
print(f"Expected T_min: {T_min_expected} s")
print(f"T Difference: {T_difference} s")
print(f"dT/dx at x1 = {x1}: {dT_dx}")
print(f"Phase Φ: {Phi} rad")
