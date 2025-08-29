# === Install (quiet) ===
!pip -q install mpmath scipy

# === Imports & helpers ===
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, hilbert
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz
from scipy.special import digamma
from mpmath import mp, zetazero
mp.dps = 50
np.set_printoptions(precision=6, suppress=True)

def ntrapz(y, dx):
    try:
        return np.trapezoid(y, dx=dx)
    except AttributeError:
        return np.trapz(y, dx=dx)

# === Config（文档口径 + 包络互关修复） ================================
c0           = 299_792_458.0
J            = 1.327652314

MATCH_DOC_LENGTH  = True
L_TARGET_M        = 320.021
MATCH_DOC_SIGMA   = True
SIGMA_DOC         = 4.2997600652

USE_FROZEN_G      = True
g_frozen          = 2.0818254069974373e-3
GR_CAP            = 0.70  # 仅当 USE_FROZEN_G=False 时用

nu           = 12000
nz           = 2000
src_offset   = 80
probe_offset = 16
gauss_t0_factor = 1.6

TIMERS = ('xcorr_env', 'gcc_env', 'phase')  # 包络互关、包络 GCC-PHAT、相位斜率

print("配置：",
      f"\n  L_TARGET_M={L_TARGET_M}, MATCH_DOC_LENGTH={MATCH_DOC_LENGTH}",
      f"\n  SIGMA_DOC={SIGMA_DOC}, MATCH_DOC_SIGMA={MATCH_DOC_SIGMA}",
      f"\n  USE_FROZEN_G={USE_FROZEN_G}, g_frozen={g_frozen}",
      f"\n  nu={nu}, nz={nz}, src_offset={src_offset}, probe_offset={probe_offset}, gauss_t0_factor={gauss_t0_factor}",
      f"\n  timers={TIMERS}",
      sep="")

# === 1) zeros & sigma ====================================================
M_seg        = 80
u_span_zeros = max(M_seg, 100)
gamma = np.array([float(zetazero(n).imag) for n in range(1, u_span_zeros + 1)], dtype=np.float64)
gamma_seg = gamma[:M_seg].copy()
delta_gamma_base = float(np.mean(np.diff(gamma_seg)))
sigma_auto = 2.0 * delta_gamma_base
sigma = float(SIGMA_DOC) if MATCH_DOC_SIGMA else sigma_auto
print(f"[Zeros] M_seg={M_seg}, mean Δγ(base)={delta_gamma_base:.9f}, sigma_auto={sigma_auto:.9f}, sigma_used={sigma:.9f}")

# === 2) u-window (exact L targeting, centered) ===========================
base_left, base_right = float(gamma_seg[0]), float(gamma_seg[-1])
base_span_u = base_right - base_left
u_center = 0.5 * (base_left + base_right)

if MATCH_DOC_LENGTH:
    desired_u_len = L_TARGET_M / J
    if desired_u_len <= base_span_u:
        desired_u_len = base_span_u * 1.000001
    u0 = u_center - desired_u_len / 2.0
    u1 = u_center + desired_u_len / 2.0
    if u0 < 0.0:
        u1 += -u0; u0 = 0.0
    beta_ref = max(0.0, (desired_u_len - base_span_u) / (2.0 * sigma))
else:
    beta_ref = 3.0
    u0 = max(0.0, base_left - beta_ref * sigma)
    u1 = base_right + beta_ref * sigma

u  = np.linspace(u0, u1, nu, dtype=np.float64)
L  = J * (u1 - u0)
print(f"[u-window] beta≈{beta_ref:.3f} (ref), u0={u0:.6f}, u1={u1:.6f}")
print(f"[Length] L_target={L_TARGET_M if MATCH_DOC_LENGTH else float('nan'):.3f} m, L_achieved={L:.3f} m, ΔL={(L - (L_TARGET_M if MATCH_DOC_LENGTH else L)):+.3f} m")

# === 3) φ'_σ(u) = analytic − Dirichlet ==================================
s  = (0.5 + sigma) + 1j * u
analytic = 0.5 * digamma(s / 2.0) - 0.5 * np.log(np.pi) + 1.0/s + 1.0/(s - 1.0)
analytic_real = np.real(analytic)

def sieve_primes(n):
    bs = np.ones(n+1, dtype=bool); bs[:2] = False
    p = 2
    while p*p <= n:
        if bs[p]: bs[p*p:n+1:p] = False
        p += 1
    return np.flatnonzero(bs)

Nmax_dirich = 12000
primes = sieve_primes(Nmax_dirich)
half_sigma = 0.5 + sigma
terms_a, terms_w = [], []
for p in primes:
    lp = np.log(p); n = p
    while n <= Nmax_dirich:
        terms_a.append(lp * (n ** (-half_sigma)))   # Λ(p^k) = log p
        terms_w.append(np.log(n))
        n *= p
terms_a = np.array(terms_a, dtype=np.float64)
terms_w = np.array(terms_w, dtype=np.float64)

cos_mat = np.cos(np.outer(terms_w, u))   # (K, nu)
sum_term = terms_a @ cos_mat             # (nu,)
phi_prime = analytic_real - sum_term

# === 4) φ_σ(u) & mean gauge =============================================
phi_sigma = cumtrapz(phi_prime, u, initial=0.0).astype(np.float64)
phi_sigma -= phi_sigma.mean()
print(f"[phi] φ_σ(u) centered (mean≈{phi_sigma.mean():+.3e}); prime-power terms={len(terms_a)}")

# === 5) g & N(u) ========================================================
def calibrate_g_cap_only(phi_centered, cap=0.70):
    xmax = float(np.max(np.abs(phi_centered)))
    if not np.isfinite(xmax) or xmax <= 0.0:
        return 0.0, np.ones_like(phi_centered)
    g = min(np.log1p(cap), -np.log1p(-cap)) / xmax
    N = np.exp(g * phi_centered)
    exc = float(np.max(np.abs(N - 1.0)))
    if exc > cap:
        g *= 0.999 * cap / (exc + 1e-18)
        N = np.exp(g * phi_centered)
    return g, N

if USE_FROZEN_G:
    g = float(g_frozen)
    N_u = np.exp(g * phi_sigma)
    mode = "frozen"
else:
    g, N_u = calibrate_g_cap_only(phi_sigma, cap=GR_CAP)
    mode = "cap-only"

max_exc = float(np.max(np.abs(N_u - 1.0)))
print(f"[Gain] mode={mode}, g={g:.12g}, max|N-1|={max_exc:.3f}")

# === 6) map to z, CFL, time axis ========================================
z  = np.linspace(0.0, L, nz, dtype=np.float64)
dz = z[1] - z[0]
u_for_z = u0 + z / J
N_z = np.interp(u_for_z, u, N_u).astype(np.float64)
n_max = float(np.max(N_z))
dt = 0.99 * dz / (c0 * n_max)           # CFL
steps_travel = int(np.ceil(1.1 * n_max * (nz - 1)))
nt = int(3 * steps_travel)
t  = np.arange(nt, dtype=np.float64) * dt
print(f"[FDTD] L={L:.3f} m, dz={dz:.3e} m, n_max={n_max:.3f}, dt={dt:.3e} s")
print(f"[FAST] nt={nt}, total_time≈{nt*dt:.3e} s")

# Gaussian source
c_eff  = c0 / max(np.mean(N_z), 1.0)
t0     = gauss_t0_factor * (L / c_eff) * 0.25
spread = t0 / 6.0
def gaussian(tt):
    x = (tt - t0) / (spread + 1e-18)
    return np.exp(-0.5 * x*x)

# === 7) 1D FDTD (TE, Mur-1 ABC) =======================================
def mur_update_left(E_new, E_old, c_b, dt, dz):
    k = (c_b*dt - dz) / (c_b*dt + dz)
    return E_old[1] + k*(E_new[1] - E_old[0])
def mur_update_right(E_new, E_old, c_b, dt, dz):
    k = (c_b*dt - dz) / (c_b*dt + dz)
    return E_old[-2] + k*(E_new[-2] - E_old[-1])
def run_fdtd(N_profile, show_progress=True):
    eps = (N_profile.astype(np.float32)**2)
    mu  = np.ones_like(eps, dtype=np.float32)
    Ez = np.zeros(nz, dtype=np.float32)
    Hy = np.zeros(nz, dtype=np.float32)
    Ez_old = Ez.copy()
    probe_idx = nz - 1 - probe_offset
    rec = np.zeros(nt, dtype=np.float32)
    c_left  = c0 / float(N_profile[0])
    c_right = c0 / float(N_profile[-1])
    rpt = max(1, nt // 10)
    for it in range(nt):
        Hy[:-1] += (dt / (mu[:-1] * dz)) * (Ez[1:] - Ez[:-1])
        Ez_old[:] = Ez
        Ez[1:]   += (dt / (eps[1:] * dz)) * (Hy[1:] - Hy[:-1])
        Ez[src_offset] += gaussian(t[it])
        Ez[0]  = mur_update_left(Ez, Ez_old,  c_left,  dt, dz)
        Ez[-1] = mur_update_right(Ez, Ez_old, c_right, dt, dz)
        rec[it] = Ez[probe_idx]
        if show_progress and (it % rpt == 0 or it == nt-1):
            print(f"[FDTD] {it+1}/{nt} ({(it+1)/nt:5.1%})", end="\r")
    if show_progress: print()
    return rec

rec_vac = run_fdtd(np.ones_like(N_z), show_progress=True)
rec_med = run_fdtd(N_z, show_progress=True)

# === 8) predictions (u-continuous & grid-aligned) ========================
z_src, z_prb = src_offset*dz, (nz-1-probe_offset)*dz
u_src, u_prb = u0 + z_src/J, u0 + z_prb/J
j0, j1 = np.searchsorted(u, [u_src, u_prb])
Iu_cum = cumtrapz(N_u - 1.0, u, initial=0.0)
Iu_path = float(Iu_cum[j1] - Iu_cum[j0])
dt_pred_u = (J / c0) * Iu_path

T_vac_grid = (z_prb - z_src) / c0
T_med_grid = ntrapz(N_z[src_offset:nz-probe_offset], dx=dz) / c0
dt_pred_grid = T_med_grid - T_vac_grid

print(f"[预测] Δt_pred_u   = {dt_pred_u:.6e} s")
print(f"[预测] Δt_pred_grid= {dt_pred_grid:.6e} s  (grid-aligned)")

# === 9) robust timing — Envelope xcorr/GCC with proper gating ============
dt_samp = t[1] - t[0]
dist      = z_prb - z_src
T_vac_est = dist / c0
T_med_est = T_vac_grid + dt_pred_grid

pad = max(4*spread, 40*dt_samp)
i0_vac = max(0, int((T_vac_est - pad)/dt_samp))
i1_vac = min(len(t), int((T_vac_est + pad)/dt_samp))
i0_med = max(0, int((T_med_est - pad)/dt_samp))
i1_med = min(len(t), int((T_med_est + pad)/dt_samp))

x = rec_vac[i0_vac:i1_vac].astype(np.float64)
y = rec_med[i0_med:i1_med].astype(np.float64)
m = min(len(x), len(y)); x = x[:m] - x[:m].mean(); y = y[:m] - y[:m].mean()
w = np.hanning(m); xw = x*w; yw = y*w

# 包络（Hilbert）
ex = np.abs(hilbert(xw)); ey = np.abs(hilbert(yw))

# — 归一化互相关（ZNCC）
def zncc_corr(a, b):
    a0 = a - a.mean(); b0 = b - b.mean()
    denom = (np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-18)
    return correlate(a0, b0, mode='full') / denom

# 窗口起点差（全局时间差基线）
offset_base = (i0_med - i0_vac) * dt_samp

# 关键：窗内期望滞后（通常很小，但不一定正好 0）
tau_loc_exp = dt_pred_grid - offset_base
k_center = int(np.round(tau_loc_exp / dt_samp))

# 窄门宽（避免副峰）：W ≈ 0.8 * spread / dt
W = min(max(int(0.8*spread/dt_samp), 24), (m-3)//2)

def peak_parabola(y_arr, k):
    if k <= 0 or k >= len(y_arr)-1: return 0.0
    y0,y1,y2 = y_arr[k-1], y_arr[k], y_arr[k+1]
    den = (y0 - 2*y1 + y2) + 1e-18
    return 0.5 * (y0 - y2) / den

def delay_xcorr_env_local(ex, ey, dt_samp, k0, W):
    r = zncc_corr(ex, ey)                   # ZNCC in [-1,1]
    lags = np.arange(-(len(ex)-1), len(ex))
    mask = (lags >= k0 - W) & (lags <= k0 + W)
    r_win = r[mask]; lags_win = lags[mask]
    k = int(np.argmax(r_win))               # 取正峰（不取绝对值）
    frac = peak_parabola(r_win, k)
    tau = (lags_win[k] + frac) * dt_samp
    # 质量控制：若命中边界或峰值过小，回退 TOA
    if k == 0 or k == len(r_win)-1 or r_win[k] < 0.2:
        ix = int(np.argmax(ex)); iy = int(np.argmax(ey))
        tau = (iy - ix) * dt_samp
    return tau

def delay_gcc_env_local(ex, ey, dt_samp, k0, W):
    n = int(1<<int(np.ceil(np.log2(len(ex)*2 - 1))))
    X = np.fft.rfft(ex, n=n); Y = np.fft.rfft(ey, n=n)
    S = X * np.conj(Y)
    C = S / (np.abs(S) + 1e-18)
    r = np.fft.irfft(C, n=n)
    # center zero-lag
    r = np.concatenate((r[-(n//2):], r[:(n - n//2)]))
    lags = np.arange(-n//2, n - n//2)
    mask = (lags >= k0 - W) & (lags <= k0 + W)
    r_win = r[mask]; lags_win = lags[mask]
    k = int(np.argmax(r_win))               # 取正峰
    frac = peak_parabola(r_win, k)
    tau = (lags_win[k] + frac) * dt_samp
    if k == 0 or k == len(r_win)-1 or r_win[k] < 0.2*np.max(np.abs(r_win)+1e-18):
        ix = int(np.argmax(ex)); iy = int(np.argmax(ey))
        tau = (iy - ix) * dt_samp
    return tau

def delay_phase_slope(xw, yw, dt_samp, fmin_frac=0.05, fmax_frac=0.8):
    n = int(1<<int(np.ceil(np.log2(len(xw)))))
    X = np.fft.rfft(xw, n=n); Y = np.fft.rfft(yw, n=n)
    S = X * np.conj(Y)
    mag = np.abs(S); phi = np.unwrap(np.angle(S))
    f = np.fft.rfftfreq(n, d=dt_samp); w = 2*np.pi*f
    i0 = int(fmin_frac * len(f)); i1 = int(fmax_frac * len(f))
    if i1 - i0 < 8: return 0.0
    Wg = (mag[i0:i1] + 1e-18)
    A = np.vstack([np.ones_like(w[i0:i1]), w[i0:i1]]).T
    ATA = (A*Wg[:,None]).T @ A
    ATb = (A*Wg[:,None]).T @ phi[i0:i1]
    try:
        coef = np.linalg.solve(ATA, ATb)
        tau = -coef[1]
        return float(tau)
    except np.linalg.LinAlgError:
        return 0.0

dt_sim = {}
if 'xcorr_env' in TIMERS:
    tau = delay_xcorr_env_local(ex, ey, dt_samp, k_center, W)
    dt_sim['xcorr_env'] = offset_base + tau
if 'gcc_env' in TIMERS:
    tau = delay_gcc_env_local(ex, ey, dt_samp, k_center, W)
    dt_sim['gcc_env'] = offset_base + tau
if 'phase' in TIMERS:
    tau = delay_phase_slope(xw, yw, dt_samp)
    dt_sim['phase'] = offset_base + tau

# === 10) report ==========================================================
print("\n=== Results ===")
print(f"Δt_pred_u      = {dt_pred_u:.6e} s")
print(f"Δt_pred_grid   = {dt_pred_grid:.6e} s  (grid-aligned)")
for k in TIMERS:
    v = dt_sim[k]
    err_u    = abs(v - dt_pred_u   ) / (abs(dt_pred_u   ) + 1e-18)
    err_grid = abs(v - dt_pred_grid) / (abs(dt_pred_grid) + 1e-18)
    print(f"Δt_sim_{k:8s} = {v:.6e} s | rel.err vs u: {err_u:.3e}, vs grid: {err_grid:.3e}")

# === 11) plots ===========================================================
plt.figure(figsize=(10,4))
plt.plot(t*1e9, rec_vac, label="vacuum")
plt.plot(t*1e9, rec_med, label="medium N(u)")
plt.xlabel("time (ns)"); plt.ylabel("E (arb.)"); plt.title("Time responses at probe")
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(10,4))
plt.plot(u, N_u)
plt.xlabel("u"); plt.ylabel("N(u)"); plt.title("Refractive index N(u) from Dirichlet side")
plt.grid(True); plt.show()
