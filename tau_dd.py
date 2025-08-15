# =================== Standing-mode derivative check (clean & robust, FIXED) ===================
# Fixes:
# - Avoid name collisions with a pre-existing array named `a` (use A_COUP instead of `a`)
# - Safe conversion from possibly-array globals to scalars via safe_scalar()
# - No reliance on np.math

import numpy as np
import matplotlib.pyplot as plt

PLOT = True  # set False to disable plots

# ---------- Helpers ----------
def is_vec(x):
    try:
        arr = np.asarray(x, float)
        return arr.ndim == 1 and arr.size > 4 and np.isfinite(arr).all()
    except Exception:
        return False

def safe_scalar(glob_name, default_val):
    """Fetch a scalar from globals(); if it's a vector or invalid, fall back to default."""
    val = globals().get(glob_name, default_val)
    try:
        # allow 0-d array / single-element array / scalar
        arr = np.asarray(val)
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 1 and arr.size == 1:
            return float(arr.item())
        # multi-element -> conflict: fall back to default
        print(f"[warn] globals()['{glob_name}'] is non-scalar (shape {arr.shape}); using default {default_val}.")
        return float(default_val)
    except Exception:
        return float(default_val)

# ---------- 0) Data & parameters ----------
if 'gamma' in globals() and is_vec(globals()['gamma']):
    gamma_arr = np.asarray(globals()['gamma'], float)
else:
    # synthetic Riemann-like zeros for demo if none provided
    def synth_riemann_like(n=1500, t0=50.0):
        g = [float(t0)]
        for _ in range(n-1):
            denom = np.log(max(g[-1], 10.0)/(2.0*np.pi))
            denom = max(denom, 1e-6)
            g.append(g[-1] + 2.0*np.pi/denom)
        return np.array(g, float)
    gamma_arr = synth_riemann_like(1500)

# Use distinct names to avoid clashing with any pre-existing array named `a`
A_COUP       = safe_scalar('a',     0.25)  # kept only for downstream scaling if needed
B_GAIN       = safe_scalar('b',     1.00)  # operate on τ(u) directly
SIGMA_SMOOTH = safe_scalar('sigma', 2.00)

# Grid: coarse φ → interpolate to fine grid (speed + stability)
pad   = 600.0
u_min = float(gamma_arr.min() - pad)
u_max = float(gamma_arr.max() + pad)
M_coarse = 2500
u_coarse = np.linspace(u_min, u_max, M_coarse)

# ---------- 1) φ(u), τ(u) on coarse grid ----------
phi_c = np.zeros_like(u_coarse)
blk = 128
for i in range(0, gamma_arr.size, blk):
    g = gamma_arr[i:i+blk]
    phi_c += np.arctan((u_coarse[:, None] - g[None, :]) / SIGMA_SMOOTH).sum(axis=1)

tau_c = B_GAIN * (phi_c - phi_c.mean())

# ---------- 2) Interp to fine grid ----------
M  = 12000
u  = np.linspace(u_min, u_max, M)
du = u[1] - u[0]
tau = np.interp(u, u_coarse, tau_c)

# ---------- 3) Dominant frequency of τ(u) (Hann window + zero-padding) ----------
win   = 0.5 - 0.5*np.cos(2.0*np.pi * np.arange(M) / (M-1))
tau_w = (tau - tau.mean()) * win
zf    = 4
M_pad = zf * M

Tau_pad  = np.fft.rfft(tau_w, n=M_pad)
freq_pad = np.fft.rfftfreq(M_pad, d=du)
PSD_pad  = (np.abs(Tau_pad)**2) / M_pad
kmax_pad = int(np.argmax(PSD_pad[1:]) + 1)   # skip DC
f_dom    = freq_pad[kmax_pad]
lam_dom  = 1.0 / f_dom if f_dom > 0 else np.nan

# Map back to nearest original-bin frequency
freq        = np.fft.rfftfreq(M, d=du)
kmax        = int(np.argmin(np.abs(freq - f_dom)))
f_dom_exact = freq[kmax]
lam_dom_exact = 1.0 / f_dom_exact if f_dom_exact > 0 else np.nan

# ---------- 4) Keep only dominant bin; reconstruct single-tone τ_dom ----------
Tau_full   = np.fft.rfft(tau - tau.mean())
Tau_single = np.zeros_like(Tau_full)
Tau_single[kmax] = Tau_full[kmax]      # dominant bin only
tau_dom    = np.fft.irfft(Tau_single, n=M)

# ---------- 5) Analytic derivatives in frequency domain ----------
i2pi_f = 1j * (2.0*np.pi * freq)
Tau1   = i2pi_f * Tau_single
Tau2   = - (2.0*np.pi * freq)**2 * Tau_single
tau1   = np.fft.irfft(Tau1, n=M)
tau2   = np.fft.irfft(Tau2, n=M)

# mid 80% stats (reduce edge effects)
lo = int(0.10 * M); hi = int(0.90 * M)
def stats(x):
    xm = x[lo:hi]
    return float(np.mean(xm)), float(np.std(xm))

m1, s1 = stats(tau1)
m2, s2 = stats(tau2)

# ---------- 6) Method A: τ'' dominant frequency via Hann+pad ----------
win2     = 0.5 - 0.5*np.cos(2.0*np.pi * np.arange(M) / (M-1))
tau2_w   = (tau2 - np.mean(tau2)) * win2
T2_pad   = np.fft.rfft(tau2_w, n=M_pad)
freq2_pad= np.fft.rfftfreq(M_pad, d=du)
PSD2_pad = (np.abs(T2_pad)**2) / M_pad
k2_pad   = int(np.argmax(PSD2_pad[1:]) + 1)
f2_pad   = freq2_pad[k2_pad]
lam2_A   = 1.0 / f2_pad if f2_pad > 0 else np.nan
rel_A    = abs(lam2_A - lam_dom_exact) / lam_dom_exact if np.isfinite(lam2_A) else np.nan

# ---------- 7) Method B: projection energy at known f0 ----------
f0 = f_dom_exact
w  = np.exp(-1j * 2.0*np.pi * f0 * (u - u[0]))
E_match = np.abs(np.vdot(w, tau2))**2 / tau2.size
lam2_B  = 1.0 / f0
rel_B   = abs(lam2_B - lam_dom_exact) / lam_dom_exact if np.isfinite(lam2_B) else np.nan

# ---------- 8) Print summary ----------
print("\n=== ANALYTIC-DERIVATIVE (single-tone) — CLEAN SUMMARY (FIXED) ===")
print(f"A_COUP={A_COUP:.3f}, B_GAIN={B_GAIN:.3f}, SIGMA={SIGMA_SMOOTH:.3f}, M={M}, du={du:.6g}, zero-pad×{zf}")
print(f"[τ]   f_dom≈{f_dom:.6g} cycles/u  (zero-padded),  λ≈{lam_dom:.6f} (Δu)")
print(f"[τ]   mapped to original grid:    f_dom={f_dom_exact:.6g}, λ={lam_dom_exact:.6f}")
print(f"[τ']  mean={m1:.3e}, std={s1:.3e}   (middle 80%)")
print(f"[τ''] mean={m2:.3e}, std={s2:.3e}   (middle 80%)")
print(f"[τ''] Method A (Hann+pad) λ={lam2_A:.6f},  rel mismatch={rel_A:.2%}")
print(f"[τ''] Method B (projection) λ={lam2_B:.6f}, rel mismatch={rel_B:.2%} (should be ~0%)")
print(f"[τ''] matched energy at f0 (arb units) E_match={E_match:.3e}")

# ---------- 9) Quick plots ----------
if PLOT:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,3))
    plt.plot(u, tau_dom)
    plt.title("Single-tone τ_dom(u) (dominant bin only)")
    plt.xlabel("u"); plt.ylabel("τ_dom"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(10,3))
    plt.plot(u[lo:hi], tau2[lo:hi])
    plt.title("τ''(u) — analytic derivative (middle 80%)")
    plt.xlabel("u"); plt.ylabel("τ''"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(10,3))
    plt.plot(freq_pad, PSD_pad); plt.axvline(f_dom, ls='--')
    plt.title("PSD of τ(u) with Hann+zero-padding (dominant marked)")
    plt.xlabel("frequency (cycles per u)"); plt.ylabel("PSD"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(10,3))
    plt.plot(freq2_pad, PSD2_pad); plt.axvline(f2_pad, ls='--')
    plt.title("PSD of τ''(u) with Hann+zero-padding (dominant marked)")
    plt.xlabel("frequency (cycles per u)"); plt.ylabel("PSD"); plt.tight_layout(); plt.show()
