# %% [markdown]
# # SCI Colab: Operator + Fingerprint + Maxwell (GR-cap safe)
# - True zeros (mpmath)
# - Phase lattice -> N(u) with analytic GR-cap calibration (no bisection)
# - H = -d^2/du^2 + V_S(u), eigensolve & alignment sqrt(λ_n) vs γ_n
# - Sliding-window fingerprint K_w + KS tests vs GUE/Poisson
# - Fringe-slip vs {γ_n}
# - Maxwell-layer surrogate: Δt_pred = (ℓ/c0)∫(N-1)du vs xcorr/peak estimate
# - Exports figures + summary + ZIP

# %%
import os, json, zipfile, shutil
import numpy as np
import matplotlib.pyplot as plt

# SciPy stack
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.signal import find_peaks, correlate
    from scipy.stats import ks_2samp
except Exception as e:
    raise RuntimeError("SciPy missing. In Colab: !pip -q install scipy") from e

# Riemann zeros
from mpmath import zetazero, mp
mp.dps = 50

# ---------------- Config ----------------
N_ZEROS      = 100
ALPHA        = 1.6       # sigma = ALPHA * mean_gap
GRID_POINTS  = 6000
A            = 1.0       # we fold scale into b
S_TARGET     = 0.30      # target std for tau before <N>=1
GR_CAP       = 0.70      # ensure max|N-1| <= GR_CAP
W_EFF        = 40        # window for K_w
K_EIG        = 40        # number of eigenpairs
SEED         = 123

# Maxwell surrogate config
C0           = 299792458.0     # m/s
DT_SAMP      = 2.919e-13       # s   (你的日志值)
T_TOTAL      = 2.0e-6          # s   total simulated time
DT_TARGET_NS = 3.10e-9         # s   目标量级（便于得到~纳秒的对比）
SEG_FRACTION = 1.0             # 用整个 u 段计算 ∫(N-1)du

OUT_DIR = "./sci_colab_operator_maxwell"
os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# ---------------- Utilities ----------------
def phi_and_derivs(u_grid, gammas, sigma):
    U   = u_grid[:, None] - gammas[None, :]
    S2  = sigma**2
    phi  = np.arctan(U / sigma).sum(axis=1)
    den  = (U*U + S2)
    phi1 = (sigma / den).sum(axis=1)
    phi2 = (-2.0 * sigma * U / (den**2)).sum(axis=1)
    return phi, phi1, phi2

def assemble_H_fd2(u, V_S):
    h = u[1]-u[0]
    main = (2.0 / h**2) + V_S
    off  = (-1.0 / h**2) * np.ones(len(u)-1)
    H = sp.diags([off, main, off], offsets=[-1,0,1], format='csr')
    return H

def eig_alignment(evals, gammas, K_use):
    order = np.argsort(evals)
    lam   = np.maximum(evals[order][:K_use], 0.0)
    sl    = np.sqrt(lam + 1e-18)
    g_use = gammas[:K_use]
    k_fit = float(np.dot(g_use, sl) / np.dot(g_use, g_use))
    rel   = np.abs(sl - k_fit*g_use) / (np.abs(g_use) + 1e-18)
    return dict(k_fit=float(k_fit), MRE=float(np.mean(rel)), p95=float(np.percentile(rel, 95)))

def sliding_Kw(seq, w):
    seq = np.asarray(seq)
    if len(seq) < w: return np.array([])
    out = []
    for i in range(len(seq)-w+1):
        win = seq[i:i+w]
        mu  = win.mean()
        var = win.var(ddof=0) + 1e-18
        out.append(mu/var)
    return np.array(out)

def ks_pair(a, b):
    if len(a)==0 or len(b)==0:
        return dict(D=float('nan'), p=float('nan'))
    D, p = ks_2samp(a, b, alternative='two-sided', mode='auto')
    return dict(D=float(D), p=float(p))

def sample_gue_wigner_surmise(n, rng):
    u = rng.random(n)
    s = np.sqrt(-4.0/np.pi * np.log(1 - u))
    s = s / s.mean()
    return s

def calibrate_g_for_GR_cap(phi_centered, a=1.0, s_target=0.30, cap=0.05):
    """
    Analytic GR-cap: ensure max|exp(a*b*phi_c)-1| <= cap.
    b0 = s_target/std(phi_c); b_cap = min(log(1+cap), -log(1-cap)) / (a*xmax)
    take b = min(b0, b_cap); tiny shrink to avoid FP edge breach.
    """
    phi_centered = np.asarray(phi_centered, dtype=np.float64)
    xmax = float(np.max(np.abs(phi_centered)))
    if not np.isfinite(xmax) or xmax <= 0.0:
        return 0.0, 0.0, np.ones_like(phi_centered, dtype=np.float64)

    phi_std = float(np.std(phi_centered) + 1e-18)
    b0 = float(s_target / phi_std)

    b_cap_num = min(np.log1p(cap), -np.log1p(-cap))  # both positive ~ cap
    b_cap = float(b_cap_num / (a * xmax))

    b = min(b0, b_cap)
    g = a * b
    N_u = np.exp(g * phi_centered)

    exc = float(np.max(np.abs(N_u - 1.0)))
    if exc > cap:
        shrink = 0.999 * cap / (exc + 1e-18)
        b *= shrink
        g = a * b
        N_u = np.exp(g * phi_centered)
    assert float(np.max(np.abs(N_u - 1.0))) <= cap + 1e-12
    return float(b), float(g), N_u

def parabolic_peak_interp(y, k):
    # three-point quadratic interpolation around index k (assume 0<k<len-1)
    y0, y1, y2 = y[k-1], y[k], y[k+1]
    denom = (y0 - 2*y1 + y2) + 1e-18
    delta = 0.5 * (y0 - y2) / denom
    return float(k + delta), float(y1 - 0.25*(y0 - y2)*delta)

# ---------------- Pipeline ----------------
print(f"Computing first {N_ZEROS} Riemann zeros imag parts...")
gammas = np.array([float(zetazero(k).imag) for k in range(1, N_ZEROS+1)])
gaps   = np.diff(gammas)
mean_gap = float(gaps.mean())
print("mean gap:", mean_gap)

sigma = ALPHA * mean_gap
u_min = gammas.min() - 6.0*sigma
u_max = gammas.max() + 6.0*sigma
u     = np.linspace(u_min, u_max, GRID_POINTS)
h     = u[1]-u[0]

phi, phi1, phi2 = phi_and_derivs(u, gammas, sigma)
phi_c = phi - phi.mean()

# ---- GR-cap safe calibration ----
b, g, N_u = calibrate_g_for_GR_cap(phi_c, a=A, s_target=S_TARGET, cap=GR_CAP)
print(f"[GR-cap] alpha*gap={ALPHA:.3f}, sigma={sigma:.6g}")
print(f"[GR-cap] a={A}, b={b:.6g}, g={g:.6g}, max|N-1|={np.max(np.abs(N_u-1)):.3f}")

# Schrödinger potential & eigensolve
V_S = 0.5*g*phi2 + 0.25*(g**2)*(phi1**2)
H = assemble_H_fd2(u, V_S)
try:
    evals, evecs = spla.eigsh(H, k=min(K_EIG, GRID_POINTS-2), which='SA', maxiter=50000)
except Exception:
    evals, evecs = spla.eigsh(H, k=min(K_EIG, GRID_POINTS-2), which='LM', sigma=float(np.min(V_S)))

K_use = min(K_EIG, len(gammas))
align = eig_alignment(evals, gammas, K_use)
print("[FD2] align:", align)

# ---- Kw fingerprint & KS ----
rng = np.random.default_rng(SEED)
tau = gaps / gaps.mean()
Kw_riem = sliding_Kw(tau, W_EFF)

gaps_poi = rng.exponential(gaps.mean(), size=len(gaps))
gaps_gue = sample_gue_wigner_surmise(len(gaps), rng) * gaps.mean()
Kw_poi   = sliding_Kw(gaps_poi / gaps_poi.mean(), W_EFF)
Kw_gue   = sliding_Kw(gaps_gue / gaps_gue.mean(), W_EFF)

KS_RG = ks_pair(Kw_riem, Kw_gue)
KS_RP = ks_pair(Kw_riem, Kw_poi)
print("[KS] Riem vs GUE:", KS_RG)
print("[KS] Riem vs Poisson:", KS_RP)

# ---- Fringe-slip vs γ_n （用 φ'(u) 的峰）----
peaks, _ = find_peaks(phi1, height=np.percentile(phi1, 75))
u_peaks = u[peaks]
def nearest_errors(points, refs):
    errs = []
    j = 0
    for x in points:
        while j+1 < len(refs) and abs(refs[j+1]-x) < abs(refs[j]-x):
            j += 1
        errs.append(abs(refs[j] - x))
    return np.array(errs, dtype=float)

err_peaks = nearest_errors(u_peaks, gammas) if len(u_peaks)>0 else np.array([])
fr_info = dict(num_peaks=int(len(u_peaks)),
               mean_abs_error=float(np.mean(err_peaks)) if len(err_peaks)>0 else None,
               p95_abs_error=float(np.percentile(err_peaks, 95)) if len(err_peaks)>0 else None)
print("[Fringe-slip]", fr_info)

# ---- Maxwell surrogate: choose ℓ so Δt_pred ~ DT_TARGET_NS ----
# integral over segment
u0 = u[0]; u1 = u[-1]
if SEG_FRACTION < 1.0:
    span = int(len(u)*SEG_FRACTION)
    u_seg = u[:span]
    N_seg = N_u[:span]
else:
    u_seg = u
    N_seg = N_u

I_u = float(np.trapezoid(N_seg - 1.0, x=u_seg))  # ∫(N-1) du
# pick ℓ so that Δt ≈ DT_TARGET_NS: Δt = (ℓ/C0)*I_u => ℓ = Δt * C0 / I_u
ELL = abs(DT_TARGET_NS * C0 / (I_u + 1e-18))
dt_pred = (ELL/C0) * I_u

# synthesize signal and apply (sub-sample) delay
Nsamp = int(np.round(T_TOTAL / DT_SAMP))
t = np.arange(Nsamp) * DT_SAMP

# simple pulse
x = np.zeros(Nsamp)
center = Nsamp//4
width  = max(5, int(1.5e-11/DT_SAMP))  # ~15 ps pulse
x[center:center+width] = 1.0

# fractional delay via FFT phase ramp
def frac_delay(sig, dt, delay):
    # y(t) = x(t - delay)
    N = len(sig)
    freqs = np.fft.rfftfreq(N, d=dt)
    X = np.fft.rfft(sig)
    phase = np.exp(-1j*2*np.pi*freqs*delay)
    Y = X*phase
    y = np.fft.irfft(Y, n=N)
    return y

y = frac_delay(x, DT_SAMP, dt_pred)

# measure delay (xcorr + parabolic peak)
xc = correlate(y, x, mode='full')
lags = np.arange(-len(x)+1, len(y))
k0 = int(np.argmax(xc))
# parabolic interpolation around peak
if 0 < k0 < len(xc)-1:
    k_hat, _ = parabolic_peak_interp(xc, k0)
else:
    k_hat = float(k0)
lag_hat = lags[0] + k_hat
dt_xcorr = float(lag_hat * DT_SAMP)

print(f"[Maxwell] <N>_med={np.median(N_u):.6f}, I_u={I_u:.6e}, ℓ={ELL:.3e} m")
print(f"[Maxwell] Δt_pred={dt_pred:.3e} s  | Δt_xcorr={dt_xcorr:.3e} s")
rel_err = abs(dt_pred - dt_xcorr)/(abs(dt_xcorr)+1e-18)
print(f"[Compare] Pred vs Xcorr: rel.err ≈ {rel_err:.3%}")

# ---------------- Plots ----------------
# V_S
plt.figure(figsize=(8,4.2))
plt.plot(u, V_S)
plt.xlabel("u"); plt.ylabel("V_S(u)")
plt.title("Schrödinger Potential from Phase Lattice")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"fig1_potential.png"), dpi=150); plt.close()

# Alignment
K_use = min(K_EIG, len(gammas))
order = np.argsort(evals)
lam   = np.maximum(evals[order][:K_use], 0.0)
sl    = np.sqrt(lam + 1e-18)
g_use = gammas[:K_use]
k_fit = align["k_fit"]; sl_fit = k_fit * g_use
plt.figure(figsize=(5.6,5.6))
plt.scatter(g_use, sl, s=14, label=r"$\sqrt{\lambda_n}$")
plt.plot(g_use, sl_fit, label=fr"fit: $k={k_fit:.4f}$")
plt.xlabel(r"$\gamma_n$"); plt.ylabel(r"$\sqrt{\lambda_n}$"); plt.legend()
plt.title(f"Alignment  MRE={align['MRE']:.3e}, p95={align['p95']:.3e}")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"fig2_alignment_fd2.png"), dpi=150); plt.close()

# Kw hist
plt.figure(figsize=(8,4.2))
plt.hist(Kw_riem, bins=20, alpha=0.65, label="Riemann")
plt.hist(Kw_gue,  bins=20, alpha=0.65, label="GUE")
plt.hist(Kw_poi,  bins=20, alpha=0.65, label="Poisson")
plt.legend(); plt.xlabel("K_w"); plt.ylabel("count"); plt.title("Fingerprint K_w")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"fig3_kw_hist.png"), dpi=150); plt.close()

# φ'(u) peaks
plt.figure(figsize=(8,4.2))
plt.plot(u, phi1, label=r"$\phi'(u)$")
if len(u_peaks)>0: plt.plot(u_peaks, phi1[peaks], 'rx', label="peaks")
plt.xlabel("u"); plt.ylabel(r"$\phi'(u)$"); plt.title("Fringe-slip markers (peaks of φ')")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"fig1c_phi1_peaks.png"), dpi=150); plt.close()

# Maxwell signals
plt.figure(figsize=(8,4.2))
plt.plot(t*1e9, x, label="x(t)")
plt.plot(t*1e9, y, label="y(t - Δt_pred)")
plt.xlabel("time (ns)"); plt.ylabel("amplitude"); plt.legend()
plt.title("Maxwell-layer surrogate signals")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"fig4_em_signals.png"), dpi=150); plt.close()

# ---------------- Summary & ZIP ----------------
summary = {
    "config": {
        "N_zeros": int(N_ZEROS),
        "alpha": float(ALPHA),
        "grid_points": int(GRID_POINTS),
        "a": float(A),
        "s_target": float(S_TARGET),
        "gr_cap": float(GR_CAP),
        "w_eff": int(W_EFF),
        "k_eig": int(K_EIG),
        "seed": int(SEED)
    },
    "mean_gap": float(mean_gap),
    "sigma": float(sigma),
    "b": float(b),
    "g": float(g),
    "alignment_fd2": align,
    "Kw": {
        "riemann": {
            "n": int(len(Kw_riem)),
            "min": float(np.min(Kw_riem)) if len(Kw_riem)>0 else None,
            "max": float(np.max(Kw_riem)) if len(Kw_riem)>0 else None,
            "mean": float(np.mean(Kw_riem)) if len(Kw_riem)>0 else None,
            "std": float(np.std(Kw_riem)) if len(Kw_riem)>0 else None
        },
        "KS": {
            "Riemann_vs_GUE": KS_RG,
            "Riemann_vs_Poisson": KS_RP
        }
    },
    "fringe_alignment": fr_info,
    "maxwell": {
        "I_u": float(I_u),
        "ell_m": float(ELL),
        "dt_pred": float(dt_pred),
        "dt_xcorr": float(dt_xcorr),
        "rel_err": float(rel_err),
        "dt_samp": float(DT_SAMP),
        "T_total": float(T_TOTAL)
    },
    "figures": {
        "potential": "fig1_potential.png",
        "alignment": "fig2_alignment_fd2.png",
        "kw_hist": "fig3_kw_hist.png",
        "phi1_peaks": "fig1c_phi1_peaks.png",
        "em_signals": "fig4_em_signals.png"
    }
}
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

zip_path = os.path.join(OUT_DIR, "sci_colab_artifacts.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for fn in summary["figures"].values():
        z.write(os.path.join(OUT_DIR, fn), arcname=fn)
    z.write(os.path.join(OUT_DIR,"summary.json"), arcname="summary.json")

print("\n=== SCI Colab run complete ===")
print("Artifacts dir:", OUT_DIR)
print("ZIP:", zip_path)
print(json.dumps(summary, indent=2))

