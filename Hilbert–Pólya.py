# ============================================
# SCI – Riemann (first 100 true zeros) Colab test
# ============================================
# What this cell does:
# 1) Compute first 100 nontrivial zeros via mpmath.zetazero
# 2) Build phase lattice φ(u), τ(u), exponential medium N(u), internal time T_int(u)
# 3) Fringe-slip alignment: peaks of φ'(u) near γ_n
# 4) Sliding-window fingerprint K_w on zero gaps; KS vs GUE/Poisson controls
# 5) Export artifacts (CSVs/PNGs/JSON) + ZIP; print concise summary
# --------------------------------------------

!pip -q install mpmath scipy numpy pandas matplotlib

import os, json, math, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import ks_2samp

# ---------- Config ----------
N_ZEROS      = 100      # number of zeta zeros
W_DEFAULT    = 40       # sliding window for Kw
S_TARGET     = 0.30     # target std of τ
KAPPA        = 1.0      # σ = κ * mean_gap
GRID_POINTS  = 6000     # u grid for fields/derivatives
U_MARGIN_SIG = 5        # extend range by ±(U_MARGIN_SIG * σ)

PEAK_HEIGHT_REL = 0.5   # relative threshold for peak detection on norm φ'
PEAK_DIST_IN_SIG = 0.5  # min distance between peaks (in units of σ)

SEED = 42
np.random.seed(SEED)

OUTDIR = Path("/content/sci_paper_scale_100")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- 1) True Riemann zeros ----------
mp.mp.dps = 80
zeros = [mp.zetazero(k) for k in range(1, N_ZEROS+1)]
gammas = np.array([float(mp.im(z)) for z in zeros], dtype=float)
gammas.sort()

# Save zeros
pd.DataFrame({"gamma": gammas}).to_csv(OUTDIR/"riemann_zeros_imag.csv", index=False)

# ---------- 2) Phase lattice & medium ----------
gaps = np.diff(gammas)
mean_gap = float(np.mean(gaps))
sigma = KAPPA * mean_gap
a = 1.0

u_min = gammas[0] - U_MARGIN_SIG * sigma
u_max = gammas[-1] + U_MARGIN_SIG * sigma
U = np.linspace(u_min, u_max, GRID_POINTS)

def phi_of_u(u_vals, gammas, sigma):
    # φ(u) = Σ arctan((u - γ_n)/σ)
    return np.sum(np.arctan((u_vals[:, None] - gammas[None, :]) / sigma), axis=1)

phi = phi_of_u(U, gammas, sigma)
phi_c = phi - phi.mean()
std_phi = float(np.std(phi_c))
b = S_TARGET / std_phi if std_phi > 0 else 1.0
tau = b * phi_c
N_u = np.exp(a * tau)
N_u /= N_u.mean()  # enforce ⟨N⟩=1 gauge
du = U[1] - U[0]
T_int = np.cumsum(N_u) * du  # c0=1 units

# Save fields
pd.DataFrame({
    "u": U,
    "phi": phi,
    "tau": tau,
    "N": N_u,
    "T_int": T_int
}).to_csv(OUTDIR/"U_phi_tau_N_Tint.csv", index=False)

# ---------- 3) Fringe-slip alignment ----------
phi_prime = np.gradient(phi, U)
# normalized for peak picking
phi_prime_norm = (phi_prime - phi_prime.min()) / (phi_prime.max() - phi_prime.min() + 1e-12)
min_distance_pts = max(1, int((PEAK_DIST_IN_SIG * sigma) / (du + 1e-12)))
peaks, props = find_peaks(phi_prime_norm, height=PEAK_HEIGHT_REL, distance=min_distance_pts)
u_peaks = U[peaks]

# distance from each detected peak to nearest gamma
def nearest_distance(xs, ys):
    ys = np.sort(ys)
    dists = []
    for x in xs:
        j = np.searchsorted(ys, x)
        cand = []
        if j < len(ys): cand.append(abs(x - ys[j]))
        if j > 0: cand.append(abs(x - ys[j-1]))
        dists.append(min(cand) if cand else np.nan)
    return np.array(dists, float)

align_dists_abs = nearest_distance(u_peaks, gammas)
align_mean_abs = float(np.nanmean(align_dists_abs)) if align_dists_abs.size else float('nan')
align_p95_abs  = float(np.nanpercentile(align_dists_abs,95)) if align_dists_abs.size else float('nan')

pd.DataFrame({
    "u_peak": u_peaks,
    "nearest_gamma_dist_abs": align_dists_abs
}).to_csv(OUTDIR/"fringe_alignment.csv", index=False)

# ---------- 4) Fingerprint Kw vs controls ----------
def sliding_kw(series, w):
    s = np.asarray(series, float)
    if len(s) < w: return np.array([])
    out = []
    for i in range(len(s)-w+1):
        win = s[i:i+w]
        m = np.mean(win); v = np.var(win, ddof=0)
        out.append(np.nan if v<=1e-15 else m/v)
    return np.array(out, float)

# unfolded gaps
gaps_unfold = gaps / mean_gap
w_eff = min(W_DEFAULT, max(10, len(gaps_unfold)//2))
Kw_riem = sliding_kw(gaps_unfold, w_eff)
Kw_riem = Kw_riem[np.isfinite(Kw_riem)]

# Controls
rng = np.random.default_rng(SEED)
def sample_poisson(n, rng):
    return rng.exponential(1.0, size=n)

# Wigner surmise (GOE-like) as a lightweight GUE control surrogate
def sample_wigner_goe(n, rng):
    # p(s) = (π/2) s exp(-π s^2 / 4)
    # acceptance–rejection
    s = []
    while len(s) < n:
        x = rng.exponential(1.0) * 3.0
        y = rng.random()
        px = (math.pi/2.0) * x * math.exp(- (math.pi/4.0) * x * x)
        if y < min(1.0, px): s.append(x)
    return np.array(s[:n])

def kw_pool_from_sampler(sampler, n_len, w, n_trials=200):
    pool = []
    for _ in range(n_trials):
        seq = sampler(n_len, rng)
        K = sliding_kw(seq, w)
        pool.extend(list(K[np.isfinite(K)]))
    return np.array(pool, float)

Kw_poi = kw_pool_from_sampler(sample_poisson, len(gaps_unfold), w_eff, n_trials=200)
Kw_gue = kw_pool_from_sampler(sample_wigner_goe, len(gaps_unfold), w_eff, n_trials=200)

# KS tests
ks_R_G = ks_2samp(Kw_riem, Kw_gue, alternative='two-sided', mode='auto')
ks_R_P = ks_2samp(Kw_riem, Kw_poi, alternative='two-sided', mode='auto')
ks_G_P = ks_2samp(Kw_gue,  Kw_poi, alternative='two-sided', mode='auto')

# Save Kw pools
pd.DataFrame({"Kw_riem": Kw_riem}).to_csv(OUTDIR/"kw_riemann.csv", index=False)
pd.DataFrame({"Kw_gue": Kw_gue}).to_csv(OUTDIR/"kw_gue.csv", index=False)
pd.DataFrame({"Kw_poisson": Kw_poi}).to_csv(OUTDIR/"kw_poisson.csv", index=False)

# ---------- 5) Plots ----------
plt.figure(figsize=(9,4))
plt.plot(U, phi)
plt.xlabel("u"); plt.ylabel("φ(u)")
plt.title("Phase lattice φ(u)")
plt.tight_layout(); plt.savefig(OUTDIR/"phi_u.png", dpi=140); plt.show()

plt.figure(figsize=(9,4))
plt.plot(U, phi_prime)
plt.xlabel("u"); plt.ylabel("φ'(u)")
plt.title("Cauchy-smoothed zero density φ'(u)")
plt.tight_layout(); plt.savefig(OUTDIR/"phi_prime_u.png", dpi=140); plt.show()

plt.figure(figsize=(9,4))
plt.plot(U, N_u)
plt.xlabel("u"); plt.ylabel("N(u)")
plt.title("Exponential medium N(u)")
plt.tight_layout(); plt.savefig(OUTDIR/"N_u.png", dpi=140); plt.show()

plt.figure(figsize=(6,5))
plt.hist(Kw_riem, bins=30, density=True, alpha=0.85, label="Riemann (gaps)")
plt.hist(Kw_gue,  bins=30, density=True, alpha=0.55, label="GUE surrogate")
plt.hist(Kw_poi,  bins=30, density=True, alpha=0.55, label="Poisson")
plt.xlabel(r"$K_w$"); plt.ylabel("Density"); plt.legend()
plt.title(f"Sliding-window fingerprint $K_w$ (w={w_eff})")
plt.tight_layout(); plt.savefig(OUTDIR/"kw_hist.png", dpi=140); plt.show()

plt.figure(figsize=(9,4))
plt.scatter(u_peaks, np.interp(u_peaks, U, phi_prime_norm), s=10)
for g in gammas: plt.axvline(g, linewidth=0.5, linestyle="--")
plt.xlabel("u"); plt.ylabel("normalized φ'(u)")
plt.title("Peak locations of φ'(u) vs zeros γ_n (vertical dashed)")
plt.tight_layout(); plt.savefig(OUTDIR/"peaks_vs_gammas.png", dpi=140); plt.show()

# ---------- 6) Summary + export ----------
summary = {
    "N_zeros": N_ZEROS,
    "mean_gap": mean_gap,
    "sigma": sigma,
    "a": a,
    "b": b,
    "g": a*b,
    "grid_points": len(U),
    "w_eff": int(w_eff),
    "Kw_riem": {
        "n": int(Kw_riem.size),
        "min": float(np.min(Kw_riem)) if Kw_riem.size else None,
        "max": float(np.max(Kw_riem)) if Kw_riem.size else None,
        "mean": float(np.mean(Kw_riem)) if Kw_riem.size else None,
        "std": float(np.std(Kw_riem)) if Kw_riem.size else None,
    },
    "KS": {
        "Riemann_vs_GUE": {"D": float(ks_R_G.statistic), "p": float(ks_R_G.pvalue)},
        "Riemann_vs_Poisson": {"D": float(ks_R_P.statistic), "p": float(ks_R_P.pvalue)},
        "GUE_vs_Poisson": {"D": float(ks_G_P.statistic), "p": float(ks_G_P.pvalue)},
    },
    "fringe_alignment": {
        "num_peaks": int(len(u_peaks)),
        "mean_abs_error": align_mean_abs,
        "p95_abs_error": align_p95_abs,
    }
}
with open(OUTDIR/"metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("=== SCI Colab test (first 100 true zeros) ===")
print(json.dumps(summary, indent=2))
print("Artifacts directory:", str(OUTDIR))

# Pack ZIP for download
zip_path = "/content/sci_paper_scale_100_artifacts.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for fn in os.listdir(OUTDIR):
        z.write(OUTDIR/fn, arcname=fn)
print("ZIP:", zip_path)

# Optional: auto-download in Colab UI
try:
    from google.colab import files
    files.download(zip_path)
except Exception as e:
    print("(Tip) In Colab, run `from google.colab import files; files.download('"+zip_path+"')` to download.")

