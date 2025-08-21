# %% [markdown]
# # SCI Colab: Hilbert–Pólya scaffold (built-in first 100 zeros)
# - Computes first 100 true zeros (mpmath)
# - Builds phase lattice, exponential medium, Schrödinger potential
# - Sparse eigen-solve & alignment sqrt(λ_n) vs γ_n
# - Sliding-window fingerprint K_w and KS tests vs GUE/Poisson
# - Fringe-slip peak alignment
# - Exports figures + summary + ZIP

# %%
import os, json, shutil, zipfile
import numpy as np
import matplotlib.pyplot as plt

# Try to import SciPy; in Colab it's preinstalled
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.signal import find_peaks
    from scipy.stats import ks_2samp
except Exception as e:
    raise RuntimeError("SciPy is required. In Colab, run: !pip -q install scipy") from e

# mpmath for Riemann zeros
from mpmath import zetazero, mp
mp.dps = 50  # precision

# ---------------- Config ----------------
N_ZEROS      = 100          # how many true zeros to compute
GRID_POINTS  = 6000         # u-grid size (increase for higher fidelity)
MARGIN_SIGMA = 6.0          # span padding in units of sigma
S_TARGET     = 0.30         # target std for tau before <N>=1 normalization
A            = 1.0          # set a=1 (we fold scale into b)
K_EIG        = 40           # how many eigenpairs to extract
W_EFF        = 40           # sliding window for K_w
OUT_DIR      = "./sci_colab_operator_l4"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------- True zeros --------------
print("Computing first", N_ZEROS, "true Riemann zeros (imag parts)...")
gammas = np.array([float(zetazero(k).imag) for k in range(1, N_ZEROS+1)])
gaps   = np.diff(gammas)
mean_gap = float(gaps.mean())
sigma = mean_gap  # mesoscopic smoothing tied to local mean spacing

# --------- Phase & derivatives ----------
def phi_and_derivs(u_grid, gammas, sigma):
    U   = u_grid[:, None] - gammas[None, :]
    S2  = sigma**2
    phi  = np.arctan(U / sigma).sum(axis=1)
    den  = (U*U + S2)
    phi1 = (sigma / den).sum(axis=1)
    phi2 = (-2.0 * sigma * U / (den**2)).sum(axis=1)
    return phi, phi1, phi2

u_min = gammas.min() - MARGIN_SIGMA * sigma
u_max = gammas.max() + MARGIN_SIGMA * sigma
u     = np.linspace(u_min, u_max, GRID_POINTS)
h     = u[1] - u[0]

phi, phi1, phi2 = phi_and_derivs(u, gammas, sigma)
phi_c = phi - phi.mean()

# scale tau so std(tau) ~ S_TARGET
b = float(S_TARGET / (phi_c.std() + 1e-18))
g = A * b

# Exponential medium (for reference) and Schrödinger potential
# N(u) = exp{ g * (phi - <phi>) }, T_int'(u) = N/c0
N_u = np.exp(g * phi_c)
V_S = 0.5*g*phi2 + 0.25*(g**2)*(phi1**2)

# ------------- Assemble H ---------------
# H = -d^2/du^2 + V_S with Dirichlet BCs
main = (2.0 / h**2) + V_S
off  = (-1.0 / h**2) * np.ones(GRID_POINTS - 1)
H = sp.diags([off, main, off], offsets=[-1, 0, 1], format="csr")

# Sparse eigensolve
print("Solving sparse eigenproblem...")
try:
    # shift-invert near min(V_S) for speed
    sigma_si = float(np.min(V_S))
    evals, evecs = spla.eigsh(H, k=min(K_EIG, GRID_POINTS-2), sigma=sigma_si, which='LM', maxiter=20000)
except Exception:
    evals, evecs = spla.eigsh(H, k=min(K_EIG, GRID_POINTS-2), which='SA', maxiter=20000)

# Sort by eigenvalue; clip tiny negatives from numerics
order = np.argsort(evals)
evals = np.maximum(evals[order], 0.0)
evecs = evecs[:, order]

# ------------- Alignment ----------------
K_use      = min(K_EIG, len(gammas))
lam        = evals[:K_use]
sqrt_lam   = np.sqrt(lam + 1e-18)
gamma_use  = gammas[:K_use]

# One-parameter scale fit: minimize || sqrt(λ) - k * γ ||
k_fit = float(np.dot(gamma_use, sqrt_lam) / np.dot(gamma_use, gamma_use))
sqrt_lam_fit = k_fit * gamma_use
rel_err = np.abs(sqrt_lam - sqrt_lam_fit) / (np.abs(gamma_use) + 1e-18)
MRE = float(np.mean(rel_err))
P95 = float(np.percentile(rel_err, 95))

# ---------- Controls (GUE/Poisson) -----
rng = np.random.default_rng(123)

def sample_gue_wigner_surmise(n):
    # Wigner surmise CDF: F(s) = 1 - exp(-π s^2 / 4) => s = sqrt(-4/π * ln(1-u))
    u = rng.random(n)
    s = np.sqrt(-4.0/np.pi * np.log(1 - u))
    s = s / s.mean()  # normalize to mean 1
    return s

def surrogate_gammas(kind, n, base_start, mean_gap):
    if kind == "poisson":
        gaps = rng.exponential(mean_gap, size=n)
    elif kind == "gue":
        s = sample_gue_wigner_surmise(n)
        gaps = s * mean_gap
    else:
        raise ValueError("unknown kind")
    gsur = np.cumsum(gaps)
    gsur -= gsur.min()
    gsur += base_start
    return gsur

def ctrl_mre(gsur):
    sl = np.sqrt(lam + 1e-18)
    k  = float(np.dot(gsur[:K_use], sl) / np.dot(gsur[:K_use], gsur[:K_use]))
    return float(np.mean(np.abs(sl - k*gsur[:K_use]) / (np.abs(gsur[:K_use]) + 1e-18)))

gsur_poi = surrogate_gammas("poisson", K_use, gammas.min(), mean_gap)
gsur_gue = surrogate_gammas("gue",     K_use, gammas.min(), mean_gap)
MRE_poi  = ctrl_mre(gsur_poi)
MRE_gue  = ctrl_mre(gsur_gue)

# ------------- Kw fingerprint -----------
def sliding_Kw(seq, w):
    # Φ_w = mean, H_w = var, K_w = Φ/H
    seq = np.asarray(seq)
    if len(seq) < w:
        return np.array([])
    out = []
    for i in range(len(seq) - w + 1):
        win = seq[i:i+w]
        mu  = win.mean()
        var = win.var(ddof=0) + 1e-18
        out.append(mu / var)
    return np.array(out)

# Use unfolded gaps (normalize to global mean)
tau = gaps / gaps.mean()
Kw_riem = sliding_Kw(tau, W_EFF)

# Controls for Kw (same length as gaps)
gaps_poi = rng.exponential(gaps.mean(), size=len(gaps))
gaps_gue = sample_gue_wigner_surmise(len(gaps)) * gaps.mean()
Kw_poi   = sliding_Kw(gaps_poi / gaps_poi.mean(), W_EFF)
Kw_gue   = sliding_Kw(gaps_gue / gaps_gue.mean(), W_EFF)

# KS tests on Kw distributions
def ks_pair(a, b):
    if len(a)==0 or len(b)==0:
        return dict(D=float('nan'), p=float('nan'))
    D, p = ks_2samp(a, b, alternative='two-sided', mode='auto')
    return dict(D=float(D), p=float(p))

KS_RG = ks_pair(Kw_riem, Kw_gue)
KS_RP = ks_pair(Kw_riem, Kw_poi)
KS_GP = ks_pair(Kw_gue, Kw_poi)

# -------- Fringe-slip alignment ---------
# Peaks in phi'(u) (which equals (1/g) d(log N)/du) should center near γ_n
peaks, _ = find_peaks(phi1, height=np.percentile(phi1, 75))  # take stronger peaks
u_peaks = u[peaks]

def nearest_errors(points, refs):
    # for each point, distance to nearest ref
    errs = []
    j = 0
    for x in points:
        while j+1 < len(refs) and abs(refs[j+1]-x) < abs(refs[j]-x):
            j += 1
        errs.append(abs(refs[j] - x))
    return np.array(errs)

err_peaks = nearest_errors(u_peaks, gammas)
FR_MEAN = float(np.mean(err_peaks)) if len(err_peaks)>0 else float('nan')
FR_P95  = float(np.percentile(err_peaks, 95)) if len(err_peaks)>0 else float('nan')

# --------------- Plots ------------------
plt.figure(figsize=(8,4.5))
plt.plot(u, V_S)
plt.xlabel("u"); plt.ylabel("V_S(u)")
plt.title("Schrödinger Potential from Phase Lattice")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig1_potential.png"), dpi=150)
plt.close()

plt.figure(figsize=(5.6,5.6))
plt.scatter(gamma_use, sqrt_lam, s=14, label=r"$\sqrt{\lambda_n}$")
plt.plot(gamma_use, sqrt_lam_fit, label=fr"fit: $k={k_fit:.4f}$")
plt.xlabel(r"$\gamma_n$"); plt.ylabel(r"$\sqrt{\lambda_n}$")
plt.legend()
plt.title(f"Alignment  MRE={MRE:.3e}, p95={P95:.3e}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig2_alignment.png"), dpi=150)
plt.close()

plt.figure(figsize=(8,4.5))
plt.plot(np.arange(1, K_use+1), rel_err, marker='o', label="Real zeros")
plt.plot(np.arange(1, K_use+1), np.full(K_use, MRE_poi), label=f"Poisson MRE={MRE_poi:.2e}")
plt.plot(np.arange(1, K_use+1), np.full(K_use, MRE_gue), label=f"GUE MRE={MRE_gue:.2e}")
plt.xlabel("index n"); plt.ylabel("relative error")
plt.title("Relative Error: Real vs Controls (flat line = control MRE)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig3_error_compare.png"), dpi=150)
plt.close()

plt.figure(figsize=(8,4.5))
for i in range(min(3, evecs.shape[1])):
    y = evecs[:, i]
    y = y / (np.max(np.abs(y)) + 1e-18)
    plt.plot(u, y + i, label=f"mode {i}")
plt.xlabel("u"); plt.ylabel("normalized modes (offset)")
plt.title("Lowest Eigenmodes")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig4_eigenmodes.png"), dpi=150)
plt.close()

# --------- Save summary + ZIP -----------
summary = {
    "N_zeros": int(N_ZEROS),
    "mean_gap": float(mean_gap),
    "sigma": float(sigma),
    "a": float(A),
    "b": float(b),
    "g": float(g),
    "grid_points": int(GRID_POINTS),
    "k_eigs": int(K_EIG),
    "alignment": {"k_fit": float(k_fit), "MRE": float(MRE), "p95": float(P95)},
    "Kw": {
        "Riemann": {"n": int(len(Kw_riem)), "min": float(np.min(Kw_riem)) if len(Kw_riem)>0 else None,
                    "max": float(np.max(Kw_riem)) if len(Kw_riem)>0 else None,
                    "mean": float(np.mean(Kw_riem)) if len(Kw_riem)>0 else None,
                    "std": float(np.std(Kw_riem)) if len(Kw_riem)>0 else None},
        "GUE":     {"n": int(len(Kw_gue)), "mean": float(np.mean(Kw_gue)) if len(Kw_gue)>0 else None,
                    "std": float(np.std(Kw_gue)) if len(Kw_gue)>0 else None},
        "Poisson": {"n": int(len(Kw_poi)), "mean": float(np.mean(Kw_poi)) if len(Kw_poi)>0 else None,
                    "std": float(np.std(Kw_poi)) if len(Kw_poi)>0 else None},
        "KS": {"Riemann_vs_GUE": KS_RG, "Riemann_vs_Poisson": KS_RP, "GUE_vs_Poisson": KS_GP}
    },
    "fringe_alignment": {
        "num_peaks": int(len(u_peaks)),
        "mean_abs_error": FR_MEAN,
        "p95_abs_error": FR_P95
    },
    "figures": {
        "potential": os.path.join(OUT_DIR, "fig1_potential.png"),
        "alignment": os.path.join(OUT_DIR, "fig2_alignment.png"),
        "error_compare": os.path.join(OUT_DIR, "fig3_error_compare.png"),
        "eigenmodes": os.path.join(OUT_DIR, "fig4_eigenmodes.png")
    }
}
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

zip_path = os.path.join(OUT_DIR, "sci_colab_l4_artifacts.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for fname in ["summary.json", "fig1_potential.png", "fig2_alignment.png",
                  "fig3_error_compare.png", "fig4_eigenmodes.png"]:
        z.write(os.path.join(OUT_DIR, fname), arcname=fname)

print("=== SCI Colab L4 run complete ===")
print(json.dumps(summary, indent=2))
print("Artifacts dir:", OUT_DIR)
print("ZIP:", zip_path)
