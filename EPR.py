# ============================================
# Phase-Lattice ↔ Fingerprint → EPR (Extended)
# Clean Colab version — NO file I/O, NO paths
# ============================================

import numpy as np
import matplotlib.pyplot as plt

# ---------- Core utilities ----------
def binary_entropy(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return -(p*np.log2(p) + (1-p)*np.log2(1-p))

def ent_entropy_theta(theta):
    # |ψ(θ)> = cos θ |00> + sin θ |11>, reduced eigenvalues {cos^2 θ, sin^2 θ}
    return binary_entropy(np.sin(theta)**2)

def CHSH_max_theta(theta):
    # Horodecki bound for this family: S_max = 2 sqrt(1 + sin^2(2θ))
    return 2.0 * np.sqrt(1.0 + np.sin(2.0*theta)**2)

def sliding(arr, w):
    arr = np.asarray(arr, dtype=float)
    if w < 2 or w > len(arr):
        raise ValueError("w must be in [2, len(arr)]")
    means = np.convolve(arr, np.ones(w)/w, mode='valid')
    sq = np.convolve(arr**2, np.ones(w)/w, mode='valid')
    var = np.clip(sq - means**2, 0, None)
    return means, var

def sliding_K(arr, w):
    m, v = sliding(arr, w)
    return m / np.where(v==0, np.nan, v)

# ---------- First ~50 imaginary parts γ_n of ζ zeros (hard-coded) ----------
gamma = np.array([
    14.134725141, 21.022039639, 25.010857580, 30.424876125, 32.935061588,
    37.586178159, 40.918719012, 43.327073281, 48.005150881, 49.773832478,
    52.970321478, 56.446247697, 59.347044003, 60.831778525, 65.112544048,
    67.079810529, 69.546401711, 72.067157674, 75.704690699, 77.144840069,
    79.337375020, 82.910380854, 84.735492981, 87.425274613, 88.809111208,
    92.491899271, 94.651344041, 95.870634228, 98.831194218, 101.317851006,
    103.725538040, 105.446623052, 107.168611184, 111.029535543, 111.874659177,
    114.320220915, 116.226680321, 118.790782866, 121.370125002, 122.946829294,
    124.256818554, 127.516683880, 129.578704200, 131.087688531, 133.497737203,
    134.756509753, 138.116042055, 139.736208952, 141.123707405
], dtype=float)

tau_n = np.diff(gamma)

# ---------- 2.1 Phase-lattice: ϕ(u) & τ(u) ----------
def phase_lattice_tau(u_grid, gamma, sigma=1.5, b=1.0):
    # ϕ(u) = Σ arctan((u-γ)/σ), τ(u) = b(ϕ - ⟨ϕ⟩)
    UU = u_grid[:, None]
    PHI = np.arctan((UU - gamma[None, :]) / sigma).sum(axis=1)
    PHI = PHI - PHI.mean()
    return b * PHI

def tau_to_theta(tau_u, kappa=0.9, theta_min=0.03, theta_max=np.pi/4 - 0.03):
    # Normalize τ via tanh to [-1,1], then map into [θ_min, θ_max]
    z = np.tanh(kappa * (tau_u / (np.std(tau_u) + 1e-9)))
    z01 = (z - z.min()) / (z.max() - z.min() + 1e-12)
    return theta_min + z01 * (theta_max - theta_min)

# Build u grid; compute τ(u) → θ(u) → S(u), S_max(u)
u_min, u_max = gamma.min() - 15.0, gamma.max() + 15.0
u = np.linspace(u_min, u_max, 1800)

tau_u = phase_lattice_tau(u, gamma, sigma=1.5, b=1.0)
theta_u = tau_to_theta(tau_u, kappa=0.9)
S_u = ent_entropy_theta(theta_u)
Smax_u = CHSH_max_theta(theta_u)
viol_rate_u = (Smax_u > 2.0).mean()

print("Phase-lattice pipeline:")
print("  <S(u)> = %.3f bits,  S_max = %.3f,  Bell violation = %.1f%%"
      % (S_u.mean(), S_u.max(), 100*viol_rate_u))

plt.figure()
plt.plot(u, S_u, label="Entropy S(u)")
plt.plot(u, Smax_u, label="CHSH max")
plt.xlabel("u"); plt.ylabel("value")
plt.title("Phase-lattice (2.1): entropy & CHSH along u")
plt.legend(); plt.show()

plt.figure()
plt.plot(theta_u, Smax_u, '.', ms=2)
plt.xlabel("θ(u)"); plt.ylabel("CHSH max")
plt.title("CHSH vs entanglement parameter θ (phase-lattice)")
plt.show()

# ---------- 2.3 Fingerprint: sliding K_w on τ_n, then map to θ ----------
def K_to_theta(K, theta_min=0.03, theta_max=np.pi/4 - 0.03, kappa=0.9):
    Kst = (K - np.nanmean(K))/(np.nanstd(K) + 1e-9)
    z = np.tanh(kappa * Kst)
    z01 = (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z) + 1e-12)
    return theta_min + z01 * (theta_max - theta_min)

w = 10  # sliding window length
K_tau = sliding_K(tau_n, w=w)
theta_K = K_to_theta(K_tau, kappa=0.9)
S_K = ent_entropy_theta(theta_K)
Smax_K = CHSH_max_theta(theta_K)
viol_rate_K = (Smax_K > 2.0).mean()

print("\nSpacing→Fingerprint pipeline:")
print("  window w = %d,  #windows = %d" % (w, len(K_tau)))
print("  <S> = %.3f bits,  S_max = %.3f,  Bell violation = %.1f%%"
      % (np.nanmean(S_K), np.nanmax(S_K), 100*viol_rate_K))

plt.figure()
plt.plot(K_tau, label="K_w(τ_n)")
plt.xlabel("window index"); plt.ylabel("K_w = mean/var")
plt.title("Fingerprint K_w from zero spacings (w=%d)" % w)
plt.legend(); plt.show()

plt.figure()
plt.plot(S_K, label="Entropy from K-driven θ")
plt.plot(Smax_K, label="CHSH max from K-driven θ")
plt.xlabel("window index"); plt.ylabel("value")
plt.title("Spacing→K pipeline: entropy and CHSH over windows")
plt.legend(); plt.show()

# ---------- (1) Depolarizing noise robustness ----------
# ρ' = (1-p)ρ + p I/4; correlations scale by (1-p)
ps = np.linspace(0.0, 0.5, 26)
viol_curve_phase = [(((1-p)*Smax_u) > 2.0).mean() for p in ps]
viol_curve_kw    = [(((1-p)*Smax_K) > 2.0).mean() for p in ps]

plt.figure()
plt.plot(ps, viol_curve_phase, label="Phase-lattice")
plt.plot(ps, viol_curve_kw, label="Fingerprint")
plt.xlabel("depolarizing p"); plt.ylabel("Violation rate")
plt.title("Bell violation rate vs depolarizing noise p")
plt.legend(); plt.show()

# Approx median critical noise: p_crit ≈ 1 - 2/median(Smax)
pcrit_phase = float(1.0 - 2.0/np.median(Smax_u))
pcrit_kw    = float(1.0 - 2.0/np.nanmedian(Smax_K))
print("\nApprox median p_crit: phase=%.3f, fingerprint=%.3f" % (pcrit_phase, pcrit_kw))

# ---------- (2) Fixed/suboptimal CHSH (x–z plane) ----------
def nz(alpha):
    return np.array([np.sin(alpha), 0.0, np.cos(alpha)])

def E_fixed(theta, a, b):
    # T = diag(s, -s, 1) for |ψ(θ)>, with s=sin(2θ); n = (sin α, 0, cos α)
    s = np.sin(2.0*theta)
    na = nz(a); nb = nz(b)
    # because nb_y = 0, the middle component vanishes
    return float(na[0]*(s*nb[0]) + na[2]*(1.0*nb[2]))

# Fixed angles (no optimization)
a, ap, b, bp = 0.0, np.pi/2, np.pi/4, -np.pi/4

def CHSH_fixed(theta, p=0.0):
    Eab   = E_fixed(theta, a,  b)
    Eabp  = E_fixed(theta, a,  bp)
    Eapb  = E_fixed(theta, ap, b)
    Eapbp = E_fixed(theta, ap, bp)
    S = abs(Eab + Eabp + Eapb - Eapbp)
    return (1.0 - p) * S

for p in [0.0, 0.1, 0.2, 0.3]:
    viol_fixed_phase = (np.array([CHSH_fixed(th, p) for th in theta_u]) > 2.0).mean()
    viol_fixed_kw    = (np.array([CHSH_fixed(th, p) for th in theta_K if not np.isnan(th)]) > 2.0).mean()
    print("Fixed-angle CHSH violation (p=%.1f): phase=%.1f%%, fingerprint=%.1f%%"
          % (p, 100*viol_fixed_phase, 100*viol_fixed_kw))

# ---------- (3) Controls: shuffled spacings & Poisson spacings ----------
rng = np.random.default_rng(123)
mean_tau = float(np.mean(tau_n))
tau_shuf = tau_n.copy(); rng.shuffle(tau_shuf)
tau_pois = rng.exponential(scale=mean_tau, size=len(tau_n))

def pipeline_from_spacings(tau, w=10, kappa=0.9):
    K = sliding_K(tau, w=w)
    # map K -> θ (same mapping as before)
    Kst = (K - np.nanmean(K))/(np.nanstd(K) + 1e-9)
    z = np.tanh(kappa * Kst)
    z01 = (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z) + 1e-12)
    theta = 0.03 + z01 * (np.pi/4 - 0.06)
    S = ent_entropy_theta(theta); Smax = CHSH_max_theta(theta)
    return K, theta, S, Smax

for name, tau in [("original", tau_n), ("shuffled", tau_shuf), ("poisson", tau_pois)]:
    Kc, thetac, Sc, Smaxc = pipeline_from_spacings(tau, w=10, kappa=0.9)
    print("%-9s: <S>=%.3f, S_max=%.3f, violation=%.1f%%"
          % (name, np.nanmean(Sc), np.nanmax(Sc), 100*np.nanmean(Smaxc>2.0)))

# ---------- (4) Sensitivity heatmaps: scan (w, kappa) ----------
def map_theta_from_K(K, kappa=0.9):
    Kst = (K - np.nanmean(K))/(np.nanstd(K) + 1e-9)
    z = np.tanh(kappa * Kst)
    z01 = (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z) + 1e-12)
    return 0.03 + z01 * (np.pi/4 - 0.06)

w_vals = np.arange(6, 21, 2)
kappa_vals = np.linspace(0.4, 1.4, 11)
avgS = np.zeros((len(w_vals), len(kappa_vals)))
viol = np.zeros_like(avgS)

for i, w_ in enumerate(w_vals):
    K_ = sliding_K(tau_n, w=w_)
    for j, k_ in enumerate(kappa_vals):
        theta_ = map_theta_from_K(K_, kappa=k_)
        S_ = ent_entropy_theta(theta_)
        Smax_ = CHSH_max_theta(theta_)
        avgS[i,j] = np.nanmean(S_)
        viol[i,j] = np.nanmean(Smax_ > 2.0)

plt.figure()
plt.imshow(avgS, aspect='auto', origin='lower',
           extent=[kappa_vals.min(), kappa_vals.max(), w_vals.min(), w_vals.max()])
plt.colorbar(label="<S> (bits)")
plt.xlabel("kappa"); plt.ylabel("window w")
plt.title("Average entropy heatmap")
plt.show()

plt.figure()
plt.imshow(viol, aspect='auto', origin='lower',
           extent=[kappa_vals.min(), kappa_vals.max(), w_vals.min(), w_vals.max()])
plt.colorbar(label="Violation rate")
plt.xlabel("kappa"); plt.ylabel("window w")
plt.title("Bell violation rate heatmap")
plt.show()

# ---------- (5) Correlations: corr(Kw, {S, Smax}) ----------
def nanpearson(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 3:
        return np.nan
    xm = x[mask] - np.mean(x[mask]); ym = y[mask] - np.mean(y[mask])
    return float(np.dot(xm, ym) / (np.sqrt(np.dot(xm, xm)) * np.sqrt(np.dot(ym, ym)) + 1e-12))

w_corr = 10
K_corr = sliding_K(tau_n, w=w_corr)
theta_corr = K_to_theta(K_corr, kappa=0.9)
S_corr = ent_entropy_theta(theta_corr); Smax_corr = CHSH_max_theta(theta_corr)

corr_K_S    = nanpearson(K_corr, S_corr)
corr_K_Smax = nanpearson(K_corr, Smax_corr)
print("\nCorrelations (w=%d): corr(K,S)=%.3f, corr(K,S_max)=%.3f" % (w_corr, corr_K_S, corr_K_Smax))
