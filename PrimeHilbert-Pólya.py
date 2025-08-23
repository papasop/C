# L4 Hilbert–Pólya (No Dependencies, print-only, 4th-order scheme)
# - numpy + mpmath only
# - Prime/Euler-side φ'σ(u) -> W -> V_S = W^2 + W'
# - 4th-order first-derivative (W') and 4th-order Laplacian (FD4)
# - Dense eigen solve, prints JSON summary only

import json, math, time
import numpy as np
from mpmath import zetazero, mp, digamma

mp.dps = 50  # precision

# ===== Config =====
SEED = 20250823
N_BASE = 120        # zeros for alignment
N_ZERO_REF = 400    # more zeros for zero-side reference (improves n_max convergence)
START_BASE = 1
K_EIG = 30
GRID_POINTS = 900
ALPHA = 2.0
A = 1.0
S_TARGET = 0.30
GR_CAP = 0.70
NMAX_LIST = [2000, 5000, 10000, 20000]
MARGIN_SIGMA = 6.0

rng = np.random.default_rng(SEED)

# ===== Utilities =====
def integrate(y, x):  # numpy>=2
    return np.trapezoid(y, x=x)

def riemann_zeros(n, start=1):
    return np.array([float(zetazero(k).imag) for k in range(start, start+n)])

def phi_prime_zero_side_sym(u, gammas, sigma):
    # symmetric ±γ zero-side reference (reduces bias)
    Up = u[:, None] - gammas[None, :]
    Un = u[:, None] + gammas[None, :]
    return (sigma / (Up*Up + sigma*sigma)).sum(axis=1) + \
           (sigma / (Un*Un + sigma*sigma)).sum(axis=1)

def sieve_primes(nmax):
    sieve = np.ones(nmax+1, dtype=bool); sieve[:2] = False
    for p in range(2, int(nmax**0.5)+1):
        if sieve[p]: sieve[p*p:nmax+1:p] = False
    return np.flatnonzero(sieve)

def von_mangoldt_terms(nmax):
    primes = sieve_primes(nmax)
    ns, L = [], []
    for p in primes:
        logp = math.log(p); pk = p
        while pk <= nmax:
            ns.append(pk); L.append(logp); pk *= p
    n_arr = np.array(ns, dtype=np.int64); L_arr = np.array(L, dtype=float)
    order = np.argsort(n_arr)
    return n_arr[order], L_arr[order]

def corr_gamma_pole(u, S_re):
    out = np.empty_like(u)
    for i, ui in enumerate(u):
        s = mp.mpf(S_re) + 1j*mp.mpf(ui)
        term = (1/(2*s)) + (1/(s-1)) - 0.5*mp.log(mp.pi) + 0.5*digamma(s/2)
        out[i] = float(mp.re(term))
    return out

def build_u_grid(gammas, sigma, margin=6.0, m=900):
    u_min = float(gammas.min()) - margin*sigma
    u_max = float(gammas.max()) + margin*sigma
    u = np.linspace(u_min, u_max, m)
    return u, (u[1]-u[0])

def calibrate_b_gr_cap(phi_c, a=1.0, cap=0.70, s_target=0.30):
    phi_std = float(np.std(phi_c)+1e-18)
    b0 = float(s_target/phi_std)
    lo, hi = 0.0, b0
    for _ in range(50):
        mid = 0.5*(lo+hi)
        Nmid = np.exp(a*mid*phi_c)
        if np.max(np.abs(Nmid-1.0)) > cap: hi = mid
        else: lo = mid
    b = lo
    N = np.exp(a*b*phi_c)
    return b, a*b, N

# ---- 4th-order derivatives & Laplacian (dense) ----
def diff1_c4(u, f):
    """ 4th-order central difference for first derivative; 2nd-order one-sided at edges. """
    h = u[1]-u[0]
    n = len(f)
    d = np.empty(n, dtype=float)
    # one-sided 2nd-order at edges
    d[0]  = (-3*f[0] + 4*f[1] - f[2])/(2*h)
    d[1]  = (-3*f[1] + 4*f[2] - f[3])/(2*h)
    d[-1] = ( 3*f[-1] - 4*f[-2] + f[-3])/(2*h)
    d[-2] = ( 3*f[-2] - 4*f[-3] + f[-4])/(2*h)
    # 4th-order central for interior
    d[2:-2] = (-f[4:] + 8*f[3:-1] - 8*f[1:-3] + f[0:-4])/(12*h)
    return d

def H_FD4_dense(u, V):
    """ 4th-order accurate -d^2/du^2 + V(u) with Dirichlet-like ends """
    h2 = (u[1]-u[0])**2
    n = len(u)
    H = np.zeros((n,n), dtype=float)
    main = (30.0/(12.0*h2)) + V
    off1 = -16.0/(12.0*h2)
    off2 = 1.0/(12.0*h2)
    # fill main
    H[np.arange(n), np.arange(n)] = main
    # ±1
    i = np.arange(n-1)
    H[i, i+1] += off1
    H[i+1, i] += off1
    # ±2
    j = np.arange(n-2)
    H[j, j+2] += off2
    H[j+2, j] += off2
    return H

def eig_smallest_dense(H, k_eig):
    vals, vecs = np.linalg.eigh(H)
    order = np.argsort(vals)
    vals = vals[order][:k_eig]
    vecs = vecs[:, order][:,:k_eig]
    return vals, vecs

def eig_align_dense(H, gammas, k_eig=30):
    lam, _ = eig_smallest_dense(H, k_eig=k_eig)
    lam = np.maximum(lam, 0.0)
    sqrtl = np.sqrt(lam + 1e-18)
    g_use = gammas[:min(len(gammas), len(sqrtl))]
    sqrtl = sqrtl[:len(g_use)]
    k_fit = float(np.dot(g_use, sqrtl) / (np.dot(g_use, g_use)+1e-18))
    rel = np.abs(sqrtl - k_fit*g_use)/(np.abs(g_use)+1e-18)
    return dict(k_fit=k_fit, MRE=float(np.mean(rel)), p95=float(np.percentile(rel,95))), lam

# ===== Main (print-only) =====
t0 = time.time()

# Zeros (compute once, reuse)
Z = riemann_zeros(max(N_BASE, N_ZERO_REF), 1)
gam = Z[:N_BASE]
gam_ref = Z[:N_ZERO_REF]

# σ & grid
gaps = np.diff(gam); mean_gap = float(gaps.mean())
sigma = ALPHA * mean_gap; S_re = 0.5 + sigma
u, h = build_u_grid(gam, sigma, margin=MARGIN_SIGMA, m=GRID_POINTS)

# zero-side φ' reference (±γ)
phi1_zero = phi_prime_zero_side_sym(u, gam_ref, sigma)

# prime-side φ': cache corr, build one big cos matrix and reuse prefixes
corr = corr_gamma_pole(u, S_re)
nmax_max = max(NMAX_LIST)
n_arr_all, L_arr_all = von_mangoldt_terms(nmax_max)
logn_all = np.log(n_arr_all)
weights_all = (L_arr_all / np.power(n_arr_all, S_re)).astype(np.float64)
C_all = np.cos(u[:, None] * logn_all[None, :]).astype(np.float32)

conv_rows = []
phi1_prime_best = None; best_nmax = None
for nmax in NMAX_LIST:
    idx = int(np.searchsorted(n_arr_all, nmax, side='right'))
    S_part = (C_all[:, :idx] @ weights_all[:idx].astype(np.float32)).astype(np.float64)
    phi1_prime = corr - S_part
    # convergence vs zero-side ref
    ref = phi1_zero
    l2 = float(np.sqrt(integrate((phi1_prime-ref)**2, u)))
    l2_rel = l2 / (np.sqrt(integrate(ref**2, u)) + 1e-18)
    linf = float(np.max(np.abs(phi1_prime - ref)))
    conv_rows.append({"nmax": int(nmax), "relL2": l2_rel, "Linf": linf})
    phi1_prime_best = phi1_prime; best_nmax = nmax

# φ, GR-cap -> b,g
phi = np.cumsum(phi1_prime_best)*h; phi -= phi.mean()
b_star, g_star, _ = calibrate_b_gr_cap(phi, a=A, cap=GR_CAP, s_target=S_TARGET)

# W and 4th-order W', potential
W  = 0.5*g_star*phi1_prime_best
Wp = diff1_c4(u, W)               # 4th-order first derivative
V_S = W*W + Wp

# FD4 operator & alignment
H4 = H_FD4_dense(u, V_S)
align4, lam4 = eig_align_dense(H4, gam, k_eig=K_EIG)

runtime = time.time()-t0

summary = {
  "config": {
    "scheme": "FD4 (4th-order)",
    "N_base": N_BASE, "N_zero_ref": N_ZERO_REF, "alpha": ALPHA, "sigma": sigma,
    "grid_points": GRID_POINTS, "k_eig": K_EIG,
    "GR_cap": GR_CAP, "s_target": S_TARGET,
    "nmax_list": NMAX_LIST, "best_nmax": int(best_nmax)
  },
  "convergence": conv_rows,
  "align_fd4": align4,
  "b_g": {"b": float(b_star), "g": float(g_star)},
  "runtime_sec": float(runtime)
}
print(json.dumps(summary, indent=2))
# === FD2 vs FD4 cross-check (append-only, print-only) ===
# 依赖于你前一格里已有的变量: u, V_S, gam, K_EIG, eig_align_dense
import json, numpy as np

def H_FD2_dense(u, V):
    h = u[1]-u[0]
    n = len(u)
    H = np.zeros((n,n), dtype=float)
    main = (2.0/h**2) + V
    off  = (-1.0/h**2)
    H[np.arange(n), np.arange(n)] = main
    H[np.arange(n-1), np.arange(1,n)] = off
    H[np.arange(1,n), np.arange(n-1)] = off
    return H

def eig_smallest_dense(H, k_eig):
    vals, vecs = np.linalg.eigh(H)
    order = np.argsort(vals)
    vals = vals[order][:k_eig]
    vecs = vecs[:, order][:,:k_eig]
    return vals, vecs

# 构造并求解
H2 = H_FD2_dense(u, V_S)
H4 = H_FD4_dense(u, V_S)  # 由上一格提供

lam2, _ = eig_smallest_dense(H2, K_EIG)
lam4, _ = eig_smallest_dense(H4, K_EIG)
s2 = np.sqrt(np.maximum(lam2,0.0)+1e-18)
s4 = np.sqrt(np.maximum(lam4,0.0)+1e-18)

# 与零点对齐指标
def align_metrics(s, gam):
    k = float(np.dot(gam[:len(s)], s) / (np.dot(gam[:len(s)], gam[:len(s)])+1e-18))
    rel = np.abs(s - k*gam[:len(s)])/(np.abs(gam[:len(s)])+1e-18)
    return {"k_fit": k, "MRE": float(np.mean(rel)), "p95": float(np.percentile(rel,95))}

align_fd2 = align_metrics(s2, gam)
align_fd4 = align_metrics(s4, gam)

# 两种离散方案的一致性
diff = s4 - s2
consistency = {
    "max_abs_diff_sqrtlam": float(np.max(np.abs(diff))),
    "l2_rel_diff": float(np.linalg.norm(diff)/ (np.linalg.norm(s4)+1e-18))
}

print(json.dumps({
    "align_fd2": align_fd2,
    "align_fd4": align_fd4,
    "scheme_consistency": consistency
}, indent=2))
# --- zero-side tail patch (append-only, print-only) ---
import numpy as np, json, math

def zero_tail_density(u, T, sigma, L_mult=400.0, step_mult=2.0):
    L = L_mult * sigma
    h = step_mult * sigma
    grid = np.arange(T, T + L + h, h, dtype=float)
    rho = (1.0/(2.0*np.pi)) * np.log(np.maximum(grid, 2.0*np.pi) / (2.0*np.pi))
    U = u[:, None]
    Pp = sigma / ((U - grid[None, :])**2 + sigma*sigma)
    Pn = sigma / ((U + grid[None, :])**2 + sigma*sigma)
    return (Pp + Pn) @ (rho * h)

# 重新计算“核心参照”并加尾
def phi_prime_zero_side_sym(u, gammas, sigma):
    Up = u[:, None] - gammas[None, :]
    Un = u[:, None] + gammas[None, :]
    return (sigma / (Up*Up + sigma*sigma)).sum(axis=1) + \
           (sigma / (Un*Un + sigma*sigma)).sum(axis=1)

phi1_zero_core = phi_prime_zero_side_sym(u, gam_ref, sigma)
T = float(gam_ref[-1] + 10.0 * sigma)  # 参照末端再留 ~10σ 缓冲
tail = zero_tail_density(u, T, sigma, L_mult=400.0, step_mult=2.0)
phi1_zero_tail = phi1_zero_core + tail

def integrate(y, x): return np.trapezoid(y, x=x)

relL2_old = float(np.sqrt(integrate((phi1_prime_best - phi1_zero_core)**2, u)) /
                  (np.sqrt(integrate(phi1_zero_core**2, u)) + 1e-18))
relL2_new = float(np.sqrt(integrate((phi1_prime_best - phi1_zero_tail)**2, u)) /
                  (np.sqrt(integrate(phi1_zero_tail**2, u)) + 1e-18))

print(json.dumps({
    "zero_side_relL2_before_tail": relL2_old,
    "zero_side_relL2_after_tail": relL2_new
}, indent=2))
# --- zero-side tail: density + geometric grid (print-only patch) ---
import numpy as np, json

def zero_tail_density_v2(u, T, sigma, K=800, r=1.003):
    """
    Tail ≈ ∫_{|γ|>T} σ/((u-γ)^2+σ^2) ρ(γ) dγ  with  ρ(γ) ≈ (1/2π)[log(γ/2π)+1]
    γ-grid: geometric grid γ_k = T * r^k  (nonuniform trapezoid weights)
    """
    grid = T * (r ** np.arange(K+1, dtype=float))  # γ ≥ T
    rho = (1.0/(2.0*np.pi)) * (np.log(np.maximum(grid, 2.0*np.pi)/(2.0*np.pi)) + 1.0)

    # Nonuniform trapezoid weights
    w = np.empty_like(grid)
    w[1:-1] = 0.5*(grid[2:] - grid[:-2])
    w[0]    = 0.5*(grid[1]  - grid[0])
    w[-1]   = 0.5*(grid[-1] - grid[-2])

    U = u[:, None]
    Pp = sigma / ((U - grid[None, :])**2 + sigma*sigma)
    Pn = sigma / ((U + grid[None, :])**2 + sigma*sigma)
    return (Pp + Pn) @ (rho * w)

def integrate(y, x): return np.trapezoid(y, x=x)

# baseline (你的核心参照，不含尾)
def phi_prime_zero_side_sym(u, gammas, sigma):
    Up = u[:, None] - gammas[None, :]
    Un = u[:, None] + gammas[None, :]
    return (sigma / (Up*Up + sigma*sigma)).sum(axis=1) + \
           (sigma / (Un*Un + sigma*sigma)).sum(axis=1)

phi1_zero_core = phi_prime_zero_side_sym(u, gam_ref, sigma)

# tail start：从参照末端再留出 ~10σ 缓冲
T = float(gam_ref[-1] + 10.0 * sigma)
tail2 = zero_tail_density_v2(u, T, sigma, K=800, r=1.003)
phi1_zero_tail2 = phi1_zero_core + tail2

relL2_before = float(np.sqrt(integrate((phi1_prime_best - phi1_zero_core)**2, u)) /
                     (np.sqrt(integrate(phi1_zero_core**2, u)) + 1e-18))
relL2_after  = float(np.sqrt(integrate((phi1_prime_best - phi1_zero_tail2)**2, u)) /
                     (np.sqrt(integrate(phi1_zero_tail2**2, u)) + 1e-18))

print(json.dumps({
  "zero_side_relL2_before_tail": relL2_before,
  "zero_side_relL2_after_tail_v2": relL2_after
}, indent=2))
# --- zero-side tail v3: correct density + geometric grid + optional fit η (print-only) ---
import numpy as np, json

def integrate(y, x): return np.trapezoid(y, x=x)

def zero_tail_density_v3(u, T, sigma, K=1000, r=1.0025):
    """
    Tail ≈ ∫_{|γ|>T} σ/((u-γ)^2+σ^2) ρ(γ) dγ,  ρ(γ)=(1/2π)log(γ/2π)
    γ-grid: geometric γ_k = T * r^k, nonuniform trapezoid weights.
    """
    grid = T * (r ** np.arange(K+1, dtype=float))   # γ ≥ T
    # correct density (no +1)
    rho = (1.0/(2.0*np.pi)) * np.log(np.maximum(grid, 2.0*np.pi)/(2.0*np.pi))

    # nonuniform trapezoid weights
    w = np.empty_like(grid)
    w[1:-1] = 0.5*(grid[2:] - grid[:-2])
    w[0]    = 0.5*(grid[1]  - grid[0])
    w[-1]   = 0.5*(grid[-1] - grid[-2])

    U = u[:, None]
    Pp = sigma / ((U - grid[None, :])**2 + sigma*sigma)
    Pn = sigma / ((U + grid[None, :])**2 + sigma*sigma)
    return (Pp + Pn) @ (rho * w)

# baseline zero-side core (±γ, no tail)
def phi_prime_zero_side_sym(u, gammas, sigma):
    Up = u[:, None] - gammas[None, :]
    Un = u[:, None] + gammas[None, :]
    return (sigma / (Up*Up + sigma*sigma)).sum(axis=1) + \
           (sigma / (Un*Un + sigma*sigma)).sum(axis=1)

phi1_zero_core = phi_prime_zero_side_sym(u, gam_ref, sigma)

# build v3 tail
T = float(gam_ref[-1] + 10.0 * sigma)           # keep a buffer from last tabulated zero
tail3 = zero_tail_density_v3(u, T, sigma, K=1000, r=1.0025)

# (A) unscaled tail
phi1_zero_tail3 = phi1_zero_core + tail3
relL2_unscaled = float(
    np.sqrt(integrate((phi1_prime_best - phi1_zero_tail3)**2, u)) /
    (np.sqrt(integrate(phi1_zero_tail3**2, u)) + 1e-18)
)

# (B) fit a single η to best match the difference (closed form least squares)
num = float(np.dot(phi1_prime_best - phi1_zero_core, tail3))
den = float(np.dot(tail3, tail3)) + 1e-18
eta = max(0.0, num / den)   # optional nonnegativity; drop max(...) if you prefer unconstrained
phi1_zero_tail3_eta = phi1_zero_core + eta*tail3
relL2_fitted = float(
    np.sqrt(integrate((phi1_prime_best - phi1_zero_tail3_eta)**2, u)) /
    (np.sqrt(integrate(phi1_zero_tail3_eta**2, u)) + 1e-18)
)

print(json.dumps({
  "zero_side_relL2_core"      : float(np.sqrt(integrate((phi1_prime_best - phi1_zero_core)**2, u)) /
                                      (np.sqrt(integrate(phi1_zero_core**2, u)) + 1e-18)),
  "relL2_tail3_unscaled"      : relL2_unscaled,
  "relL2_tail3_fitted"        : relL2_fitted,
  "eta_fitted"                : eta
}, indent=2))
# --- zero-side tail v4: geometric grid + log density × fitted η (print-only) ---
import numpy as np, json

def integrate(y, x): return np.trapezoid(y, x=x)

# baseline zero-side core (±γ, no tail)
def phi_prime_zero_side_sym(u, gammas, sigma):
    Up = u[:, None] - gammas[None, :]
    Un = u[:, None] + gammas[None, :]
    return (sigma / (Up*Up + sigma*sigma)).sum(axis=1) + \
           (sigma / (Un*Un + sigma*sigma)).sum(axis=1)

def zero_tail_log_geom(u, T, sigma, K=1200, r=1.002):
    """
    Tail ≈ ∫_{|γ|>T} σ/((u-γ)^2+σ^2) * [(1/2π) log(γ/2π)] dγ
    γ-grid: geometric γ_k = T * r^k with nonuniform trapezoid weights.
    """
    grid = T * (r ** np.arange(K+1, dtype=float))
    rho  = (1.0/(2.0*np.pi)) * np.log(np.maximum(grid, 2.0*np.pi)/(2.0*np.pi))

    w = np.empty_like(grid)
    w[1:-1] = 0.5*(grid[2:] - grid[:-2])
    w[0]    = 0.5*(grid[1]  - grid[0])
    w[-1]   = 0.5*(grid[-1] - grid[-2])

    U  = u[:, None]
    Pp = sigma / ((U - grid[None, :])**2 + sigma*sigma)
    Pn = sigma / ((U + grid[None, :])**2 + sigma*sigma)
    return (Pp + Pn) @ (rho * w)

# core (与你主流程一致)
phi1_zero_core = phi_prime_zero_side_sym(u, gam_ref, sigma)

# v4 tail + η 拟合
T = float(gam_ref[-1] + 10.0 * sigma)
tail4 = zero_tail_log_geom(u, T, sigma, K=1200, r=1.002)

num = float(np.dot(phi1_prime_best - phi1_zero_core, tail4))
den = float(np.dot(tail4, tail4)) + 1e-18
eta = max(0.0, num/den)  # 可去掉 max(...) 变成无约束拟合
phi1_zero_tail4_eta = phi1_zero_core + eta*tail4

relL2_core = float(np.sqrt(integrate((phi1_prime_best - phi1_zero_core)**2, u)) /
                   (np.sqrt(integrate(phi1_zero_core**2, u)) + 1e-18))
relL2_v4   = float(np.sqrt(integrate((phi1_prime_best - phi1_zero_tail4_eta)**2, u)) /
                   (np.sqrt(integrate(phi1_zero_tail4_eta**2, u)) + 1e-18))

print(json.dumps({
  "zero_side_relL2_core": relL2_core,
  "zero_side_relL2_tail_v4_fitted": relL2_v4,
  "eta_v4": eta
}, indent=2))
