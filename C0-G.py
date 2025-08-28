# ===================== Section 3 (No Fitting): boundary-safe t*, strong LPF, multiscale cdiff + OLS median =====================
!pip -q install mpmath scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz
from mpmath import mp, zetazero

# --- NumPy trapezoid compat (for old/new versions)
def trapezoid_compat(y, x=None, dx=None, axis=-1):
    try:
        if x is not None: return np.trapezoid(y, x=x, axis=axis)
        return np.trapezoid(y, dx=dx, axis=axis)
    except AttributeError:
        if x is not None: return np.trapz(y, x=x, axis=axis)
        return np.trapz(y, dx=dx, axis=axis)

# ===================== Physical / grid params =====================
c0 = 299_792_458.0
J  = 1.327652314            # z = J * u
Lx = 320.0                  # m
nx = 1200
x  = np.linspace(0.0, Lx, nx); dx = x[1]-x[0]

T_total = 3.3e-6            # s
dt      = 5.5e-10           # s
nt      = int(np.ceil(T_total/dt))
t       = np.arange(nt)*dt

rng = np.random.default_rng(123)

# ===================== True zeros -> phase-lattice φσ(u) =====================
mp.dps = 40
N_ZEROS = 120
M_SEG   = 80
zeros = np.array([float(zetazero(k).imag) for k in range(1, N_ZEROS+1)])
dgam  = float(np.mean(np.diff(zeros)))
alpha = 1.6
sigma = alpha * dgam

u0 = zeros[0]       - 6.0*sigma
u1 = zeros[M_SEG-1] + 6.0*sigma
nu = 8000
u  = np.linspace(u0, u1, nu)

def phi_prime_sigma(u_grid, gammas, sigma):
    U = u_grid[:,None] - gammas[None,:]
    return np.sum(sigma / (U*U + sigma*sigma), axis=1)

phi_p = phi_prime_sigma(u, zeros[:M_SEG], sigma)
phi   = cumtrapz(phi_p, u, initial=0.0)
phi  -= phi.mean()

# δ(x,t) = φσ(u(x) - v_u t)
v_u = 50.0
u_x = u0 + x/J
def phi_at_u(uq):
    uq = np.clip(uq, u0, u1)
    return np.interp(uq, u, phi)

delta_clean = np.empty((nt, nx))
for it, tt in enumerate(t):
    delta_clean[it] = phi_at_u(u_x - v_u*tt)

# ===================== 2D fixed band-limit -> structural field δ̃ =====================
def _odd_cap(n, w, minw=5):
    # ensure 1) <= n, 2) odd, 3) >= minw
    w = max(minw, min(w, n - (1 - n%2)))
    if w % 2 == 0: w -= 1
    return max(minw, w)

def _sg_window(n, frac=0.03, minw=31):
    return _odd_cap(n, max(minw, int(frac*n)), minw=minw)

LPF_WIN_X_REQ = _sg_window(nx, frac=0.03, minw=31)   # ~3% of x
LPF_WIN_T_REQ = 6001                                  # strong time LPF (auto-capped)

def lpf_x(A, win=LPF_WIN_X_REQ, poly=3):
    w = _odd_cap(A.shape[1], win, minw=31)
    return savgol_filter(A, window_length=w, polyorder=poly, axis=1, mode="interp")

def lpf_t(A, win=LPF_WIN_T_REQ, poly=3):
    w = _odd_cap(A.shape[0], win, minw=51)
    return savgol_filter(A, window_length=w, polyorder=poly, axis=0, mode="interp")

def structural_field(A):
    return lpf_t(lpf_x(A))

# ===================== Functionals on δ̃ (K / locking / display) =====================
def robust_dt(y, dt, win=None, poly=3):
    if win is None: win = _sg_window(len(y), frac=0.02, minw=51)
    w = _odd_cap(len(y), win, minw=51)
    y_s = savgol_filter(y, window_length=w, polyorder=poly, mode="interp")
    dy  = savgol_filter(y, window_length=w, polyorder=poly, deriv=1, delta=dt, mode="interp")
    return y_s, dy

def compute_functionals_from_struct(delta_tilde):
    phi_t = trapezoid_compat(delta_tilde, x=x, axis=1)
    H_t   = trapezoid_compat(delta_tilde*delta_tilde, x=x, axis=1)
    win   = _sg_window(len(phi_t), frac=0.02, minw=51)
    phi_t_s, phi_c = robust_dt(phi_t, dt, win=win, poly=3)
    H_t_s          = savgol_filter(H_t, window_length=_odd_cap(len(H_t), win, 51), polyorder=3, mode="interp")
    K = (phi_t_s*phi_t_s)/(H_t_s + 1e-18)
    return phi_t, H_t, phi_t_s, H_t_s, phi_c, K

delta_clean_struct = structural_field(delta_clean)
phi_t_clean_raw, H_t_clean_raw, phi_t_clean, H_t_clean, phi_c_clean, K_clean = compute_functionals_from_struct(delta_clean_struct)

# ===================== OLS derivative for φ_t (coherent 1D) =====================
def ols_derivative_series(y, dt, w_req):
    w = _odd_cap(len(y), w_req, minw=101)
    n = len(y); h = w // 2
    slopes = np.zeros(n, dtype=float)
    tgrid  = np.arange(n) * dt
    for i in range(n):
        i0 = max(0, i - h); i1 = min(n, i + h + 1)
        tt = tgrid[i0:i1]; yy = y[i0:i1]
        t0 = tt.mean()
        denom = np.sum((tt - t0)**2) + 1e-30
        slopes[i] = np.sum((tt - t0) * (yy - yy.mean())) / denom
    return slopes

phi_c_ols_clean = ols_derivative_series(phi_t_clean, dt, w_req=6001)

# ===================== Clean-only t* locking（边界裕度 30%） =====================
def choose_t_star_weighted(K, phi_c_ols, H_t, eps=0.35, sK=0.20, floor_H_frac=0.50, margin_frac=0.30):
    """
    在 |K-2|<=eps 带内最大化 |φ̇_OLS|*exp(-((K-2)^2)/(2 sK^2))；仅在 clean 档执行一次。
    强制 t* ∈ [margin_frac, 1-margin_frac] 区间，避免边界失真。
    """
    n = len(K)
    M = int(n * margin_frac)
    band = np.where(np.abs(K - 2.0) <= eps)[0]
    if band.size:
        thr_H = np.median(H_t) * floor_H_frac
        band  = band[(band >= M) & (band <= n - 1 - M) & (H_t[band] >= thr_H)]
    if band.size == 0:
        band = np.arange(M, n - M)
    weight = np.exp(-((K[band]-2.0)**2)/(2.0*sK*sK))
    score  = np.abs(phi_c_ols[band]) * weight
    return int(band[np.argmax(score)])

k_idx_clean = choose_t_star_weighted(K_clean, phi_c_ols_clean, H_t_clean,
                                     eps=0.35, sK=0.20, floor_H_frac=0.50, margin_frac=0.30)

# ===================== Freeze φ^2 in clean (K≈2 band, sliding-median) =====================
def freeze_phi2_from_band(phi_t, K, w=101, eps=0.35):
    w = _odd_cap(len(phi_t), w, minw=51)
    idxs = np.where(np.abs(K - 2.0) <= eps)[0]
    if idxs.size == 0:
        order = np.argsort(np.abs(K - 2.0))
        idxs  = order[:max(5, len(K)//200)]
    vals = []
    for idx in idxs:
        i0 = max(0, idx - w//2); i1 = min(len(phi_t), idx + w//2 + 1)
        vals.append(np.mean(phi_t[i0:i1]**2))
    return float(np.median(vals)) if vals else float(np.mean(phi_t**2))

phi2_ref = freeze_phi2_from_band(phi_t_clean, K_clean, w=101, eps=0.35)

print(f"[info] zeros={N_ZEROS}, M_SEG={M_SEG}, alpha={alpha:.3f}, sigma={sigma:.6f}")
print(f"[info] u∈[{u0:.4f},{u1:.4f}],  t* (locked) at {t[k_idx_clean]*1e6:.3f} µs")
print(f"[sanity] t* idx = {k_idx_clean}/{len(t)}  ({t[k_idx_clean]/t[-1]:.1%} of record)")

# ===================== Spatial MoM for H, and fixed time averaging =====================
W_H_REQ = 1201   # time-average window around t*（固定；自动封顶为奇数）
def H_series_mom(delta_tilde, n_blocks=8):
    nt, n = delta_tilde.shape
    b = max(1, n // n_blocks)
    blocks=[]
    for i in range(0, n, b):
        j = min(n, i + b)
        if j > i: blocks.append(np.mean(delta_tilde[:,i:j]**2, axis=1))
    return Lx * np.median(np.stack(blocks, axis=1), axis=1)

def H_bar_window(H_series, idx, w_req=W_H_REQ):
    w = _odd_cap(len(H_series), w_req, minw=101)
    i0 = max(0, idx - w//2); i1 = min(len(H_series), idx + w//2 + 1)
    return float(np.mean(H_series[i0:i1]))

# ===================== |φ̇| from φ_t：multiscale central differences（内域裁剪） =====================
def phi_c_bar_from_phi_series_cdiff(phi_t_series, dt, idx, w_mean_req=W_H_REQ, margin_frac=0.30):
    """
    固定 Δ 集与固定时间窗；对称差分在“所有 Δ 都合法”的内域上求中位数。
    全部常量一次设定，不随数据或噪声改变。
    """
    n = len(phi_t_series)
    w = _odd_cap(n, w_mean_req, minw=101)
    i0 = max(0, idx - w//2); i1 = min(n, idx + w//2 + 1)

    # 固定 Δ 集（由记录长度与 margin_frac 决定一个上界，仍与数据无关）
    dmax = int(0.8 * margin_frac * n)  # 留 20% buffer
    deltas = sorted(set([max(100, dmax//3), max(200, (2*dmax)//3), max(300, dmax)]))

    # 关键：裁剪到所有 Δ 都可用的内域
    dM = max(deltas)
    i0 = max(i0, dM)
    i1 = min(i1, n - dM)

    vals=[]
    for k in range(i0, i1):
        for d in deltas:
            vals.append(abs(phi_t_series[k+d] - phi_t_series[k-d]) / (2 * d * dt))
    return float(np.median(vals)) if vals else 0.0

# ===================== OLS φ̇（用于与 cdiff 融合） =====================
def phi_c_bar_from_phi_series_OLS(phi_t_series, dt, idx, w_ols_req=6001, w_mean_req=W_H_REQ):
    pc = ols_derivative_series(phi_t_series, dt, w_req=w_ols_req)
    w  = _odd_cap(len(pc), w_mean_req, minw=101)
    i0 = max(0, idx - w//2); i1 = min(len(pc), idx + w//2 + 1)
    return float(np.mean(np.abs(pc[i0:i1])))

# ===================== c0-link diagnostic Π（固定 βn） =====================
g_fixed = 2.9e-3
beta_n  = g_fixed

def grad_x(a, dx, scheme="central"):
    g = np.empty_like(a)
    if scheme == "central":
        g[1:-1] = (a[2:] - a[:-2])/(2*dx)
        g[0]    = (a[1]  - a[0]) / dx
        g[-1]   = (a[-1] - a[-2])/ dx
    else:  # upwind
        g[0]  = (a[1] - a[0]) / dx
        g[1:] = (a[1:] - a[:-1]) / dx
    return g

def Pi_closure(delta_tilde, t_idx):
    lnN   = beta_n * delta_tilde[t_idx]
    dlnN  = grad_x(lnN, dx, "central")
    g_str = beta_n * grad_x(delta_tilde[t_idx], dx, "upwind")
    num = np.median(dlnN); den = np.median(g_str)
    return float(num / (den + 1e-18))

# ===================== Monte Carlo（噪声参照 δ̃；一次冻结 φ2 与 t*） =====================
NOISE_LEVELS   = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
delta_std_ref  = np.std(delta_clean_struct)

rows = []
for sig in NOISE_LEVELS:
    noise = rng.normal(0.0, sig*delta_std_ref, size=delta_clean.shape)
    delta_struct = structural_field(delta_clean + noise)

    # functionals（显示/语义）
    phi_t_raw, H_t_raw, phi_t_s, H_t_s, phi_c_sg, K = compute_functionals_from_struct(delta_struct)

    # 分母组件（固定窗）
    H_ser = H_series_mom(delta_struct)
    Hbar  = H_bar_window(H_ser, k_idx_clean, w_req=W_H_REQ)

    # 斜率两路：cdiff 与 OLS；固定融合规则：标量中位数
    phi_c_cdiff = phi_c_bar_from_phi_series_cdiff(phi_t_s, dt, idx=k_idx_clean, w_mean_req=W_H_REQ, margin_frac=0.30)
    phi_c_ols   = phi_c_bar_from_phi_series_OLS  (phi_t_s, dt, idx=k_idx_clean, w_ols_req=6001, w_mean_req=W_H_REQ)
    phi_c_bar   = float(np.median([phi_c_cdiff, phi_c_ols]))

    Ghat = float(phi2_ref / (Hbar * (phi_c_bar + 1e-18)))
    Pi_  = Pi_closure(delta_struct, k_idx_clean)

    rows.append({"noise":sig, "K_star":float(K[k_idx_clean]), "Ghat":Ghat, "Pi":Pi_,
                 "Hbar":Hbar, "|phidot|_cdiff":phi_c_cdiff, "|phidot|_OLS":phi_c_ols})

G0 = rows[0]["Ghat"]
H0 = rows[0]["Hbar"]
P0_c = rows[0]["|phidot|_cdiff"]
P0_o = rows[0]["|phidot|_OLS"]

print("\n=== Monte Carlo (no fitting; true zeros; boundary-safe t*; δ̃; MoM(H); φ̇ via cdiff+OLS median) ===")
print("noise\tK*(clean t*)\tGhat_norm\tPi(t*)")
for r in rows:
    print(f"{r['noise']:<6.0e}\t{r['K_star']:.3f}\t\t{r['Ghat']/G0:.6f}\t{r['Pi']:.6f}")

print("\n[debug] component drifts (relative to clean):")
print("noise\tHbar_norm\t|φ̇|_cdiff_norm\t|φ̇|_OLS_norm")
for r in rows:
    Hn = r["Hbar"]/H0
    Pc = r["|phidot|_cdiff"]/P0_c
    Po = r["|phidot|_OLS"]/P0_o
    print(f"{r['noise']:<6.0e}\t{Hn:.3f}\t\t{Pc:.3f}\t\t{Po:.3f}")

# ===================== Clean plots =====================
plt.figure(figsize=(10,4))
plt.plot(t*1e6, K_clean, label="K(t) on δ̃ (clean)")
plt.axhline(2.0, ls="--", lw=1)
plt.axvline(t[k_idx_clean]*1e6, ls=":", lw=1)
plt.xlabel("time (µs)"); plt.ylabel("K")
plt.title("K selector (clean, structural field)")
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(10,4))
plt.plot(t*1e6, phi_t_clean, label="φ_t (δ̃)")
plt.plot(t*1e6, H_t_clean,   label="H_t (δ̃)")
plt.axvline(t[k_idx_clean]*1e6, ls=":", lw=1)
plt.xlabel("time (µs)"); plt.title("Functionals on δ̃ (clean)")
plt.legend(); plt.grid(True); plt.show()
#
