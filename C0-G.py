# ===================== c0-link 闭包（Π）：真正的双轴闭包版（无拟合/一次冻结） =====================
!pip -q install mpmath scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz
from mpmath import mp, zetazero

# ---- NumPy trapezoid 兼容（老/新版）----
def trapezoid_compat(y, x=None, dx=None, axis=-1):
    try:
        if x is not None: return np.trapezoid(y, x=x, axis=axis)
        return np.trapezoid(y, dx=dx, axis=axis)
    except AttributeError:
        if x is not None: return np.trapz(y, x=x, axis=axis)
        return np.trapz(y, dx=dx, axis=axis)

# ===================== 物理/网格参数（固定常量） =====================
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

# ===================== 真零点 -> 相位格 φσ(u)（无任何调参） =====================
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

# δ(x,t) = φσ(u(x) - v_u t)（固定传播速率 v_u）
v_u = 50.0
u_x = u0 + x/J
def phi_at_u(uq):
    uq = np.clip(uq, u0, u1)
    return np.interp(uq, u, phi)

delta_clean = np.empty((nt, nx))
for it, tt in enumerate(t):
    delta_clean[it] = phi_at_u(u_x - v_u*tt)

# ===================== 2D 固定带限 -> 结构场 δ̃（仅作平滑，非拟合） =====================
def _odd_cap(n, w, minw=5):
    w = max(minw, min(w, n - (1 - n%2)))
    if w % 2 == 0: w -= 1
    return max(minw, w)

def _sg_window(n, frac=0.03, minw=31):
    return _odd_cap(n, max(minw, int(frac*n)), minw=minw)

LPF_WIN_X_REQ = _sg_window(nx, frac=0.03, minw=31)  # ~3% 空间窗
LPF_WIN_T_REQ = 6001                                 # 强时间平滑

def lpf_x(A, win=LPF_WIN_X_REQ, poly=3):
    w = _odd_cap(A.shape[1], win, minw=31)
    return savgol_filter(A, window_length=w, polyorder=poly, axis=1, mode="interp")

def lpf_t(A, win=LPF_WIN_T_REQ, poly=3):
    w = _odd_cap(A.shape[0], win, minw=51)
    return savgol_filter(A, window_length=w, polyorder=poly, axis=0, mode="interp")

def structural_field(A):
    return lpf_t(lpf_x(A))

delta_clean_struct = structural_field(delta_clean)

# ===================== 功能量（φ_t, H_t, φ̇_t, K） =====================
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

phi_t_clean_raw, H_t_clean_raw, phi_t_clean, H_t_clean, phi_c_clean, K_clean = compute_functionals_from_struct(delta_clean_struct)

# ===================== OLS φ̇_t（与 cdiff 融合用） =====================
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

# ===================== clean-only：锁定 t*（边界安全 + K≈2 带内） =====================
def choose_t_star_weighted(K, phi_c_ols, H_t, eps=0.35, sK=0.20, floor_H_frac=0.50, margin_frac=0.30):
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

# ===================== 冻结 φ^2（仅用于 Ghat 对照；Π 不依赖此项） =====================
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

# ===================== 分母稳健化组件（MoM(H) + 固定窗）、φ̇ 多尺度 =====================
W_H_REQ = 1201
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

def phi_c_bar_from_phi_series_cdiff(phi_t_series, dt, idx, w_mean_req=W_H_REQ, margin_frac=0.30):
    n = len(phi_t_series)
    w = _odd_cap(n, w_mean_req, minw=101)
    i0 = max(0, idx - w//2); i1 = min(n, idx + w//2 + 1)
    dmax = int(0.8 * margin_frac * n)
    deltas = sorted(set([max(100, dmax//3), max(200, (2*dmax)//3), max(300, dmax)]))
    dM = max(deltas)
    i0 = max(i0, dM); i1 = min(i1, n - dM)
    vals=[]
    for k in range(i0, i1):
        for d in deltas:
            vals.append(abs(phi_t_series[k+d] - phi_t_series[k-d]) / (2 * d * dt))
    return float(np.median(vals)) if vals else 0.0

def phi_c_bar_from_phi_series_OLS(phi_t_series, dt, idx, w_ols_req=6001, w_mean_req=W_H_REQ):
    pc = ols_derivative_series(phi_t_series, dt, w_req=w_ols_req)
    w  = _odd_cap(len(pc), w_mean_req, minw=101)
    i0 = max(0, idx - w//2); i1 = min(len(pc), idx + w//2 + 1)
    return float(np.mean(np.abs(pc[i0:i1])))

# ===================== 关键：一次性“结构轴尺度”λ_str 冻结（真正双轴闭包） =====================
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

beta_n   = 2.9e-3          # 光学轴桥接常数（固定）
lnN_clean = beta_n * delta_clean_struct[k_idx_clean]
dlnN_med  = np.median(grad_x(lnN_clean, dx, "central"))                 # 光学轴斜率
ddel_med  = np.median(grad_x(delta_clean_struct[k_idx_clean], dx, "upwind"))  # 结构轴斜率
lambda_str = (c0**2/2) * dlnN_med / (ddel_med + 1e-18)                 # —— 冻结！——
print(f"[freeze] beta_n={beta_n:.3e}, lambda_str={lambda_str:.6e} (SI: m/s^2 per unit ∂xδ)")

def Pi_closure(delta_tilde, t_idx):
    lnN   = beta_n * delta_tilde[t_idx]                       # 光学轴
    num   = (c0**2/2) * np.median(grad_x(lnN, dx, "central"))
    den   = lambda_str * np.median(grad_x(delta_tilde[t_idx], dx, "upwind"))  # 结构轴
    return float(num / (den + 1e-18))

# ===================== Monte Carlo：常数冻结后检验 Π（以及可选 Ghat 归一） =====================
NOISE_LEVELS   = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
delta_std_ref  = np.std(delta_clean_struct)

rows = []
for sig in NOISE_LEVELS:
    noise = rng.normal(0.0, sig*delta_std_ref, size=delta_clean.shape)
    delta_struct = structural_field(delta_clean + noise)

    # 功能量（用于显示/对照）
    phi_t_raw, H_t_raw, phi_t_s, H_t_s, phi_c_sg, K = compute_functionals_from_struct(delta_struct)

    # 分母组件（MoM(H) + 固定窗）：只做 Ghat 的可选对照
    H_ser  = H_series_mom(delta_struct)
    Hbar   = H_bar_window(H_ser, k_idx_clean, w_req=W_H_REQ)
    ph_c_c = phi_c_bar_from_phi_series_cdiff(phi_t_s, dt, idx=k_idx_clean, w_mean_req=W_H_REQ, margin_frac=0.30)
    ph_c_o = phi_c_bar_from_phi_series_OLS  (phi_t_s, dt, idx=k_idx_clean, w_ols_req=6001, w_mean_req=W_H_REQ)
    ph_bar = float(np.median([ph_c_c, ph_c_o]))
    Ghat   = float(phi2_ref / (Hbar * (ph_bar + 1e-18)))  # 仅作无量纲对照

    Pi_  = Pi_closure(delta_struct, k_idx_clean)

    rows.append({"noise":sig, "K_star":float(K[k_idx_clean]),
                 "Ghat":Ghat, "Pi":Pi_,
                 "Hbar":Hbar, "|phidot|_cdiff":ph_c_c, "|phidot|_OLS":ph_c_o})

G0 = rows[0]["Ghat"]; H0 = rows[0]["Hbar"]
P0_c = rows[0]["|phidot|_cdiff"]; P0_o = rows[0]["|phidot|_OLS"]

print("\n=== Monte Carlo (no fitting; true zeros; boundary-safe t*; 双轴 Π 闭包) ===")
print("noise\tK*(clean t*)\tPi(t*)\t\tGhat_norm")
for r in rows:
    print(f"{r['noise']:<6.0e}\t{r['K_star']:.3f}\t\t{r['Pi']:.6f}\t{r['Ghat']/G0:.6f}")

print("\n[debug] component drifts (relative to clean):")
print("noise\tHbar_norm\t|φ̇|_cdiff_norm\t|φ̇|_OLS_norm")
for r in rows:
    Hn = r["Hbar"]/H0
    Pc = r["|phidot|_cdiff"]/P0_c
    Po = r["|phidot|_OLS"]/P0_o
    print(f"{r['noise']:<6.0e}\t{Hn:.3f}\t\t{Pc:.3f}\t\t{Po:.3f}")

# ===================== 可视化：clean 的 K / φ_t / H_t（用于审阅） =====================
plt.figure(figsize=(10,4))
plt.plot(t*1e6, K_clean, label="K(t) on δ̃ (clean)")
plt.axhline(2.0, ls="--", lw=1)
plt.axvline(t[k_idx_clean]*1e6, ls=":", lw=1)
plt.xlabel("time (µs)"); plt.ylabel("K"); plt.title("K selector (clean, structural field)")
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(10,4))
plt.plot(t*1e6, phi_t_clean, label="φ_t (δ̃)")
plt.plot(t*1e6, H_t_clean,   label="H_t (δ̃)")
plt.axvline(t[k_idx_clean]*1e6, ls=":", lw=1)
plt.xlabel("time (µs)"); plt.title("Functionals on δ̃ (clean)")
plt.legend(); plt.grid(True); plt.show()
