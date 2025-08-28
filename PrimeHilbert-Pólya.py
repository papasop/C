# 安装依赖（Colab 通常自带 scipy/matplotlib，这里仅确保 mpmath/scipy 存在）
!pip -q install mpmath scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, hilbert

# ---- cumtrapz 兼容（SciPy 新版 cumulative_trapezoid 更名）----
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz

# ---- numpy.trapezoid 兼容（老版无 trapezoid，用 trapz 兜底）----
def ntrapz(y, dx):
    try:
        return np.trapezoid(y, dx=dx)  # 新版
    except AttributeError:
        return np.trapz(y, dx=dx)      # 旧版

# ===================== 参数（素数侧 / Dirichlet） =====================
c0           = 299_792_458.0     # m/s
J            = 1.327652314       # m（z = J*u）
alpha        = 2.0               # σ = alpha * 平均零点间隙
g            = 2.0818254069974373e-3   # 冻结幅度（无拟合）
Nmax_dirich  = 12000             # Dirichlet 级数截断（到 n ≤ Nmax，含素数幂）
u_span_zeros = 100               # 用前多少个零点估计 Δγ（仅用于 σ 的尺度）
nu           = 6000              # u 网格数量
M_seg        = 80                # 测试段覆盖前 M_seg 个零点 + 两端缓冲
nz           = 1200              # FDTD 空间网格
src_offset   = 50                # 源格点（距左端）
probe_offset = 8                 # 探测点离右端
gauss_t0_factor = 1.3            # 脉冲中心时间比例（越大越靠后）

# ===================== 工具：零点（仅为 σ 尺度） =====================
from mpmath import mp, zetazero
mp.dps = 30
gamma = np.array([float(zetazero(n).imag) for n in range(1, u_span_zeros+1)])
delta_gamma = float(np.mean(np.diff(gamma)))
sigma = alpha * delta_gamma

# ===================== 工具：素数幂与 von Mangoldt =====================
def sieve_primes(n):
    bs = np.ones(n+1, dtype=bool)
    bs[:2] = False
    p = 2
    while p*p <= n:
        if bs[p]:
            bs[p*p:n+1:p] = False
        p += 1
    return np.flatnonzero(bs)

primes = sieve_primes(Nmax_dirich)
# 预生成素数幂项：n = p^k <= Nmax，系数 a = (ln p) * n^{-(1/2+σ)}，频率 w = ln n
half_sigma = 0.5 + sigma
terms_a = []
terms_w = []
for p in primes:
    lp = np.log(p)
    n = p
    k = 1
    while n <= Nmax_dirich:
        a = lp * (n ** (-half_sigma))   # Λ(p^k)=ln p
        w = np.log(n)
        terms_a.append(a)
        terms_w.append(w)
        k += 1
        n *= p
terms_a = np.array(terms_a, dtype=np.float64)
terms_w = np.array(terms_w, dtype=np.float64)
print(f"[Dirichlet] prime-power terms: {len(terms_a)} (Nmax={Nmax_dirich})")

# === 原来是：===
# phi_sigma = cumtrapz(phi_prime, u, initial=0.0)
# phi_sigma -= np.mean(phi_sigma)
# N_u = np.exp(g * phi_sigma)

# === 换成：GR-cap 安全标定（不拟合）===
GR_CAP = 0.70     # |N-1| 的上限预算
S_TARGET = 0.30   # 让 phi_sigma 的标准差映射后有稳健对比度（可保持这个值）

phi_sigma = cumtrapz(phi_prime, u, initial=0.0).astype(np.float64)
phi_sigma -= phi_sigma.mean()

def calibrate_g_for_GR_cap(phi_centered, a=1.0, s_target=S_TARGET, cap=GR_CAP):
    xmax = float(np.max(np.abs(phi_centered)))
    if not np.isfinite(xmax) or xmax <= 0.0:
        return 0.0, np.ones_like(phi_centered)
    b0 = float(s_target / (np.std(phi_centered) + 1e-18))
    b_cap = float(min(np.log1p(cap), -np.log1p(-cap)) / (a * xmax))
    g = a * min(b0, b_cap)
    N = np.exp(g * phi_centered)
    exc = float(np.max(np.abs(N - 1.0)))
    if exc > cap:
        g *= 0.999 * cap / (exc + 1e-18)
        N = np.exp(g * phi_centered)
    return g, N

g_calib, N_u = calibrate_g_for_GR_cap(phi_sigma, a=1.0, s_target=S_TARGET, cap=GR_CAP)
print(f"[Dirichlet-GRcap] g_calib={g_calib:.6g}, max|N-1|={np.max(np.abs(N_u-1)):.3f}")

# ===================== 2) FDTD 网格与参数 =====================
L  = J * (u1 - u0)
z  = np.linspace(0.0, L, nz)
dz = z[1] - z[0]
# N(z)：z→u 线性映射再插值
u_for_z = u0 + z / J
N_z = np.interp(u_for_z, u, N_u)
n_max = float(np.max(N_z))
dt = 0.99 * dz / (c0 * n_max)
steps_travel = int(np.ceil(1.1 * n_max * (nz-1)))
nt = int(3 * steps_travel)
t  = np.arange(nt) * dt
print(f"[FDTD] L={L:.3f} m, dz={dz:.3e} m, n_max={n_max:.3f}, dt={dt:.3e} s")
print(f"[FAST] nt={nt}, total_time≈{nt*dt:.3e} s")

# 高斯源
c_eff  = c0 / max(np.mean(N_z), 1.0)
t0     = gauss_t0_factor * (L / c_eff) * 0.25
spread = t0 / 6.0
def gaussian(tt):
    x = (tt - t0) / (spread + 1e-18)
    return np.exp(-0.5 * x*x)

# ===================== 3) 1D FDTD（Mur-1 ABC） =====================
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

rec_vac = run_fdtd(np.ones_like(N_z))
rec_med = run_fdtd(N_z)

# ===================== 4) 路径匹配的 eikonal 预测 & 稳健测时 =====================
# eikonal（与 FDTD 源→探测点路径严格一致）
z_src, z_prb = src_offset*dz, (nz-1-probe_offset)*dz
u_src, u_prb = u0 + z_src/J, u0 + z_prb/J
j0, j1 = np.searchsorted(u, [u_src, u_prb])
Iu_cum = cumtrapz(N_u - 1.0, u, initial=0.0)
Iu_path = float(Iu_cum[j1] - Iu_cum[j0])
dt_pred_path = (J / c0) * Iu_path
print(f"[预测/路径匹配] Iu_path={Iu_path:.6e},  Δt_pred_path={dt_pred_path:.6e} s")

# 估计到达时间（用于分窗）
dist      = z_prb - z_src
T_vac_est = dist / c0
T_med_est = ntrapz(N_z[src_offset:nz-probe_offset], dx=dz) / c0
dT_est    = T_med_est - T_vac_est
k_est     = int(np.round(dT_est / dt))
pred_samp = dt_pred_path / dt

# 为两条记录各取窗（围绕各自到达时刻）
pad = max(4*spread, 40*dt)
i0_vac = max(0, int((T_vac_est - pad)/dt))
i1_vac = min(nt, int((T_vac_est + pad)/dt))
i0_med = max(0, int((T_med_est - pad)/dt))
i1_med = min(nt, int((T_med_est + pad)/dt))
x = rec_vac[i0_vac:i1_vac].astype(np.float64)
y = rec_med[i0_med:i1_med].astype(np.float64)
m = min(len(x), len(y))
x = x[:m] - x[:m].mean()
y = y[:m] - y[:m].mean()
w = np.hanning(m); xw = x*w; yw = y*w

# 相关：只在 k_est ± W 内找峰；必要时包络 fallback
xc   = correlate(xw, yw, mode='full')     # lag>0: y 比 x 晚
lags = np.arange(-(m-1), m)
W = max(int(2.5*spread/dt), 12); W = min(W, (m-3)//2)
mask = (lags >= k_est - W) & (lags <= k_est + W)
xc_win, lags_win = xc[mask], lags[mask]

def _parabolic_abs(y_arr, k):
    if k <= 0 or k >= len(y_arr)-1: return 0.0
    y0,y1,y2 = y_arr[k-1], y_arr[k], y_arr[k+1]
    den = (y0 - 2*y1 + y2) + 1e-18
    return 0.5*(y0 - y2)/den

use_env = False
if len(xc_win) >= 3:
    k_rel = int(np.argmax(np.abs(xc_win)))
    delta_rel = _parabolic_abs(np.abs(xc_win), k_rel)
    lag_win_hat = (lags_win[k_rel] + delta_rel)
    if abs(lag_win_hat - k_est) >= (W - 1):
        use_env = True
else:
    use_env = True

if not use_env:
    dt_sim = ((i0_med - i0_vac) + lag_win_hat) * dt
else:
    ex = np.abs(hilbert(x))
    ey = np.abs(hilbert(y))
    ix = int(np.argmax(ex)); iy = int(np.argmax(ey))
    dt_sim = ((i0_med + iy) - (i0_vac + ix)) * dt
    lag_win_hat = (iy - ix)

rel_err = abs(dt_sim - dt_pred_path) / (abs(dt_pred_path) + 1e-18)
print(f"[debug] T_vac_est={T_vac_est:.3e}s, T_med_est={T_med_est:.3e}s, ΔT_est≈{dT_est:.3e}s, "
      f"k_est≈{k_est} samp, 期望≈{pred_samp:.1f} samp, W={W}, 窗长={m} samp, 选中滞后≈{lag_win_hat:.2f} samp")
print(f"[结果] Δt_sim={dt_sim:.6e} s,  Δt_pred_path={dt_pred_path:.6e} s,  相对误差≈{rel_err:.3e}")

# ===================== 5) 可视化 =====================
plt.figure(figsize=(10,4))
plt.plot(t*1e9, rec_vac, label="vacuum")
plt.plot(t*1e9, rec_med, label="medium N(u)")
plt.xlabel("time (ns)"); plt.ylabel("E (arb.)"); plt.title("Time responses at probe (Dirichlet side)")
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(10,4))
plt.plot(u, N_u)
plt.xlabel("u"); plt.ylabel("N(u)"); plt.title("Refractive index N(u) from Dirichlet side")
plt.grid(True); plt.show()

