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

from mpmath import mp, zetazero

# ===================== 配置参数（快跑版） =====================
mp.dps = 30
num_zeros    = 100           # 获取的真零点个数（用于构造/可视化）
M_seg        = 80            # 仅用前 M_seg 个零点构造 φσ 与 N(u)
alpha        = 1.6           # σ = alpha * 平均间隙
g            = 2.9078744151369885e-3   # 文献给定幅度
c0           = 299_792_458.0 # m/s
J            = 1.327652314   # m（z = J * u）
nu           = 8000          # u 网格点数（相位/折射率计算）
nz           = 1200          # z 网格点数（FDTD 空间网格）
src_offset   = 50            # 源位置（距左端格点数）
probe_offset = 5             # 探测点离右端的偏移（避开边界）
gauss_t0_factor = 1.2        # 脉冲中心时间的经验比例

# ===================== 小工具函数 =====================
def parabolic_peak_interp(y, k):
    """三点抛物线插值，返回亚样本峰位置 k_hat（可能非整数）"""
    if k <= 0 or k >= len(y)-1:
        return float(k)
    y0, y1, y2 = y[k-1], y[k], y[k+1]
    denom = (y0 - 2*y1 + y2) + 1e-18
    delta = 0.5 * (y0 - y2) / denom
    return float(k + delta)

# ===================== 1) 构造 N(u)（无拟合、固定区间） =====================
# 真零点
gamma = np.array([float(zetazero(n).imag) for n in range(1, num_zeros + 1)])
delta_gamma = float(np.mean(np.diff(gamma)))
sigma = alpha * delta_gamma

# 固定区间：覆盖前 M_seg 个零点，并在两端各加 6σ 缓冲（不看任何“目标积分”）
u0 = gamma[0]       - 6.0*sigma
u1 = gamma[M_seg-1] + 6.0*sigma
u  = np.linspace(u0, u1, nu)

# ϕ'_σ(u) 与 ϕ_σ(u) —— 仅用前 M_seg 个零点
phi_prime = np.zeros_like(u)
for gam in gamma[:M_seg]:
    phi_prime += sigma / ((u - gam)**2 + sigma**2)

phi_sigma = cumtrapz(phi_prime, u, initial=0.0)
mean_phi  = float(np.mean(phi_sigma))

# 折射率 N(u) = exp(g * (ϕ_σ - 平均))（无任何拟合）
N_u = np.exp(g * (phi_sigma - mean_phi))

# ===== （A）全段预测（仅用于展示差异，不用于比对）=====
Iu_total = float(cumtrapz(N_u - 1.0, u, initial=0.0)[-1])
dt_pred_total = (J / c0) * Iu_total
print(f"[预测/全段] u∈[{u0:.4f}, {u1:.4f}],  σ={sigma:.6f},  Iu_total={Iu_total:.6e},  Δt_pred_total={dt_pred_total:.6e} s")

# ===================== 2) FDTD 网格与参数（CFL 稳定） =====================
# u→z 映射：z ∈ [0, L]
L  = J * (u1 - u0)
z  = np.linspace(0.0, L, nz)
dz = z[1] - z[0]

# N(z)：把 z 映射回 u，再插值
u_for_z = u0 + z / J
N_z = np.interp(u_for_z, u, N_u)

# CFL 稳定步长：考虑 n_max
n_max = float(np.max(N_z))
dt = 0.99 * dz / (c0 * n_max)
print(f"[FDTD] L={L:.3f} m, dz={dz:.3e} m, n_max={n_max:.3f}, dt={dt:.3e} s")

# 预估首波传播步数并给足时长（快跑：3 × 路程时间）
steps_travel = int(np.ceil(1.1 * n_max * (nz-1)))
nt = int(3 * steps_travel)
t  = np.arange(nt) * dt
print(f"[FAST] nt={nt}, total_time≈{nt*dt:.3e} s")

# 高斯脉冲源（时间宽度/位置）
c_eff  = c0 / max(np.mean(N_z), 1.0)
t0     = gauss_t0_factor * (L / c_eff) * 0.25
spread = t0 / 6.0
def gaussian(tt):
    x = (tt - t0) / (spread + 1e-18)
    return np.exp(-0.5 * x*x)

# ===================== 3) FDTD 主循环（Mur 一阶 ABC + 进度条） =====================
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

    report_every = max(1, nt // 10)
    for it in range(nt):
        # H 更新：长度匹配 (nz-1)
        Hy[:-1] += (dt / (mu[:-1] * dz)) * (Ez[1:] - Ez[:-1])

        # 备份旧 Ez，用于 Mur
        Ez_old[:] = Ez

        # E 更新：对齐 (Hy[1:] - Hy[:-1]) 的长度 (nz-1)
        Ez[1:] += (dt / (eps[1:] * dz)) * (Hy[1:] - Hy[:-1])

        # 源（软源）
        Ez[src_offset] += gaussian(t[it])

        # 边界（Mur 一阶）—— 覆盖 Ez[0] 与 Ez[-1]
        Ez[0]  = mur_update_left(Ez, Ez_old,  c_left,  dt, dz)
        Ez[-1] = mur_update_right(Ez, Ez_old, c_right, dt, dz)

        # 记录探测点
        rec[it] = Ez[probe_idx]

        if show_progress and (it % report_every == 0 or it == nt-1):
            print(f"[FDTD] {it+1}/{nt} ({(it+1)/nt:5.1%})", end="\r")

    if show_progress:
        print()
    return rec

# 真空与介质分别跑
rec_vac = run_fdtd(np.ones_like(N_z))
rec_med = run_fdtd(N_z)

# ===================== 4) 互相关测时并对比（路径匹配的理论预测 + 稳健测时） =====================
# ---- (B) 路径匹配的 eikonal 预测（与 FDTD 源→探测点的路径一致）----
z_src    = src_offset * dz
z_probe  = (nz-1 - probe_offset) * dz
u_src    = u0 + z_src   / J
u_probe  = u0 + z_probe / J
j0, j1   = np.searchsorted(u, [u_src, u_probe])
Iu_cum   = cumtrapz(N_u - 1.0, u, initial=0.0)
Iu_path  = float(Iu_cum[j1] - Iu_cum[j0])
dt_pred_path = (J / c0) * Iu_path
print(f"[预测/路径匹配] Iu_path={Iu_path:.6e},  Δt_pred_path={dt_pred_path:.6e} s")

# ---- 估计到达时间（仅用于分窗）----
dist      = z_probe - z_src
T_vac_est = dist / c0
T_med_est = ntrapz(N_z[src_offset:nz-probe_offset], dx=dz) / c0
dT_est    = T_med_est - T_vac_est
k_est     = int(np.round(dT_est / dt))  # 预计样本滞后
pred_samp = dt_pred_path / dt           # 期望样本滞后（理论）

# ---- 为两条记录各自取窗（围绕各自到达时刻）----
pad = max(4*spread, 40*dt)   # 适度裕量
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

# ---- 互相关：只在预计滞后 k_est ± W 的“小窗口”中找峰；必要时用包络 fallback ----
xc   = correlate(xw, yw, mode='full')     # lag>0 表示 y 比 x 晚（介质更慢）
lags = np.arange(-(m-1), m)

W = max(int(2.5*spread/dt), 12)                    # 搜索半宽（样本）
W = min(W, (m-3)//2)                               # 不能超过窗长
mask = (lags >= k_est - W) & (lags <= k_est + W)
xc_win   = xc[mask]
lags_win = lags[mask]

def _parabolic_abs(y_arr, k):
    if k <= 0 or k >= len(y_arr)-1: return 0.0
    y0, y1, y2 = y_arr[k-1], y_arr[k], y_arr[k+1]
    den = (y0 - 2*y1 + y2) + 1e-18
    return 0.5*(y0 - y2)/den

use_envelope = False
if len(xc_win) >= 3:
    k_rel = int(np.argmax(np.abs(xc_win)))          # 用 |xcorr| 选峰（抗反相）
    delta_rel = _parabolic_abs(np.abs(xc_win), k_rel)
    lag_win_hat = (lags_win[k_rel] + delta_rel)     # 带符号的亚样本滞后
    # 若峰仍然贴边，改用包络 fallback
    if abs(lag_win_hat - k_est) >= (W - 1):
        use_envelope = True
else:
    use_envelope = True

if not use_envelope:
    delta_t_sim = ((i0_med - i0_vac) + lag_win_hat) * dt
else:
    # 包络 fallback：分别对两条记录取包络峰对齐
    ex = np.abs(hilbert(x))
    ey = np.abs(hilbert(y))
    ix = int(np.argmax(ex)); iy = int(np.argmax(ey))
    delta_t_sim = ((i0_med + iy) - (i0_vac + ix)) * dt
    lag_win_hat = (iy - ix)

rel_err = abs(delta_t_sim - dt_pred_path) / (abs(dt_pred_path) + 1e-18)
print(f"[debug] T_vac_est={T_vac_est:.3e}s, T_med_est={T_med_est:.3e}s, "
      f"ΔT_est≈{dT_est:.3e}s, k_est≈{k_est} samp, 期望≈{pred_samp:.1f} samp, "
      f"W={W}, 窗长={m} samp, 选中滞后≈{lag_win_hat:.2f} samp")
print(f"[结果] Δt_sim={delta_t_sim:.6e} s,  Δt_pred_path={dt_pred_path:.6e} s,  相对误差≈{rel_err:.3e}")
print(f"[提示] 全段预测 Δt_pred_total={dt_pred_total:.6e} s（与路径预测不同，不用于比对）")

# ===================== 5) 可视化 =====================
plt.figure(figsize=(10,4))
plt.plot(t*1e9, rec_vac, label="vacuum")
plt.plot(t*1e9, rec_med, label="medium N(u)")
plt.xlabel("time (ns)"); plt.ylabel("E (arb.)"); plt.title("Time responses at probe")
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(10,4))
plt.plot(u, N_u)
plt.xlabel("u"); plt.ylabel("N(u)"); plt.title("Refractive index N(u)")
plt.grid(True); plt.show()

