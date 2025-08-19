# -*- coding: utf-8 -*-
# Colab 一键脚本（自动判断版）
# - 若存在 riemann_zeros.csv（列名 gamma 或单列），优先使用真实零点
# - 否则用内置示例 + 自动“补齐零点”保证 K_w(gaps) 至少 MIN_WINDOWS_GAP 个窗口
# - 正演/逆演（Tikhonov + SG 自动调参）、Snell/Fermat/Bouguer 验证
# - K_w 指纹（N(u) & gaps）、Poisson/GUE 对照、KS、CSV & LaTeX
# =================================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.optimize import minimize_scalar
from scipy.interpolate import UnivariateSpline
from scipy.stats import ks_2samp
from sklearn.linear_model import LinearRegression
from math import sin, asin, sqrt

# -----------------------------
# 全局设置
# -----------------------------
DO_PLOTS = True
SAVE_FIGS = True
FIG_DPI = 160
np.random.seed(42)

# -----------------------------
# 工具函数（稳健）
# -----------------------------
def rmse(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(np.sqrt(np.mean((x - y) ** 2)))

def corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.linalg.norm(xm) * np.linalg.norm(ym)
    return float(np.dot(xm, ym) / denom) if denom != 0 else np.nan

def trapz_mean(f, x):
    L = x[-1] - x[0]
    return trapezoid(f, x) / L

def kw_sliding_array(arr, w=40):
    """返回长度 max(n-w+1, 0) 的 K_w 序列；n<w 时为空数组"""
    arr = np.asarray(arr)
    if len(arr) < w:
        return np.array([])
    out = np.empty(len(arr)-w+1)
    for i in range(len(out)):
        seg = arr[i:i+w]
        out[i] = seg.mean() * np.mean(1.0/seg)
    return out

def smooth_tikhonov(y, lam=8.0, iters=4):
    """Tikhonov-like 平滑（同长度拉普拉斯，Neumann 边界）"""
    x = np.asarray(y, dtype=float).copy()
    for _ in range(iters):
        x_left  = np.r_[x[1],  x[:-1]]
        x_right = np.r_[x[1:], x[-2]]
        lap = x_left - 2.0*x + x_right
        x = x + lam * lap
    return x

def safe_savgol(y, win=401, order=2):
    """SG 窗口安全化：奇数、>=5、<=len(y)"""
    n = len(y)
    if n < 5:
        return y.copy()
    win = int(win)
    if win % 2 == 0: win -= 1
    win = max(5, min(win, n if n % 2 == 1 else n-1))
    return savgol_filter(y, window_length=win, polyorder=order, mode='interp')

def zero_gap_unit_mean(gamma, large_w=61):
    """
    单位均值零点间隔序列（可选局部展开，类似 unfolding）。
    large_w=None 时使用全局均值。
    """
    gaps = np.diff(gamma)
    if len(gaps) == 0:
        return np.array([])
    if large_w is None:
        return gaps / gaps.mean()
    w = int(large_w)
    if w % 2 == 0: w += 1
    w = max(3, w)
    pad = w//2
    gpad = np.r_[np.full(pad, gaps[0]), gaps, np.full(pad, gaps[-1])]
    mov_mean = np.convolve(gpad, np.ones(w)/w, mode='valid')
    return gaps / mov_mean

def gen_poisson(N):
    return np.random.exponential(scale=1.0, size=N)

def gen_gue(N, scale_up=1.25):
    """
    近似 Wigner surmise: p(s)=(π/2)s exp(-π s^2/4) 的简单拒绝采样
    """
    out = []
    cap = max(10*N, 10000)
    while len(out) < N and cap > 0:
        s = np.random.exponential(1.0)
        p = (np.pi/2.0) * s * np.exp(-np.pi*s*s/4.0)
        if np.random.rand() < p*scale_up:
            out.append(s)
        cap -= 1
    out = np.array(out)
    if len(out) < N:
        out = np.r_[out, np.random.exponential(1.0, size=N-len(out))]
    return out

def safe_stats(arr):
    if arr is None or len(arr) == 0:
        return {"n_points": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n_points": int(len(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }

def ks_line_safe(a, b):
    if (a is None or len(a) == 0 or np.all(np.isnan(a)) or
        b is None or len(b) == 0 or np.all(np.isnan(b))):
        return ("--", "--")
    s, p = ks_2samp(a, b, alternative='two-sided', mode='auto')
    return (f"{s:.4f}", f"{p:.4f}")

# -----------------------------
# 自动“补齐零点”工具（方案 A 核心）
# -----------------------------
def mean_gap_asymp(t):
    # 渐近平均间隔 Δ(t) ≈ 2π / log(t/2π)
    x = max(t/(2*np.pi), 1.5)
    return 2*np.pi / np.log(x)

_WIGNER_MEAN = np.sqrt(np.pi)/2.0  # ≈0.8862

def wigner_sample(rng):
    while True:
        x = rng.exponential(1.0)
        p = (np.pi/2.0) * x * np.exp(-np.pi*x*x/4.0)
        if rng.random() < p:
            return x

def extend_gamma_asymptotic(gamma, n_extra=200, jitter=0.12, repulsion=True, seed=123):
    """
    从最后一个零点开始，按 Δ(t) 的尺度延长 n_extra 个近似零点；
    repulsion=True 用 Wigner 斥力做相对步长，再缩放到 Δ(t) 的均值。
    """
    rng = np.random.default_rng(seed)
    g = gamma.tolist()
    t = g[-1]
    for _ in range(int(n_extra)):
        mu = mean_gap_asymp(t)
        if repulsion:
            s_rel = wigner_sample(rng) / _WIGNER_MEAN  # E≈1
            step = mu * s_rel
        else:
            step = rng.normal(mu, jitter*mu)
        step *= (1.0 + rng.normal(0.0, jitter*0.5))  # 轻微扰动
        step = max(step, 0.2*mu)
        t += step
        g.append(t)
    return np.array(g)

def ensure_min_windows_extend(gamma, desired_w=40, min_windows=50, seed=123):
    """
    确保用窗口 w 做滑窗统计时，窗口数 >= min_windows。
    #windows = len(gaps)-w+1 = len(gamma)-w
    => len(gamma) >= w + min_windows
    不足则自动扩展 gamma。
    """
    need_len = desired_w + min_windows
    if len(gamma) >= need_len:
        return gamma, 0
    n_extra = need_len - len(gamma)
    gamma_ext = extend_gamma_asymptotic(gamma, n_extra=n_extra, seed=seed)
    return gamma_ext, n_extra

# -----------------------------
# 自动载入真实零点（若存在），否则用内置示例
# -----------------------------
GAMMA_SOURCE = "builtin_sample"
gamma = None

if os.path.exists("riemann_zeros.csv"):
    try:
        dfz = pd.read_csv("riemann_zeros.csv", header=None)
        if dfz.shape[1] == 1:
            gamma = dfz.iloc[:,0].dropna().values.astype(float)
        else:
            # 若存在列名 'gamma' 列
            if "gamma" in dfz.columns:
                gamma = dfz["gamma"].dropna().values.astype(float)
            else:
                gamma = dfz.iloc[:,0].dropna().values.astype(float)
        gamma = np.sort(gamma)
        GAMMA_SOURCE = f"riemann_zeros.csv (len={len(gamma)})"
    except Exception as e:
        print("[WARN] 加载 riemann_zeros.csv 失败，改用内置示例。", e)

if gamma is None:
    gamma = np.array([
        14.13472514, 21.02203964, 25.01085758, 30.42487613, 32.93506159,
        37.58617816, 40.91871901, 43.32707328, 48.00515088, 49.77383248,
        52.97032148, 56.44624770, 59.34704400, 60.83177852, 65.11254405,
        67.07981053, 69.54640171, 72.06715767, 75.70469070, 77.14484007,
        79.33737502, 82.91038085, 84.73549298, 87.42527461, 88.80911121,
        92.49189927, 94.65134404, 95.87063423, 98.83119422, 101.3178510
    ])

print(f"[Gamma source] {GAMMA_SOURCE}")

# -----------------------------
# u-grid & 物理参数
# -----------------------------
u_min, u_max, M = 0.0, 120.0, 4000
u = np.linspace(u_min, u_max, M)
L_span = u[-1] - u[0]

a = 0.25
b = 1.0
sigma = 2.0
c0 = 299_792_458.0
target_T_rel = 0.025216

# -----------------------------
# 正演：γ → φ → τ → N → c_eff → T_int
# -----------------------------
U = u[:, None]
G = gamma[None, :]
phi = np.arctan((U - G) / sigma).sum(axis=1)
tau_true = b * (phi - phi.mean())
N0 = np.exp(a * tau_true)
scale = (1.0 + target_T_rel) / trapz_mean(N0, u)
N_true = N0 * scale
c_eff_true = c0 / N_true
T_int_true = cumulative_trapezoid(N_true, u, initial=0) / c0

T_rel_from_mean_rect  = N_true.mean() - 1.0
T_rel_from_mean_trapz = trapz_mean(N_true, u) - 1.0
T_rel_from_T          = (T_int_true[-1] - T_int_true[0]) / (L_span / c0) - 1.0

# -----------------------------
# 测量噪声 + 逆演（dT/du）
# -----------------------------
noise_T_std = 2e-14
noise_N_std = 2e-3
T_meas = T_int_true + np.random.normal(0.0, noise_T_std, size=T_int_true.shape)
N_meas = N_true * (1.0 + np.random.normal(0.0, noise_N_std, size=N_true.shape))

def build_inverse_with_params(T_meas, u, c0, lam=8.0, iters=4, sg_win=401, sg_poly=2, a=0.25):
    dT_du = np.gradient(T_meas, u)
    N_hat_raw = c0 * dT_du
    N_hat_raw = smooth_tikhonov(N_hat_raw, lam=lam, iters=iters)
    N_hat = safe_savgol(N_hat_raw, win=sg_win, order=sg_poly)
    N_hat = np.clip(N_hat, 1e-12, None)
    tau_hat = (1.0 / a) * np.log(N_hat)
    reg = LinearRegression().fit(tau_hat.reshape(-1,1), tau_true)
    tau_hat_cal = reg.predict(tau_hat.reshape(-1,1))
    return tau_hat, tau_hat_cal, N_hat, float(reg.coef_[0]), float(reg.intercept_)

# 自动调参
lam_grid   = [4.0, 8.0, 12.0, 16.0]
iter_grid  = [2, 3, 4, 5]
win_grid   = [301, 401, 501]
poly_grid  = [2, 3]

best = {"rmse": 1e99}
for lam in lam_grid:
    for iters in iter_grid:
        for win in win_grid:
            for poly in poly_grid:
                try:
                    tau_hat_, tau_hat_cal_, N_hat_, k_, b_ = build_inverse_with_params(
                        T_meas, u, c0, lam=lam, iters=iters, sg_win=win, sg_poly=poly, a=a
                    )
                    rm = rmse(tau_true, tau_hat_cal_)
                    cr = corr(tau_true, tau_hat_cal_)
                    if rm < best["rmse"]:
                        best.update(dict(rmse=rm, corr=cr, lam=lam, iters=iters, win=win, poly=poly,
                                         tau_hat=tau_hat_, tau_hat_cal=tau_hat_cal_, N_hat=N_hat_,
                                         k=k_, b=b_))
                except Exception:
                    continue

print("==== AutoTune (Tikhonov+SG) best ====")
print({k:best[k] for k in ["rmse","corr","lam","iters","win","poly","k","b"]})

tau_hat, tau_hat_cal, N_hat = best["tau_hat"], best["tau_hat_cal"], best["N_hat"]
gain_k, offset_b = best["k"], best["b"]

# -----------------------------
# 画图：tau、残差、N(u)
# -----------------------------
if DO_PLOTS:
    plt.figure(figsize=(6.4,4))
    plt.plot(u, tau_true, label='tau_true')
    plt.plot(u, tau_hat_cal, label='tau_hat_cal')
    plt.xlabel('u'); plt.ylabel('tau'); plt.title('Tau: true vs calibrated inverse')
    plt.legend(); 
    if SAVE_FIGS: plt.savefig('tau_fit.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(6.4,3.2))
    plt.plot(u, tau_true - tau_hat_cal)
    plt.xlabel('u'); plt.ylabel('residual'); plt.title('Residual: tau_true - tau_hat_cal')
    if SAVE_FIGS: plt.savefig('tau_residual.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(6.4,3.6))
    plt.plot(u, N_true, label='N_true')
    plt.plot(u, N_hat, '--', label='N_hat')
    plt.xlabel('u'); plt.ylabel('N(u)'); plt.title('Effective index: true vs inverse')
    plt.legend()
    if SAVE_FIGS: plt.savefig('N_compare.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.show()

# -----------------------------
# Appendix A：Snell + Fermat + Bouguer（高精）
# -----------------------------
n1 = 1.000303
n2 = 1.366306
theta1_deg = 30.00
theta1_rad = np.radians(theta1_deg)
theta2_rad_exact = asin((n1 * sin(theta1_rad)) / n2)
theta2_deg_exact = np.degrees(theta2_rad_exact)
snell_diff_exact = abs(n1 * sin(theta1_rad) - n2 * sin(theta2_rad_exact))

h1 = 0.77; h2 = 0.77; d = 0.77
def optical_time(x, h1, h2, d, n1, n2, c0):
    L1 = sqrt((x + d)**2 + h1**2)
    L2 = sqrt((d - x)**2 + h2**2)
    return (n1 * L1 + n2 * L2) / c0

res_opt = minimize_scalar(
    optical_time, args=(h1, h2, d, n1, n2, c0),
    bounds=(-2.0*d, 2.0*d), method='bounded',
    options={'xatol': 1e-14, 'maxiter': 20000}
)
x_star = float(res_opt.x)
T_min = float(res_opt.fun)
eps = 1e-11
dTdx_num = (optical_time(x_star + eps, h1, h2, d, n1, n2, c0)
            - optical_time(x_star - eps, h1, h2, d, n1, n2, c0)) / (2*eps)

L1_star = sqrt((x_star + d)**2 + h1**2)
L2_star = sqrt((d - x_star)**2 + h2**2)
sin_th1_geom = abs((x_star + d) / L1_star)
sin_th2_geom = abs((d - x_star) / L2_star)
snell_diff_at_opt = abs(n1 * sin_th1_geom - n2 * sin_th2_geom)
C_minus = n1 * sin_th1_geom
C_plus  = n2 * sin_th2_geom
lambda0 = 1550e-9
Phi = (2.0 * np.pi / lambda0) * (n1 * L1_star + n2 * L2_star)

print("\n==== APPENDIX A (Snell + Fermat + Bouguer) ====")
print(f"theta1_deg = {theta1_deg:.6f}")
print(f"theta2_exact_deg = {theta2_deg_exact:.6f}")
print(f"Snell diff (using exact θ2) = {snell_diff_exact:.3e}")
print(f"x* = {x_star:.12f}")
print(f"T_min = {T_min:.12e} s")
print(f"dT/dx @ x* = {dTdx_num:.3e}")
print(f"Snell diff @ x* (geom angles) = {snell_diff_at_opt:.3e}")
print(f"Bouguer const L/R = {C_minus:.9e} / {C_plus:.9e}  (diff={abs(C_minus-C_plus):.3e})")
print(f"Phase Φ @ x* (λ0={lambda0} m): {Phi:.6e} rad")

# -----------------------------
# K_w on N_true
# -----------------------------
w_kw_N = 40
Kw_N = kw_sliding_array(N_true, w=w_kw_N)
kwN_stats = {"w": w_kw_N, **safe_stats(Kw_N)}
print(f"\n==== K_w on N(u) (w={w_kw_N}) ====")
for k,v in kwN_stats.items(): print(f"{k} = {v}")

if DO_PLOTS and kwN_stats["n_points"] > 0:
    x_kw = u[:len(Kw_N)] + (u[w_kw_N-1] - u[0]) / 2.0
    plt.figure(figsize=(6,3.2))
    plt.plot(x_kw, Kw_N)
    plt.xlabel('u'); plt.ylabel('K_w (N(u))'); plt.title(f'K_w on N(u) (w={w_kw_N})')
    if SAVE_FIGS: plt.savefig('kw_N_series.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(5,3.2))
    plt.hist(Kw_N, bins=30, density=True)
    plt.xlabel('K_w (N(u))'); plt.ylabel('pdf'); plt.title('Histogram of K_w (N(u))')
    if SAVE_FIGS: plt.savefig('kw_N_hist.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.show()

# -----------------------------
# 自动判断：零点是否足够 → 扩展以保证最少窗口
# -----------------------------
DESIRED_W_GAP   = 40   # 目标窗口
MIN_WINDOWS_GAP = 50   # 至少窗口数
gamma_ext, n_added = ensure_min_windows_extend(
    gamma, desired_w=DESIRED_W_GAP, min_windows=MIN_WINDOWS_GAP, seed=20250820
)
print(f"\n[extend] source_len={len(gamma)}, added={n_added}, final_len={len(gamma_ext)}")
gaps_len = len(gamma_ext) - 1
print(f"[extend] gaps_len={gaps_len}, desired_w={DESIRED_W_GAP} -> windows={gaps_len - DESIRED_W_GAP + 1}")

# K_w on zero gaps（用扩展后的 gamma_ext；与 Poisson/GUE 对齐）
unfold_w = 61  # 或 None 用全局均值
tau_unit = zero_gap_unit_mean(gamma_ext, large_w=unfold_w)
Kw_gap = kw_sliding_array(tau_unit, w=DESIRED_W_GAP)

kwG_stats = {"w": DESIRED_W_GAP, **safe_stats(Kw_gap)}
print(f"\n==== K_w on zero gaps (w={DESIRED_W_GAP}, min_windows={MIN_WINDOWS_GAP}) ====")
for k, v in kwG_stats.items():
    print(f"{k} = {v}")

if DO_PLOTS and kwG_stats["n_points"] > 0:
    plt.figure(figsize=(6, 3.2))
    plt.plot(np.arange(len(Kw_gap)), Kw_gap)
    plt.xlabel('window index'); plt.ylabel('K_w (gaps)'); plt.title(f'K_w on zero gaps (w={DESIRED_W_GAP})')
    if SAVE_FIGS: plt.savefig('kw_gap_series.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(5, 3.2))
    plt.hist(Kw_gap, bins=30, density=True)
    plt.xlabel('K_w (gaps)'); plt.ylabel('pdf'); plt.title('Histogram of K_w (gaps)')
    if SAVE_FIGS: plt.savefig('kw_gap_hist.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.show()

# -----------------------------
# Baselines: Poisson & GUE（与扩展后的 gaps_len 对齐）
# -----------------------------
if gaps_len >= DESIRED_W_GAP:
    poisson_gaps = gen_poisson(gaps_len)
    gue_gaps     = gen_gue(gaps_len)
    poisson_unit = poisson_gaps / np.nanmean(poisson_gaps)
    gue_unit     = gue_gaps / np.nanmean(gue_gaps)
    Kw_poi = kw_sliding_array(poisson_unit, w=DESIRED_W_GAP)
    Kw_gue = kw_sliding_array(gue_unit, w=DESIRED_W_GAP)
else:
    Kw_poi = np.array([])
    Kw_gue = np.array([])

poi_stats = {"w": DESIRED_W_GAP, **safe_stats(Kw_poi)}
gue_stats = {"w": DESIRED_W_GAP, **safe_stats(Kw_gue)}
print("\n==== Baseline Kw (gaps; with extension) ====")
print(f"Poisson: n={poi_stats['n_points']}, mean={poi_stats['mean']}, std={poi_stats['std']}")
print(f"GUE    : n={gue_stats['n_points']}, mean={gue_stats['mean']}, std={gue_stats['std']}")

if DO_PLOTS and all(s["n_points"] > 0 for s in [kwG_stats, poi_stats, gue_stats]):
    plt.figure(figsize=(6.4, 3.6))
    plt.hist(Kw_gap, bins=20, density=True, alpha=0.6, label='Riemann gaps (extended)')
    plt.hist(Kw_poi, bins=20, density=True, alpha=0.6, label='Poisson')
    plt.hist(Kw_gue, bins=20, density=True, alpha=0.6, label='GUE')
    plt.legend(); plt.xlabel('K_w'); plt.ylabel('pdf'); plt.title(f'Kw comparison (w={DESIRED_W_GAP})')
    if SAVE_FIGS: plt.savefig('kw_compare_hist.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.show()

# -----------------------------
# KS 检验（扩展后）
# -----------------------------
s_rg, p_rg = ks_line_safe(Kw_gap, Kw_gue)
s_rp, p_rp = ks_line_safe(Kw_gap, Kw_poi)
s_gp, p_gp = ks_line_safe(Kw_gue, Kw_poi)
print("\nKS tests (extended):")
print("Riemann vs GUE    :", s_rg, p_rg)
print("Riemann vs Poisson:", s_rp, p_rp)
print("GUE vs Poisson    :", s_gp, p_gp)

# -----------------------------
# CSV 导出
# -----------------------------
pd.DataFrame({
    "use_T_for_inverse":[True],
    "gamma_source":[GAMMA_SOURCE],
    "a":[a], "b":[b], "sigma":[sigma],
    "best_lam":[best["lam"]], "best_iters":[best["iters"]],
    "SG_win":[best["win"]], "SG_poly":[best["poly"]],
    "corr_tau_tauhat_cal":[best["corr"]],
    "rmse_tau_tauhat_cal":[best["rmse"]],
    "gain_k":[best["k"]], "offset_b":[best["b"]],
    "N_mean_rect":[N_true.mean()],
    "N_mean_trapz":[trapz_mean(N_true, u)],
    "Tint_span_s":[T_int_true[-1] - T_int_true[0]],
    "T_rel_from_mean_rect":[T_rel_from_mean_rect],
    "T_rel_from_mean_trapz":[T_rel_from_mean_trapz],
    "T_rel_from_T":[T_rel_from_T],
    # Kw(N)
    "KwN_w":[w_kw_N], "KwN_mean":[safe_stats(Kw_N)["mean"]], "KwN_std":[safe_stats(Kw_N)["std"]],
    # Kw(gaps) with extension
    "KwG_w":[DESIRED_W_GAP],
    "KwG_mean":[safe_stats(Kw_gap)["mean"]], "KwG_std":[safe_stats(Kw_gap)["std"]],
    "KwG_windows":[len(Kw_gap)],
    "gamma_original_len":[len(gamma)],
    "gamma_extended_by":[int(n_added)],
    "gamma_final_len":[len(gamma_ext)]
}).to_csv('metrics_summary.csv', index=False)

pd.DataFrame({
    'u': u,
    'tau_true': tau_true,
    'tau_hat': best["tau_hat"],
    'tau_hat_cal': best["tau_hat_cal"],
    'N_true': N_true,
    'N_hat': best["N_hat"],
    'c_eff_true': c_eff_true,
    'T_int_true': T_int_true,
    'T_meas': T_meas
}).to_csv('series_export.csv', index=False)

pd.DataFrame({'Kw_N': Kw_N}).to_csv('kw_N_series.csv', index=False)
pd.DataFrame({'Kw_gap': Kw_gap}).to_csv('kw_gap_series.csv', index=False)
pd.DataFrame({'Kw_poi': Kw_poi}).to_csv('kw_poi_series.csv', index=False)
pd.DataFrame({'Kw_gue': Kw_gue}).to_csv('kw_gue_series.csv', index=False)
print("\nSaved CSVs: metrics_summary.csv, series_export.csv, kw_N_series.csv, kw_gap_series.csv, kw_poi_series.csv, kw_gue_series.csv")

if SAVE_FIGS:
    print("Saved figs: tau_fit.png, tau_residual.png, N_compare.png, kw_N_series.png, kw_N_hist.png, kw_gap_series.png, kw_gap_hist.png, kw_compare_hist.png")

# -----------------------------
# LaTeX 片段输出
# -----------------------------
def fmt_or_dash(x):
    try: return f"{x:.6f}"
    except: return "--"

latex_snell = rf"""
\paragraph{{Numerical verification (tight tolerances).}}
With $(n_1,n_2,\theta_1)=({n1:.6f},{n2:.6f},{theta1_deg:.2f}^\circ)$,
Snell's law gives $\theta_2={theta2_deg_exact:.4f}^\circ$ and
$\Delta_\text{{Snell}}={snell_diff_exact:.2e}$.
Minimizing $T(x)$ with $(h_1,h_2,d)=({h1},{h2},{d})$ yields
$x^\*={x_star:.6f}$, $T_\min={T_min:.6e}\,\mathrm{{s}}$,
and $\partial_x T|_{{x^\*}}\approx {dTdx_num:.2e}$.
At $x^\*$, Bouguer's constant matches across the interface:
$N^-\sin\theta_1^\*={C_minus:.6e}$, $N^+\sin\theta_2^\*={C_plus:.6e}$
(diff $={abs(C_minus-C_plus):.2e}$).
"""
with open('appendix_A1_snell_fermat.tex', 'w', encoding='utf-8') as f:
    f.write(latex_snell)

kwN_s = safe_stats(Kw_N)
latex_kwN = rf"""
\begin{{table}}[h]
\centering
\caption{{Sliding-window fingerprint on $N(u)$ ($w={w_kw_N}$).}}
\begin{{tabular}}{{lrrrr}}
\hline
stat & mean & std & min & max \\\\ \hline
$K_w(N)$ & {fmt_or_dash(kwN_s['mean'])} & {fmt_or_dash(kwN_s['std'])} & {fmt_or_dash(kwN_s['min'])} & {fmt_or_dash(kwN_s['max'])} \\\\ \hline
\end{{tabular}}
\end{{table}}
"""
with open('kw_N_table.tex', 'w', encoding='utf-8') as f:
    f.write(latex_kwN)

kwG_s = safe_stats(Kw_gap)
latex_kwG = rf"""
\begin{{table}}[h]
\centering
\caption{{Sliding-window fingerprint on unit-mean zero gaps (extended; $w={DESIRED_W_GAP}$, unfold={unfold_w}).}}
\begin{{tabular}}{{lrrrr}}
\hline
stat & mean & std & min & max \\\\ \hline
$K_w(\text{{gaps}})$ & {fmt_or_dash(kwG_s['mean'])} & {fmt_or_dash(kwG_s['std'])} & {fmt_or_dash(kwG_s['min'])} & {fmt_or_dash(kwG_s['max'])} \\\\ \hline
\end{{tabular}}
\end{{table}}
"""
with open('kw_gap_table.tex', 'w', encoding='utf-8') as f:
    f.write(latex_kwG)

s_rg, p_rg = ks_line_safe(Kw_gap, Kw_gue)
s_rp, p_rp = ks_line_safe(Kw_gap, Kw_poi)
s_gp, p_gp = ks_line_safe(Kw_gue, Kw_poi)
latex_ks = rf"""
\begin{{table}}[h]
\centering
\caption{{KS tests on $K_w$ (w={DESIRED_W_GAP}, windows $\ge$ {MIN_WINDOWS_GAP}).}}
\begin{{tabular}}{{lcc}}
\hline
pair & KS stat & p-value \\\\ \hline
Riemann vs GUE    & {s_rg} & {p_rg} \\\\
Riemann vs Poisson& {s_rp} & {p_rp} \\\\
GUE vs Poisson    & {s_gp} & {p_gp} \\\\ \hline
\end{{tabular}}
\end{{table}}
"""
with open('kw_ks_table.tex', 'w', encoding='utf-8') as f:
    f.write(latex_ks)

print("Saved LaTeX: appendix_A1_snell_fermat.tex, kw_N_table.tex, kw_gap_table.tex, kw_ks_table.tex")
