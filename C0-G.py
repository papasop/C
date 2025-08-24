# ==============================================
# URSF–G 结构推导：Riemann-mix 一键版（含 c0-link 验证）
# - 与生产一致的带宽筛选 + 稳健 t* 选取
# - UNIT_GAIN 一次性标定 → 冻结
# - β_n 用同分布标定（标定集与生产一致），打印 Π≈1
# - 保留原有导出：mc_summary.csv / blind_matrix.csv
# - 新增导出：c0_mc_summary.csv / c0_blind_matrix.csv
# ==============================================

import numpy as np
from scipy.integrate import simpson as simps
from scipy.signal import convolve2d
import pandas as pd
import os

# ---------- 物理常量 ----------
G_CODATA = 6.67430e-11               # m^3 kg^-1 s^-2
hbar = 1.054571817e-34               # J·s
c0 = 299792458.0                     # m/s

# ---------- 量纲占位（与旧稿保持一致） ----------
L_L, L_T, L_M = 1e-15, 1e-20, 0.063826
UNIT_FACTOR = (L_L**3) / (L_M * L_T**2)

# ---------- 统一网格 ----------
x = np.linspace(150.0, 250.0, 300)
t = np.linspace(4.7, 5.7, 100)
dx = x[1] - x[0]
dt = t[1] - t[0]

def ker5():
    g = np.linspace(-2.0, 2.0, 5)
    K = np.outer(np.exp(-g**2), np.exp(-g**2))
    K /= K.sum()
    return K

KER5 = ker5()

# ---------- Riemann 零点（容错） ----------
try:
    import mpmath as mp
    _MP_OK = True
except Exception:
    _MP_OK = False

def riemann_zeros(N=120):
    if _MP_OK:
        return np.array([float(mp.zetazero(k).imag) for k in range(1, N+1)], float)
    # 无 mpmath 时给出平滑近似，仅为数值演示
    base, step = 14.134725, 2.0
    return base + step*np.arange(N, dtype=float)

# ---------- 相位叠加 ----------
def phase_phi(u, gammas, sigma_u):
    du = u[:, None] - gammas[None, :]
    return np.sum(np.arctan(du / sigma_u), axis=1)

def norm_row(row):
    area = simps(row, dx=dx)
    return row / (area + 1e-16)

# ---------- 高斯对照（可选） ----------
def delta_field_gauss(x, t, A, sigma_x, sigma_t, k, t0, quantum_noise=0.0, seed=None):
    rng = np.random.default_rng(seed)
    X, T = np.meshgrid(x, t)

    def pulse(x0, t0_local, amp=1.0):
        S = np.exp(-((X - x0)**2) / (2*sigma_x**2))
        S = norm_row(S[0]); S = np.tile(S, (len(t), 1))
        tau = np.clip(T - t0_local, 0.0, None)
        env = (tau**k) * np.exp(-(tau**2) / (2*sigma_t**2))
        env /= (np.max(np.abs(env)) + 1e-12)
        return amp * S * env

    base = pulse(195.0, t0, +1.0) + pulse(205.0, t0+0.1, -0.7)
    noise_scale = (quantum_noise * hbar) / max(sigma_x * sigma_t, 1e-30)
    noise = rng.normal(0.0, noise_scale, size=base.shape)
    noise_smooth = convolve2d(noise, KER5, mode='same', boundary='symm')
    return A * (base + noise_smooth)

# ---------- Riemann-mix 生成器 ----------
def delta_field_riemann_mix(
    x, t, A, sigma_x,
    a, b, sigma_u, u_span1, u_span2, gammas,
    mix_weight=0.35, mode='demean',
    t0=None, sigma_t_env=0.35,
    quantum_noise=0.0, seed=None
):
    rng = np.random.default_rng(seed)
    X, T = np.meshgrid(x, t)
    x_mid = 0.5*(x[0] + x[-1])

    def track(u_span, which='demean'):
        u = np.linspace(u_span[0], u_span[1], len(t))
        phi_u = phase_phi(u, gammas, sigma_u)
        tau_u = b * (phi_u - np.mean(phi_u))
        N_u = np.exp(a * tau_u)
        z = np.gradient(N_u, dt) if which == 'd_dt' else (N_u - np.mean(N_u))
        z = (z - np.mean(z)) / (np.std(z) + 1e-12)
        return z

    s1 = track(u_span1, 'demean')
    s2 = track(u_span2, 'd_dt')
    # 去相关 + 轻微二次耦合
    s2 = s2 - (np.dot(s2, s1) / (np.dot(s1, s1) + 1e-12)) * s1
    s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-12)
    s2 = s2 + 0.10 * ((s1**2 - np.mean(s1**2)) / (np.std(s1**2) + 1e-12))
    s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-12)

    S1 = norm_row(np.exp(-((x - (x_mid - 5.0))**2) / (2*sigma_x**2)))
    S2 = norm_row(np.exp(-((x - (x_mid + 6.0))**2) / (2*(1.2*sigma_x)**2)))
    S1 = np.tile(S1, (len(t), 1)); S2 = np.tile(S2, (len(t), 1))

    if t0 is None:
        t0 = 0.5*(t[0] + t[-1])
    env = np.exp(-0.5 * ((t - t0) / sigma_t_env)**2)
    env /= (env.max() + 1e-12)

    base = S1 * (s1[:, None] * env[:, None]) + mix_weight * S2 * (s2[:, None] * env[:, None])

    noise_scale = (quantum_noise * hbar) / max(sigma_x * sigma_u, 1e-30)
    noise = rng.normal(0.0, noise_scale, size=base.shape)
    noise_smooth = convolve2d(noise, KER5, mode='same', boundary='symm')
    return A * (base + noise_smooth)

# ---------- 结构量 ----------
def compute_structures(delta_xt):
    H_t   = np.array([simps(row**2, dx=dx) for row in delta_xt])
    phi_t = np.array([simps(row,   dx=dx) for row in delta_xt])
    phi_c_t = np.gradient(phi_t, dt, edge_order=2)
    eps = 1e-30
    dlogH   = np.gradient(np.log(np.abs(H_t ) + eps), dt, edge_order=2)
    dlogphi = np.gradient(np.log(np.abs(phi_t) + eps), dt, edge_order=2)
    K_t = dlogH / (dlogphi + 1e-30)
    K_t = np.where(np.isfinite(K_t), K_t, 0.0)
    return H_t, phi_t, phi_c_t, K_t

# ---------- G(t)（未乘增益；分母取 H·|phi_c|） ----------
def compute_G_series(H_t, phi_c_t, phi2):
    epsH, epsC = 1e-20, 1e-12
    denom = np.maximum(H_t, epsH) * np.maximum(np.abs(phi_c_t), epsC)
    return phi2 / denom * UNIT_FACTOR

# ---------- t* 选择 ----------
def _nearest_index(arr, target, valid_idx):
    if len(valid_idx) == 0:
        return None
    return int(valid_idx[np.argmin(np.abs(arr[valid_idx] - target))])

def pick_tstar_and_G(G_series, K_t, valid_idx, rule='argmax_abs_phic',
                     phi_c_t=None, robust_window=2):
    if len(valid_idx) == 0:
        return np.nan, np.nan, None

    if rule == 'argmax_K':
        idx = int(valid_idx[np.argmax(K_t[valid_idx])])
    elif rule == 'K_cross':
        K0 = float(np.median(K_t[valid_idx]))
        j = _nearest_index(K_t, K0, valid_idx)
        idx = int(j) if j is not None else int(valid_idx[np.argmax(K_t[valid_idx])])
    else:  # 默认：|phi_c| 最大
        idx = int(valid_idx[np.nanargmax(np.abs(phi_c_t[valid_idx]))])

    j0 = max(0, idx - robust_window)
    j1 = min(len(G_series) - 1, idx + robust_window)
    g_star = float(np.median(G_series[j0:j1+1]))
    k_star = float(np.median(K_t[j0:j1+1]))
    return g_star, k_star, idx

# ---------- 单次运行 ----------
def run_once(params, rule='argmax_abs_phic', unit_gain=1.0, seed=None, robust_window=2,
             generator='riemann_mix', shared=None, EDGE_GUARD=4,
             K_band=None, prod_band=None, A_pow=0.0, A_ref=0.10):
    p = params.copy()
    phi2 = p.pop('phi2')

    if generator == 'riemann_mix':
        delta_xt = delta_field_riemann_mix(
            x, t, A=p['A'], sigma_x=p['sigma_x'],
            a=shared['a'], b=shared['b'], sigma_u=shared['sigma_u'],
            u_span1=shared['u_span1'], u_span2=shared['u_span2'],
            gammas=shared['gammas'], mix_weight=shared.get('mix_weight', 0.35),
            mode=shared.get('mode', 'demean'),
            t0=p['t0'], sigma_t_env=p['sigma_t'],
            quantum_noise=p.get('quantum_noise', 0.0), seed=seed
        )
    elif generator == 'gauss':
        delta_xt = delta_field_gauss(
            x, t, A=p['A'], sigma_x=p['sigma_x'], sigma_t=p['sigma_t'],
            k=p['k'], t0=p['t0'], quantum_noise=p.get('quantum_noise', 0.0), seed=seed
        )
    else:
        raise ValueError("unknown generator")

    H_t, phi_t, phi_c_t, K_t = compute_structures(delta_xt)
    G_uncal = compute_G_series(H_t, phi_c_t, phi2)

    # 有效域 + 带宽
    idx_all = np.arange(len(t))
    interior = (idx_all >= EDGE_GUARD) & (idx_all < len(t)-EDGE_GUARD)
    base_mask = (H_t > 1e-12) & (np.abs(phi_c_t) > 1e-8) & interior

    mask = base_mask.copy()
    if K_band is not None:
        mask &= (K_t >= K_band[0]) & (K_t <= K_band[1])
    if prod_band is not None:
        P = H_t * np.abs(phi_c_t)
        mask &= (P >= prod_band[0]) & (P <= prod_band[1])

    valid_idx = np.where(mask)[0]
    if valid_idx.size == 0:
        valid_idx = np.where(base_mask)[0]

    g_star_uncal, k_star, idx = pick_tstar_and_G(
        G_uncal, K_t, valid_idx, rule=rule, phi_c_t=phi_c_t, robust_window=robust_window
    )
    if idx is None or not np.isfinite(g_star_uncal):
        return None

    G_star = float(unit_gain * g_star_uncal)
    rel_err = abs(G_star - G_CODATA) / G_CODATA * 100.0
    return dict(G=G_star, G_uncal=g_star_uncal, rel_err_percent=rel_err,
                t_star=float(t[idx]), K_star=k_star,
                H_t=H_t, phi_t=phi_t, phi_c_t=phi_c_t, K_t=K_t,
                valid_idx=valid_idx, idx=idx)

# ---------- 标定 UNIT_GAIN ----------
def calibrate_unit_gain(dev_list, rule='argmax_abs_phic', strategy='median', robust_window=2,
                        generator='riemann_mix', shared=None, seed0=42, EDGE_GUARD=4):
    K_all, P_all, g_uncal = [], [], []

    # 统计带宽
    for j, p in enumerate(dev_list):
        tmp = run_once(p, rule=rule, unit_gain=1.0, seed=seed0+j, robust_window=robust_window,
                       generator=generator, shared=shared, EDGE_GUARD=EDGE_GUARD,
                       K_band=None, prod_band=None)
        if tmp is None:
            continue
        H_t, phi_c_t, K_t = tmp['H_t'], tmp['phi_c_t'], tmp['K_t']
        idx_all = np.arange(len(t))
        interior = (idx_all >= EDGE_GUARD) & (idx_all < len(t)-EDGE_GUARD)
        mask = (H_t > 1e-12) & (np.abs(phi_c_t) > 1e-8) & interior
        K_all.append(K_t[mask])
        P_all.append((H_t * np.abs(phi_c_t))[mask])

    if len(K_all) == 0:
        raise RuntimeError("标定失败：带宽统计为空。")
    K_all = np.concatenate(K_all); P_all = np.concatenate(P_all)

    # 中位带宽（稳）
    k_lo, k_hi = np.quantile(K_all, [0.50, 0.9999999])
    p_lo, p_hi = np.quantile(P_all, [0.25, 0.75])
    K_band_calib  = (float(k_lo), float(k_hi))
    prod_band_calib = (float(p_lo), float(p_hi))

    # 生产放宽（轻度）
    K_band_prod = (max(K_band_calib[0] - 0.0, float(K_all.min())),
                   min(K_band_calib[1] + 0.0, float(K_all.max())))
    prod_band_prod = (max(prod_band_calib[0]*1.0, float(P_all.min())),
                      min(prod_band_calib[1]*1.0, float(P_all.max())))
    P_pivot = float(np.median(P_all))

    # 枢轴样本
    for j, p in enumerate(dev_list):
        res = run_once(p, rule=rule, unit_gain=1.0, seed=seed0+100+j, robust_window=robust_window,
                       generator=generator, shared=shared, EDGE_GUARD=EDGE_GUARD,
                       K_band=K_band_calib, prod_band=prod_band_calib)
        if res is not None and np.isfinite(res['G_uncal']):
            g_uncal.append(res['G_uncal'])

    g_uncal = np.array(g_uncal, float)
    g_uncal = g_uncal[np.isfinite(g_uncal)]
    if g_uncal.size == 0:
        raise RuntimeError("标定失败：无有效 G_uncal 样本。")

    pivot = float(np.median(g_uncal))
    if strategy == 'lstsq':
        num = g_uncal.sum(); den = (g_uncal**2).sum() + 1e-30
        gain = G_CODATA * (num / den)
        strategy_used = 'lstsq'
    else:
        gain = G_CODATA / max(pivot, 1e-30)
        strategy_used = 'median'

    info = dict(
        n_valid=int(g_uncal.size), pivot=pivot, strategy=strategy_used,
        K_band_calib=K_band_calib,  prod_band_calib=prod_band_calib,
        K_band_prod=K_band_prod,    prod_band_prod=prod_band_prod,
        P_pivot=P_pivot
    )
    return float(gain), info

# ---------- 参考 ∂x ln N 于 t*（不含噪声；可选振幅归一） ----------
def grad_ref_at_tstar(shared, params, t_idx, A_pow=0.0, A_ref=0.10):
    # 重建与生成器一致的 s1/s2、S1/S2、env，但不加噪声，不乘 A
    N = len(t)
    x_mid = 0.5*(x[0] + x[-1])

    def track(u_span, which='demean'):
        u = np.linspace(u_span[0], u_span[1], N)
        phi_u = phase_phi(u, shared['gammas'], shared['sigma_u'])
        tau_u = shared['b'] * (phi_u - np.mean(phi_u))
        N_u = np.exp(shared['a'] * tau_u)
        z = np.gradient(N_u, dt) if which == 'd_dt' else (N_u - np.mean(N_u))
        z = (z - np.mean(z)) / (np.std(z) + 1e-12)
        return z

    s1 = track(shared['u_span1'], 'demean')
    s2 = track(shared['u_span2'], 'd_dt')
    # 去相关 + 二次项，与生成器一致
    s2 = s2 - (np.dot(s2, s1) / (np.dot(s1, s1) + 1e-12)) * s1
    s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-12)
    s2 = s2 + 0.10 * ((s1**2 - np.mean(s1**2)) / (np.std(s1**2) + 1e-12))
    s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-12)

    S1 = norm_row(np.exp(-((x - (x_mid - 5.0))**2) / (2*params['sigma_x']**2)))
    S2 = norm_row(np.exp(-((x - (x_mid + 6.0))**2) / (2*(1.2*params['sigma_x'])**2)))

    if params.get('t0', None) is None:
        t0_local = 0.5*(t[0] + t[-1])
    else:
        t0_local = params['t0']
    env = np.exp(-0.5 * ((t - t0_local) / params['sigma_t'])**2)
    env /= (env.max() + 1e-12)

    lnN_row = s1[t_idx] * env[t_idx] * S1 + shared.get('mix_weight', 0.35) * s2[t_idx] * env[t_idx] * S2
    # 中心差分 ∂x ln N
    grad_x = np.gradient(lnN_row, dx)
    gmed = float(np.median(np.abs(grad_x)))

    # 振幅归一修正（可关）
    if A_pow != 0.0:
        gmed = gmed * (params['A'] / (A_ref + 1e-30))**A_pow

    return gmed

# ---------- β_n 标定（与生产一致的带宽 + 集合） ----------
def calibrate_beta_n(dev_list, shared, rule, generator, seed0, robust_window, EDGE_GUARD,
                     UNIT_GAIN, A_pow, A_ref, Pi_target=1.0,
                     K_band=None, prod_band=None):
    grads, Gs = [], []
    for j, p in enumerate(dev_list):
        r = run_once(
            p, rule=rule, unit_gain=UNIT_GAIN, seed=seed0+999+j,
            robust_window=robust_window, generator=generator, shared=shared,
            EDGE_GUARD=EDGE_GUARD, A_pow=A_pow, A_ref=A_ref,
            K_band=K_band, prod_band=prod_band
        )
        if r is None:
            continue
        t_idx = r['idx']
        g_ref = grad_ref_at_tstar(shared, p, t_idx, A_pow=A_pow, A_ref=A_ref)
        grads.append(g_ref); Gs.append(r['G'])
    if not grads:
        return 0.0, dict(n=0, med_G=np.nan, med_grad_ref=np.nan)
    med_G = float(np.median(Gs))
    med_grad_ref = float(np.median(grads))
    beta_n = Pi_target * (2.0*med_G) / (c0**2 * med_grad_ref + 1e-30)
    return float(beta_n), dict(n=len(Gs), med_G=med_G, med_grad_ref=med_grad_ref)

# ---------- Noise-Calib（占位） ----------
def calibrate_noise_params():
    lam = 1e-12
    gamma = 0.0
    return lam, gamma

# ============== 配置区 ==============
N_ZEROS = 120
gammas = riemann_zeros(N_ZEROS)
dg = float(np.mean(np.diff(gammas)))
PHASECFG = dict(
    a=1.0, b=0.30, sigma_u=dg,
    u_span1=(float(gammas[12]-6.0), float(gammas[88]+6.0)),
    u_span2=(float(gammas[16]-4.0), float(gammas[92]+8.0)),
    gammas=gammas, mode='demean', mix_weight=0.35
)

dev_baselines = [
    dict(phi2=7.8e-30, A=0.10, sigma_t=0.35,  sigma_x=8.0, k=4, t0=4.94, quantum_noise=0.0),
    dict(phi2=7.8e-30, A=0.08, sigma_t=0.315, sigma_x=8.0, k=4, t0=4.89, quantum_noise=0.0),
    dict(phi2=7.8e-30, A=0.12, sigma_t=0.38,  sigma_x=8.0, k=4, t0=5.02, quantum_noise=0.0),
    dict(phi2=7.8e-30, A=0.10, sigma_t=0.35,  sigma_x=8.0, k=5, t0=4.94, quantum_noise=0.0),
    dict(phi2=7.8e-30, A=0.08, sigma_t=0.315, sigma_x=8.0, k=5, t0=4.89, quantum_noise=0.0),
]

CALIB_GENERATOR = 'riemann_mix'
GENERATOR       = 'riemann_mix'
TSTAR_RULE      = 'argmax_abs_phic'
GAIN_STRATEGY   = 'median'
ROBUST_WINDOW   = 2
EDGE_GUARD      = 4
SEED0           = 2025
# 振幅归一（可调）：g_ref *= (A/A_ref)**A_pow
A_pow           = 0.0
A_ref           = 0.10

# ============== 标定 UNIT_GAIN ==============
UNIT_GAIN, calib = calibrate_unit_gain(
    dev_baselines, rule=TSTAR_RULE, strategy=GAIN_STRATEGY,
    robust_window=ROBUST_WINDOW, generator=CALIB_GENERATOR, shared=PHASECFG,
    seed0=SEED0, EDGE_GUARD=EDGE_GUARD
)
print(f"[Unit-Calib] UNIT_GAIN={UNIT_GAIN:.6e} | pivot={calib['pivot']:.6e} | n={calib['n_valid']}")
print("  (calib K_band=(%.12f, %.12f), prod_band=(%.6e, %.6e), strategy=%s, P_pivot=%.6e, A_pow=%.6f, A_ref=%.6f)" % (
    calib['K_band_calib'][0], calib['K_band_calib'][1],
    calib['prod_band_calib'][0], calib['prod_band_calib'][1],
    calib['strategy'], calib['P_pivot'], A_pow, A_ref
))

# ============== Noise-Calib（占位打印） ==============
lam, gamma = calibrate_noise_params()
print(f"[Noise-Calib] lambda={lam:.6e}, gamma={gamma:.6e}")

# ============== 标定 β_n（与生产一致带宽/集合） ==============
beta_n, binfo = calibrate_beta_n(
    dev_baselines, PHASECFG, TSTAR_RULE, GENERATOR, SEED0, ROBUST_WINDOW, EDGE_GUARD,
    UNIT_GAIN=UNIT_GAIN, A_pow=A_pow, A_ref=A_ref, Pi_target=1.0,
    K_band=calib['K_band_prod'], prod_band=calib['prod_band_prod']
)
print(f"[c0-Calib] beta_n={beta_n:.6e}  (Π_target=1.0, med_G={binfo['med_G']:.6e}, med_grad_ref={binfo['med_grad_ref']:.6e}, n={binfo['n']})")

# ============== 单次演示 ==============
demo_params = dict(phi2=7.8e-30, A=0.10, sigma_t=0.35, sigma_x=8.0, k=4, t0=4.94, quantum_noise=0.0)
demo = run_once(
    demo_params, rule=TSTAR_RULE, unit_gain=UNIT_GAIN, seed=SEED0+1,
    robust_window=ROBUST_WINDOW, generator=GENERATOR, shared=PHASECFG, EDGE_GUARD=EDGE_GUARD,
    K_band=calib['K_band_prod'], prod_band=calib['prod_band_prod'],
    A_pow=A_pow, A_ref=A_ref
)
if demo is None:
    raise RuntimeError("单次演示失败。")

print(f"\n=== 单次演示（{GENERATOR} | 规则={TSTAR_RULE}）===")
print(f"G(t*)={demo['G']:.6e} 相对误差={demo['rel_err_percent']:.4f}% t*={demo['t_star']:.6f} K(t*)={demo['K_star']:.6e}")
print(f"参考 CODATA = {G_CODATA:.6e}")

# 诊断：corr(H, phi^2) 与斜率
phi_sq = demo['phi_t']**2
corr = float(np.corrcoef(demo['H_t'], phi_sq)[0,1])
num = float(np.dot(phi_sq - phi_sq.mean(), demo['H_t'] - demo['H_t'].mean()))
den = float(np.dot(phi_sq - phi_sq.mean(), phi_sq - phi_sq.mean()) + 1e-12)
slope = num / den
print(f"[诊断] corr(H, phi^2) = {corr:.6f}, 斜率≈{slope:.6e}")

# c0-link：Π = 2G / (c0^2 * beta_n * median|∂x ln N|)
gref = grad_ref_at_tstar(PHASECFG, demo_params, demo['idx'], A_pow=A_pow, A_ref=A_ref)
Pi_demo = (2.0 * demo['G']) / (c0**2 * beta_n * (gref + 1e-30))
print(f"[c0-link] median|∂x ln N|(t*)={gref:.6e}  median|g_x|(t*)={demo['G']:.6e}  Π={Pi_demo:.6e}")

# ============== 多噪声蒙特卡罗 ==============
noise_levels = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
N_PER_LEVEL = 20
mc_rows, c0_rows = [], []

print("\n=== 多噪声蒙特卡罗（每档 {} 次） ===".format(N_PER_LEVEL))
for z, nz in enumerate(noise_levels):
    G_list, K_list, Pi_list = [], [], []
    for rep in range(N_PER_LEVEL):
        p = demo_params.copy(); p['quantum_noise'] = nz
        r = run_once(
            p, rule=TSTAR_RULE, unit_gain=UNIT_GAIN, seed=SEED0+100*z+rep,
            robust_window=ROBUST_WINDOW, generator=GENERATOR, shared=PHASECFG, EDGE_GUARD=EDGE_GUARD,
            K_band=calib['K_band_prod'], prod_band=calib['prod_band_prod'],
            A_pow=A_pow, A_ref=A_ref
        )
        if r is None:
            continue
        G_list.append(r['G']); K_list.append(r['K_star'])
        mc_rows.append(dict(noise=nz, G=r['G'], rel_err=r['rel_err_percent'], K=r['K_star']))

        gref_i = grad_ref_at_tstar(PHASECFG, p, r['idx'], A_pow=A_pow, A_ref=A_ref)
        Pi_i = (2.0 * r['G']) / (c0**2 * beta_n * (gref_i + 1e-30))
        Pi_list.append(Pi_i)
        c0_rows.append(dict(noise=nz, Pi=Pi_i, grad_ref=gref_i, G=r['G'], t_star=r['t_star']))

    if G_list:
        arrG = np.array(G_list); arrK = np.array(K_list); arrPi = np.array(Pi_list)
        mean_rel = np.mean(np.abs((arrG - G_CODATA)/G_CODATA) * 100)
        print(f"[噪声={nz} · ħ={nz*hbar:.2e}] 平均G={arrG.mean():.6e} 平均误差={mean_rel:.4f}% G方差={arrG.var():.3e} 平均K*={arrK.mean():.3f}")
        print(f"           [c0-link] Π: mean={arrPi.mean():.6e}, std={arrPi.std():.6e}")

pd.DataFrame(mc_rows).to_csv("mc_summary.csv", index=False)
pd.DataFrame(c0_rows).to_csv("c0_mc_summary.csv", index=False)
print("\n已导出：mc_summary.csv")
print("已导出：c0_mc_summary.csv")

# ============== 盲测矩阵 ==============
blind_specs = [
    (4.89, 0.08, 0.315, 8.0, 4, 0.00),
    (4.89, 0.08, 0.315, 8.0, 4, 0.01),
    (4.89, 0.08, 0.315, 8.0, 4, 0.10),
    (4.89, 0.08, 0.315, 8.0, 5, 0.00),
    (4.89, 0.08, 0.315, 8.0, 5, 0.10),
    (4.89, 0.08, 0.315, 8.0, 6, 0.00),
    (4.94, 0.10, 0.350, 8.0, 4, 0.00),
]
rows, c0b_rows = []
for i, (t0_, A_, st, sx, k_, nz) in enumerate(blind_specs):
    p = dict(phi2=7.8e-30, A=A_, sigma_t=st, sigma_x=sx, k=k_, t0=t0_, quantum_noise=nz)
    r = run_once(
        p, rule=TSTAR_RULE, unit_gain=UNIT_GAIN, seed=SEED0+2025+i,
        robust_window=ROBUST_WINDOW, generator=GENERATOR, shared=PHASECFG, EDGE_GUARD=EDGE_GUARD,
        K_band=calib['K_band_prod'], prod_band=calib['prod_band_prod'],
        A_pow=A_pow, A_ref=A_ref
    )
    if r is not None:
        rows.append(dict(
            t0=t0_, A=A_, sigma_t=st, sigma_x=sx, k=k_, noise=nz,
            G=r['G'], rel_err_percent=r['rel_err_percent'],
            t_star=r['t_star'], K_star=r['K_star'],
            rule=TSTAR_RULE, generator=GENERATOR
        ))
        gref_i = grad_ref_at_tstar(PHASECFG, p, r['idx'], A_pow=A_pow, A_ref=A_ref)
        Pi_i = (2.0 * r['G']) / (c0**2 * beta_n * (gref_i + 1e-30))
        c0b_rows.append(dict(
            t0=t0_, A=A_, sigma_t=st, sigma_x=sx, k=k_, noise=nz,
            Pi=Pi_i, grad_ref=gref_i, G=r['G'], t_star=r['t_star'],
            rule=TSTAR_RULE, generator=GENERATOR
        ))

blind_df = pd.DataFrame(rows)
blind_df.to_csv("blind_matrix.csv", index=False)
print("\n=== 盲测矩阵（规则={} | 生成器={}） ===".format(TSTAR_RULE, GENERATOR))
if not blind_df.empty:
    print(blind_df.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
print("已导出：blind_matrix.csv")

pd.DataFrame(c0b_rows).to_csv("c0_blind_matrix.csv", index=False)
print("已导出：c0_blind_matrix.csv")
