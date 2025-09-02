# -*- coding: utf-8 -*-
# ===================== ζ/π 连续介质层析 —— 完整修复版（含参数显示） =====================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from scipy.stats import ttest_1samp, wilcoxon

# ---------------- 0) 核心参数 y（可改后重跑） ----------------
y = dict(
    beta=1.00,        # ln N = beta * zscore(dev)
    pi_shape=0.30,    # π-surrogate Gamma shape（越小越“抖/簇”）
    n_rays=22,        # 射线条数
    K=18,             # 样条控制点数量（模型容量）
    lam=2.0e-2,       # 二阶差分正则强度
    alpha=1.6,        # arctan 核宽度因子：sigma = alpha * Δ
    p_pow=1.8,        # 非 cheby 采样时的幂律偏置
    z_points=900,     # z 网格分辨率
    M=48,             # 使用的 ζ 零点个数
    N_RUNS=48,        # 批次数（加大可提高功效）
    noise_rel=0.010,  # 相对噪声（飞时）
    weight_mode='up', # 高-p 权重：'up'/'balanced'/'down'
    weight_clip=3.0,  # 权重裁剪
    rng_seed=11000    # 批次随机种子
)

print(">>> CORE PARAMS (y)")
for k in y: print(f"  {k:>10s} : {y[k]}")

# ---------------- 1) ζ 零点（前若干个） ----------------
gamma_zeta_full = np.array([
    14.1347251417, 21.0220396388, 25.0108575801, 30.4248761259, 32.9350615877,
    37.5861781588, 40.9187190121, 43.3270732809, 48.0051508812, 49.7738324777,
    52.9703214777, 56.4462476971, 59.3470440026, 60.8317785246, 65.1125440481,
    67.0798105295, 69.5464017112, 72.0671576745, 75.7046906991, 77.1448400689,
    79.3373750202, 82.9103808541, 84.7354929805, 87.4252746131, 88.8091112076,
    92.4918992706, 94.6513440405, 95.8706342282, 98.8311942182, 101.3178510057,
    103.7255380405, 105.4466230523, 107.1686111843, 111.0295355432, 111.8746591770,
    114.3202209155, 116.2266803209, 118.7907828660, 121.3701250024, 122.9468292936,
    124.2568185543, 127.5166838796, 129.5787042000, 131.0876885309, 133.4977372030,
    134.7565097534, 138.1160420545, 139.7362089521, 141.1237074040, 143.1118458076
])
gamma_zeta_base = gamma_zeta_full[:y['M']]

# ---------------- 2) π surrogate：簇状短间隙 + 长尾混合 ----------------
def gamma_mixture_bursty(M=48, mean_gap=None,
                         pi_shape=0.30, mix_prob=0.50, burst_prob=0.60, burst_geom_p=0.5,
                         lognorm_sigma=1.4, clip_low=0.05, clip_high=8.0,
                         seed=123, align_start=None):
    rng = np.random.default_rng(seed)
    if mean_gap is None:
        mean_gap = (gamma_zeta_base[-1] - gamma_zeta_base[0]) / (len(gamma_zeta_base) - 1)
    gaps, i = [], 0
    while i < M:
        if rng.random() < burst_prob:
            L = max(1, int(rng.geometric(burst_geom_p)))
            L = min(L, M - i)
            small = np.clip(rng.beta(1, 9, size=L) * mean_gap, clip_low*mean_gap, 0.6*mean_gap)
            gaps.extend(small); i += L
        else:
            if rng.random() < mix_prob:
                g = rng.lognormal(np.log(mean_gap), lognorm_sigma)
            else:
                g = rng.gamma(shape=pi_shape, scale=mean_gap/max(pi_shape,1e-6))
            gaps.append(np.clip(g, clip_low*mean_gap, clip_high*mean_gap)); i += 1
    gaps = np.array(gaps[:M])
    if align_start is None:
        align_start = gamma_zeta_base[0]
    return align_start + np.cumsum(gaps)

# ---------------- 3) 相位 → 折射率 N(z) ----------------
def phase_to_index_from_zeros_scaled(gamma, beta=1.00, alpha=1.6, z_points=900, pad_sigma=5.0):
    M = len(gamma)
    Delta = (gamma[-1] - gamma[0]) / max(M - 1, 1)
    sigma = alpha * Delta
    z_min = gamma.min() - pad_sigma * sigma
    z_max = gamma.max() + pad_sigma * sigma
    z = np.linspace(z_min, z_max, z_points)
    phi = np.zeros_like(z)
    for gn in gamma:
        phi += np.arctan((z - gn) / sigma)
    dev = phi - np.mean(phi)
    dev_hat = dev / (np.std(dev) + 1e-12)
    lnN = beta * dev_hat
    return z, np.exp(lnN)

# ---------------- 4) 飞时积分（近 turning：NaN 掩蔽） ----------------
def _T_of_p_vectorized(z, n, p_list, margin=0.985):
    nmin = float(np.min(n))
    dz = np.gradient(z)
    out = np.empty(len(p_list), dtype=float)
    for i, p in enumerate(p_list):
        if not (0 < p < margin*nmin):
            out[i] = np.nan; continue
        denom = np.sqrt(np.maximum(n**2 - p**2, 0.0))
        zero = (denom == 0)
        if np.any(zero): denom[zero] = 1e-12
        out[i] = float(np.sum(((n**2)/denom) * dz))
    return out

# ---------------- 5) 射线采样 ----------------
def p_list_cheby(nmin, n_rays, pmin=0.20, pmax=0.985):
    k = np.arange(1, n_rays+1)
    x = np.cos((2*k - 1) * np.pi / (2*n_rays))  # [-1,1]
    u = (x + 1)/2                                # [0,1]
    return nmin * (pmin + (pmax - pmin) * u)

# ---------------- 6) 高-p 权重 ----------------
def make_weights(p_list, nmin, mode='up', clip=3.0):
    t = 1.0 - (p_list/(nmin+1e-12))**2
    t = np.clip(t, 1e-4, 1.0)
    if mode == 'down':
        w = np.sqrt(t)
    elif mode == 'balanced':
        w = t**(-0.25)
    else:
        w = t**(-0.6)  # aggressive
    w = np.clip(w, 0.2, clip)
    return w / (np.mean(w)+1e-12)

# ---------------- 7) 稳健反演（Huber + 二阶差分 + turning barrier） ----------------
def invert_fixed_p_FULL(z, n_true, p_list, K=18, lam=2.0e-2, max_nfev=400,
                        add_noise_rel=0.010, rng=None, huber_fscale=1.0,
                        margin=0.985, barrier_weight=4.0,
                        noise_eps=None, weight_mode='up', weight_clip=3.0):
    if rng is None: rng = np.random.default_rng(0)
    T_obs = _T_of_p_vectorized(z, n_true, p_list, margin=margin)
    T_scale = np.nanmedian(np.abs(T_obs)) + 1e-12
    if noise_eps is None:
        noise_eps = rng.normal(0.0, add_noise_rel*T_scale, size=T_obs.shape)
    T_obs_noisy = T_obs + noise_eps

    zk = np.linspace(z[0], z[-1], K)
    ln_n0 = np.full(K, np.log(np.mean(n_true)))

    L = np.zeros((K-2, K))
    for i in range(K-2): L[i, i:i+3] = [1, -2, 1]

    nmin_true = float(np.min(n_true))
    w = make_weights(p_list, nmin_true, mode=weight_mode, clip=weight_clip)

    def interp_ln_n(ln_nk):
        cs = CubicSpline(zk, ln_nk, bc_type='natural'); return cs(z)

    def residuals(ln_nk):
        ln_n = interp_ln_n(ln_nk); n = np.exp(ln_n)
        T_pred = _T_of_p_vectorized(z, n, p_list, margin=margin)
        nmin = float(np.min(n))
        bad = ~np.isfinite(T_pred); good = ~bad

        r = np.zeros_like(T_pred, dtype=float)
        r[good] = w[good] * ((T_pred[good] - T_obs_noisy[good]) / T_scale)

        viol = np.maximum(p_list/(margin*(nmin+1e-12)) - 1.0, 0.0)
        r[bad] = barrier_weight * viol[bad]

        r_reg = np.sqrt(lam) * (L @ ln_nk)
        return np.concatenate([r, r_reg])

    lo = float(np.log(np.min(n_true)*0.6))
    hi = float(np.log(np.max(n_true)*1.7))
    bounds = (np.full(K, lo), np.full(K, hi))

    res = least_squares(residuals, ln_n0, bounds=bounds, method='trf',
                        loss='huber', f_scale=huber_fscale,
                        max_nfev=max_nfev, xtol=1e-9, ftol=1e-9, gtol=1e-9)

    n_hat = np.exp(interp_ln_n(res.x))
    nrmse = np.sqrt(np.mean((n_hat - n_true)**2)) / (np.mean(n_true)+1e-12)
    info = dict(nrmse=float(nrmse), success=bool(res.success), nfev=int(res.nfev),
                theta=res.x, zk=zk, L=L, w=w, T_scale=T_scale, lam=lam)
    return n_hat, info

# ---------------- 8) 雅可比（安全）与增广条件数 ----------------
def jacobian_data_SAFE(z, p_list, theta_opt, zk, h=8e-4):
    K = len(theta_opt); m = len(p_list)
    J = np.zeros((m, K), dtype=float)

    def T_pred(theta):
        cs = CubicSpline(zk, theta, bc_type='natural')
        ln_n = cs(z); n = np.exp(ln_n)
        return _T_of_p_vectorized(z, n, p_list)

    T0 = T_pred(theta_opt)
    for k in range(K):
        thp = theta_opt.copy(); thm = theta_opt.copy()
        thp[k] += h; thm[k] -= h
        Tp = T_pred(thp); Tm = T_pred(thm)
        J[:, k] = (Tp - Tm)/(2*h)

    mask_rows = np.isfinite(T0) & np.isfinite(J).all(axis=1)
    return J, T0, mask_rows

def augmented_stability_metrics_MASKED(J_full, w, T_scale, L, lam, mask_rows):
    W = (w.reshape(-1,1)/(T_scale+1e-12))
    J_aug = np.vstack([W[mask_rows]*J_full[mask_rows], np.sqrt(lam)*L])
    ok = np.isfinite(J_aug).all(axis=1); J_aug = J_aug[ok,:]
    if J_aug.size==0 or J_aug.shape[0]<2 or J_aug.shape[1]<2:
        return 0.0, np.inf
    s = np.linalg.svd(J_aug, compute_uv=False)
    smin = float(s.min()); cond = float(s.max()/max(s.min(),1e-18))
    return smin, cond

# ---------------- 9) 指标 ----------------
def safe_log10(x, eps=1e-18):
    x = np.asarray(x, float)
    return np.log10(np.clip(x, eps, np.inf))

def FWNRMSE_mid(z, n_true, n_hat, k0_frac=0.15, k1_frac=0.60, p=1.0):
    y = n_true - n_hat
    Y = np.fft.rfft(y - y.mean()); N = len(Y)
    k0 = int(k0_frac * N); k1 = int(k1_frac * N)
    w = (np.arange(N) / max(N-1,1))**p
    w[:k0] = 0.0
    if k1+1 < N: w[k1+1:] = 0.0
    num = np.sum(w * (np.abs(Y)**2))
    den = np.sum(np.abs(np.fft.rfft(n_true - n_true.mean()))**2) + 1e-12
    return np.sqrt(num / (den + 1e-12))

def FWNRMSE_high(z, n_true, n_hat, k0_frac=0.25, k1_frac=0.95, p=2.0):
    y = n_true - n_hat
    Y = np.fft.rfft(y - y.mean()); N = len(Y)
    k0 = int(k0_frac * N); k1 = int(k1_frac * N)
    w = (np.arange(N) / max(N-1,1))**p
    w[:k0] = 0.0
    if k1+1 < N: w[k1+1:] = 0.0
    num = np.sum(w * (np.abs(Y)**2))
    den = np.sum(np.abs(np.fft.rfft(n_true - n_true.mean()))**2) + 1e-12
    return np.sqrt(num / (den + 1e-12))

def data_space_rel_error(z, n_true, n_hat, p_list, margin=0.985):
    T_true = _T_of_p_vectorized(z, n_true, p_list, margin=margin)
    T_hat  = _T_of_p_vectorized(z, n_hat,  p_list, margin=margin)
    m = np.isfinite(T_true) & np.isfinite(T_hat)
    if not np.any(m): return np.nan
    num = np.linalg.norm(T_hat[m] - T_true[m])
    den = np.linalg.norm(T_true[m]) + 1e-12
    return num / den

# ---------------- 10) 效应大小 & CI ----------------
def cohens_d_paired(x, y):
    x = np.asarray(x); y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    diff = x - y
    n = len(diff)
    if n < 2:
        return dict(dz=np.nan, gz=np.nan, dav=np.nan, gav=np.nan)
    dz = diff.mean() / (diff.std(ddof=1) + 1e-12)
    sx = x.std(ddof=1); sy = y.std(ddof=1)
    s_av = np.sqrt((sx**2 + sy**2)/2.0)
    dav = (x.mean() - y.mean()) / (s_av + 1e-12)
    J = 1.0 - 3.0/(4.0*(n-1) - 1.0) if n > 2 else 1.0
    gz = J * dz; gav = J * dav
    return dict(dz=dz, gz=gz, dav=dav, gav=gav)

def rank_biserial_from_diff(diff):
    diff = np.asarray(diff); diff = diff[np.isfinite(diff)]
    n_pos = np.sum(diff > 0); n_neg = np.sum(diff < 0)
    n = n_pos + n_neg
    if n == 0: return np.nan
    return (n_pos - n_neg) / n

def bootstrap_ci(diff, func=np.mean, B=2000, alpha=0.05, rng=None):
    diff = np.asarray(diff); diff = diff[np.isfinite(diff)]
    if rng is None: rng = np.random.default_rng(0)
    if diff.size == 0: return (np.nan, np.nan, np.nan)
    boots = [func(rng.choice(diff, size=diff.size, replace=True)) for _ in range(B)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return (func(diff), lo, hi)

# ---------------- 11) 单次实例（ζ/π 反演 + 指标） ----------------
def run_instance_augmented(seed=0,
                           beta=1.00, alpha=1.6, z_points=900,
                           n_rays=22, sampling='cheby', p_pow=1.8,
                           K=18, lam=2.0e-2,
                           pi_shape=0.30, mix_prob=0.50, burst_prob=0.60, burst_geom_p=0.5,
                           lognorm_sigma=1.4,
                           noise_rel=0.010, unify_mask=True,
                           weight_mode='up', weight_clip=3.0):
    rng = np.random.default_rng(seed)
    # 平移避免边界重合
    Delta = (gamma_zeta_base[-1] - gamma_zeta_base[0])/(len(gamma_zeta_base)-1)
    shift = rng.uniform(-0.5*Delta, 0.5*Delta)
    gamma_zeta = gamma_zeta_base + shift

    # π：簇状长尾 surrogate
    mean_gap = (gamma_zeta_base[-1] - gamma_zeta_base[0])/(len(gamma_zeta_base)-1)
    gamma_pi = gamma_mixture_bursty(
        M=len(gamma_zeta_base), mean_gap=mean_gap,
        pi_shape=pi_shape, mix_prob=mix_prob, burst_prob=burst_prob, burst_geom_p=burst_geom_p,
        lognorm_sigma=lognorm_sigma, clip_low=0.05, clip_high=8.0,
        seed=seed+1337, align_start=gamma_zeta_base[0] + shift
    )

    # 相位→折射率；取中段窗口
    z1,n1 = phase_to_index_from_zeros_scaled(gamma_zeta, beta=beta, alpha=alpha, z_points=z_points)
    z2,n2 = phase_to_index_from_zeros_scaled(gamma_pi,   beta=beta, alpha=alpha, z_points=z_points)
    def mid(z,n,frac=0.65):
        L=len(z); a=int((1-frac)/2*L); b=int((1+frac)/2*L); return z[a:b], n[a:b]
    z1,n1 = mid(z1,n1); z2,n2 = mid(z2,n2)

    # 射线 p（Chebyshev）
    nmin_common = float(min(np.min(n1), np.min(n2)))
    p_list = p_list_cheby(nmin_common, n_rays, pmin=0.20, pmax=0.985)

    # 配对噪声
    rng_eps = np.random.default_rng(seed+2024)
    T_dummy = _T_of_p_vectorized(z1, n1, p_list)
    T_scale_dummy = np.nanmedian(np.abs(T_dummy)) + 1e-12
    eps = rng_eps.normal(0.0, noise_rel*T_scale_dummy, size=T_dummy.shape)

    # 反演
    n_hat_z, info_z = invert_fixed_p_FULL(z1, n1, p_list, K=K, lam=lam,
                                          add_noise_rel=noise_rel, rng=rng, noise_eps=eps,
                                          margin=0.985, barrier_weight=4.0,
                                          weight_mode=weight_mode, weight_clip=weight_clip)
    n_hat_p, info_p = invert_fixed_p_FULL(z2, n2, p_list, K=K, lam=lam,
                                          add_noise_rel=noise_rel, rng=rng, noise_eps=eps,
                                          margin=0.985, barrier_weight=4.0,
                                          weight_mode=weight_mode, weight_clip=weight_clip)

    # 雅可比 & 稳定性（统一可观测掩码）
    Jz_full, Tz0, mz = jacobian_data_SAFE(z1, p_list, info_z['theta'], info_z['zk'], h=8e-4)
    Jp_full, Tp0, mp = jacobian_data_SAFE(z2, p_list, info_p['theta'], info_p['zk'], h=8e-4)
    mask_common = mz & mp

    smin_z, cond_z = augmented_stability_metrics_MASKED(Jz_full, info_z['w'], info_z['T_scale'], info_z['L'], info_z['lam'], mask_common)
    smin_p, cond_p = augmented_stability_metrics_MASKED(Jp_full, info_p['w'], info_p['T_scale'], info_p['L'], info_p['lam'], mask_common)

    # 频域 & 数据空间误差
    fw_mid_z = FWNRMSE_mid (z1, n1, n_hat_z); fw_mid_p = FWNRMSE_mid (z2, n2, n_hat_p)
    fw_hi_z  = FWNRMSE_high(z1, n1, n_hat_z); fw_hi_p  = FWNRMSE_high(z2, n2, n_hat_p)
    ds_err_z = data_space_rel_error(z1, n1, n_hat_z, p_list)
    ds_err_p = data_space_rel_error(z2, n2, n_hat_p, p_list)

    out = dict(
        # 参数空间
        NRMSE_z=info_z['nrmse'], NRMSE_p=info_p['nrmse'],
        FW_mid_z=fw_mid_z, FW_mid_p=fw_mid_p,
        FW_high_z=fw_hi_z, FW_high_p=fw_hi_p,
        # 数据空间
        DSERR_z=ds_err_z, DSERR_p=ds_err_p,
        # 稳定性
        sigmin_aug_z=smin_z, sigmin_aug_p=smin_p,
        cond_aug_z=cond_z, cond_aug_p=cond_p,
        keep_ratio=float(mask_common.mean()),
        # 示例（非数值，后面会过滤）
        _example=(z1,n1,n_hat_z,z2,n2,n_hat_p,p_list)
    )
    return out

# ---------------- 12) 批量运行（修复：仅收集数值标量键；不 pop cfg） ----------------
def run_batch_paper(N_RUNS=48, **cfg):
    seed_base = cfg.get('seed_base', 11000)
    # 传给实例的参数副本（去掉 seed_base 以免多余参数报错）
    run_cfg = dict(cfg)
    if 'seed_base' in run_cfg: run_cfg.pop('seed_base')

    stats = []
    example_pack = None
    for i in range(N_RUNS):
        m = run_instance_augmented(seed=seed_base+i, **run_cfg)
        if example_pack is None and '_example' in m:
            example_pack = m['_example']
        stats.append(m)

    # —— 仅收集“可安全转换为 float 的标量键” —— #
    def is_numeric_scalar(v):
        if np.isscalar(v):
            try:
                float(v)
                return True
            except Exception:
                return False
        return False

    first_keys = list(stats[0].keys())
    numeric_keys = [k for k in first_keys if k != '_example' and is_numeric_scalar(stats[0][k])]
    M = {k: np.array([float(s[k]) for s in stats], dtype=float) for k in numeric_keys}

    # 计算差分（方向：π−ζ）
    d_cndlog = safe_log10(M['cond_aug_p']) - safe_log10(M['cond_aug_z'])
    d_ds     = M['DSERR_p'] - M['DSERR_z']
    d_fwm    = M['FW_mid_p'] - M['FW_mid_z']
    d_fwh    = M['FW_high_p'] - M['FW_high_z']
    d_nrmse  = M['NRMSE_p'] - M['NRMSE_z']

    # 统计工具
    def pstats(diff):
        m = np.isfinite(diff)
        t = ttest_1samp(diff[m], 0.0, nan_policy='omit') if m.any() else None
        w = wilcoxon(diff[m]) if m.sum()>=2 else None
        mu, lo, hi = bootstrap_ci(diff)
        return t, w, mu, lo, hi

    t_cnd, w_cnd, mu_cnd, lo_cnd, hi_cnd = pstats(d_cndlog)
    t_dse, w_dse, mu_dse, lo_dse, hi_dse = pstats(d_ds)
    t_fwm, w_fwm, mu_fwm, lo_fwm, hi_fwm = pstats(d_fwm)
    t_fwh, w_fwh, mu_fwh, lo_fwh, hi_fwh = pstats(d_fwh)
    t_nrm, w_nrm, mu_nrm, lo_nrm, hi_nrm = pstats(d_nrmse)

    es_cnd = cohens_d_paired(safe_log10(M['cond_aug_p']), safe_log10(M['cond_aug_z']))
    es_dse = cohens_d_paired(M['DSERR_p'], M['DSERR_z'])
    es_fwm = cohens_d_paired(M['FW_mid_p'], M['FW_mid_z'])
    es_fwh = cohens_d_paired(M['FW_high_p'], M['FW_high_z'])
    es_nrm = cohens_d_paired(M['NRMSE_p'], M['NRMSE_z'])

    rb_cnd = rank_biserial_from_diff(d_cndlog)
    rb_dse = rank_biserial_from_diff(d_ds)
    rb_fwm = rank_biserial_from_diff(d_fwm)
    rb_fwh = rank_biserial_from_diff(d_fwh)
    rb_nrm = rank_biserial_from_diff(d_nrmse)

    # Holm-Bonferroni 校正
    def holm_bonferroni(pvals, labels, alpha=0.05):
        order = np.argsort(pvals)
        out = []
        for i, idx in enumerate(order):
            thr = alpha / (len(pvals) - i)
            out.append((labels[idx], pvals[idx], pvals[idx] <= thr, thr))
        return out

    labels = ["Δlog10(cond)", "ΔDSERR", "ΔFW_mid", "ΔFW_high", "ΔNRMSE"]
    pvals = [getattr(w,'pvalue',np.nan) for w in [w_cnd,w_dse,w_fwm,w_fwh,w_nrm]]
    holm = holm_bonferroni(np.array(pvals), labels)

    # —— 打印摘要 —— #
    print("\n=== PAPER BATCH SUMMARY (N={}, keep≈{:.2f}) ===".format(
        len(stats), np.mean([s['keep_ratio'] for s in stats])
    ))
    print("CORE PARAMS (y):", y)
    print("Mean(log10(cond)) : ζ={:.2f}, π={:.2f}".format(
        np.nanmean(safe_log10(M['cond_aug_z'])), np.nanmean(safe_log10(M['cond_aug_p']))
    ))
    print("Mean(DSERR)       : ζ={:.3e}, π={:.3e}".format(np.nanmean(M['DSERR_z']), np.nanmean(M['DSERR_p'])))
    print("Mean(FW_mid)      : ζ={:.3e}, π={:.3e}".format(np.nanmean(M['FW_mid_z']), np.nanmean(M['FW_mid_p'])))
    print("Mean(FW_high)     : ζ={:.3e}, π={:.3e}".format(np.nanmean(M['FW_high_z']), np.nanmean(M['FW_high_p'])))
    print("Mean(NRMSE)       : ζ={:.3e}, π={:.3e}".format(np.nanmean(M['NRMSE_z']), np.nanmean(M['NRMSE_p'])))

    def pr(name, t, w, mu, lo, hi, es, rb):
        print(f"[{name}]  Δmean={mu:+.3e}  95%CI[{lo:+.3e},{hi:+.3e}]  "
              f"t={getattr(t,'statistic',np.nan):.2f}, p_t={getattr(t,'pvalue',np.nan):.3f} ; "
              f"p_W={getattr(w,'pvalue',np.nan):.3f}  |  "
              f"ES: dz={es['dz']:+.3f}, gz={es['gz']:+.3f}, dav={es['dav']:+.3f}, gav={es['gav']:+.3f}, r_rb={rb:+.3f}")
    pr("Δlog10(cond)  (π-ζ)", t_cnd, w_cnd, mu_cnd, lo_cnd, hi_cnd, es_cnd, rb_cnd)
    pr("ΔDSERR        (π-ζ)", t_dse, w_dse, mu_dse, lo_dse, hi_dse, es_dse, rb_dse)
    pr("ΔFW_mid       (π-ζ)", t_fwm, w_fwm, mu_fwm, lo_fwm, hi_fwm, es_fwm, rb_fwm)
    pr("ΔFW_high      (π-ζ)", t_fwh, w_fwh, mu_fwh, lo_fwh, hi_fwh, es_fwh, rb_fwh)
    pr("ΔNRMSE        (π-ζ)", t_nrm, w_nrm, mu_nrm, lo_nrm, hi_nrm, es_nrm, rb_nrm)

    print("\n[Holm-Bonferroni over 5 tests]")
    for name, p, ok, thr in holm:
        print(f"  {name:<13s}: p={p:.3g}, threshold={thr:.3g}  -> {'significant' if ok else 'ns'}")

    # —— 图：差分直方图（主+次终点） —— #
    plt.figure(figsize=(11,8))
    for i,(d,lab) in enumerate([
        (d_cndlog,"Δlog10(cond) (π-ζ)"),
        (d_ds,"ΔDSERR (π-ζ)"),
        (d_fwm,"ΔFW_mid (π-ζ)"),
        (d_fwh,"ΔFW_high (π-ζ)")
    ],1):
        plt.subplot(2,2,i)
        plt.hist(d[np.isfinite(d)], bins=10); plt.title(lab); plt.grid(True)
    plt.tight_layout(); plt.show()

    # —— 示例曲线 —— #
    if example_pack is not None:
        z1,n1,n1_hat,z2,n2,n2_hat,p_list = example_pack
        plt.figure(); plt.plot(z1,n1,label='zeta true'); plt.plot(z1,n1_hat,label='zeta recon')
        plt.xlabel('z'); plt.ylabel('n(z)'); plt.title('Zeta: true vs recon (example)')
        plt.legend(); plt.grid(True); plt.show()
        plt.figure(); plt.plot(z2,n2,label='pi true'); plt.plot(z2,n2_hat,label='pi recon')
        plt.xlabel('z'); plt.ylabel('n(z)'); plt.title('Pi-bursty: true vs recon (example)')
        plt.legend(); plt.grid(True); plt.show()

    return dict(M=M,
                diffs=dict(cndlog=d_cndlog, ds=d_ds, fwm=d_fwm, fwh=d_fwh, nrmse=d_nrmse),
                tests=dict(t_cnd=t_cnd, w_cnd=w_cnd, t_dse=t_dse, w_dse=w_dse,
                           t_fwm=t_fwm, w_fwm=w_fwm, t_fwh=t_fwh, w_fwh=w_fwh, t_nrm=t_nrm, w_nrm=w_nrm),
                holm=holm)

# ---------------- 13) 一键运行（使用 y 的参数） ----------------
OUT = run_batch_paper(
    N_RUNS=y['N_RUNS'],
    beta=y['beta'], alpha=y['alpha'], z_points=y['z_points'],
    n_rays=y['n_rays'], sampling='cheby', p_pow=y['p_pow'],
    K=y['K'], lam=y['lam'],
    pi_shape=y['pi_shape'], mix_prob=0.50, burst_prob=0.60, burst_geom_p=0.5,
    lognorm_sigma=1.4,
    noise_rel=y['noise_rel'],
    weight_mode=y['weight_mode'], weight_clip=y['weight_clip'],
    seed_base=y['rng_seed']
)

print("\n>>> USED CORE PARAMS (y)")
for k in y: print(f"  {k:>10s} : {y[k]}")
