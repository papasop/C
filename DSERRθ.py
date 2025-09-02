# zeta_phase_angle_domain_fullbatch.py
# 真实 ζ 零点 + Chebyshev 角域 + T(p) 射线积分 + FW_mid/high
# 批量统计 & “默认 vs 激进”配对差 + Holm–Bonferroni

import numpy as np, math

# ===================== 配置（可改） =====================
CFG_DEFAULT = dict(
    # 零点/格子/介质
    M=48,                 # ζ 零点数量
    alpha=1.6,            # σ = α Δ
    beta=1.0,             # ln N = β·stdz(ddev)
    crop_frac=0.65,       # 中心窗裁剪
    z_span=1.0,           # γ 映射的总跨度（相对单位）
    Nz=900,               # z 采样点数
    # π-surrogate（A.2）
    pburst=0.20, pgeom=0.30, pmix=0.50, lognorm_scale=0.40, pi_shape=0.30,
    # 角域射线
    n_rays=22, p_margin=0.96, p_pow=1.8,
    # 成对噪声与权重
    noise_rel=0.01, weight_mode="up", weight_clip=3.0,
    # FW 指标（A.8）
    fw_mid=(0.15, 0.60, 1.0),
    fw_high=(0.25, 0.95, 2.0),
)

# “激进 knobs”（A.11 风格：放大 HF 粗糙度 & 近临界敏感）
CFG_AGGRESSIVE = dict(CFG_DEFAULT)
CFG_AGGRESSIVE.update({
    "alpha": 1.2,
    "pburst": 0.45,
    "pi_shape": 0.15,
    "lognorm_scale": 0.80,
    "n_rays": 32,
    "p_pow": 2.2,
    "p_margin": 0.99,
})

# ===================== 真实 ζ 零点（mpmath 优先，内置表后备） =====================
def get_zeta_zeros(M):
    try:
        import mpmath as mp
        mp.mp.dps = 50
        return np.array([float(mp.zetazero(n)) for n in range(1, M+1)], dtype=float)
    except Exception:
        fallback = [
            14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
            52.970321, 56.446248, 59.347044, 60.831779, 65.112544, 67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
            79.337375, 82.910381, 84.735493, 87.425275, 88.809111, 92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
            103.725539, 105.446623, 107.168612, 111.029536, 111.874659, 114.320221, 116.226680, 118.790783, 121.370130, 122.946829,
            124.256818, 127.516684, 129.578704, 131.087689, 133.497737, 134.756508, 138.116042, 139.736208, 141.123707, 143.111845,
            146.000983, 147.422765, 150.053520, 150.925257, 153.024694, 156.112908, 157.597591, 158.849988
        ]
        if M > len(fallback):
            raise RuntimeError("后备零点列表不足；请安装 mpmath 以获取更多 ζ 零点。")
        return np.array(fallback[:M], dtype=float)

# ===================== A.1：相位 → 折射率 =====================
def build_N_from_gammas(gammas, z_min, z_max, Nz, alpha, beta, crop_frac):
    z = np.linspace(z_min, z_max, Nz)
    M = len(gammas)
    Delta = (gammas[-1] - gammas[0])/(M-1)
    sigma = alpha * Delta
    phi = np.zeros_like(z)
    for gn in gammas:
        phi += np.arctan((z - gn) / sigma)
    ddev = phi - np.mean(phi)
    dstd = ddev / (np.std(ddev) + 1e-12)
    lnN = beta * dstd
    N = np.exp(lnN)
    # 中心窗裁剪
    L = z_max - z_min; mid = 0.5*(z_min+z_max); half = 0.5*crop_frac*L
    mask = (z >= mid-half) & (z <= mid+half)
    return z[mask], N[mask]

# ===================== A.2：π-surrogate gaps 混合 =====================
def make_pi_surrogate_gammas(M, Delta, start, *, pburst, pgeom, pmix, lognorm_scale, pi_shape, rng):
    gaps=[]; n=0
    while n < M-1:
        if rng.random() < pburst:
            L = 1 + rng.geometric(pgeom)
            for _ in range(min(L, M-1-n)):
                g = rng.uniform(0.0, 0.6*Delta)
                gaps.append(g); n += 1
                if n >= M-1: break
        else:
            if rng.random() < pmix:
                g = Delta * math.exp(rng.normal(0.0, lognorm_scale))
            else:
                g = rng.gamma(pi_shape, Delta/max(pi_shape, 1e-9))
            gaps.append(np.clip(g, 0.1*Delta, 3.0*Delta)); n += 1
    gaps = np.array(gaps)
    scale = (Delta*(M-1))/(gaps.sum()+1e-18)
    gaps *= scale
    gammas = np.empty(M); gammas[0] = start; gammas[1:] = start + np.cumsum(gaps)
    return gammas

# ===================== A.3：Chebyshev 角域 & 射线 T(p) =====================
def cheb_p_list(n_rays, pmin, pmax, p_pow=1.8):
    j = np.arange(n_rays)
    x = -np.cos((j+0.5)*math.pi/n_rays)   # Chebyshev in [-1,1]
    u = 0.5*(x+1.0)                       # -> [0,1]
    u = u**p_pow                          # 近 pmax 增密
    return pmin + u*(pmax - pmin)

def ray_time_T(z, N, p_list, margin):
    Nmin = float(np.min(N))
    pmax_feasible = margin * Nmin
    feasible = p_list < pmax_feasible
    T = np.full_like(p_list, np.nan, dtype=float)
    N2 = N**2
    for i,p in enumerate(p_list):
        if not feasible[i]: continue
        denom = N2 - p**2
        mask = denom > 0.0
        integrand = np.zeros_like(N)
        integrand[mask] = N2[mask] / np.sqrt(denom[mask])
        T[i] = np.trapezoid(integrand, z)
    return T, feasible

# ===================== A.4：成对噪声 & 权重 =====================
def paired_noise(T_ref, noise_rel, rng):
    Tscale = np.median(np.abs(T_ref[~np.isnan(T_ref)]))
    sigma = noise_rel * Tscale
    return rng.normal(0.0, sigma, size=T_ref.shape), Tscale

def make_weights(p_list, mode="up", clip=3.0):
    if mode == "none": return np.ones_like(p_list)
    p = p_list.copy()
    p -= np.nanmin(p); p /= (np.nanmax(p)+1e-12)
    w = 1.0 + 2.0*(p**2)
    return np.clip(w, 1.0, clip)

# ===================== A.8：FW_mid / FW_high =====================
def fw_band_metric(e_vec, band, denom_ref=None, eps=1e-12):
    e = e_vec - np.nanmean(e_vec)
    F = np.fft.rfft(e)
    P = np.abs(F)**2
    N = len(e_vec)
    k0f,k1f,pw = band
    kmax = len(P) - 1
    k0 = int(max(1, round(k0f*kmax)))
    k1 = int(min(kmax, round(k1f*kmax)))
    ks = np.arange(len(P))
    w = (ks/(N-1+1e-12))**pw
    num = np.sum(w[k0:k1+1] * P[k0:k1+1])
    if denom_ref is None:
        den = np.sum(P) + eps
    else:
        Fref = np.fft.rfft(denom_ref - np.mean(denom_ref))
        den = np.sum(np.abs(Fref)**2) + eps
    return math.sqrt(num/den)

# ===================== 单次前向：角域指标 =====================
def forward_once(cfg, rng_seed=11000):
    rng = np.random.default_rng(rng_seed)

    gamma_zeta = get_zeta_zeros(cfg["M"])
    Delta = (gamma_zeta[-1]-gamma_zeta[0])/(cfg["M"]-1)
    gamma_pi = make_pi_surrogate_gammas(
        cfg["M"], Delta, start=gamma_zeta[0],
        pburst=cfg["pburst"], pgeom=cfg["pgeom"], pmix=cfg["pmix"],
        lognorm_scale=cfg["lognorm_scale"], pi_shape=cfg["pi_shape"], rng=rng
    )

    z_min, z_max = 0.0, cfg["z_span"]
    z_z, N_z = build_N_from_gammas(gamma_zeta, z_min, z_max, cfg["Nz"], cfg["alpha"], cfg["beta"], cfg["crop_frac"])
    z_p, N_p = build_N_from_gammas(gamma_pi,   z_min, z_max, cfg["Nz"], cfg["alpha"], cfg["beta"], cfg["crop_frac"])
    # 对齐
    L = min(len(z_z), len(z_p))
    z = z_z[:L]; N_z = N_z[:L]; N_p = N_p[:L]

    Nmin_common = min(float(N_z.min()), float(N_p.min()))
    pmin = 0.10*Nmin_common
    pmax = cfg["p_margin"]*Nmin_common
    p_list = cheb_p_list(cfg["n_rays"], pmin, pmax, cfg["p_pow"])

    Tz, fz = ray_time_T(z, N_z, p_list, cfg["p_margin"])
    Tp, fp = ray_time_T(z, N_p, p_list, cfg["p_margin"])
    mask = fz & fp & np.isfinite(Tz) & np.isfinite(Tp)
    keep = np.count_nonzero(mask)/len(p_list)

    noise, Tscale = paired_noise(Tz[mask], cfg["noise_rel"], rng)
    Tz_obs = Tz[mask] + noise
    Tp_obs = Tp[mask] + noise

    w = make_weights(p_list[mask], cfg["weight_mode"], cfg["weight_clip"])
    def wnorm(x): return math.sqrt(np.sum((w*x)**2))
    DSERR_theta = wnorm(Tp_obs - Tz_obs) / (wnorm(Tz_obs) + 1e-18)

    e_theta = (Tp_obs - Tz_obs) - np.mean(Tp_obs - Tz_obs)
    FWmid_theta  = fw_band_metric(e_theta, cfg["fw_mid"])
    FWhigh_theta = fw_band_metric(e_theta, cfg["fw_high"])

    e_space = (N_p - N_z) - np.mean(N_p - N_z)
    FWmid_space  = fw_band_metric(e_space, cfg["fw_mid"])
    FWhigh_space = fw_band_metric(e_space, cfg["fw_high"])

    return dict(
        keep=keep, pmin=pmin, pmax=pmax, Nmin_common=Nmin_common, Tscale=Tscale,
        DSERR_theta=DSERR_theta,
        FWmid_theta=FWmid_theta, FWhigh_theta=FWhigh_theta,
        FWmid_space=FWmid_space, FWhigh_space=FWhigh_space
    )

# ===================== 批量：分布与“默认 vs 激进”的配对差 =====================
def run_batch(cfg, N_RUNS=48, base_seed=11000):
    outs=[]; 
    for i in range(N_RUNS):
        outs.append(forward_once(cfg, rng_seed=base_seed+i))
    return outs

def summarize_batch(outs, label="BATCH"):
    def arr(key): return np.array([o[key] for o in outs], dtype=float)
    keep_mean = float(np.mean(arr("keep")))
    print(f"\n=== {label} (N={len(outs)}, keep≈{keep_mean:.2f}) ===")
    for k in ["DSERR_theta","FWmid_theta","FWhigh_theta","FWmid_space","FWhigh_space"]:
        a = arr(k)
        mean = a.mean(); sd = a.std(ddof=1)
        lo, hi = np.quantile(a,[0.025,0.975])
        print(f"{k:>14}: mean={mean:.4e}, sd={sd:.4e}, 95% PI[{lo:.4e},{hi:.4e}]")
    return {k: arr(k) for k in ["DSERR_theta","FWmid_theta","FWhigh_theta","FWmid_space","FWhigh_space"]}

def paired_stats(d, alpha=0.05):
    n = len(d); m = float(np.mean(d)); s = float(np.std(d, ddof=1))
    t = m / (s/np.sqrt(n) + 1e-18)
    # SciPy 优先
    try:
        from scipy import stats
        p_t = 2*stats.t.sf(abs(t), df=n-1)
        tcrit = stats.t.ppf(1-alpha/2, df=n-1)
        w, p_w = stats.wilcoxon(d, zero_method="wilcox", correction=False, alternative="two-sided", method="auto")
    except Exception:
        import math
        p_t = 2*(1-0.5*(1+math.erf(abs(t)/math.sqrt(2))))
        tcrit = 1.96
        # 退化为符号检验近似
        sgn = np.sum(d>0); mu, var = 0.5*n, 0.25*n
        z = (sgn-mu)/math.sqrt(var+1e-18)
        p_w = 2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))
    half = tcrit * s/np.sqrt(n + 1e-18)
    ci = (m - half, m + half)
    dz = m / (s + 1e-18)  # Cohen's dz
    return dict(mean=m, t=t, p_t=p_t, p_w=p_w, ci=ci, dz=dz)

def holm_bonferroni(pvals, alpha=0.05, names=None):
    order = np.argsort(pvals)
    results=[]
    for rank,i in enumerate(order, start=1):
        thr = alpha / (len(pvals) - rank + 1)
        results.append((i, pvals[i], thr, pvals[i] <= thr))
    if names:
        for i,p,thr,ok in results:
            print(f"  {names[i]:<12}: p={p:.3g}, threshold={thr:.3g}  -> {'significant' if ok else 'ns'}")
    return results

def paired_compare(default_stats, aggressive_stats, names=("ΔDSERRθ","ΔFW_midθ","ΔFW_highθ")):
    d_DS = aggressive_stats["DSERR_theta"] - default_stats["DSERR_theta"]
    d_FM = aggressive_stats["FWmid_theta"]  - default_stats["FWmid_theta"]
    d_FH = aggressive_stats["FWhigh_theta"] - default_stats["FWhigh_theta"]
    diffs = [d_DS, d_FM, d_FH]
    print("\n[Paired differences] aggressive - default")
    pvals=[]
    for d,name in zip(diffs, names):
        st = paired_stats(d)
        print(f"{name:<10}: Δmean={st['mean']:.3e}  95%CI[{st['ci'][0]:.3e},{st['ci'][1]:.3e}]  "
              f"t={st['t']:.2f}  p_t={st['p_t']:.3g} ; p_W={st['p_w']:.3g}  | ES dz={st['dz']:.3f}")
        pvals.append(st["p_w"])  # 用 Wilcoxon 做 Holm
    print("\n[Holm-Bonferroni over 3 tests]")
    holm_bonferroni(np.array(pvals), alpha=0.05, names=list(names))

# ===================== 主入口 =====================
if __name__ == "__main__":
    BASE_SEED = 11000
    N_RUNS = 48

    # 单次示例（默认）
    print("=== ANGLE-DOMAIN FORWARD SUMMARY (single run, DEFAULT knobs) ===")
    one = forward_once(CFG_DEFAULT, rng_seed=BASE_SEED)
    print(f"M={CFG_DEFAULT['M']}, Nz={CFG_DEFAULT['Nz']}, n_rays={CFG_DEFAULT['n_rays']}, keep≈{one['keep']:.2f}")
    print(f"p range: [{one['pmin']:.4f}, {one['pmax']:.4f}]  (min N common={one['Nmin_common']:.4f})")
    print(f"T scale (median|T|): {one['Tscale']:.4e}, noise_rel={CFG_DEFAULT['noise_rel']:.3f}")
    print(f"DSERR_theta (||Tπ-Tζ||/||Tζ||, weighted): {one['DSERR_theta']:.4e}")
    print(f"FW_mid_theta:  {one['FWmid_theta']:.4e}   (band={CFG_DEFAULT['fw_mid']})")
    print(f"FW_high_theta: {one['FWhigh_theta']:.4e}   (band={CFG_DEFAULT['fw_high']})")

    # 批量：默认
    outs_def = run_batch(CFG_DEFAULT, N_RUNS=N_RUNS, base_seed=BASE_SEED)
    stats_def = summarize_batch(outs_def, label="BATCH (default)")

    # 批量：激进
    outs_agg = run_batch(CFG_AGGRESSIVE, N_RUNS=N_RUNS, base_seed=BASE_SEED)
    stats_agg = summarize_batch(outs_agg, label="BATCH (aggressive)")

    # 配对差 + Holm–Bonferroni
    paired_compare(stats_def, stats_agg)
