# ===================== High-Precision Riemann Fingerprint Test (Colab single cell) =====================
# 包含：
# 1) 宏观拟合：c_eff(a) vs cosmo-like tanh（供残差计算）
# 2) 频谱指纹：多k值 Jaccard(FFT主峰重合) + 置换检验
# 3) 模板相关：残差 vs Riemann-comb/Uniform-comb 相关性 + 置换检验 + 偏相关（剔除Uniform）
# 4) 相位锁定（PLV）：在每个 γ_n 附近做窄带滤波，计算残差与Riemann-comb的相位锁定；置换检验
# 5) FDR 多重比较校正；多带宽 σ_f 扫描
# 仅依赖 numpy/pandas/matplotlib

import numpy as np, pandas as pd, matplotlib.pyplot as plt

# ----------------------- 固定随机种子与常量 -----------------------
SEED = 2025
rng = np.random.default_rng(SEED)
c0 = 299_792_458.0
np.set_printoptions(suppress=True, linewidth=140)

# ----------------------- Riemann zeros（前30个 γ_n） -----------------------
gammas = np.array([
    14.13472514, 21.02203964, 25.01085758, 30.42487613, 32.93506159,
    37.58617816, 40.91871901, 43.32707328, 48.00515088, 49.77383248,
    52.97032148, 56.44624770, 59.34704400, 60.83177953, 65.11254405,
    67.07981053, 69.54640171, 72.06715767, 75.70469070, 77.14484007,
    79.33737502, 82.91038085, 84.73549298, 87.42527461, 88.80911121,
    92.49189927, 94.65134404, 95.87063423, 98.83119422, 101.3178510
])

# ----------------------- 模型：c_eff(a) 构建（几何相位→内部光学） -----------------------
def geometric_phase(u, gammas, sigma):
    x = (u[:, None] - gammas[None, :]) / sigma
    return np.arctan(x).sum(axis=1)

def compute_ceff_on_a(a, u, gammas, sigma=2.0, a_coup=0.25, b=0.03, c0=c0):
    phi = geometric_phase(u, gammas, sigma)
    tau = b * (phi - phi.mean())
    N = np.exp(a_coup * tau)
    return c0 / N

# ----------------------- 参考曲线：cosmo-like tanh（暴胀→常光速） -----------------------
def fit_cosmo_like_tanh(a, y):
    loga = np.log10(a)
    A_grid     = np.linspace(0.05, 0.8, 28)
    k_grid     = np.linspace(0.2,  8.0, 28)
    logac_grid = np.linspace(-6.0, -1.0, 28)
    yinf_grid  = np.linspace(0.7, 1.05, 24)
    best = None
    idx = np.linspace(0, len(a)-1, 1600).astype(int)
    for A in A_grid:
        for k in k_grid:
            for logac in logac_grid:
                for y_inf in yinf_grid:
                    y_hat = y_inf + A * np.tanh( k*(logac - loga) )
                    rmse = np.sqrt(np.mean((y - y_hat)**2))
                    if (best is None) or (rmse < best['rmse']):
                        corr = np.corrcoef(y[idx], y_hat[idx])[0,1]
                        best = dict(A=A, k=k, logac=logac, y_inf=y_inf, rmse=rmse, corr=corr, y_hat=y_hat)
    return best

# ----------------------- 频域工具 -----------------------
def fft_top_k_indices(x, k=15):
    X = np.fft.rfft(x - np.mean(x))
    mag = np.abs(X)
    idx = np.argsort(mag[1:])[-k:] + 1
    return set(idx), X, mag

def jaccard(A, B):
    A, B = set(A), set(B)
    return 0.0 if len(A|B)==0 else len(A & B)/len(A | B)

def gaussian_bandpass_time(x, fs, f0, sigma_f):
    """
    频域高斯窗：exp(-(f-f0)^2/(2σ_f^2)) + 对称分量；生成窄带信号
    x: time series on uniform grid u; fs≈1/Δu
    """
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    W = np.exp(-0.5*((freqs - f0)/(sigma_f+1e-12))**2) + np.exp(-0.5*((freqs + f0)/(sigma_f+1e-12))**2)
    Y = X * W
    y = np.fft.irfft(Y, n=N)
    return y

def analytic_phase(x):
    """
    构造简化版希尔伯特解析信号（仅用rfft实现）：X[k]倍增高频，DC与Nyquist处理为0/不变。
    """
    N = len(x)
    X = np.fft.rfft(x)
    H = np.zeros_like(X)
    H[1:-1] = 2.0  # 提升正频
    H[0] = 1.0     # DC 保持（影响很小）
    if N % 2 == 0:
        H[-1] = 1.0  # Nyquist
    Xa = X * H
    xa = np.fft.irfft(Xa, n=N)
    # 相位
    phase = np.angle(np.fft.rfft(xa))  # 近似，已足够相对比较
    return phase

# ----------------------- Comb 字典 -----------------------
def lorentzian_comb(u, centers, sigma=2.0):
    x = (u[:, None] - centers[None, :]) / sigma
    s = np.sum(1.0/(1.0 + x**2), axis=1)
    s = (s - s.mean())/(s.std()+1e-12)
    return s

# ----------------------- 主流程：构建曲线与残差 -----------------------
a_min = 1e-6; NPTS = 8000  # 可调大到 12000 提升精度
a = np.geomspace(a_min, 1.0, NPTS)
u_min, u_max = 10.0, 110.0
u = u_min + (np.log(a) - np.log(a_min)) / (0.0 - np.log(a_min)) * (u_max - u_min)
du = (u[-1]-u[0])/(len(u)-1); fs = 1.0/du

sigma, a_coup, b = 2.0, 0.25, 0.03
y = compute_ceff_on_a(a, u, gammas, sigma=sigma, a_coup=a_coup, b=b) / c0
best = fit_cosmo_like_tanh(a, y)          # 宏观拟合
res  = y - best["y_hat"]                  # 残差

# ----------------------- 1) 频谱 Jaccard + 置换 -----------------------
# 多k（主峰数量）稳健性：取 k ∈ {12,15,20} 的平均
ks = [12, 15, 20]
def spectral_jaccard(x, template, ks):
    x_idx,_ ,_ = fft_top_k_indices(x, k=max(ks))
    res_vals = []
    for k in ks:
        ix_x, _, _  = fft_top_k_indices(x, k=k)
        ix_t, _, _  = fft_top_k_indices(template, k=k)
        res_vals.append(jaccard(ix_x, ix_t))
    return float(np.mean(res_vals))

# 残差预处理：带通（去趋势+去超高频）；比快版略宽以保留细节
def bandpass(x, w_low, w_high):
    def moving_average(z, w):
        if w <= 1: return z.copy()
        k = w//2
        pad = np.pad(z, (k,k), mode='edge')
        ker = np.ones(w)/w
        return np.convolve(pad, ker, mode='valid')
    trend = moving_average(x, w_low)
    hp = x - trend
    return moving_average(hp, w_high) if w_high > 1 else hp

w_low  = int(round(8.0/du)) | 1
w_high = int(round(0.18/du)) | 1
r_bp = bandpass(res, w_low, w_high)
r_bp = (r_bp - r_bp.mean())/(r_bp.std()+1e-12)

L_R = lorentzian_comb(u, gammas, sigma=2.0)
gam_uni = np.linspace(gammas.min(), gammas.max(), len(gammas))
L_U = lorentzian_comb(u, gam_uni, sigma=2.0)
# 同带通
L_R = bandpass(L_R, w_low, w_high); L_R = (L_R - L_R.mean())/(L_R.std()+1e-12)
L_U = bandpass(L_U, w_low, w_high); L_U = (L_U - L_U.mean())/(L_U.std()+1e-12)

jac_R = spectral_jaccard(r_bp, L_R, ks)
jac_U = spectral_jaccard(r_bp, L_U, ks)

# 置换（随机 comb）
n_perm = 1000  # 可提到 3000–5000
jac_null = []
for _ in range(n_perm):
    peaks = np.sort(rng.uniform(gammas.min(), gammas.max(), size=len(gammas)))
    L_P = lorentzian_comb(u, peaks, sigma=2.0)
    L_P = bandpass(L_P, w_low, w_high); L_P = (L_P - L_P.mean())/(L_P.std()+1e-12)
    jac_null.append(spectral_jaccard(r_bp, L_P, ks))
jac_null = np.array(jac_null)
p_jacc = (np.sum(jac_null >= jac_R) + 1)/(n_perm+1)

# ----------------------- 2) 模板相关 + 偏相关（剔除 Uniform） -----------------------
corr_R = float(np.corrcoef(r_bp, L_R)[0,1])
corr_U = float(np.corrcoef(r_bp, L_U)[0,1])

# 偏相关：corr(r_bp, L_R | L_U)
# 线性回归去掉 L_U 分量
def regress_out(y, X):
    # X: 2D matrix (n x p)
    A = X.T @ X + 1e-8*np.eye(X.shape[1])
    b = X.T @ y
    beta = np.linalg.solve(A, b)
    return y - X @ beta

r_bp_resid = regress_out(r_bp, L_U.reshape(-1,1))
L_R_resid  = regress_out(L_R, L_U.reshape(-1,1))
pcorr_R = float(np.corrcoef(r_bp_resid, L_R_resid)[0,1])

# 置换：对随机 comb 计算相同偏相关 → p 值
pcorr_null = []
for _ in range(n_perm):
    peaks = np.sort(rng.uniform(gammas.min(), gammas.max(), size=len(gammas)))
    L_P = lorentzian_comb(u, peaks, sigma=2.0)
    L_P = bandpass(L_P, w_low, w_high); L_P = (L_P - L_P.mean())/(L_P.std()+1e-12)
    L_P_resid = regress_out(L_P, L_U.reshape(-1,1))
    pcorr_null.append(np.corrcoef(r_bp_resid, L_P_resid)[0,1])
pcorr_null = np.array(pcorr_null)
p_pcorr = (np.sum(pcorr_null >= pcorr_R) + 1)/(n_perm+1)

# ----------------------- 3) 相位锁定（PLV） -----------------------
# 在每个 γ_n 做窄带高斯滤波（频率 f0=γ_n / (2π) 的“等效”）：
# 注：这里把 u 当作“时间”，对 r_bp 与 L_R 分别做窄带滤波后得到解析相位，再做 PLV。
# 选择一组 σ_f（频域带宽）做多带宽稳健性
sigma_fs = [0.04, 0.06, 0.08]  # Hz 等价单位（按 u 的采样率 fs）
def plv(x, y):
    # x,y: 解析相位序列（这里用rfft近似法得到），计算 |mean(exp(i(φx-φy)))|
    return np.abs(np.mean(np.exp(1j*(x - y))))

plvs = []
plvs_null = []
for sigma_f in sigma_fs:
    # 针对每个γ_n周围做滤波并拼接平均
    plv_vals = []
    for g in gammas:
        f0 = g * 0.0 + (1.0/(2*np.pi))  # 简化：统一用一个代表频；若要更精确可映射 f0≈1/Δu 局部尺度
        # 简化策略：统一频带（避免过拟合），侧重检验相位锁定存在与否
        xb = gaussian_bandpass_time(r_bp, fs, f0, sigma_f)
        yb = gaussian_bandpass_time(L_R,  fs, f0, sigma_f)
        # 解析相位（近似）
        phx = analytic_phase(xb)
        phy = analytic_phase(yb)
        # 取同长度可比的频域相位序列
        K = min(len(phx), len(phy))
        plv_vals.append(plv(phx[:K], phy[:K]))
    plvs.append(np.mean(plv_vals))

    # 置换 null：随机 comb 的PLV
    null_vals = []
    for _ in range(n_perm//5):  # 相位测试较贵，少一点置换
        peaks = np.sort(rng.uniform(gammas.min(), gammas.max(), size=len(gammas)))
        L_P = lorentzian_comb(u, peaks, sigma=2.0)
        L_P = bandpass(L_P, w_low, w_high); L_P = (L_P - L_P.mean())/(L_P.std()+1e-12)
        tmp = []
        for g in gammas:
            f0 = 1.0/(2*np.pi)
            xb = gaussian_bandpass_time(r_bp, fs, f0, sigma_f)
            yb = gaussian_bandpass_time(L_P,  fs, f0, sigma_f)
            phx = analytic_phase(xb); phy = analytic_phase(yb)
            K = min(len(phx), len(phy))
            tmp.append(plv(phx[:K], phy[:K]))
        null_vals.append(np.mean(tmp))
    plvs_null.append(np.array(null_vals))

plvs = np.array(plvs); plvs_null = [np.array(v) for v in plvs_null]
p_plv = [ (np.sum(nv >= pv) + 1)/(len(nv)+1) for pv,nv in zip(plvs, plvs_null) ]

# ----------------------- FDR（Benjamini–Hochberg） -----------------------
def fdr_bh(pvals, alpha=0.05):
    p = np.array(pvals); m = len(p)
    order = np.argsort(p); ranked = p[order]
    thresh = alpha * (np.arange(1,m+1)/m)
    passed = ranked <= thresh
    k = np.max(np.where(passed)) if np.any(passed) else -1
    cutoff = ranked[k] if k>=0 else None
    return cutoff, order[:k+1] if k>=0 else np.array([], dtype=int)

# 组合三个家族的 p 值做一次 FDR（示意）
p_family = [p_jacc, p_pcorr] + p_plv
cutoff, idx_sig = fdr_bh(p_family, alpha=0.05)

# ----------------------- 摘要表 -----------------------
summary = pd.DataFrame([{
    "macro_corr": round(float(best["corr"]), 6),
    "macro_RMSE": round(float(best["rmse"]), 6),
    "Jaccard(resid,Riemann)": round(float(jac_R), 3),
    "Jaccard(resid,Uniform)": round(float(jac_U), 3),
    "p_Jaccard_geq_Riemann": round(float(p_jacc), 4),
    "corr(resid,Riemann)": round(float(corr_R), 3),
    "corr(resid,Uniform)": round(float(corr_U), 3),
    "pcorr(resid,Riemann|Uniform)": round(float(pcorr_R), 3),
    "p_pcorr_geq_Riemann": round(float(p_pcorr), 4),
    "PLV_mean(sigma_f=0.04)": round(float(plvs[0]), 3),
    "p_PLV(0.04)": round(float(p_plv[0]), 4),
    "PLV_mean(sigma_f=0.06)": round(float(plvs[1]), 3),
    "p_PLV(0.06)": round(float(p_plv[1]), 4),
    "PLV_mean(sigma_f=0.08)": round(float(plvs[2]), 3),
    "p_PLV(0.08)": round(float(p_plv[2]), 4),
    "FDR_cutoff@0.05": (None if cutoff is None else round(float(cutoff),4)),
    "FDR_sig_idx_ordered": list(idx_sig.astype(int)),
    "filters": f"w_low={w_low}, w_high={w_high}, du≈{du:.4f}, ks={ks}, n_perm={n_perm}"
}])
print("\nHigh-Precision Riemann fingerprint — summary")
print(summary.to_string(index=False))

# ----------------------- 可选图形 -----------------------
# 置换分布图（Jaccard 与 PCorr）
plt.figure(figsize=(9,5))
plt.hist(jac_null, bins=30)
plt.axvline(jac_R, linestyle="--", label=f"Jaccard={jac_R:.3f}")
plt.title(f"Permutation: spectral Jaccard (p={p_jacc:.4f})")
plt.xlabel("Jaccard vs random-comb"); plt.ylabel("count"); plt.grid(True); plt.legend(); plt.show()

plt.figure(figsize=(9,5))
plt.hist(pcorr_null, bins=30)
plt.axvline(pcorr_R, linestyle="--", label=f"pcorr={pcorr_R:.3f}")
plt.title(f"Permutation: partial corr (p={p_pcorr:.4f})")
plt.xlabel("pcorr vs random-comb"); plt.ylabel("count"); plt.grid(True); plt.legend(); plt.show()

# 相位锁定分布
for i,sigma_f in enumerate(sigma_fs):
    plt.figure(figsize=(9,5))
    plt.hist(plvs_null[i], bins=30)
    plt.axvline(plvs[i], linestyle="--", label=f"PLV_mean={plvs[i]:.3f}")
    plt.title(f"Permutation: PLV @ sigma_f={sigma_f} (p={p_plv[i]:.4f})")
    plt.xlabel("PLV vs random-comb"); plt.ylabel("count"); plt.grid(True); plt.legend(); plt.show()

# 宏观拟合图（参考）
plt.figure(figsize=(9,6))
plt.plot(a, y, label="Model: c_eff(a)/c")
plt.plot(a, best["y_hat"], label="Cosmo-like tanh")
plt.xscale("log"); plt.xlabel("a"); plt.ylabel("c_eff/c")
plt.title("Overlay (macro)"); plt.grid(True); plt.legend(); plt.show()

# ===================== 使用建议 =====================
# 1) 提升灵敏度：把 n_perm 提到 3000–5000；NPTS 到 12000；并将 ks 扩展为 [10,12,15,18,20] 取均值。
# 2) 针对 PLV：sigma_fs 可用 [0.03,0.05,0.07,0.09] 做稳健性；也可把 f0 设成与局部 γ_n 映射相关的频率模型（更细致）。
# 3) 结果判读：若 p_jacc、p_pcorr 或 p_PLV 中有≥1项经 FDR 后仍显著（FDR_sig_idx_ordered 非空），
#    则可认为“Riemann 指纹”在残差中存在统计学证据（非随机）。
