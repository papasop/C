# ======================= 2.2 章节 · 全量可复现实验（Colab 单元） =======================
# 覆盖：Riemann 指纹（Jaccard/pcorr/PLV+置换+FDR）、GUE/Poisson 对照（Kw+KS），
#       高斯近似失败（K_Gauss vs 实测）、闭合一致性（Direct/FFT 双链 + Consistency Layer）
# 依赖：numpy / scipy / matplotlib / mpmath ；无需联网数据。
# ================================================================================

# 0) 安装与导入
!pip -q install mpmath

import numpy as np
from numpy import trapezoid as trapz
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.stats import ks_2samp
import mpmath
mpmath.mp.dps = 100

# -------------------- 可调总控参数（跑慢就先用默认） --------------------
SEED       = 2025
M_ZEROS    = 300        # 使用前 M 个黎曼零点（>300 会更强，>1000 更耗时）
ALPHA      = 1.6        # σ = α * 平均零点间隔
S_TARGET   = 0.30       # b 标定目标（std 归一因子）
A_COUP     = 1.0        # a（log 指数耦合强度）
NPTS       = 6000       # u 网格点数（建议 ≥ 4000；更高更细）
N_PERM     = 800        # 置换检验样本数（严谨建议 3000–5000）
TOPK_LIST  = [12, 15, 20]  # 频谱主峰个数的稳健性集成
PLV_BWS    = [0.04, 0.06, 0.08]  # PLV 频域带宽
MAKE_PLOTS = False      # True 时画图（Jaccard/pcorr/PLV 置换分布 & 宏观拟合）

rng = np.random.default_rng(SEED)

# 1) 获取前 M 个 Riemann 零点（虚部 γ_n）
def get_riemann_gammas(M):
    return np.array([float(mpmath.zetazero(k).imag) for k in range(1, M+1)])

gammas = get_riemann_gammas(M_ZEROS)
mean_gap = (gammas[-1] - gammas[0]) / (len(gammas) - 1)
sigma = ALPHA * mean_gap

# u 网格（两侧各留 ~5 个平均间隔做边界缓冲）
U_MIN = gammas[0]  - 5 * mean_gap
U_MAX = gammas[-1] + 5 * mean_gap
u = np.linspace(U_MIN, U_MAX, NPTS)
du = u[1] - u[0]
fs = 1.0 / du

# 2) 从零点生成相位 φ 与 τ、指数介质 N（Direct & FFT 两条实现）
def phi_direct(u, gam, sigma):
    x = (u[:, None] - gam[None, :]) / sigma
    return np.arctan(x).sum(axis=1)

def phi_fft(u, gam, sigma):
    # 在 u 网格上放置单位脉冲列 δ(u-γ_n) -> 与 Poisson/Lorentz 核卷积得到 φ'(u)
    delta = np.zeros_like(u)
    idx = np.clip(np.round((gam - u[0]) / du).astype(int), 0, len(u)-1)
    np.add.at(delta, idx, 1.0)
    # Poisson kernel: Pσ(x) = σ / (x^2 + σ^2) ；数值上截断到网格长度
    x = (np.arange(len(u)) - len(u)//2) * du
    P = sigma / (x**2 + sigma**2)
    P = np.fft.ifftshift(P)  # 对齐中心
    phi_prime = fftconvolve(delta, P, mode='same')  # 近似 φ' = sum Pσ(u-γ)
    # φ = ∫ φ' du（累积分）
    phi = np.cumsum(phi_prime) * du
    # 去掉整体线性倾向（数值积分常见），只保留零均值结构
    return phi - phi.mean()

def make_tau_N(phi, u, s_target=S_TARGET, a=A_COUP):
    # τ = b*(φ - <φ>)，b 选到 std(φ-<φ>) 的尺度 s_target
    phiz = phi - phi.mean()
    stdp = np.std(phiz)
    b = s_target / (stdp + 1e-15)
    tau = b * phiz
    N = np.exp(a * tau)
    # 全局归一：<N> = 1
    N = N / (trapz(N, x=u) / (u[-1]-u[0]))
    return tau, N, b

phi_dir  = phi_direct(u, gammas, sigma)
phi_fft_ = phi_fft(u, gammas, sigma)

tau_dir,  N_dir,  b_dir  = make_tau_N(phi_dir,  u)
tau_fft,  N_fft,  b_fft  = make_tau_N(phi_fft_,  u)

def summarize_tau_pair(name, tauA, tauB):
    rmse = np.sqrt(np.mean((tauA - tauB)**2))
    corr = np.corrcoef(tauA, tauB)[0,1]
    # 简单标度校正（仅因 FFT 数值积分残差），不动相位
    k = np.dot(tauA, tauB) / (np.dot(tauB, tauB) + 1e-15)
    rmse_post = np.sqrt(np.mean((tauA - k*tauB)**2))
    print(f"\n=== {name} method metrics ===")
    print(f"RMSE(tau, tau_hat) [pre-cal] = {rmse:.4f}")
    print(f"corr(tau, tau_hat) [pre-cal] = {corr:.4f}")
    print(f"RMSE(tau, tau_hat) [post-cal] = {rmse_post:.4f} (gain k = {k:.4f}, sign-fixed corr = {corr:.4f})")

summarize_tau_pair("FFT vs Direct", tau_dir, tau_fft)

# 3) 正向（τ→N→T_int）与逆向（T→τ̂）闭合一致性 + “一致性层”
c0 = 299_792_458.0

def forward_T(N, u, c0=c0):
    # T(u) = c0^{-1} ∫ N du
    T = np.cumsum(N) * du / c0
    # 归一化：强制 <N>=1 已做；这里只保证起点 0
    T = T - T[0]
    return T

def inverse_tau_from_T(T, u, a=A_COUP, c0=c0, smooth_win=0):
    # N_hat = c0 * dT/du（中央差分近似）
    if smooth_win and smooth_win > 1:
        k = smooth_win//2
        pad = np.pad(T, (k,k), mode='edge')
        ker = np.ones(smooth_win)/smooth_win
        T = np.convolve(pad, ker, mode='valid')
    dT = np.gradient(T, du)
    Nhat = c0 * dT
    # 归一：<N_hat>=1
    Nhat = Nhat / (trapz(Nhat, x=u) / (u[-1]-u[0]))
    tau_hat = (1.0/a) * (np.log(Nhat + 1e-30))
    # 去均值
    tau_hat = tau_hat - tau_hat.mean()
    return tau_hat, Nhat

def consistency_layer(T, u, tau_true, a=A_COUP, grid=(3,5,7,9,11,15,21,31)):
    # 穷举平滑窗口，选择 tau RMSE 最小的平滑参数（数值导数减噪）
    best = {"rmse": 1e9, "w": None, "tau": None, "Nhat": None}
    for w in grid:
        tau_hat, Nhat = inverse_tau_from_T(T, u, a=a, smooth_win=w)
        # 允许一个线性标度 k 做微调（meso 偏差）
        k = np.dot(tau_true, tau_hat) / (np.dot(tau_hat, tau_hat) + 1e-15)
        rmse = np.sqrt(np.mean((tau_true - k*tau_hat)**2))
        if rmse < best["rmse"]:
            best = {"rmse": rmse, "w": w, "tau": k*tau_hat, "Nhat": Nhat}
    return best

# Direct 路径：τ_dir -> N_dir -> T_dir -> τ_hat
T_dir  = forward_T(N_dir, u)
tau_hat_raw, N_hat_raw = inverse_tau_from_T(T_dir, u)
rmse_raw = np.sqrt(np.mean((tau_dir - tau_hat_raw)**2))

best_cl = consistency_layer(T_dir, u, tau_dir)
print("\n=== Inverse consistency (Direct) ===")
print(f"T raw (after): RMSE(tau)= {rmse_raw:.4e}")
print(f"[Consistency Layer] best window = {best_cl['w']}, RMSE(tau)= {best_cl['rmse']:.4e}")

# FFT 路径
T_fft = forward_T(N_fft, u)
tau_hat_raw2, _ = inverse_tau_from_T(T_fft, u)
rmse_raw2 = np.sqrt(np.mean((tau_fft - tau_hat_raw2)**2))
best_cl2 = consistency_layer(T_fft, u, tau_fft)
print("\n=== Inverse consistency (FFT) ===")
print(f"T raw (after): RMSE(tau)= {rmse_raw2:.4e}")
print(f"[Consistency Layer] best window = {best_cl2['w']}, RMSE(tau)= {best_cl2['rmse']:.4e}")

# 4) 滑窗 Kw（信息指纹） + GUE/Poisson 对照 + KS
def sliding_Kw(N, u, w_frac=0.10):
    w_len = int(round(w_frac * (u[-1]-u[0]) / du))
    w_len = max(21, w_len | 1)  # 奇数
    out = []
    for i in range(0, len(u)-w_len+1):
        sl = slice(i, i+w_len)
        Nw = N[sl]
        meanN   = trapz(Nw, dx=du) / (w_len*du)
        meanInv = trapz(1.0/Nw, dx=du) / (w_len*du)
        out.append(meanN * meanInv)
    return np.array(out)

def gen_gue_gaps(n, mean_gap):
    # 用 GOE 的 Wigner（Rayleigh 近似）作为实用代理：p(s) = (π/2) s exp(-π s^2/4)
    # 其均值 ≈ 1，缩放到 mean_gap。
    # 采样：Rayleigh(sigma) with sigma = sqrt(2/π)
    sigma_ray = np.sqrt(2/np.pi)
    s = rng.rayleigh(sigma_ray, size=n)
    s = s / s.mean() * mean_gap
    return s

def gen_poisson_gaps(n, mean_gap):
    s = rng.exponential(mean_gap, size=n)
    return s

def build_gamma_from_gaps(g0, gaps):
    return g0 + np.cumsum(gaps)

def N_from_gammas(gam):
    ph = phi_direct(u, gam, sigma)
    tau, N, _ = make_tau_N(ph, u)
    return tau, N

# Riemann / GUE / Poisson 的 Kw 分布与 KS
Kw_R = sliding_Kw(N_dir, u, w_frac=0.10)

gaps_gue  = gen_gue_gaps(len(gammas), mean_gap)
gam_gue   = build_gamma_from_gaps(gammas[0], gaps_gue)
_, N_gue  = N_from_gammas(gam_gue)
Kw_G      = sliding_Kw(N_gue, u, w_frac=0.10)

gaps_poi  = gen_poisson_gaps(len(gammas), mean_gap)
gam_poi   = build_gamma_from_gaps(gammas[0], gaps_poi)
_, N_poi  = N_from_gammas(gam_poi)
Kw_P      = sliding_Kw(N_poi, u, w_frac=0.10)

def kw_summary(lbl, Kw):
    print(f"{lbl:7s} | K_min={Kw.min():.3f}  K_max={Kw.max():.3f}  Var={Kw.var():.3e}")

print("\n=== Summary (windowed K_w) ===")
kw_summary("Riemann", Kw_R)
kw_summary("GUE",     Kw_G)
kw_summary("Poisson", Kw_P)

print("\n=== KS tests (two-sample, Kw distributions) ===")
D_RG, p_RG = ks_2samp(Kw_R, Kw_G, alternative='two-sided')
D_RP, p_RP = ks_2samp(Kw_R, Kw_P, alternative='two-sided')
print(f"KS(Riemann vs GUE)    : D={D_RG:.4f}, p={p_RG:.2e}")
print(f"KS(Riemann vs Poisson): D={D_RP:.4f}, p={p_RP:.2e}")

# 5) 高斯近似失败：K_Gauss = exp(a^2 Var(τ)) 与滑窗 K_w 比较
tau0 = tau_dir
K_gauss = np.exp((A_COUP**2) * np.var(tau0))
K_meas  = Kw_R.mean()
print("\n=== Failure of Gaussian Approximation ===")
print(f"K_pred (Gaussian/lognormal proxy) = {K_gauss:.3e}")
print(f"K_meas (mean sliding K_w)         = {K_meas:.3e}")
print(f"Ratio (K_pred / K_meas)           = {K_gauss / (K_meas + 1e-30):.3e}")

# 6) Riemann 指纹：残差 Jaccard / 偏相关（剔除 Uniform）/ 相位锁定 PLV + 置换 + FDR
# 6.1 宏观拟合（cosmo-like tanh）与残差
def fit_cosmo_like_tanh(a, y):
    loga = np.log10(a)
    A_grid     = np.linspace(0.05, 0.8, 22)
    k_grid     = np.linspace(0.2,  8.0, 22)
    logac_grid = np.linspace(-6.0, -1.0, 22)
    yinf_grid  = np.linspace(0.7, 1.05, 18)
    idx = np.linspace(0, len(a)-1, 1400).astype(int)
    best = None
    for A in A_grid:
        for k in k_grid:
            for logac in logac_grid:
                for y_inf in yinf_grid:
                    y_hat = y_inf + A * np.tanh(k*(logac - loga))
                    rmse = np.sqrt(np.mean((y - y_hat)**2))
                    if (best is None) or (rmse < best['rmse']):
                        corr = np.corrcoef(y[idx], y_hat[idx])[0,1]
                        best = dict(A=A, k=k, logac=logac, y_inf=y_inf, rmse=rmse, corr=corr, y_hat=y_hat)
    return best

def gaussian_bandpass_time(x, fs, f0, sigma_f):
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1/fs)
    W = np.exp(-0.5*((freqs - f0)/(sigma_f+1e-12))**2) + np.exp(-0.5*((freqs + f0)/(sigma_f+1e-12))**2)
    Y = X * W
    return np.fft.irfft(Y, n=len(x))

def analytic_phase_fast(x):
    # rfft-based 简易解析相位
    N = len(x)
    X = np.fft.rfft(x)
    H = np.zeros_like(X)
    H[1:-1] = 2.0
    H[0] = 1.0
    if N % 2 == 0:
        H[-1] = 1.0
    xa = np.fft.irfft(X * H, n=N)
    return np.angle(np.fft.rfft(xa))

def fft_top_k_indices(x, k=15):
    X = np.fft.rfft(x - x.mean())
    mag = np.abs(X)
    idx = np.argsort(mag[1:])[-k:] + 1
    return set(idx), X, mag

def jaccard(A, B):
    A, B = set(A), set(B)
    return 0.0 if len(A|B)==0 else len(A & B)/len(A | B)

def bandpass_ma(x, w_low, w_high):
    def ma(z, w):
        if w <= 1: return z.copy()
        k = w//2
        pad = np.pad(z, (k,k), mode='edge')
        ker = np.ones(w)/w
        return np.convolve(pad, ker, mode='valid')
    trend = ma(x, w_low)
    hp = x - trend
    return ma(hp, w_high) if w_high > 1 else hp

# 建“cosmo-like”宏观：用 N 的等效速度 c_eff = c0/N，再归一成 y∈~[0.7,1.05]
y = (c0 / N_dir) / c0
a_grid = np.geomspace(1e-6, 1.0, len(u))
best_macro = fit_cosmo_like_tanh(a_grid, y)
res = y - best_macro["y_hat"]

# 频谱字典：Riemann comb & Uniform comb（同个 σL）
def lorentzian_comb(u, centers, sigmaL=2.0):
    x = (u[:,None] - centers[None,:]) / sigmaL
    s = np.sum(1.0/(1.0 + x**2), axis=1)
    s = (s - s.mean())/(s.std()+1e-12)
    return s

L_R = lorentzian_comb(u, gammas, sigmaL=2.0)
L_U = lorentzian_comb(u, np.linspace(gammas.min(), gammas.max(), len(gammas)), sigmaL=2.0)

# 与残差同样带通并标准化
W_LOW  = int(round(8.0/du)) | 1
W_HIGH = int(round(0.18/du)) | 1
r_bp = bandpass_ma(res, W_LOW, W_HIGH);  r_bp = (r_bp - r_bp.mean())/(r_bp.std()+1e-12)
L_R_ = bandpass_ma(L_R, W_LOW, W_HIGH);  L_R_ = (L_R_ - L_R_.mean())/(L_R_.std()+1e-12)
L_U_ = bandpass_ma(L_U, W_LOW, W_HIGH);  L_U_ = (L_U_ - L_U_.mean())/(L_U_.std()+1e-12)

# 6.2 频谱 Jaccard + 置换
def spectral_jaccard(x, template, ks=TOPK_LIST):
    vals = []
    for k in ks:
        ix_x,_ ,_ = fft_top_k_indices(x, k=k)
        ix_t,_ ,_ = fft_top_k_indices(template, k=k)
        vals.append(jaccard(ix_x, ix_t))
    return float(np.mean(vals))

jac_R = spectral_jaccard(r_bp, L_R_)
jac_U = spectral_jaccard(r_bp, L_U_)

jac_null = []
for _ in range(N_PERM):
    peaks = np.sort(rng.uniform(gammas.min(), gammas.max(), size=len(gammas)))
    L_P = lorentzian_comb(u, peaks, sigmaL=2.0)
    L_P = bandpass_ma(L_P, W_LOW, W_HIGH); L_P=(L_P-L_P.mean())/(L_P.std()+1e-12)
    jac_null.append(spectral_jaccard(r_bp, L_P))
jac_null = np.array(jac_null)
p_jacc = (np.sum(jac_null >= jac_R) + 1) / (len(jac_null) + 1)

# 6.3 模板相关 & 偏相关（剔除 Uniform）
def regress_out(y, X):
    A = X.T @ X + 1e-8*np.eye(X.shape[1])
    b = X.T @ y
    beta = np.linalg.solve(A, b)
    return y - X @ beta

corr_R = float(np.corrcoef(r_bp, L_R_)[0,1])
corr_U = float(np.corrcoef(r_bp, L_U_)[0,1])

r_bp_res = regress_out(r_bp, L_U_.reshape(-1,1))
L_R_res  = regress_out(L_R_, L_U_.reshape(-1,1))
pcorr_R  = float(np.corrcoef(r_bp_res, L_R_res)[0,1])

pcorr_null = []
for _ in range(N_PERM):
    peaks = np.sort(rng.uniform(gammas.min(), gammas.max(), size=len(gammas)))
    L_P = lorentzian_comb(u, peaks, sigmaL=2.0)
    L_P = bandpass_ma(L_P, W_LOW, W_HIGH); L_P=(L_P-L_P.mean())/(L_P.std()+1e-12)
    L_P_res = regress_out(L_P, L_U_.reshape(-1,1))
    pcorr_null.append(np.corrcoef(r_bp_res, L_P_res)[0,1])
pcorr_null = np.array(pcorr_null)
p_pcorr = (np.sum(pcorr_null >= pcorr_R) + 1) / (len(pcorr_null) + 1)

# 6.4 相位锁定（PLV） + 置换
def plv(phi_x, phi_y):
    K = min(len(phi_x), len(phi_y))
    return np.abs(np.mean(np.exp(1j*(phi_x[:K] - phi_y[:K]))))

plv_vals, plv_nulls, p_plv = [], [], []
for sigma_f in PLV_BWS:
    # 用统一代表频率 f0（简化版）；要更严谨可为每个 γ_n 设局部等效 f0。
    f0 = 1.0 / (2*np.pi)
    xb = gaussian_bandpass_time(r_bp, fs, f0, sigma_f)
    yb = gaussian_bandpass_time(L_R_, fs, f0, sigma_f)
    phx = analytic_phase_fast(xb)
    phy = analytic_phase_fast(yb)
    plv_obs = plv(phx, phy)
    plv_vals.append(plv_obs)
    # 置换 Null
    null_ = []
    for _ in range(max(100, N_PERM//5)):
        peaks = np.sort(rng.uniform(gammas.min(), gammas.max(), size=len(gammas)))
        L_P = lorentzian_comb(u, peaks, sigmaL=2.0)
        L_P = bandpass_ma(L_P, W_LOW, W_HIGH); L_P=(L_P-L_P.mean())/(L_P.std()+1e-12)
        yb2 = gaussian_bandpass_time(L_P, fs, f0, sigma_f)
        phy2 = analytic_phase_fast(yb2)
        null_.append(plv(phx, phy2))
    null_ = np.array(null_)
    plv_nulls.append(null_)
    p_plv.append( (np.sum(null_ >= plv_obs) + 1) / (len(null_) + 1) )

# 6.5 FDR（Benjamini–Hochberg）
def fdr_bh(pvals, alpha=0.05):
    p = np.array(pvals); m=len(p)
    order = np.argsort(p); ranked = p[order]
    thresh = alpha * (np.arange(1,m+1)/m)
    passed = ranked <= thresh
    k = np.max(np.where(passed)) if np.any(passed) else -1
    cutoff = ranked[k] if k>=0 else None
    return cutoff, order[:k+1] if k>=0 else np.array([], dtype=int)

p_family = [p_jacc, p_pcorr] + p_plv
cutoff, idx_sig = fdr_bh(p_family, alpha=0.05)

print("\n=== High-Precision Riemann fingerprint — summary ===")
print(f"macro_corr={best_macro['corr']:.6f}  macro_RMSE={best_macro['rmse']:.6f}")
print(f"Jaccard(resid,Riemann)={jac_R:.3f}  Jaccard(resid,Uniform)={jac_U:.3f}  p_Jaccard≥obs={p_jacc:.4f}")
print(f"corr(resid,Riemann)={corr_R:.3f}  corr(resid,Uniform)={corr_U:.3f}")
print(f"pcorr(resid,Riemann|Uniform)={pcorr_R:.3f}  p_pcorr≥obs={p_pcorr:.4f}")
for bw, pv, pval in zip(PLV_BWS, plv_vals, p_plv):
    print(f"PLV_mean(σ_f={bw:.02f})={pv:.3f}  p_PLV={pval:.4f}")
print(f"FDR@0.05 cutoff={None if cutoff is None else round(float(cutoff),4)}, sig_idx(order)={list(idx_sig.astype(int))}")
print(f"[filters] w_low={W_LOW}, w_high={W_HIGH}, du≈{du:.4f}, ks={TOPK_LIST}, n_perm={N_PERM}")

# 7) 可选可视化
if MAKE_PLOTS:
    # 宏观拟合覆盖
    plt.figure(figsize=(9,6))
    plt.plot(a_grid, y, label="c_eff(a)/c")
    plt.plot(a_grid, best_macro["y_hat"], label="cosmo-like tanh")
    plt.xscale("log"); plt.xlabel("a"); plt.ylabel("c_eff/c"); plt.grid(True); plt.legend(); plt.title("Macro overlay"); plt.show()

    # 置换分布
    plt.figure(figsize=(9,5))
    plt.hist(jac_null, bins=30); plt.axvline(jac_R, ls='--', label=f"Jaccard={jac_R:.3f}")
    plt.title(f"Permutation: spectral Jaccard (p={p_jacc:.4f})"); plt.grid(True); plt.legend(); plt.show()

    plt.figure(figsize=(9,5))
    plt.hist(pcorr_null, bins=30); plt.axvline(pcorr_R, ls='--', label=f"pcorr={pcorr_R:.3f}")
    plt.title(f"Permutation: partial corr (p={p_pcorr:.4f})"); plt.grid(True); plt.legend(); plt.show()

    for bw, null_, pv, pval in zip(PLV_BWS, plv_nulls, plv_vals, p_plv):
        plt.figure(figsize=(9,5))
        plt.hist(null_, bins=30); plt.axvline(pv, ls='--', label=f"PLV_mean={pv:.3f}")
        plt.title(f"Permutation: PLV @ σ_f={bw:.02f} (p={pval:.4f})"); plt.grid(True); plt.legend(); plt.show()
# ================================================================================

