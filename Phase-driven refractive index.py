# === Riemann-phase pipeline (fixed derivative bias + real sensitivity) ===
import numpy as np, math, time, warnings, csv
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
from scipy.interpolate import UnivariateSpline
from scipy.signal import coherence
from scipy.linalg import lstsq
from scipy.integrate import simpson

warnings.filterwarnings('ignore')
np.set_printoptions(precision=6, suppress=True)

# ---------------- Global toggles ----------------
SEED            = 42
PLOT_FIGS       = True
RUN_DIAGNOSTICS = True
RUN_NOISE_SCAN  = True
RUN_N_EXTRAP    = True
EXPORT_CSV      = True

# ===== 默认值（建议先用这组） =====
DEFAULT_DERIV_MODE     = "fft"   # "fft" | "spline" | "fd5"
DEFAULT_T_SMOOTH_GAIN  = 2.0     # 仅对 spline 导数器有效：s_T = gain * M * Var(T)
DEFAULT_FFT_CUTOFF     = 0.30    # 仅对 fft 导数器有效：低通截止占 Nyquist 比例 ∈ (0,1)
DEFAULT_SX_GAIN        = 2.0     # log(N_hat) 样条平滑强度；=0 表示关闭
DEFAULT_SX_USE_VAR     = True    # True: sX = gain * M * Var(X)；False: sX = 常数 gain

# ---------------- Configs ----------------
@dataclass
class ForwardConfig:
    M: int = 6000
    u_min: float = -50.0
    u_max: float =  50.0
    Nzeros: int = 1200
    gamma_spacing: float = 1.0
    jitter: float = 0.05
    sigma: float = 1.0
    b: float = 1.0
    a: float = 0.30
    c0: float = 1.0
    noise_T_std: float = 2e-14

@dataclass
class InverseConfig:
    deriv_mode: str = DEFAULT_DERIV_MODE          # "fft" | "spline" | "fd5"
    T_smooth_gain: float = DEFAULT_T_SMOOTH_GAIN  # for spline-deriv
    fft_cutoff_ratio: float = DEFAULT_FFT_CUTOFF  # for fft-deriv
    spline_k: int = 3
    enforce_meanN: bool = True
    log_clip: float = 40.0
    sX_gain: float = DEFAULT_SX_GAIN              # log-domain smoothing strength
    sX_use_variance: bool = DEFAULT_SX_USE_VAR    # True: gain * M * Var(X) ; False: constant sX_gain

# ---------------- Helpers ----------------
def rmse(a, b) -> float:
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a-b)**2)))

def nrmse(a, b) -> float:
    a = np.asarray(a); b = np.asarray(b)
    return float(rmse(a,b) / (np.max(a)-np.min(a) + 1e-18))

def center_int_mean(x, u):
    return x - (np.trapz(x, x=u) / (u[-1]-u[0]))

def calibrate_gain_on_tau(u, tau_true, tau_hat, a, N_hat):
    """积分均值对齐后线性回归求 k，并重建 N_hat；保持 <N>=1。"""
    tau_c  = center_int_mean(tau_true, u)
    tauh_c = center_int_mean(tau_hat,  u)
    num = np.trapz(tau_c * tauh_c, x=u)
    den = np.trapz(tauh_c * tauh_c, x=u) + 1e-18
    k = float(num / den)
    tau_hat_cal = k * tau_hat
    N_hat_cal   = np.exp(a * tau_hat_cal)
    meanN = np.trapz(N_hat_cal, x=u) / (u[-1]-u[0])
    if meanN != 0:
        N_hat_cal /= meanN
    return tau_hat_cal, N_hat_cal, k

# ---------------- Phase φ from arctan lattice ----------------
def phi_arctan_direct(u, gamma, sigma):
    UU = u[:, None]; GG = gamma[None, :]
    return np.arctan2(UU - GG, sigma).sum(axis=1)

def _linear_bin_gamma_to_grid(gamma, u0, du, M):
    pos = (gamma - u0) / du
    i = np.floor(pos).astype(int)
    w = pos - i
    counts = np.zeros(M, dtype=np.float64)
    m0 = (i >= 0) & (i < M);         np.add.at(counts, i[m0], 1.0 - w[m0])
    m1 = (i+1 >= 0) & (i+1 < M);     np.add.at(counts, i[m1]+1, w[m1])
    return counts

class KernelCache:
    def __init__(self): self.key=None; self.K=None; self.L=None; self.M=None
KC = KernelCache()

def _prepare_kernel(u, sigma, pad_factor=2):
    M = len(u); du = float(u[1]-u[0])
    n = np.arange(M); x = (n - M//2) * du
    k = np.arctan(x / sigma)
    L = 1 << int(np.ceil(np.log2(pad_factor * M)))
    K = np.fft.rfft(np.pad(np.roll(k, M//2), (0, L - M)))
    return K, L, M

def phi_arctan_fft_cached(u, gamma, sigma, pad_factor=2):
    key = (len(u), float(u[0]), float(u[1]-u[0]), float(sigma), int(pad_factor))
    if KC.key != key:
        KC.K, KC.L, KC.M = _prepare_kernel(u, sigma, pad_factor)
        KC.key = key
    counts = _linear_bin_gamma_to_grid(gamma, u[0], float(u[1]-u[0]), KC.M)
    C = np.fft.rfft(np.pad(counts, (0, KC.L - KC.M)))
    conv = np.fft.irfft(C * KC.K)[:KC.L]
    return conv[:KC.M]

def phi_arctan_auto(u, gamma, sigma, pad_factor=2, threshold=8e6):
    M = len(u); N = len(gamma)
    if M * N <= threshold:
        return phi_arctan_direct(u, gamma, sigma), 'Direct'
    else:
        return phi_arctan_fft_cached(u, gamma, sigma, pad_factor=pad_factor), 'FFT'

# ---------------- Forward：构造 tau / N / T ----------------
def make_synthetic_forward(cfg: ForwardConfig, seed: int = 0,
                           mode: str = 'auto', pad_factor: int = 2,
                           timeit: bool = True, noise_scale: float = 1.0):
    rng = np.random.default_rng(seed)
    u = np.linspace(cfg.u_min, cfg.u_max, cfg.M)
    gamma = np.arange(cfg.Nzeros)*cfg.gamma_spacing + rng.normal(0, cfg.jitter, cfg.Nzeros)

    t0 = time.perf_counter()
    if mode == 'direct':
        phi = phi_arctan_direct(u, gamma, cfg.sigma); method = 'Direct'
    elif mode == 'fft':
        phi = phi_arctan_fft_cached(u, gamma, cfg.sigma, pad_factor=pad_factor); method = 'FFT'
    else:
        phi, method = phi_arctan_auto(u, gamma, cfg.sigma, pad_factor=pad_factor)
    t1 = time.perf_counter()

    # 相位标准化 -> tau0
    phi_mean = np.trapz(phi, u) / (u[-1]-u[0])
    phi_c = phi - phi_mean
    phi_std = np.std(phi_c) + 1e-18
    tau0 = cfg.b * (phi_c / phi_std)

    # N: 以 tau0 为原型，但强制 <N>=1
    logN_raw = cfg.a * tau0
    logN_raw = np.clip(logN_raw, -40.0, 40.0)
    N_raw = np.exp(logN_raw)
    meanN = np.trapz(N_raw, x=u) / (u[-1]-u[0])
    N = N_raw / (meanN + 1e-18)

    # 真值 tau 与 N 自洽
    tau = (1.0/cfg.a) * np.log(N + 1e-300)

    # 内部时间 + 测量噪声
    du = float(u[1]-u[0])
    T_int = (1.0/cfg.c0) * np.cumsum(N) * du
    T_meas = T_int + rng.normal(0, cfg.noise_T_std * noise_scale, cfg.M)

    if timeit:
        print(f"[Forward] method={method:6s} time={((t1-t0)*1e3):.2f} ms")
    return u, gamma, phi, tau, N, T_int, T_meas, method

# ---------------- Derivative estimators ----------------
def derivative_spline(u, T, k=3, gain=DEFAULT_T_SMOOTH_GAIN):
    """对 T 做平滑样条 S_T，返回 S_T'(u)"""
    m = len(u)
    varT = float(np.var(T))
    sT = gain * m * varT
    sp = UnivariateSpline(u, T, s=sT, k=min(k, m-1))
    return sp.derivative(1)(u)

def derivative_fft(u, T, cutoff_ratio=DEFAULT_FFT_CUTOFF):
    """Hann 窗 + FFT 导数 + 低通"""
    T = np.asarray(T)
    m = len(u); du = float(u[1]-u[0])
    # Hann 窗
    w = 0.5 - 0.5*np.cos(2*np.pi*np.arange(m)/m)
    Tw = T * w
    F = np.fft.rfft(Tw)
    freqs = np.fft.rfftfreq(m, d=du)
    omega = 2*np.pi*freqs
    # 低通窗
    fN = 0.5/du
    fc = cutoff_ratio * fN
    H = (freqs <= fc).astype(float)
    dT = np.fft.irfft(1j*omega * F * H, n=m)
    return dT

def derivative_fd5(u, T):
    """五点中心差分（边界退化为一侧差分），数值稳定，不引入先验平滑。"""
    T = np.asarray(T); m=len(T); du = float(u[1]-u[0])
    d = np.zeros_like(T)
    if m>=5:
        d[2:-2] = ( -T[4:] + 8*T[3:-1] - 8*T[1:-3] + T[0:-4])/(12*du)
        # 边界：三点二阶准确
        d[0] = (-3*T[0] + 4*T[1] - T[2])/(2*du)
        d[1] = (-3*T[1] + 4*T[2] - T[3])/(2*du)
        d[-2]= ( 3*T[-2] -4*T[-3] + T[-4])/(2*du)
        d[-1]= ( 3*T[-1] -4*T[-2] + T[-3])/(2*du)
    else:
        d[:] = np.gradient(T, du)
    return d

# ---------------- Inverse（可选三种导数器；log 平滑可关） ----------------
def inverse_pipeline(u, T_meas, cfg_fwd: ForwardConfig, cfg_inv: InverseConfig):
    # ① 导数估计
    if cfg_inv.deriv_mode == "spline":
        dT = derivative_spline(u, T_meas, k=cfg_inv.spline_k, gain=cfg_inv.T_smooth_gain)
    elif cfg_inv.deriv_mode == "fft":
        dT = derivative_fft(u, T_meas, cutoff_ratio=cfg_inv.fft_cutoff_ratio)
    elif cfg_inv.deriv_mode == "fd5":
        dT = derivative_fd5(u, T_meas)
    else:
        raise ValueError("deriv_mode must be 'fft' | 'spline' | 'fd5'")

    # ② 初值 N_hat + 剪裁
    N_hat = cfg_fwd.c0 * dT
    N_hat = np.clip(N_hat, np.exp(-cfg_inv.log_clip), np.exp(cfg_inv.log_clip))

    # ③ log 域样条平滑（可关）
    X = np.log(N_hat + 1e-300)
    if cfg_inv.sX_gain > 0:
        if cfg_inv.sX_use_variance:
            sX = cfg_inv.sX_gain * len(u) * float(np.var(X))
        else:
            sX = float(cfg_inv.sX_gain)  # 绝对强度（常数）
        spX = UnivariateSpline(u, X, s=sX, k=min(cfg_inv.spline_k, len(u)-1))
        Xs = spX(u)
        Xs -= (np.trapz(Xs, x=u) / (u[-1]-u[0]))
        tau_hat = (1.0 / cfg_fwd.a) * Xs
        N_hat = np.exp(cfg_fwd.a * tau_hat)
    else:
        # 关闭平滑：只做基线对齐
        X -= (np.trapz(X, x=u) / (u[-1]-u[0]))
        tau_hat = (1.0 / cfg_fwd.a) * X
        N_hat = np.exp(cfg_fwd.a * tau_hat)

    if cfg_inv.enforce_meanN:
        domain = (u[-1]-u[0])
        meanN = simpson(N_hat, x=u) / domain
        if meanN != 0:
            N_hat /= meanN

    return {'dT': dT, 'N_hat': N_hat, 'tau_hat': tau_hat}

# ---------------- 跑一次：Direct vs FFT ----------------
cfg_fwd = ForwardConfig()
cfg_inv = InverseConfig()  # 使用上方 DEFAULT_* 的默认值

# Direct
u, gamma, phi, tau, N, T_int, T_meas, _m1 = make_synthetic_forward(cfg_fwd, seed=SEED, mode='direct')
res1 = inverse_pipeline(u, T_meas, cfg_fwd, cfg_inv)
tau_hat1, N_hat1 = res1['tau_hat'], res1['N_hat']
tau_c   = center_int_mean(tau, u)
tauh_c1 = center_int_mean(tau_hat1, u)
rmse_pre1 = rmse(tau_c, tauh_c1)
tau_hat1_cal, N_hat1_cal, k1 = calibrate_gain_on_tau(u, tau, tau_hat1, cfg_fwd.a, N_hat1)
tauh_c1_cal = center_int_mean(tau_hat1_cal, u)
rmse_post1 = rmse(tau_c, tauh_c1_cal)

print("\n=== Direct method metrics ===")
print(f"RMSE(tau, tau_hat) [pre-cal]  = {rmse_pre1:.4f}")
print(f"RMSE(tau, tau_hat) [post-cal] = {rmse_post1:.4f}   (gain k = {k1:.4f})")
print(f"<N_true> (trapz)              = {np.trapz(N, x=u)/(u[-1]-u[0]):.4f}")
print(f"<N_hat>  (trapz) [pre-cal]    = {np.trapz(N_hat1, x=u)/(u[-1]-u[0]):.4f}")
print(f"<N_hat>  (trapz) [post-cal]   = {np.trapz(N_hat1_cal, x=u)/(u[-1]-u[0]):.4f}")

# FFT
u2, gamma2, phi2, tau2, N2, T_int2, T_meas2, _m2 = make_synthetic_forward(cfg_fwd, seed=SEED, mode='fft')
res2 = inverse_pipeline(u2, T_meas2, cfg_fwd, cfg_inv)
tau_hat2, N_hat2 = res2['tau_hat'], res2['N_hat']
tau2_c   = center_int_mean(tau2, u2)
tauh2_c  = center_int_mean(tau_hat2, u2)
rmse_pre2 = rmse(tau2_c, tauh2_c)
tau_hat2_cal, N_hat2_cal, k2 = calibrate_gain_on_tau(u2, tau2, tau_hat2, cfg_fwd.a, N_hat2)
tauh2_c_cal = center_int_mean(tau_hat2_cal, u2)
rmse_post2 = rmse(tau2_c, tauh2_c_cal)

print("\n=== FFT method metrics ===")
print(f"RMSE(tau, tau_hat) [pre-cal]  = {rmse_pre2:.4f}")
print(f"RMSE(tau, tau_hat) [post-cal] = {rmse_post2:.4f}   (gain k = {k2:.4f})")
print(f"<N_true> (trapz)              = {np.trapz(N2, x=u2)/(u2[-1]-u2[0]):.4f}")
print(f"<N_hat>  (trapz) [pre-cal]    = {np.trapz(N_hat2, x=u2)/(u2[-1]-u2[0]):.4f}")
print(f"<N_hat>  (trapz) [post-cal]   = {np.trapz(N_hat2_cal, x=u2)/(u2[-1]-u2[0]):.4f}")

# 供后续使用的“校正后”结果
tau_hat1_use, N_hat1_use = tau_hat1_cal, N_hat1_cal

# ---------------- 诊断：K 指纹 / 一致性 / 敏感度 ----------------
if RUN_DIAGNOSTICS:
    # A) K 指纹
    def sliding_K_from_gaps(gaps, w=40):
        gaps = np.asarray(gaps); out=[]
        for i in range(0, len(gaps)-w+1):
            seg = gaps[i:i+w]
            Phi = float(np.mean(seg)); H = float(np.var(seg, ddof=1))
            out.append(Phi / (H + 1e-18))
        return np.array(out)

    def wigner_gue_gaps(n=10000, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        out=[]; C=1.2
        while len(out) < n:
            x = rng.exponential(1.0)
            p = (32/np.pi**2)*(x**2)*np.exp(-4*(x**2)/np.pi)
            q = np.exp(-x)
            if rng.random() < p/(C*q): out.append(x)
        s = np.array(out[:n]); return s/np.mean(s)

    gaps_real = np.diff(gamma); gaps_real = gaps_real/np.mean(gaps_real)
    gaps_gue  = wigner_gue_gaps(n=len(gaps_real))
    Kw_real   = sliding_K_from_gaps(gaps_real, w=40)
    Kw_gue    = sliding_K_from_gaps(gaps_gue,  w=40)

    print("\n=== K fingerprint (unit-mean gaps) ===")
    print(f"[synthetic γ] n={len(Kw_real)}  K_min={Kw_real.min():.4f}  K_max={Kw_real.max():.4f}  "
          f"K_mean={Kw_real.mean():.4f}  K_var={Kw_real.var(ddof=1):.4f}")
    print(f"[Wigner(GUE)] n={len(Kw_gue)}  K_min={Kw_gue.min():.4f}  K_max={Kw_gue.max():.4f}  "
          f"K_mean={Kw_gue.mean():.4f}  K_var={Kw_gue.var(ddof=1):.4f}")

    # B) 逆向一致性（用 Direct 的校正后结果）
    du = float(u[1]-u[0])
    T_hat = (1.0/cfg_fwd.c0) * np.cumsum(N_hat1_use) * du
    rmse_raw  = rmse(T_meas, T_hat);  nrmse_raw = nrmse(T_meas, T_hat)
    A = np.vstack([T_hat, np.ones_like(T_hat)]).T
    alpha, beta = lstsq(A, T_meas)[0]
    T_hat_aff = alpha*T_hat + beta
    rmse_aff  = rmse(T_meas, T_hat_aff);  nrmse_aff = nrmse(T_meas, T_hat_aff)
    print("\n=== Inverse consistency ===")
    print(f"T raw:    RMSE={rmse_raw:.6e}  NRMSE={nrmse_raw:.6e}")
    print(f"T affine: RMSE={rmse_aff:.6e}  NRMSE={nrmse_aff:.6e}")

    # C) 三维敏感度（噪声 × 导数强度 × sX_gain）——post-cal
    if RUN_NOISE_SCAN:
        if cfg_inv.deriv_mode == "spline":
            Tscale_list = (0.5, 1.0, 2.0, 4.0)   # 放大 T_smooth_gain
            label_cols  = "T_smooth_gain × (0.5,1,2,4)"
            def tweak_cfg(tscale, sxgain):
                return InverseConfig(deriv_mode="spline",
                                     T_smooth_gain=DEFAULT_T_SMOOTH_GAIN*tscale,
                                     sX_gain=sxgain,
                                     sX_use_variance=DEFAULT_SX_USE_VAR)
        elif cfg_inv.deriv_mode == "fft":
            Tscale_list = (0.15, 0.25, 0.35, 0.50)  # 不同 FFT 截止
            label_cols  = "fft_cutoff_ratio × (0.15,0.25,0.35,0.50)"
            def tweak_cfg(tscale, sxgain):
                return InverseConfig(deriv_mode="fft",
                                     fft_cutoff_ratio=tscale,
                                     sX_gain=sxgain,
                                     sX_use_variance=DEFAULT_SX_USE_VAR)
        else:  # fd5
            Tscale_list = (1.0, 1.0, 1.0, 1.0)
            label_cols  = "fd5 (no strength)"
            def tweak_cfg(tscale, sxgain):
                return InverseConfig(deriv_mode="fd5",
                                     sX_gain=sxgain,
                                     sX_use_variance=DEFAULT_SX_USE_VAR)

        # 关键：让 sX_gain 真正改变，包括 0（关闭）
        sX_scales = (0.0, 0.5, 1.0, 2.0, 4.0)
        noise_scales = (0.5, 1.0, 2.0, 4.0)

        RM3 = np.zeros((len(noise_scales), len(Tscale_list), len(sX_scales)))
        for a_idx, nz in enumerate(noise_scales):
            uf, gf, pf, tf, Nf, Tf, Tm, _ = make_synthetic_forward(cfg_fwd, seed=SEED, mode='fft', timeit=False, noise_scale=nz)
            for i, tscale in enumerate(Tscale_list):
                for j, sx in enumerate(sX_scales):
                    cfg_tmp = tweak_cfg(tscale, sx)
                    res_tmp = inverse_pipeline(uf, Tm, cfg_fwd, cfg_tmp)
                    tau_hat_tmp = res_tmp['tau_hat']; N_hat_tmp = res_tmp['N_hat']
                    tau_hat_tmp_cal, N_hat_tmp_cal, _ = calibrate_gain_on_tau(uf, tf, tau_hat_tmp, cfg_fwd.a, N_hat_tmp)
                    RM3[a_idx,i,j] = rmse(center_int_mean(tf, uf),
                                          center_int_mean(tau_hat_tmp_cal, uf))

        print("\n=== Sensitivity RMSE cube (noise × %s × sX_gain) ===" % label_cols)
        for a_idx, nz in enumerate(noise_scales):
            print(f"noise×{nz}:")
            for i, tscale in enumerate(Tscale_list):
                row = "  ".join(f"{RM3[a_idx,i,j]:.4f}" for j in range(len(sX_scales)))
                tag = f"{tscale:g}"
                print(f"  T-tune×{tag:<4}:  {row}")
            print("  cols = sX_gain:", sX_scales)

# ---------------- n→∞ extrapolation（post-cal RMSE） ----------------
def run_scale(cfg_base: ForwardConfig, cfg_inv: InverseConfig, Nzeros_list: List[int], scale_M: bool=True):
    rec = []
    for Nz in Nzeros_list:
        cfg = ForwardConfig(**{**cfg_base.__dict__, "Nzeros": Nz, "M": int(cfg_base.M * (Nz/cfg_base.Nzeros)) if scale_M else cfg_base.M})
        u, gamma, phi, tau, N, T_int, T_meas, _ = make_synthetic_forward(cfg, seed=SEED, mode='fft', timeit=False)
        res = inverse_pipeline(u, T_meas, cfg, cfg_inv)
        tau_hat = res['tau_hat']; N_hat = res['N_hat']
        tau_hat_cal, N_hat_cal, _ = calibrate_gain_on_tau(u, tau, tau_hat, cfg.a, N_hat)

        gaps = np.diff(gamma); gaps = gaps/np.mean(gaps)
        def sliding_K(g, w=40):
            out=[]
            for i in range(0, len(g)-w+1):
                seg = g[i:i+w]; Phi=np.mean(seg); H=np.var(seg, ddof=1); out.append(Phi/(H+1e-18))
            return np.array(out)
        Kw = sliding_K(gaps, 40)

        rec.append({
            "Nzeros": Nz,
            "M": cfg.M,
            "rmse_tau": rmse(center_int_mean(tau, u), center_int_mean(tau_hat_cal, u)),
            "K_mean": float(np.mean(Kw)) if len(Kw)>0 else np.nan,
            "K_var":  float(np.var(Kw, ddof=1)) if len(Kw)>1 else np.nan
        })
    return rec

if RUN_N_EXTRAP:
    Nzeros_list = [300, 600, 1200, 2400, 4800]
    rec = run_scale(cfg_fwd, cfg_inv, Nzeros_list, scale_M=True)
    print("\n=== n→∞ extrapolation (scaled M, post-cal RMSE) ===")
    for r in rec:
        print(f"Nzeros={r['Nzeros']:>5d}  M={r['M']:>6d}  RMSE={r['rmse_tau']:.5f}  K_mean={r['K_mean']:.2f}")

# ---------------- Export CSVs（post-cal） ----------------
if EXPORT_CSV:
    def run_once(mode, cfg_inv_run: InverseConfig):
        u, gamma, phi, tau, N, T_int, T_meas, m = make_synthetic_forward(cfg_fwd, seed=SEED, mode=mode, timeit=False)
        t0 = time.perf_counter()
        if mode == "direct": _ = phi_arctan_direct(u, gamma, cfg_fwd.sigma)
        else: _ = phi_arctan_fft_cached(u, gamma, cfg_fwd.sigma)
        t1 = time.perf_counter()
        res = inverse_pipeline(u, T_meas, cfg_fwd, cfg_inv_run)
        tau_hat, N_hat = res['tau_hat'], res['N_hat']
        tau_hat_cal, N_hat_cal, _ = calibrate_gain_on_tau(u, tau, tau_hat, cfg_fwd.a, N_hat)
        N_true = np.exp(np.clip(cfg_fwd.a * tau, -40, 40))
        N_true /= (np.trapz(N_true, x=u)/(u[-1]-u[0]) + 1e-18)
        return {
            "method": m,
            "time_ms": (t1-t0)*1e3,
            "rmse_tau": rmse(center_int_mean(tau, u), center_int_mean(tau_hat_cal, u)),
            "N_true_trapz": np.trapz(N_true, x=u)/(u[-1]-u[0]),
            "N_hat_trapz": np.trapz(N_hat_cal, x=u)/(u[-1]-u[0]),
        }

    d = run_once("direct", cfg_inv)
    f = run_once("fft",    cfg_inv)
    with open("metrics_summary.csv","w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method","time_ms","rmse_tau","N_true_trapz","N_hat_trapz"])
        writer.writerow([d["method"], f"{d['time_ms']:.2f}", f"{d['rmse_tau']:.4f}", f"{d['N_true_trapz']:.4f}", f"{d['N_hat_trapz']:.4f}"])
        writer.writerow([f["method"], f"{f['time_ms']:.2f}", f"{f['rmse_tau']:.4f}", f"{f['N_true_trapz']:.4f}", f"{f['N_hat_trapz']:.4f}"])
    print("Saved metrics_summary.csv")
