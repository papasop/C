# === Riemann-phase inverse pipeline (Full fixed Colab; single cell) ===
import numpy as np, math, time, warnings, csv
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
from scipy.interpolate import UnivariateSpline
from scipy.linalg import lstsq
from scipy.integrate import simpson

warnings.filterwarnings('ignore')
np.set_printoptions(precision=6, suppress=True)

# ---------------- Global toggles ----------------
SEED            = 42
PLOT_FIGS       = False    # True 可看曲线
RUN_DIAGNOSTICS = True     # K 指纹 / 一致性 / 敏感度 / 自动调参
RUN_NOISE_SCAN  = True     # 三维敏感度：noise × (导数强度) × sX_gain
RUN_N_EXTRAP    = True     # n→∞ 外推
EXPORT_CSV      = True

# ===== 默认值（建议先用；AutoTune 会给更优） =====
DEFAULT_DERIV_MODE     = "fft"    # "fft" | "spline" | "fd5"
DEFAULT_T_SMOOTH_GAIN  = 3.0      # spline 导数器：s_T = gain * M * Var(T)
DEFAULT_FFT_CUTOFF     = 0.35     # fft 导数器：低通截止占 Nyquist 比例 ∈ (0,1)
DEFAULT_SX_GAIN        = 1.0      # log(N_hat) 样条平滑强度；=0 关闭（基线列）
DEFAULT_SX_USE_VAR     = True     # True: sX = gain * M * Var(X)；False: 常数 gain

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
    sX_gain: float = DEFAULT_SX_GAIN              # log-domain smoothing strength (0=off)
    sX_use_variance: bool = DEFAULT_SX_USE_VAR    # True: gain * M * Var(X); False: constant sX_gain

# ---------------- Utils ----------------
def rmse(a, b) -> float:
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a-b)**2)))

def rmse_trim(a, b, trim=0.02) -> float:
    """裁掉两端各 2% 评估 RMSE，抑制导数的边界伪差影响。"""
    a = np.asarray(a); b = np.asarray(b)
    m = len(a); k = max(1, int(m*trim))
    sl = slice(k, -k) if m > 2*k else slice(0, m)
    return float(np.sqrt(np.mean((a[sl]-b[sl])**2)))

def nrmse(a, b) -> float:
    a = np.asarray(a); b = np.asarray(b)
    return float(rmse(a,b) / (np.max(a)-np.min(a) + 1e-18))

def center_int_mean(x, u):
    return x - (np.trapz(x, x=u) / (u[-1]-u[0]))

def sign_safe_calibrate_gain(u, tau_true, tau_hat, a):
    """若相关为负先整体翻转，再线性增益拟合；返回 tau_hat_cal, k, corr。"""
    t_c  = center_int_mean(tau_true, u)
    h_c  = center_int_mean(tau_hat,  u)
    num  = np.trapz(t_c * h_c, x=u)
    den  = math.sqrt(np.trapz(t_c*t_c, x=u) * np.trapz(h_c*h_c, x=u) + 1e-18)
    corr = float(num/(den+1e-18))
    if corr < 0:
        tau_hat = -tau_hat
        h_c     = -h_c
        num     = -num
        corr    = -corr
    k_num = np.trapz(t_c * h_c, x=u)
    k_den = np.trapz(h_c * h_c, x=u) + 1e-18
    k = float(k_num / k_den)
    tau_hat_cal = k * tau_hat
    return tau_hat_cal, k, corr

def calibrate_with_truth(u, tau_true, tau_hat, a, N_hat):
    """符号安全 + 增益拟合 + <N>=1 归一"""
    tau_hat_fix, k, corr = sign_safe_calibrate_gain(u, tau_true, tau_hat, a)
    N_hat_cal   = np.exp(a * tau_hat_fix)
    meanN = np.trapz(N_hat_cal, x=u)/(u[-1]-u[0])
    if meanN != 0:
        N_hat_cal /= meanN
    return tau_hat_fix, N_hat_cal, k, corr

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
    m = len(u)
    varT = float(np.var(T))
    sT = gain * m * varT
    sp = UnivariateSpline(u, T, s=sT, k=min(k, m-1))
    return sp.derivative(1)(u)

def derivative_fft(u, T, cutoff_ratio=DEFAULT_FFT_CUTOFF, pad_multiple=2):
    """
    频域导数（镜像扩展 + 无窗）：
    1) 偶对称扩展，避免周期化边界；
    2) rFFT -> 乘 i*omega 做导数；
    3) 频域低通（按 cutoff_ratio）；
    4) 裁回原长度。
    """
    T = np.asarray(T)
    m = len(u); du = float(u[1]-u[0])
    T_reflect = np.concatenate([T, T[-2:0:-1]])  # reflect
    M = len(T_reflect)
    F = np.fft.rfft(T_reflect)
    freqs = np.fft.rfftfreq(M, d=du)
    omega = 2*np.pi*freqs
    if 0 < cutoff_ratio < 1:
        fN = 0.5/du; fc = cutoff_ratio * fN
        H = (freqs <= fc).astype(float)
    else:
        H = 1.0
    dT_reflect = np.fft.irfft(1j * omega * F * H, n=M)
    dT = dT_reflect[:m]
    return dT

def derivative_fd5(u, T):
    T = np.asarray(T); m=len(T); du = float(u[1]-u[0])
    d = np.zeros_like(T)
    if m>=5:
        d[2:-2] = ( -T[4:] + 8*T[3:-1] - 8*T[1:-3] + T[0:-4])/(12*du)
        d[0] = (-3*T[0] + 4*T[1] - T[2])/(2*du)
        d[1] = (-3*T[1] + 4*T[2] - T[3])/(2*du)
        d[-2]= ( 3*T[-2] -4*T[-3] + T[-4])/(2*du)
        d[-1]= ( 3*T[-1] -4*T[-2] + T[-3])/(2*du)
    else:
        d[:] = np.gradient(T, du)
    return d

# ---------------- 一致性层工具（低通/维纳 + TV） ----------------
def lowpass_fft(y, du, cutoff_ratio=0.35):
    y = np.asarray(y); m=len(y)
    y_ref = np.concatenate([y, y[-2:0:-1]])
    M = len(y_ref)
    F = np.fft.rfft(y_ref)
    freqs = np.fft.rfftfreq(M, d=du)
    fN = 0.5/du; fc = cutoff_ratio * fN
    H = (freqs <= fc).astype(float)
    y_lp_ref = np.fft.irfft(F*H, n=M)
    return y_lp_ref[:m]

def wiener1d(y, noise_var=None, du=1.0, cutoff_ratio=0.5, eps=1e-12):
    y = np.asarray(y); m=len(y)
    y_ref = np.concatenate([y, y[-2:0:-1]]); M=len(y_ref)
    Y = np.fft.rfft(y_ref); freqs = np.fft.rfftfreq(M, d=du)
    Pyy = (np.abs(Y)**2)/M
    if noise_var is None:
        fN = 0.5/du; fc = cutoff_ratio * fN
        mask = freqs >= fc
        noise_var = float(np.mean(Pyy[mask])) if np.any(mask) else float(np.mean(Pyy[-10:]))
    Sn = noise_var
    H = Pyy / (Pyy + Sn + eps)
    Yf = H * Y
    y_hat = np.fft.irfft(Yf, n=M)[:m]
    return y_hat

def tv_denoise_logN(logN, lam=5e-3, iters=30):
    """
    1D TV-L2 on logN（Rudin–Osher–Fatemi 简化迭代）
    修复：使用 len(x)-1 的前向差分，并用原位散度填回，避免长度 +1 的拼接错误。
    """
    x = logN.copy().astype(float)
    for _ in range(iters):
        dx = np.diff(x)                    # len = m-1
        g  = dx / np.sqrt(dx*dx + 1e-8)    # len = m-1
        # 散度回填到 len = m 的数组
        divg = np.zeros_like(x)
        if len(x) >= 2:
            divg[0]      = g[0]
            divg[1:-1]   = g[1:] - g[:-1]
            divg[-1]     = -g[-1]
        # 轻度 L2 约束防漂移（1e-6 很弱），与 TV 项权衡
        x = (x + lam * divg) / (1.0 + 1e-6)
    return x

def refine_N_with_T_consistency(u, N_init, T_meas, c0=1.0,
                                step=0.6, iters=4, lp_ratio=0.35,
                                use_wiener=True, tv_on_log=True, tv_lam=5e-3):
    """
    最小化 || T_meas - (1/c0)CumSum(N) ||^2 的一致性修正：
    N <- N + step * c0 * d/du[ LP(Wiener(T_meas - T_hat)) ]
    每步保持 <N>=1；在 logN 上做轻微 TV 去噪。
    """
    u = np.asarray(u); N = N_init.copy().astype(float); du = float(u[1]-u[0])
    for _ in range(iters):
        T_hat = (1.0/c0) * np.cumsum(N) * du
        r = T_meas - T_hat
        if use_wiener:
            r_f = wiener1d(r, noise_var=None, du=du, cutoff_ratio=max(lp_ratio, 0.45))
        else:
            r_f = r
        r_f = lowpass_fft(r_f, du, cutoff_ratio=lp_ratio)
        dr = derivative_fd5(u, r_f)
        N += step * c0 * dr
        logN = np.log(np.clip(N, 1e-300, 1e300))
        if tv_on_log:
            logN = tv_denoise_logN(logN, lam=tv_lam, iters=20)
        logN -= (np.trapz(logN, x=u)/(u[-1]-u[0]))
        N = np.exp(logN)
    meanN = np.trapz(N, x=u)/(u[-1]-u[0])
    if meanN != 0:
        N /= meanN
    return N

# ---------------- Inverse（含 log 平滑可关；<N>=1 归一） ----------------
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

    # ③ log 域样条平滑（可关）；只做“去积分均值”
    X = np.log(N_hat + 1e-300)
    if cfg_inv.sX_gain > 0:
        if cfg_inv.sX_use_variance:
            sX = cfg_inv.sX_gain * len(u) * float(np.var(X))
        else:
            sX = float(cfg_inv.sX_gain)
        spX = UnivariateSpline(u, X, s=sX, k=min(cfg_inv.spline_k, len(u)-1))
        Xs = spX(u)
    else:
        Xs = X.copy()
    Xs -= (np.trapz(Xs, x=u) / (u[-1]-u[0]))   # 保持 <log N> = 0

    tau_hat = (1.0 / cfg_fwd.a) * Xs
    N_hat = np.exp(cfg_fwd.a * tau_hat)

    if cfg_inv.enforce_meanN:
        domain = (u[-1]-u[0])
        meanN = simpson(N_hat, x=u) / domain
        if meanN != 0:
            N_hat /= meanN

    return {'dT': dT, 'N_hat': N_hat, 'tau_hat': tau_hat}

# ---------------- 主流程：Direct vs FFT，指标 & 自动调参 & 一致性层 ----------------
cfg_fwd = ForwardConfig()
cfg_inv = InverseConfig(
    deriv_mode=DEFAULT_DERIV_MODE,
    T_smooth_gain=DEFAULT_T_SMOOTH_GAIN,
    fft_cutoff_ratio=DEFAULT_FFT_CUTOFF,
    sX_gain=DEFAULT_SX_GAIN,
    sX_use_variance=DEFAULT_SX_USE_VAR,
)

# Direct
u, gamma, phi, tau, N, T_int, T_meas, _m1 = make_synthetic_forward(cfg_fwd, seed=SEED, mode='direct')
res1 = inverse_pipeline(u, T_meas, cfg_fwd, cfg_inv)
tau_hat1, N_hat1 = res1['tau_hat'], res1['N_hat']

tau_c   = center_int_mean(tau, u)
tauh_c1 = center_int_mean(tau_hat1, u)
num  = np.trapz(tau_c * tauh_c1, x=u)
den  = math.sqrt(np.trapz(tau_c*tau_c, x=u) * np.trapz(tauh_c1*tauh_c1, x=u) + 1e-18)
corr1 = float(num/(den+1e-18))
rmse_pre1  = rmse_trim(tau_c, tauh_c1, trim=0.02)

tau_hat1_cal, N_hat1_cal, k1, corr1_fix = calibrate_with_truth(u, tau, tau_hat1, cfg_fwd.a, N_hat1)
tauh_c1_cal = center_int_mean(tau_hat1_cal, u)
rmse_post1 = rmse_trim(tau_c, tauh_c1_cal, trim=0.02)

print("\n=== Direct method metrics ===")
print(f"RMSE(tau, tau_hat) [pre-cal]  = {rmse_pre1:.4f}")
print(f"corr(tau, tau_hat) [pre-cal]  = {corr1:.4f}")
print(f"RMSE(tau, tau_hat) [post-cal] = {rmse_post1:.4f}   (gain k = {k1:.4f}, sign-fixed corr = {corr1_fix:.4f})")
print(f"<N_true> (trapz)              = {np.trapz(N, x=u)/(u[-1]-u[0]):.4f}")
print(f"<N_hat>  (trapz) [pre-cal]    = {np.trapz(N_hat1, x=u)/(u[-1]-u[0]):.4f}")
print(f"<N_hat>  (trapz) [post-cal]   = {np.trapz(N_hat1_cal, x=u)/(u[-1]-u[0]):.4f}")

# FFT
u2, gamma2, phi2, tau2, N2, T_int2, T_meas2, _m2 = make_synthetic_forward(cfg_fwd, seed=SEED, mode='fft')
res2 = inverse_pipeline(u2, T_meas2, cfg_fwd, cfg_inv)
tau_hat2, N_hat2 = res2['tau_hat'], res2['N_hat']

tau2_c   = center_int_mean(tau2, u2)
tauh2_c  = center_int_mean(tau_hat2, u2)
num2  = np.trapz(tau2_c * tauh2_c, x=u2)
den2  = math.sqrt(np.trapz(tau2_c*tau2_c, x=u2) * np.trapz(tauh2_c*tauh2_c, x=u2) + 1e-18)
corr2 = float(num2/(den2+1e-18))
rmse_pre2 = rmse_trim(tau2_c, tauh2_c, trim=0.02)

tau_hat2_cal, N_hat2_cal, k2, corr2_fix = calibrate_with_truth(u2, tau2, tau_hat2, cfg_fwd.a, N_hat2)
tauh2_c_cal = center_int_mean(tau_hat2_cal, u2)
rmse_post2 = rmse_trim(tau2_c, tauh2_c_cal, trim=0.02)

print("\n=== FFT method metrics ===")
print(f"RMSE(tau, tau_hat) [pre-cal]  = {rmse_pre2:.4f}")
print(f"corr(tau, tau_hat) [pre-cal]  = {corr2:.4f}")
print(f"RMSE(tau, tau_hat) [post-cal] = {rmse_post2:.4f}   (gain k = {k2:.4f}, sign-fixed corr = {corr2_fix:.4f})")
print(f"<N_true> (trapz)              = {np.trapz(N2, x=u2)/(u2[-1]-u2[0]):.4f}")
print(f"<N_hat>  (trapz) [pre-cal]    = {np.trapz(N_hat2, x=u2)/(u2[-1]-u2[0]):.4f}")
print(f"<N_hat>  (trapz) [post-cal]   = {np.trapz(N_hat2_cal, x=u2)/(u2[-1]-u2[0]):.4f}")

# 自动调参（cutoff × sX_gain，小网格；用于 FFT）
def autotune_fft_and_sX(u, tau_true, T_meas, cfg_fwd,
                        cutoff_list=(0.20,0.25,0.30,0.35,0.40),
                        sX_list=(0.0,0.25,0.5,1.0,2.0,4.0)):
    best = None
    for fc in cutoff_list:
        for sg in sX_list:
            cfg_try = InverseConfig(deriv_mode="fft",
                                    fft_cutoff_ratio=fc,
                                    sX_gain=sg,
                                    sX_use_variance=True)
            res = inverse_pipeline(u, T_meas, cfg_fwd, cfg_try)
            tau_hat = res['tau_hat']; N_hat = res['N_hat']
            tau_hat_cal, N_hat_cal, k, corr = calibrate_with_truth(u, tau_true, tau_hat, cfg_fwd.a, N_hat)
            r = rmse_trim(center_int_mean(tau_true, u), center_int_mean(tau_hat_cal, u), trim=0.02)
            cand = (r, fc, sg, k, corr)
            if (best is None) or (r < best[0]): best = cand
    return best  # (rmse, fc, sg, k, corr)

if RUN_DIAGNOSTICS:
    best = autotune_fft_and_sX(u2, tau2, T_meas2, cfg_fwd)
    print(f"\n[AutoTune] best post-cal RMSE={best[0]:.4f} at cutoff={best[1]}  sX_gain={best[2]}  (k={best[3]:.3f}, corr={best[4]:.4f})")

# 一致性层：Direct
du = float(u[1]-u[0])
T_hat1 = (1.0/cfg_fwd.c0) * np.cumsum(N_hat1_cal) * du
rmse_raw1  = rmse(T_meas, T_hat1);  nrmse_raw1 = nrmse(T_meas, T_hat1)
A = np.vstack([T_hat1, np.ones_like(T_hat1)]).T
alpha1, beta1 = lstsq(A, T_meas)[0]
T_hat1_aff = alpha1*T_hat1 + beta1
rmse_aff1  = rmse(T_meas, T_hat1_aff);  nrmse_aff1 = nrmse(T_meas, T_hat1_aff)

N_hat1_ref = refine_N_with_T_consistency(u, N_hat1_cal, T_meas, c0=cfg_fwd.c0,
                                         step=0.6, iters=4, lp_ratio=0.35,
                                         use_wiener=True, tv_on_log=True, tv_lam=5e-3)
T_hat1_ref = (1.0/cfg_fwd.c0) * np.cumsum(N_hat1_ref) * du
rmse_raw1_ref  = rmse(T_meas, T_hat1_ref);  nrmse_raw1_ref = nrmse(T_meas, T_hat1_ref)
A = np.vstack([T_hat1_ref, np.ones_like(T_hat1_ref)]).T
alpha1r, beta1r = lstsq(A, T_meas)[0]
T_hat1_aff_ref = alpha1r*T_hat1_ref + beta1r
rmse_aff1_ref  = rmse(T_meas, T_hat1_aff_ref);  nrmse_aff1_ref = nrmse(T_meas, T_hat1_aff_ref)

print("\n=== Inverse consistency (Direct) ===")
print(f"T raw (before):    RMSE={rmse_raw1:.6e}  NRMSE={nrmse_raw1:.6e}")
print(f"T affine (before): RMSE={rmse_aff1:.6e}  NRMSE={nrmse_aff1:.6e}")
print(f"[Consistency Layer] T raw (after):    RMSE={rmse_raw1_ref:.6e}  NRMSE={nrmse_raw1_ref:.6e}")
print(f"[Consistency Layer] T affine (after): RMSE={rmse_aff1_ref:.6e}  NRMSE={nrmse_aff1_ref:.6e}")
noise_floor = np.std(T_meas - T_int)
NRMSE_floor = noise_floor / (np.max(T_int)-np.min(T_int)+1e-18)
print(f"Theoretical noise floor (NRMSE) ≈ {NRMSE_floor:.3e}")

# 一致性层：FFT
du2 = float(u2[1]-u2[0])
T_hat2 = (1.0/cfg_fwd.c0) * np.cumsum(N_hat2_cal) * du2
rmse_raw2  = rmse(T_meas2, T_hat2);  nrmse_raw2 = nrmse(T_meas2, T_hat2)
A = np.vstack([T_hat2, np.ones_like(T_hat2)]).T
alpha2, beta2 = lstsq(A, T_meas2)[0]
T_hat2_aff = alpha2*T_hat2 + beta2
rmse_aff2  = rmse(T_meas2, T_hat2_aff);  nrmse_aff2 = nrmse(T_meas2, T_hat2_aff)

N_hat2_ref = refine_N_with_T_consistency(u2, N_hat2_cal, T_meas2, c0=cfg_fwd.c0,
                                         step=0.6, iters=4, lp_ratio=0.35,
                                         use_wiener=True, tv_on_log=True, tv_lam=5e-3)
T_hat2_ref = (1.0/cfg_fwd.c0) * np.cumsum(N_hat2_ref) * du2
rmse_raw2_ref  = rmse(T_meas2, T_hat2_ref);  nrmse_raw2_ref = nrmse(T_meas2, T_hat2_ref)
A = np.vstack([T_hat2_ref, np.ones_like(T_hat2_ref)]).T
alpha2r, beta2r = lstsq(A, T_meas2)[0]
T_hat2_aff_ref = alpha2r*T_hat2_ref + beta2r
rmse_aff2_ref  = rmse(T_meas2, T_hat2_aff_ref);  nrmse_aff2_ref = nrmse(T_meas2, T_hat2_aff_ref)

print("\n=== Inverse consistency (FFT) ===")
print(f"T raw (before):    RMSE={rmse_raw2:.6e}  NRMSE={nrmse_raw2:.6e}")
print(f"T affine (before): RMSE={rmse_aff2:.6e}  NRMSE={nrmse_aff2:.6e}")
print(f"[Consistency Layer] T raw (after):    RMSE={rmse_raw2_ref:.6e}  NRMSE={nrmse_raw2_ref:.6e}")
print(f"[Consistency Layer] T affine (after): RMSE={rmse_aff2_ref:.6e}  NRMSE={nrmse_aff2_ref:.6e}")

# ---------------- 诊断：K 指纹 / 敏感度 ----------------
if RUN_DIAGNOSTICS:
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
            p = (32/np.pi**2)*(x**2)*np.exp(-4*(x**2)/np.pi)  # 近似 GUE 邻距
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

    # 敏感度立方体（噪声 × fft_cutoff_ratio × sX_gain），trim 评估
    if RUN_NOISE_SCAN:
        Daxis = (0.10, 0.15, 0.25, 0.35, 0.50, 0.70)    # 更宽更分辨
        sX_scales = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
        noise_scales = (0.5, 1.0, 2.0, 4.0)
        RM3 = np.zeros((len(noise_scales), len(Daxis), len(sX_scales)))
        for a_idx, nz in enumerate(noise_scales):
            uf, gf, pf, tf, Nf, Tf, Tm, _ = make_synthetic_forward(cfg_fwd, seed=SEED, mode='fft', timeit=False, noise_scale=nz)
            for i, dstr in enumerate(Daxis):
                for j, sx in enumerate(sX_scales):
                    cfg_tmp = InverseConfig(deriv_mode="fft",
                                            fft_cutoff_ratio=dstr,
                                            sX_gain=sx,
                                            sX_use_variance=True)
                    res_tmp = inverse_pipeline(uf, Tm, cfg_fwd, cfg_tmp)
                    tau_hat_tmp = res_tmp['tau_hat']; N_hat_tmp = res_tmp['N_hat']
                    tau_hat_tmp_cal, N_hat_tmp_cal, ktmp, corr_tmp = calibrate_with_truth(uf, tf, tau_hat_tmp, cfg_fwd.a, N_hat_tmp)
                    RM3[a_idx,i,j] = rmse_trim(center_int_mean(tf, uf),
                                               center_int_mean(tau_hat_tmp_cal, uf),
                                               trim=0.02)

        print("\n=== Sensitivity RMSE cube (noise × fft_cutoff_ratio × sX_gain) ===")
        for a_idx, nz in enumerate(noise_scales):
            print(f"noise×{nz}:")
            for i, dstr in enumerate(Daxis):
                row = "  ".join(f"{RM3[a_idx,i,j]:.4f}" for j in range(len(sX_scales)))
                tag = f"{dstr:g}"
                print(f"  D-tune×{tag:<4}:  {row}")
            print("  cols = sX_gain:", sX_scales)

# ---------------- n→∞ extrapolation（post-cal RMSE，trim 评估） ----------------
def run_scale(cfg_base: ForwardConfig, Nzeros_list: List[int], scale_M: bool=True):
    rec = []
    for Nz in Nzeros_list:
        cfg = ForwardConfig(**{**cfg_base.__dict__, "Nzeros": Nz, "M": int(cfg_base.M * (Nz/cfg_base.Nzeros)) if scale_M else cfg_base.M})
        u, gamma, phi, tau, N, T_int, T_meas, _ = make_synthetic_forward(cfg, seed=SEED, mode='fft', timeit=False)
        res = inverse_pipeline(u, T_meas, cfg, InverseConfig(deriv_mode="fft",
                                                             fft_cutoff_ratio=DEFAULT_FFT_CUTOFF,
                                                             sX_gain=DEFAULT_SX_GAIN,
                                                             sX_use_variance=True))
        tau_hat = res['tau_hat']; N_hat = res['N_hat']
        tau_hat_cal, N_hat_cal, kfit, corrfit = calibrate_with_truth(u, tau, tau_hat, cfg.a, N_hat)

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
            "rmse_tau": rmse_trim(center_int_mean(tau, u), center_int_mean(tau_hat_cal, u), trim=0.02),
            "K_mean": float(np.mean(Kw)) if len(Kw)>0 else np.nan,
            "K_var":  float(np.var(Kw, ddof=1)) if len(Kw)>1 else np.nan
        })
    return rec

if RUN_N_EXTRAP:
    Nzeros_list = [300, 600, 1200, 2400, 4800]
    rec = run_scale(cfg_fwd, Nzeros_list, scale_M=True)
    print("\n=== n→∞ extrapolation (scaled M, post-cal RMSE) ===")
    for r in rec:
        print(f"Nzeros={r['Nzeros']:>5d}  M={r['M']:>6d}  RMSE={r['rmse_tau']:.5f}  K_mean={r['K_mean']:.2f}")

# ---------------- Export CSVs（post-cal + 一致性层摘要） ----------------
if EXPORT_CSV:
    def run_once(mode, cfg_inv_run: InverseConfig):
        u, gamma, phi, tau, N, T_int, T_meas, m = make_synthetic_forward(cfg_fwd, seed=SEED, mode=mode, timeit=False)
        t0 = time.perf_counter()
        if mode == "direct": _ = phi_arctan_direct(u, gamma, cfg_fwd.sigma)
        else: _ = phi_arctan_fft_cached(u, gamma, cfg_fwd.sigma)
        t1 = time.perf_counter()
        res = inverse_pipeline(u, T_meas, cfg_fwd, cfg_inv_run)
        tau_hat, N_hat = res['tau_hat'], res['N_hat']
        tau_hat_cal, N_hat_cal, _k, _corr = calibrate_with_truth(u, tau, tau_hat, cfg_fwd.a, N_hat)
        # 一致性层一次
        N_ref = refine_N_with_T_consistency(u, N_hat_cal, T_meas, c0=cfg_fwd.c0,
                                            step=0.6, iters=4, lp_ratio=0.35,
                                            use_wiener=True, tv_on_log=True, tv_lam=5e-3)
        du = float(u[1]-u[0])
        T_hat = (1.0/cfg_fwd.c0) * np.cumsum(N_hat_cal) * du
        T_hat_ref = (1.0/cfg_fwd.c0) * np.cumsum(N_ref) * du
        return {
            "method": m,
            "time_ms": (t1-t0)*1e3,
            "rmse_tau_trim": rmse_trim(center_int_mean(tau, u), center_int_mean(tau_hat_cal, u), trim=0.02),
            "N_true_trapz": np.trapz(N, x=u)/(u[-1]-u[0]),
            "N_hat_trapz": np.trapz(N_hat_cal, x=u)/(u[-1]-u[0]),
            "T_raw_NRMSE_before": nrmse(T_meas, T_hat),
            "T_raw_NRMSE_after":  nrmse(T_meas, T_hat_ref),
        }

    d = run_once("direct", InverseConfig(deriv_mode="fft", fft_cutoff_ratio=DEFAULT_FFT_CUTOFF, sX_gain=DEFAULT_SX_GAIN, sX_use_variance=True))
    f = run_once("fft",    InverseConfig(deriv_mode="fft", fft_cutoff_ratio=DEFAULT_FFT_CUTOFF, sX_gain=DEFAULT_SX_GAIN, sX_use_variance=True))
    with open("metrics_summary.csv","w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method","time_ms","rmse_tau_trim","N_true_trapz","N_hat_trapz","T_raw_NRMSE_before","T_raw_NRMSE_after"])
        writer.writerow([d["method"], f"{d['time_ms']:.2f}", f"{d['rmse_tau_trim']:.4f}", f"{d['N_true_trapz']:.4f}", f"{d['N_hat_trapz']:.4f}", f"{d['T_raw_NRMSE_before']:.6e}", f"{d['T_raw_NRMSE_after']:.6e}"])
        writer.writerow([f["method"], f"{f['time_ms']:.2f}", f"{f['rmse_tau_trim']:.4f}", f"{f['N_true_trapz']:.4f}", f"{f['N_hat_trapz']:.4f}", f"{f['T_raw_NRMSE_before']:.6e}", f"{f['T_raw_NRMSE_after']:.6e}"])
    print("Saved metrics_summary.csv")

    # 保存一个敏感度切片（noise=1.0）
    Daxis = (0.10, 0.15, 0.25, 0.35, 0.50, 0.70)
    sX_scales = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    uf, gf, pf, tf, Nf, Tf, Tm, _ = make_synthetic_forward(cfg_fwd, seed=SEED, mode='fft', timeit=False, noise_scale=1.0)
    RM_slice = []
    for i, dstr in enumerate(Daxis):
        row=[]
        for j, sx in enumerate(sX_scales):
            cfg_tmp = InverseConfig(deriv_mode="fft", fft_cutoff_ratio=dstr, sX_gain=sx, sX_use_variance=True)
            res_tmp = inverse_pipeline(uf, Tm, cfg_fwd, cfg_tmp)
            tau_hat_tmp = res_tmp['tau_hat']; N_hat_tmp = res_tmp['N_hat']
            tau_hat_tmp_cal, N_hat_tmp_cal, ktmp, corr_tmp = calibrate_with_truth(uf, tf, tau_hat_tmp, cfg_fwd.a, N_hat_tmp)
            r = rmse_trim(center_int_mean(tf, uf), center_int_mean(tau_hat_tmp_cal, uf), trim=0.02)
            row.append(r)
        RM_slice.append(row)
    with open("rmse_grid.csv","w", newline="") as fh:
        writer = csv.writer(fh); 
        writer.writerow(["cutoff\\sX"] + list(sX_scales))
        for i, dstr in enumerate(Daxis):
            writer.writerow([dstr] + [f"{x:.6f}" for x in RM_slice[i]])
    print("Saved rmse_grid.csv (noise=1.0 slice)")
