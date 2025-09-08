# %% [single-cell Colab]  1D c=1 geometry: blind windows, zeta-phase vs baselines (pi/phase)
# ------------------------------------------------------------
# CONFIG
TRUTH_MODEL   = 'zeta'   # 'zeta' | 'pi' | 'phase' （可切换做“可证伪”负控）
L             = 100.0    # u-轴长度
NPTS_PHI      = 600      # 计算 φ'(u) 的粗网格点数（越大越准但更慢）
NPTS_EVAL     = 1600     # 评估/积分网格（越大越细）
SIGMA         = 0.20     # s(u)=1/2+σ+iu 里的 σ
G             = 0.12     # 幅度耦合 g
NWINDOWS      = 80       # 盲窗数量
MIN_LEN       = 4.0      # 每窗最短长度
MAX_LEN       = 30.0     # 每窗最长长度
NOISE_STD     = 0.005    # 观测噪声（对 ∆t_meas）
SEED          = 2025     # 随机种子
MI_BINS       = 32       # 互信息的分箱数
DPS           = 60       # mpmath 小数精度（越大越稳）
# ------------------------------------------------------------

import numpy as np, mpmath as mp
from scipy.stats import wilcoxon, spearmanr
np.set_printoptions(suppress=True, precision=4)

# ---------- helpers ----------
def xi(s):
    # Riemann xi(s) = 1/2 * s(s-1) * π^{-s/2} * Γ(s/2) * ζ(s)
    return mp.mpf('0.5') * s*(s-1) * mp.power(mp.pi, -s/2) * mp.gamma(s/2) * mp.zeta(s)

def dlog_xi_real(s):
    # ϕ'(u) = Re d/ds log xi(s), s=1/2+σ+i u
    return mp.re( mp.diff(lambda z: mp.log(xi(z)), s) )

def cumtrapz(y, x):
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(0.5*(y[1:]+y[:-1])*(x[1:]-x[:-1]))
    return out

def trapz(x, y):
    return float(np.trapz(y, x))

def zscore(x):
    mu, sd = np.mean(x), np.std(x)
    return (x - mu) / (sd + 1e-12)

def mutual_information_1lag(x, nbins=32):
    # x 已是实数序列（这里用 z-score 后的 N_truth 或 τ_truth）
    if len(x) < 8: return np.nan
    x = zscore(np.asarray(x, float))
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    x = np.clip(x, lo, hi)
    edges = np.linspace(lo, hi, nbins+1)
    x0, x1 = x[:-1], x[1:]
    H2, _, _ = np.histogram2d(x0, x1, bins=[edges, edges], density=False)
    Hx, _ = np.histogram(x0, bins=edges, density=False)
    Hy, _ = np.histogram(x1, bins=edges, density=False)
    N = H2.sum()
    if N == 0: return np.nan
    Pxy = H2 / N
    Px  = Hx / N
    Py  = Hy / N
    with np.errstate(divide='ignore', invalid='ignore'):
        MI = np.nansum( Pxy * (np.log(Pxy + 1e-15) - np.log(Px[:,None] + 1e-15) - np.log(Py[None,:] + 1e-15)) )
    return float(MI)

def fft_phase_scramble(x, seed=0):
    # 保幅度谱，随机相位（实序列->实序列）
    rng = np.random.default_rng(seed)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    ph  = rng.uniform(0, 2*np.pi, size=mag.shape)
    ph[0] = 0.0
    if mag.size > 1: ph[-1] = 0.0
    Y = mag * np.exp(1j*ph)
    y = np.fft.irfft(Y, n=len(x))
    return y

def smooth_gaussian(x, sigma=5.0):
    # 简易频域高斯平滑
    n = len(x)
    k = np.fft.rfftfreq(n)
    X = np.fft.rfft(x)
    G = np.exp(-(k**2)*(sigma**2))
    y = np.fft.irfft(X*G, n=n)
    return y

# ---------- ζ-phase on coarse grid then interp ----------
def build_phi_and_N(u_eval, sigma=0.2, g=0.12, dps=60, verbose=True):
    u0, u1 = float(u_eval[0]), float(u_eval[-1])
    u_phi = np.linspace(u0, u1, NPTS_PHI)
    phi_p = np.zeros_like(u_phi, dtype=float)
    if verbose:
        print("\n[STEP] computing ζ-phase (coarse then interp) …")
    with mp.workdps(dps):
        for i, uu in enumerate(u_phi):
            s = mp.mpf('0.5') + sigma + 1j*uu
            phi_p[i] = float(dlog_xi_real(s))
            # 进度条
            if verbose and NPTS_PHI>=100 and (i % max(1, NPTS_PHI//10) == 0) and i>0:
                pct = int(100*i/NPTS_PHI)
                print(f"  ζ-phase coarse: {pct}% done")
    # integrate to phi
    phi_c = cumtrapz(phi_p, u_phi)
    phi_c -= phi_c.mean()
    # interp to eval grid
    phi = np.interp(u_eval, u_phi, phi_c)
    phi -= phi.mean()
    N   = np.exp(g * phi)
    N  /= np.mean(N)  # enforce <N>=1
    return phi, N

# ---------- surrogates ----------
def build_pi_tau(u_eval, seed=0, digits=4096, smooth=6.0):
    # 取 π 的小数 digits 位，映射到 [-1,1]，再平滑到连续 τ(u)
    mp.mp.dps = digits + 10
    s = mp.nstr(mp.pi, digits+5)  # "3.1415..."
    s = s.replace('.', '')[1:]    # 去掉 "3."
    arr = np.array([ord(c)-48 for c in s[:digits]], dtype=float)  # 0-9
    v = (arr - 4.5)/4.5
    # 展开到 eval 长度
    idx = (np.linspace(0, len(v)-1e-6, len(u_eval))).astype(int)
    raw = v[idx]
    raw = smooth_gaussian(raw, sigma=smooth)
    raw -= raw.mean()
    raw /= (np.std(raw)+1e-12)
    return raw

def build_gue_tau(u_eval, seed=0, smooth=6.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(len(u_eval))
    y = smooth_gaussian(x, sigma=smooth)
    y -= y.mean()
    y /= (np.std(y)+1e-12)
    return y

def build_poisson_tau(u_eval, seed=0, smooth=6.0):
    rng = np.random.default_rng(seed)
    x = rng.exponential(scale=1.0, size=len(u_eval)) - 1.0
    y = smooth_gaussian(x, sigma=smooth)
    y -= y.mean()
    y /= (np.std(y)+1e-12)
    return y

# ---------- build N(u) for each model ----------
def N_from_tau(tau, g=0.12):
    tau = tau - np.mean(tau)
    N = np.exp(g * tau)
    N /= np.mean(N)
    return N

# ---------- main pipeline ----------
def main():
    rng = np.random.default_rng(SEED)
    u    = np.linspace(0.0, L, NPTS_EVAL)
    # ζ
    phi_z, N_z = build_phi_and_N(u, sigma=SIGMA, g=G, dps=DPS, verbose=True)
    # phase-scramble 基于 ζ 的 φ
    tau_phase = fft_phase_scramble(phi_z, seed=SEED)
    tau_phase = smooth_gaussian(tau_phase, sigma=6.0)
    tau_phase = zscore(tau_phase)
    # π / GUE / Poisson
    tau_pi  = build_pi_tau(u,  seed=SEED+1, digits=4096, smooth=6.0)
    tau_gue = build_gue_tau(u, seed=SEED+2, smooth=6.0)
    tau_poi = build_poisson_tau(u, seed=SEED+3, smooth=6.0)

    N_pi     = N_from_tau(tau_pi,  g=G)
    N_gue    = N_from_tau(tau_gue, g=G)
    N_poi    = N_from_tau(tau_poi, g=G)
    N_phase  = N_from_tau(tau_phase, g=G)

    # 选“真模型”
    truth_map = {
        'zeta'  : N_z,
        'pi'    : N_pi,
        'gue'   : N_gue,
        'poisson': N_poi,
        'phase' : N_phase
    }
    N_truth = truth_map[TRUTH_MODEL]

    print(f"\n[SETUP]\n  TRUTH_MODEL={TRUTH_MODEL},  L={L},  Npts_eval={NPTS_EVAL},  Npts_phi={NPTS_PHI},  sigma={SIGMA}, g={G}")

    # 随机盲窗
    wins = []
    for k in range(NWINDOWS):
        length = rng.uniform(MIN_LEN, MAX_LEN)
        a = rng.uniform(0, L-length)
        b = a + length
        wins.append((a,b))

    # 逐窗评估
    headers = "win,u1,u2,err_zeta,err_pi,err_phase,MI_zeta,dt_meas,dt_pred_zeta,dt_pred_pi,dt_pred_phase"
    print("\n"+headers)
    errZ, errP, errF = [], [], []
    MI_list = []

    def seg_int(Nfield, a, b):
        m = (u>=a) & (u<=b)
        return trapz(u[m], (Nfield[m] - 1.0))

    for i,(a,b) in enumerate(wins, 1):
        # “观测” ∆t（真模型 + 噪声）
        dt_true = seg_int(N_truth, a, b)
        dt_meas = dt_true + rng.normal(0.0, NOISE_STD)

        # 各模型预测
        dt_z = seg_int(N_z   , a, b)
        dt_p = seg_int(N_pi  , a, b)
        dt_f = seg_int(N_phase, a, b)

        eZ = abs(dt_meas - dt_z)
        eP = abs(dt_meas - dt_p)
        eF = abs(dt_meas - dt_f)

        # 窗口 MI（用“真模型”的 N 做 1-lag MI）
        mwin = (u>=a)&(u<=b)
        MIz  = mutual_information_1lag(N_truth[mwin], nbins=MI_BINS)

        errZ.append(eZ); errP.append(eP); errF.append(eF); MI_list.append(MIz)
        print(f"{i:02d},{a:.2f},{b:.2f},{eZ:.4f},{eP:.4f},{eF:.4f},{MIz:.4f},{dt_meas:.4f},{dt_z:.4f},{dt_p:.4f},{dt_f:.4f}")

    errZ = np.array(errZ); errP = np.array(errP); errF = np.array(errF); MI_arr = np.array(MI_list)

    # 汇总
    def summary_row(name, e):
        return f"{name:<8s} MAE={np.nanmean(e):.6f}  RMSE={np.sqrt(np.nanmean(e**2)):.6f}"

    print("\n[ERROR SUMMARY  (lower is better)]")
    print(summary_row('zeta',  errZ))
    print(summary_row('pi',    errP))
    print(summary_row('phase', errF))

    # Wilcoxon (paired, H1: |err_zeta| < |err_alt|), Holm 校正
    def wilcoxon_less(a, b):
        # 比较 |a| 与 |b|，检验 a 更小
        A, B = np.abs(a), np.abs(b)
        # 避免全相等导致报错
        if np.allclose(A, B):
            return 0.5, 0.0, 0.0
        stat, p = wilcoxon(A, B, alternative='less', zero_method='wilcox')  # 配对
        # rank-biserial
        diff = A - B
        npos = np.count_nonzero(diff > 0)
        nneg = np.count_nonzero(diff < 0)
        r_rb = (npos - nneg) / (npos + nneg + 1e-12)
        med  = float(np.nanmedian(A - B))
        return p, r_rb, med

    pairs = [('zeta','pi',errZ,errP), ('zeta','phase',errZ,errF)]
    pvals = []
    stats = []
    for a,b,Ea,Eb in pairs:
        p, r, med = wilcoxon_less(Ea, Eb)
        pvals.append(p); stats.append((a,b,p,r,med))
    # Holm
    m = len(pvals)
    order = np.argsort(pvals)
    holm = np.ones(m)
    for rank,idx in enumerate(order, start=1):
        holm[idx] = min(1.0, pvals[idx]*(m-rank+1))

    print("\n[WILCOXON:  |err_zeta|  <  |err_alt|   (paired, Holm-corrected)]")
    for (a,b,p,r,med), ph in zip(stats, holm):
        print(f" {a} vs {b:5s}  p={p:.3e},  p_holm={ph:.3e},  r_rb={r:+.3f},  median(Δ|err|)={med:.3f}")

    # 相关性（Spearman）
    def sp(x, y):
        v = np.isfinite(x) & np.isfinite(y)
        rho, p = spearmanr(x[v], y[v])
        return float(rho), float(p)

    rhoZ, pZ = sp(MI_arr, np.abs(errZ))
    rhoP, pP = sp(MI_arr, np.abs(errP))
    rhoF, pF = sp(MI_arr, np.abs(errF))

    print("\n[CORRELATION  (Spearman, MI_truth vs |err|)]")
    print(f" rho(MI, |err_zeta|)  = {rhoZ:+.3f}  (p={pZ:.3e})")
    print(f" rho(MI, |err_pi|)    = {rhoP:+.3f}  (p={pP:.3e})")
    print(f" rho(MI, |err_phase|) = {rhoF:+.3f}  (p={pF:.3e})")

# ---------- run ----------
if __name__ == "__main__":
    main()

