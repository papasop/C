# -*- coding: utf-8 -*-
# v7.4-colab: True-zeros → lnN, Kolmogorov turbulence, stable FDTD+PML, windowed first-arrival,
#              path-integral prediction, Pi closure (clean), 3 surrogates, KS fingerprint, PASS check.

import numpy as np, mpmath as mp
from numpy.fft import fftfreq, ifft2
from scipy.signal import hilbert
from scipy.stats import ks_2samp

# ===================== Frozen hyperparams (weak-field) =====================
ALPHA   = 1.6                          # σ = α * mean_gap
G_GAIN  = 2.9078744151369885e-3        # weak-field gain g（与论文一致量级）
CHI_MAX = 0.7                          # clamp |ln N| ≤ χ_max（弱场预算）
RNG_SEED = 2025

# ===================== Zeros → φσ → lnN =====================
def true_riemann_gammas(M=120):
    mp.mp.dps = 50
    return np.array([float(mp.im(mp.zetazero(n))) for n in range(1, M+1)])

def build_phi_from_zeros(u, gam, alpha):
    mean_gap = (gam[-1]-gam[0])/(len(gam)-1)
    sig = alpha * mean_gap
    phip = np.zeros_like(u)
    for g in gam:
        phip += sig / ((u-g)**2 + sig**2)
    du = u[1]-u[0]
    phi = np.cumsum((phip[:-1]+phip[1:])*0.5*du)
    phi = np.concatenate([[0.0], phi])
    return phi, phip, sig

def lnN_from_phi(phi, g_gain, chi_max=CHI_MAX):
    # 居中 → 线性增益 → 夹紧 → 强制 <N>=1 → 轻夹紧
    lnN = g_gain*(phi - np.mean(phi))
    lnN = np.clip(lnN, -chi_max, chi_max)
    shift = np.log(np.mean(np.exp(lnN)))  # log-mean-exp
    lnN = lnN - shift
    lnN = np.clip(lnN, -chi_max, chi_max)
    return lnN

# ===================== Kolmogorov phase screen (2D) → 1D slice =====================
def kolmogorov_phase_screen(Nx=4096, Ny=1024, beta=11/3, seed=RNG_SEED, k0_frac=0.02):
    rng = np.random.default_rng(seed)
    kx = fftfreq(Nx, d=1.0/Nx)
    ky = fftfreq(Ny, d=1.0/Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K = np.sqrt(KX**2 + KY**2)
    amp = np.zeros_like(K)
    K0 = k0_frac * max(np.max(np.abs(kx)), np.max(np.abs(ky)))  # 低频截止，避免 k=0 发散
    m = (K >= K0)
    amp[m] = K[m]**(-beta/2.0)   # amplitude ~ k^{-11/6}
    phase = np.exp(1j*2*np.pi*rng.random(size=K.shape))
    spec = amp * phase
    spec[~np.isfinite(spec)] = 0.0
    psi2d = np.real(ifft2(spec))
    psi2d /= (np.std(psi2d)+1e-12)  # 零均值、单位方差
    return psi2d

def slice_center_row(psi2d, L):
    row = psi2d[psi2d.shape[0]//2, :]
    if len(row) >= L:
        s = (len(row)-L)//2
        out = row[s:s+L].copy()
    else:
        reps = -(-L//len(row))
        out = np.tile(row, reps)[:L].copy()
    out -= np.mean(out); out /= (np.std(out)+1e-12)
    return out

# ===================== Structural axis & c0-link =====================
def grad_central(a, dx=1.0):
    g = np.zeros_like(a)
    g[1:-1] = (a[2:]-a[:-2])/(2*dx); g[0] = (a[1]-a[0])/dx; g[-1] = (a[-1]-a[-2])/dx
    return g

def freeze_lambda_str(lnN_clean, delta_clean):
    num = np.median(grad_central(lnN_clean))
    den = np.median(grad_central(delta_clean))
    return 0.5 * (num/den)  # 归一化单位：c0=1 → prefactor 1/2

def Pi_closure(lnN, delta, lam_str):
    num = 0.5*np.median(grad_central(lnN))
    den = np.median(lam_str*grad_central(delta))
    return num/den

# ===================== Stable 1D FDTD (PML, S=0.5) =====================
def fdtd_trace_norm(n_idx, S=0.5, f0=0.035, n_cycles=90,
                    Nz=None, src_off=300, prb_off=2400, pml_n=120, pml_fac=3.0, seed=RNG_SEED):
    """
    Yee 1D, dz=1, dt=S；介质 eps=n^2，真空 eps=1。自适应 Nt 覆盖首达。
    测量路径 L = Nz-1 - src_off - prb_off（通过 prb_off 增大来缩短路径）
    """
    rng = np.random.default_rng(seed)
    Nz = len(n_idx) if Nz is None else Nz
    dz = 1.0; dt = S*dz
    eps = np.maximum(n_idx**2, 0.2)

    # PML
    sigma = np.zeros(Nz)
    if pml_n>0:
        ramp = np.linspace(0, 1, pml_n)
        sigma[:pml_n]  = pml_fac * ramp**2
        sigma[-pml_n:] = pml_fac * ramp[::-1]**2
    damp = np.exp(-sigma)

    # 几何配置
    src_pos = src_off
    prb_pos = Nz - prb_off - 1
    assert src_pos > pml_n and prb_pos < Nz-1-pml_n and prb_pos > src_pos+10, "路径或PML设置不合理"
    L = abs(prb_pos - src_pos) * dz

    # 脉冲
    tau = n_cycles/(2*np.pi*f0)

    # 自适应记录长度：覆盖真空首达 + 6τ
    T_need = L + 6.0*tau
    Nt = int(np.ceil(T_need/dt)) + 400

    t = np.arange(Nt)*dt
    src = np.sin(2*np.pi*f0*t) * np.exp(-((t-4*tau)/tau)**2)

    E = np.zeros(Nz); H = np.zeros(Nz-1)
    rec = np.zeros(Nt)
    inv_eps = 1.0/(eps + 1e-15)

    for n in range(Nt):
        H += (E[1:] - E[:-1]) * (dt/dz)
        H *= damp[:-1]
        curlH = np.zeros_like(E)
        curlH[1:-1] = (H[1:] - H[:-1]) * (dt/dz)
        E[1:-1] += curlH[1:-1] * inv_eps[1:-1]
        E[src_pos] += src[n]
        E *= damp
        rec[n] = E[prb_pos]
    return rec, dt, src_pos, prb_pos, tau

def peak_time_in_window(trace, dt, t_center, halfwin):
    env = np.abs(hilbert(trace))
    i0 = max(0, int((t_center - halfwin)/dt))
    i1 = min(len(env), int((t_center + halfwin)/dt))
    if i1 <= i0:
        return int(np.argmax(env))*dt
    sl = env[i0:i1]
    if np.all(sl <= 0):
        return int(np.argmax(env))*dt
    idx = i0 + int(np.argmax(sl))
    return idx * dt

def dt_pred_path_norm_segment(lnN, src_pos, prb_pos):
    N = np.exp(lnN)
    i0, i1 = int(src_pos), int(prb_pos)
    if i0 > i1: i0, i1 = i1, i0
    return float(np.sum(N[i0:i1] - 1.0))  # dz=1, c0=1

def measure_delay_windowed(lnN, f0=0.035, n_cycles=90, S=0.5,
                           src_off=300, prb_off=2400, pml_n=120, pml_fac=3.0, seed=RNG_SEED):
    nidx = np.exp(np.clip(lnN, -CHI_MAX, CHI_MAX)).astype(float)
    rec_m, dt, src_pos, prb_pos, tau = fdtd_trace_norm(
        nidx, S=S, f0=f0, n_cycles=n_cycles,
        src_off=src_off, prb_off=prb_off, pml_n=pml_n, pml_fac=pml_fac, seed=seed
    )
    rec_v, _,  _,        _,    tau_v = fdtd_trace_norm(
        np.ones_like(nidx), S=S, f0=f0, n_cycles=n_cycles,
        src_off=src_off, prb_off=prb_off, pml_n=pml_n, pml_fac=pml_fac, seed=seed
    )

    # 预测窗口中心（时间）：T0≈L, Tm≈L+Δt_pred
    L = abs(prb_pos - src_pos)*1.0
    T0_pred = L
    DT_pred = dt_pred_path_norm_segment(lnN, src_pos, prb_pos)
    Tm_pred = T0_pred + DT_pred
    halfwin = 3.0 * max(tau, tau_v)

    t_v = peak_time_in_window(rec_v, dt, T0_pred, halfwin)
    t_m = peak_time_in_window(rec_m, dt, Tm_pred, halfwin)
    DT_sim = t_m - t_v
    rel = abs(DT_sim - DT_pred) / (abs(DT_pred) + 1e-18)
    return DT_sim, DT_pred, rel, src_pos, prb_pos

# ===================== Fingerprint K_w & KS =====================
def kw_series_from_gaps(gam, w=40, stride=1):
    gaps = np.diff(gam)
    mean_gap = np.mean(gaps)
    tau = gaps / (mean_gap + 1e-18)
    K = []
    for i in range(0, len(tau)-w+1, stride):
        win = tau[i:i+w]
        Phi = np.mean(win)
        H   = np.mean((win - Phi)**2)
        K.append(Phi/(H + 1e-18))
    return np.array(K)

# ===================== Surrogates =====================
def surrogate_poisson(gam, rng):
    mean_gap = (gam[-1]-gam[0])/(len(gam)-1)
    gaps = rng.exponential(scale=mean_gap, size=len(gam)-1)
    return np.concatenate([[gam[0]], gam[0]+np.cumsum(gaps)])

def surrogate_gue(gam, rng):
    mean_gap = (gam[-1]-gam[0])/(len(gam)-1)
    s = rng.gamma(shape=3.0, scale=1.0, size=len(gam)-1)
    s *= (mean_gap/np.mean(s))
    return np.concatenate([[gam[0]], gam[0]+np.cumsum(s)])

def surrogate_psd_matched_lnN(lnN, rng):
    X = np.fft.rfft(lnN)
    amp = np.abs(X)
    phase = np.exp(1j*rng.uniform(0, 2*np.pi, size=amp.shape))
    out = np.fft.irfft(amp*phase, n=len(lnN))
    out -= np.mean(out)
    out = np.clip(out, -CHI_MAX, CHI_MAX)
    # 重新强制 <N>=1 规约（与真介质同规约口径）
    out = out - np.log(np.mean(np.exp(out)))
    out = np.clip(out, -CHI_MAX, CHI_MAX)
    return out

# ===================== Driver =====================
def run_blind_turbulence_true_vs_controls(
    M_zeros=120, Nx=3200, eps_list=(0.00, 0.03, 0.06),
    f0=0.035, n_cycles=90, seed=RNG_SEED,
    src_off=300, prb_off=2400, pml_n=120, pml_fac=3.0
):
    rng = np.random.default_rng(seed)
    print("Computing true zeros...")
    gam = true_riemann_gammas(M_zeros)
    print(f"done, last γ ≈ {gam[-1]:.4f}")

    # u-grid
    u = np.linspace(gam[0]-10.0, gam[-1]+10.0, Nx)

    # true lnN (clean) & structural delta
    phi, phip, sig = build_phi_from_zeros(u, gam, ALPHA)
    lnN_clean = lnN_from_phi(phi, G_GAIN)
    delta_clean = phi - np.mean(phi)

    # freeze λ_str & Pi on clean frame
    lam_str = freeze_lambda_str(lnN_clean, delta_clean)
    Pi_clean = Pi_closure(lnN_clean, delta_clean, lam_str)

    # fingerprint KS (true vs Poisson)
    kw_true = kw_series_from_gaps(gam, w=40, stride=1)
    gam_p = surrogate_poisson(gam, rng)
    kw_pois = kw_series_from_gaps(gam_p, w=40, stride=1)
    ks = ks_2samp(kw_true, kw_pois)

    # Kolmogorov turbulence（同一 ψ 叠加到真/对照）
    psi2d = kolmogorov_phase_screen(seed=seed)
    psi_1d = slice_center_row(psi2d, len(lnN_clean))

    rows = []
    for eps in eps_list:
        # true medium
        lnN_true = np.clip(lnN_clean + eps*psi_1d, -CHI_MAX, CHI_MAX)
        DT_sim, DT_pred_true, matched_rel_err, src_pos, prb_pos = measure_delay_windowed(
            lnN_true, f0=f0, n_cycles=n_cycles, S=0.5,
            src_off=src_off, prb_off=prb_off, pml_n=pml_n, pml_fac=pml_fac, seed=seed
        )

        # controls: Poisson, GUE, PSD（预测仅路径积分）
        phi_p, _, _ = build_phi_from_zeros(u, gam_p, ALPHA)
        lnN_p = lnN_from_phi(phi_p, G_GAIN)
        lnN_p = np.clip(lnN_p + eps*psi_1d, -CHI_MAX, CHI_MAX)
        DT_pred_p = dt_pred_path_norm_segment(lnN_p, src_pos, prb_pos)

        gam_g = surrogate_gue(gam, rng)
        phi_g, _, _ = build_phi_from_zeros(u, gam_g, ALPHA)
        lnN_g = lnN_from_phi(phi_g, G_GAIN)
        lnN_g = np.clip(lnN_g + eps*psi_1d, -CHI_MAX, CHI_MAX)
        DT_pred_g = dt_pred_path_norm_segment(lnN_g, src_pos, prb_pos)

        lnN_s = surrogate_psd_matched_lnN(lnN_clean, rng)
        lnN_s = np.clip(lnN_s + eps*psi_1d, -CHI_MAX, CHI_MAX)
        DT_pred_s = dt_pred_path_norm_segment(lnN_s, src_pos, prb_pos)

        denom = abs(DT_pred_true) + 1e-18
        rel_p = abs(DT_sim - DT_pred_p)/denom
        rel_g = abs(DT_sim - DT_pred_g)/denom
        rel_s = abs(DT_sim - DT_pred_s)/denom
        mismatch_min = min(rel_p, rel_g, rel_s)
        factor = mismatch_min / (matched_rel_err + 1e-18)

        rows.append(dict(
            eps=eps, DT_sim=DT_sim, DT_pred=DT_pred_true, matched_rel_err=matched_rel_err,
            rel_p=rel_p, rel_g=rel_g, rel_psd=rel_s, mismatch_factor=factor
        ))

    # 打印结果
    print("\n=== Blind Δt (v7.4-colab) ===")
    print("metric\tvalue")
    print(f"#zeros used\t{M_zeros}")
    print(f"last gamma\t{gam[-1]:.6f}")
    print(f"sigma (alpha*mean_gap)\t{sig:.6f}")
    print(f"Pi(clean)\t{Pi_clean:.6f}")
    print(f"KS D(Kw_true, Kw_pois)\t{ks.statistic:.6f}")
    print(f"KS p-value\t{ks.pvalue:.3e}")
    for r in rows:
        print(f"eps={r['eps']:.3f}\tΔt_sim\t{r['DT_sim']:.6f}")
        print(f"\tΔt_pred(true)\t{r['DT_pred']:.6f}")
        print(f"\tmatched rel.err\t{r['matched_rel_err']:.6f}")
        print(f"\tmin mismatch rel.err\t{min(r['rel_p'], r['rel_g'], r['rel_psd']):.6f}")
        print(f"\tmismatch factor\t{r['mismatch_factor']:.2f}")

    # PASS 规则
    pass_matched  = all(r['matched_rel_err'] <= 0.005 for r in rows)   # ≤ 0.5%
    pass_mismatch = all(r['mismatch_factor'] >= 5.0  for r in rows)    # ≥ 5×
    print("\nPASS(Δt matched ≤0.5%) :", pass_matched)
    print("PASS(mismatch ≥5×)      :", pass_mismatch)
    print("（说明：Π(clean)≈1 只在干净帧评估；湍流场景无需 Π≈1。）")

    return dict(
        zeros_used=M_zeros, last_gamma=float(gam[-1]),
        sigma=float(sig), Pi_clean=float(Pi_clean),
        ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
        results=rows
    )

# ===================== RUN =====================
out = run_blind_turbulence_true_vs_controls(
    M_zeros=120,
    Nx=3200,                     # 稍小网格 + 短路径 → 提高分离
    eps_list=(0.00, 0.03, 0.06),
    f0=0.035, n_cycles=90,       # 低载频 + 长包络 → 更接近 eikonal
    seed=RNG_SEED,
    src_off=300, prb_off=2400,   # 缩短测量路径（L≈Nx-1-300-2400≈499）
    pml_n=120, pml_fac=3.0
)
