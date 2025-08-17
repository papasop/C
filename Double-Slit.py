import numpy as np
import matplotlib.pyplot as plt

# ================== Fraunhofer 双缝与稳健测距（与你前面一致/增强） ==================
def simulate_double_slit(lam=633e-9, L=1.2, d=250e-6, a=60e-6,
                         Nx=8192, X=5e-3, pad=6,
                         riemann_phase=False, phi_scale=0.25, sigma=1.2):
    x = np.linspace(-X/2, X/2, Nx)
    dx = x[1]-x[0]
    rect = lambda z: (np.abs(z) <= 0.5).astype(float)
    slit1 = rect((x + d/2)/a)
    slit2 = rect((x - d/2)/a)
    A = slit1 + slit2
    if riemann_phase:
        gammas = np.array([
            14.134725141, 21.022039639, 25.010857580, 30.424876125, 32.935061588,
            37.586178159, 40.918719012, 43.327073281, 48.005150881, 49.773832478,
            52.970321478, 56.446247697, 59.347044003, 60.831778525, 65.112544048,
            67.079810529, 69.546401711, 72.067157674, 75.704690699, 77.144840069,
            79.337375020, 82.910380854, 84.735492981, 87.425274613, 88.809111208,
            92.491899271, 94.651344041, 95.870634228, 98.831194218, 101.317851006,
            103.725538040, 105.446623052, 107.168611184, 111.029535543, 111.874659177,
            114.320220915, 116.226680321, 118.790782866, 121.370125002, 122.946829294,
            124.256818554, 127.516683880, 129.578704200, 131.087688531, 133.497737203,
            134.756509753, 138.116042055, 139.736208952, 141.123707404, 143.111845808
        ])
        u = (x - x.mean())/(a/2 + 1e-15) * 40.0 + 60.0
        U = (u[:, None] - gammas[None, :]) / sigma
        phi = phi_scale * np.arctan(U).sum(axis=1)
        phi -= phi.mean()
        A = A * np.exp(1j*phi)
    Nfft = int(pad * Nx)
    E_f = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A), n=Nfft)) * dx
    f = np.fft.fftshift(np.fft.fftfreq(Nfft, d=dx))
    x_screen = lam * L * f
    I = np.abs(E_f)**2
    I /= I.max() + 1e-30
    meta = {"lam": lam, "L": L, "d": d, "a": a, "dx": dx, "delta_x_theory": lam*L/d}
    return x_screen, I, meta

def _parabolic_peak_refine(f, mag, k):
    if k <= 0 or k >= len(mag)-1: return f[k]
    y1,y2,y3 = mag[k-1],mag[k],mag[k+1]
    denom = (y1 - 2*y2 + y3)
    delta = 0.5*(y1 - y3)/denom if abs(denom) > 1e-20 else 0.0
    return f[k] + delta*(f[k] - f[k-1])

def _peak_in_band_parabolic(f, mag, f_center, rel_bw):
    f_lo, f_hi = f_center*(1.0-rel_bw), f_center*(1.0+rel_bw)
    idx = np.where((f >= max(0.0, f_lo)) & (f <= f_hi))[0]
    if idx.size < 3: return None, None
    loc = idx[np.argmax(mag[idx])]
    loc = max(1, min(loc, len(mag)-2))
    f_ref = _parabolic_peak_refine(f, mag, loc)
    return f_ref, mag[loc]

def measure_fringe_spacing_harmonic_aware(x, I, dx_hint, rel_bw=0.08, max_harmonic=16):
    x = np.asarray(x); I = np.asarray(I)
    dx = x[1]-x[0]; N = len(x)
    I0 = I - I.mean()
    win = 0.5 - 0.5*np.cos(2*np.pi*np.arange(N)/(N-1))
    Iw = I0 * win
    F  = np.fft.rfft(Iw); f = np.fft.rfftfreq(N, d=dx); mag = np.abs(F)
    f1 = 1.0/(dx_hint + 1e-30)
    cands = []
    for h in range(1, max_harmonic+1):
        f_dom, m = _peak_in_band_parabolic(f, mag, h*f1, rel_bw)
        if f_dom is None or f_dom <= 0: continue
        dx_h = h / f_dom
        err = abs(dx_h - dx_hint)
        cands.append((err, -m, dx_h))
    if not cands: return np.nan
    cands.sort()
    return float(cands[0][2])

def measure_fringe_spacing_autocorr_windowed(x, I, dx_hint, win_ratio=(0.8,1.25)):
    x = np.asarray(x); I = np.asarray(I)
    dx = x[1]-x[0]
    I0 = I - I.mean()
    F = np.fft.rfft(I0); S = np.abs(F)**2
    R = np.fft.irfft(S, n=len(I0))
    R = R/(R[0] + 1e-30)
    k_lo = max(2, int(win_ratio[0]*dx_hint/dx))
    k_hi = min(len(R)-2, int(win_ratio[1]*dx_hint/dx))
    if k_hi <= k_lo+2: return np.nan
    peaks = []
    for k in range(k_lo+1, k_hi):
        if R[k] >= R[k-1] and R[k] >= R[k+1]:
            y1,y2,y3 = R[k-1],R[k],R[k+1]
            denom = (y1 - 2*y2 + y3)
            delta = 0.5*(y1 - y3)/denom if abs(denom) > 1e-20 else 0.0
            peaks.append((abs((k+delta)*dx - dx_hint), -(y2), k+delta))
    if not peaks: return np.nan
    peaks.sort()
    return float(peaks[0][2]*dx)

def measure_fringe_spacing_spatial_peaks(x, I, dx_hint, thr=0.2, band=(0.85,1.2)):
    x = np.asarray(x); I = np.asarray(I)
    dx = x[1]-x[0]
    kernel = np.array([0.25,0.5,0.25])
    I_s = np.convolve(I, kernel, mode='same')
    peaks, last_ix = [], -10**9
    min_sep = max(1, int(0.35*dx_hint/dx))
    thr_abs = thr * (I_s.max() + 1e-30)
    for i in range(1,len(I_s)-1):
        if I_s[i] >= I_s[i-1] and I_s[i] >= I_s[i+1] and I_s[i] >= thr_abs:
            if i - last_ix >= min_sep:
                peaks.append(i); last_ix = i
            else:
                if I_s[i] > I_s[peaks[-1]]:
                    peaks[-1] = i; last_ix = i
    if len(peaks) < 2: return np.nan
    xs = np.array([x[p] for p in peaks])
    spacings = np.diff(xs)
    sel = (spacings >= band[0]*dx_hint) & (spacings <= band[1]*dx_hint)
    spacings = spacings[sel]
    if spacings.size == 0: return np.nan
    k = len(spacings)
    if k >= 5:
        a,b = int(0.2*k), int(0.8*k)
        spacings = spacings[a:b] if b > a else spacings
    return float(np.median(np.abs(spacings)))

def measure_fringe_spacing_cosfit(x, I, dx_hint, window_mm=1.2e-3, f_scan_ppm=6000):
    x = np.asarray(x); I = np.asarray(I)
    mask = np.abs(x) <= window_mm/2
    xs, Is = x[mask], I[mask]
    if xs.size < 16: return np.nan
    f0 = 1.0/(dx_hint + 1e-30)
    df = f0 * (f_scan_ppm * 1e-6)
    fs = np.linspace(f0 - 3*df, f0 + 3*df, 121)
    def rss(freq):
        c = np.cos(2*np.pi*freq*xs); s = np.sin(2*np.pi*freq*xs)
        M = np.column_stack([np.ones_like(xs), c, s])
        p, *_ = np.linalg.lstsq(M, Is, rcond=None)
        r = Is - M.dot(p)
        return float(np.dot(r,r))
    vals = np.array([rss(f) for f in fs])
    j = int(np.argmin(vals))
    if 0 < j < len(fs)-1:
        f1,f2,f3 = fs[j-1],fs[j],fs[j+1]
        y1,y2,y3 = vals[j-1],vals[j],vals[j+1]
        denom = (y1 - 2*y2 + y3)
        delta = 0.5*(y1 - y3)/denom if abs(denom) > 1e-20 else 0.0
        f_ref = f2 + delta*(f2 - f1)
    else:
        f_ref = fs[j]
    return float(1.0 / f_ref) if f_ref > 0 else np.nan

def combine_measurements(cands, dx_ref, tol=0.12):
    vals = [v for v in cands if np.isfinite(v) and v > 0]
    if not vals: return np.nan
    med = float(np.median(vals))
    devs = [abs(v - med)/(dx_ref + 1e-30) for v in vals]
    if len(vals)>=2 and max(devs)>tol:
        idx = np.argsort(np.abs(np.array(vals)-med))
        v1,v2 = vals[idx[0]], vals[idx[1]]
        return float(np.sqrt(v1*v2))
    return med

def estimate_dx(x, I, dx_th,
                rel_bw=0.08, win_ratio=(0.8,1.25), band=(0.85,1.2),
                window_mm=1.2e-3, f_scan_ppm=6000):
    dx_f   = measure_fringe_spacing_harmonic_aware(x, I, dx_hint=dx_th, rel_bw=rel_bw, max_harmonic=16)
    dx_c   = measure_fringe_spacing_autocorr_windowed(x, I, dx_hint=dx_th, win_ratio=win_ratio)
    dx_s   = measure_fringe_spacing_spatial_peaks(x, I, dx_hint=dx_th, thr=0.2, band=band)
    dx_fit = measure_fringe_spacing_cosfit(x, I, dx_hint=dx_th, window_mm=window_mm, f_scan_ppm=f_scan_ppm)
    return combine_measurements([dx_f,dx_c,dx_s,dx_fit], dx_ref=dx_th, tol=0.12)

# ================== 1) 三条线性标度检验：扫 d、扫 λ、扫 L ==================
def line_fit(x, y):
    # 线性拟合 y = a*x + b，并给 R²
    p = np.polyfit(x, y, 1)
    yhat = np.polyval(p, x)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-30
    r2 = 1.0 - ss_res/ss_tot
    return p, r2

def scan_and_fit_scalings(a=60e-6, Nx=8192, pad=6,
                          lam0=633e-9, L0=1.2, d0=250e-6,
                          rel_bw=0.08, window_mm=1.2e-3):
    results = {}

    # 扫 d：Δx ~ (λL)/d 线性对应于 Δx vs 1/d
    ds = np.array([200e-6, 225e-6, 250e-6, 300e-6, 350e-6, 400e-6])
    invd = 1.0/ds
    dx_meas = []
    for d in ds:
        x,I,meta = simulate_double_slit(lam=lam0, L=L0, d=d, a=a, Nx=Nx, pad=pad, X=5e-3, riemann_phase=False)
        dxt = meta["delta_x_theory"]
        dx_meas.append(estimate_dx(x,I,dxt, rel_bw=rel_bw, window_mm=window_mm))
    dx_meas = np.array(dx_meas)
    p_d, r2_d = line_fit(invd, dx_meas)
    results['d'] = (ds, invd, dx_meas, p_d, r2_d)

    # 扫 λ：Δx ~ (λL)/d 线性对应于 Δx vs λ
    lams = np.array([488e-9, 532e-9, 594e-9, 633e-9, 700e-9])
    dx_meas = []
    for lam in lams:
        x,I,meta = simulate_double_slit(lam=lam, L=L0, d=d0, a=a, Nx=Nx, pad=pad, X=5e-3, riemann_phase=False)
        dxt = meta["delta_x_theory"]
        dx_meas.append(estimate_dx(x,I,dxt, rel_bw=rel_bw, window_mm=window_mm))
    dx_meas = np.array(dx_meas)
    p_l, r2_l = line_fit(lams, dx_meas)
    results['lam'] = (lams, lams, dx_meas, p_l, r2_l)

    # 扫 L：Δx ~ (λL)/d 线性对应于 Δx vs L
    Ls = np.array([0.8, 1.0, 1.2, 1.4, 1.6])
    dx_meas = []
    for L in Ls:
        x,I,meta = simulate_double_slit(lam=lam0, L=L, d=d0, a=a, Nx=Nx, pad=pad, X=5e-3, riemann_phase=False)
        dxt = meta["delta_x_theory"]
        dx_meas.append(estimate_dx(x,I,dxt, rel_bw=rel_bw, window_mm=window_mm))
    dx_meas = np.array(dx_meas)
    p_L, r2_L = line_fit(Ls, dx_meas)
    results['L'] = (Ls, Ls, dx_meas, p_L, r2_L)

    return results

# ================== 2) 对数回归：检验指数（应近 1,1,1） ==================
def log_regression_exponents(a=60e-6, Nx=8192, pad=6,
                             lam_grid=(488e-9, 532e-9, 633e-9),
                             L_grid=(0.9, 1.2, 1.5),
                             d_grid=(200e-6, 250e-6, 350e-6),
                             rel_bw=0.08, window_mm=1.2e-3):
    rows = []
    for lam in lam_grid:
        for L in L_grid:
            for d in d_grid:
                x,I,meta = simulate_double_slit(lam=lam, L=L, d=d, a=a, Nx=Nx, pad=pad, X=5e-3, riemann_phase=False)
                dxt = meta["delta_x_theory"]
                dxm = estimate_dx(x,I,dxt, rel_bw=rel_bw, window_mm=window_mm)
                rows.append([lam, L, d, dxm])
    M = np.array(rows)               # columns: lam, L, d, dx_meas
    lamv, Lv, dv, y = M[:,0], M[:,1], M[:,2], M[:,3]
    # log regression: log y = α log λ + β log L - γ log d + c
    X = np.column_stack([np.log(lamv), np.log(Lv), -np.log(dv), np.ones_like(y)])
    coef, *_ = np.linalg.lstsq(X, np.log(y), rcond=None)
    alpha, beta, gamma, c = coef
    yhat = np.exp(X @ coef)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-30
    r2 = 1.0 - ss_res/ss_tot
    return (alpha, beta, gamma, c), r2, (lamv, Lv, dv, y, yhat)

# ================== 3) 系统不确定度扫描 ==================
def systematics_sweep(lam=633e-9, L=1.2, d=250e-6, a=60e-6,
                      Nx_list=(4096,8192), pad_list=(4,6,8),
                      rel_bw_list=(0.06,0.08,0.10),
                      window_mm_list=(0.8e-3,1.0e-3,1.2e-3,1.5e-3),
                      noise_sigma_list=(0.0,0.01,0.02),
                      trials=2, seed=123):
    rng = np.random.default_rng(seed)
    records = []
    for Nx in Nx_list:
        for pad in pad_list:
            x,I,meta = simulate_double_slit(lam=lam, L=L, d=d, a=a, Nx=Nx, pad=pad, X=5e-3, riemann_phase=False)
            dxt = meta["delta_x_theory"]
            for rel_bw in rel_bw_list:
                for window_mm in window_mm_list:
                    for ns in noise_sigma_list:
                        for _ in range(trials):
                            if ns>0:
                                In = I + ns*rng.standard_normal(I.shape)
                                In = np.clip(In, 0, None)
                                In /= In.max() + 1e-30
                            else:
                                In = I
                            dxm = estimate_dx(x, In, dxt, rel_bw=rel_bw, window_mm=window_mm)
                            rel_err = (dxm - dxt)/(dxt + 1e-30)
                            records.append([Nx, pad, rel_bw, window_mm, ns, dxm, rel_err])
    return np.array(records)

# ================== 运行并可视化 ==================
def run_colab_block():
    # —— 基本参数（可按需改动）——
    lam0, L0, d0, a0 = 633e-9, 1.2, 250e-6, 60e-6

    # 1) 三条线性标度检验
    res = scan_and_fit_scalings(a=a0, Nx=8192, pad=6,
                                lam0=lam0, L0=L0, d0=d0,
                                rel_bw=0.08, window_mm=1.2e-3)

    # 打印与作图：d 扫（Δx vs 1/d）
    ds, invd, y_d, p_d, r2_d = res['d']
    print(f"[Scan d] Δx vs 1/d : slope={p_d[0]:.6e} m^2, intercept={p_d[1]:.3e} m, R^2={r2_d:.5f}")
    xx = np.linspace(invd.min(), invd.max(), 200)
    plt.figure(figsize=(4.8,3.4))
    plt.plot(invd*1e6, y_d*1e3, 'o', label='meas')
    plt.plot(xx*1e6, np.polyval(p_d, xx)*1e3, '-', label=f'fit, R$^2$={r2_d:.4f}')
    plt.xlabel('1/d (1/m) × 1e6'); plt.ylabel('Δx (mm)')
    plt.title('Scaling: Δx vs 1/d'); plt.legend(); plt.tight_layout(); plt.show()

    # λ 扫（Δx vs λ）
    lams, xx_l, y_l, p_l, r2_l = res['lam']
    print(f"[Scan λ] Δx vs λ : slope={p_l[0]:.6e} (m/m), intercept={p_l[1]:.3e} m, R^2={r2_l:.5f}")
    xx = np.linspace(lams.min(), lams.max(), 200)
    plt.figure(figsize=(4.8,3.4))
    plt.plot(lams*1e9, y_l*1e3, 'o', label='meas')
    plt.plot(xx*1e9, np.polyval(p_l, xx)*1e3, '-', label=f'fit, R$^2$={r2_l:.4f}')
    plt.xlabel('λ (nm)'); plt.ylabel('Δx (mm)')
    plt.title('Scaling: Δx vs λ'); plt.legend(); plt.tight_layout(); plt.show()

    # L 扫（Δx vs L）
    Ls, xx_L, y_L, p_L, r2_L = res['L']
    print(f"[Scan L] Δx vs L : slope={p_L[0]:.6e} (m/m), intercept={p_L[1]:.3e} m, R^2={r2_L:.5f}")
    xx = np.linspace(Ls.min(), Ls.max(), 200)
    plt.figure(figsize=(4.8,3.4))
    plt.plot(Ls, y_L*1e3, 'o', label='meas')
    plt.plot(xx, np.polyval(p_L, xx)*1e3, '-', label=f'fit, R$^2$={r2_L:.4f}')
    plt.xlabel('L (m)'); plt.ylabel('Δx (mm)')
    plt.title('Scaling: Δx vs L'); plt.legend(); plt.tight_layout(); plt.show()

    # 2) 对数回归（指数应接近 1,1,1）
    (alpha,beta,gamma,c), r2_log, (lamv,Lv,dv,y,yhat) = log_regression_exponents(
        a=a0, Nx=8192, pad=6,
        lam_grid=(488e-9, 532e-9, 633e-9, 700e-9),
        L_grid=(0.9, 1.2, 1.5),
        d_grid=(200e-6, 250e-6, 350e-6),
        rel_bw=0.08, window_mm=1.2e-3
    )
    print(f"[Log-Regression] log Δx ~ α log λ + β log L - γ log d + c")
    print(f"               α={alpha:.4f} (→1), β={beta:.4f} (→1), γ={gamma:.4f} (→1), R^2={r2_log:.5f}")
    plt.figure(figsize=(4.8,3.4))
    plt.scatter(y*1e3, yhat*1e3, s=22)
    m = max(y.max(), yhat.max())*1.05
    plt.plot([0,m*1e3],[0,m*1e3],'--',lw=1)
    plt.xlabel('Δx_meas (mm)'); plt.ylabel('Δx_pred (mm)')
    plt.title('Log-Regression: measured vs predicted'); plt.tight_layout(); plt.show()

    # 3) 系统不确定度扫描
    rec = systematics_sweep(lam=lam0, L=L0, d=d0, a=a0,
                            Nx_list=(4096,8192), pad_list=(4,6,8),
                            rel_bw_list=(0.06,0.08,0.10),
                            window_mm_list=(0.8e-3,1.0e-3,1.2e-3,1.5e-3),
                            noise_sigma_list=(0.0,0.01,0.02),
                            trials=2, seed=2025)
    # rec columns: Nx, pad, rel_bw, window_mm, noise, dx_meas, rel_err
    Nx_c, pad_c, rbw_c, win_c, noise_c, dxm_c, err_c = [rec[:,i] for i in range(rec.shape[1])]

    # 概要统计
    med_err = 100*np.median(err_c)
    mad_err = 100*np.median(np.abs(err_c - np.median(err_c)))
    print(f"[Systematics] median rel. error = {med_err:.3f}% ; MAD = {mad_err:.3f}%")
    # 误差直方图
    plt.figure(figsize=(5.0,3.2))
    plt.hist(100*err_c, bins=30, alpha=0.9)
    plt.xlabel('Relative error (%)'); plt.ylabel('Count')
    plt.title('Systematics sweep: Δx relative error distribution')
    plt.tight_layout(); plt.show()

    # 误差 vs 参数散点（四张小图）
    fig, axs = plt.subplots(2,2, figsize=(8.4,6.2))
    axs[0,0].scatter(Nx_c, 100*err_c, s=14); axs[0,0].set_xlabel('Nx'); axs[0,0].set_ylabel('Rel. error (%)'); axs[0,0].set_title('Error vs Nx')
    axs[0,1].scatter(pad_c, 100*err_c, s=14); axs[0,1].set_xlabel('pad'); axs[0,1].set_title('Error vs pad')
    axs[1,0].scatter(rbw_c, 100*err_c, s=14); axs[1,0].set_xlabel('rel_bw'); axs[1,0].set_ylabel('Rel. error (%)'); axs[1,0].set_title('Error vs rel_bw')
    axs[1,1].scatter(win_c*1e3, 100*err_c, s=14); axs[1,1].set_xlabel('window_mm (mm)'); axs[1,1].set_title('Error vs window_mm')
    plt.tight_layout(); plt.show()

    # 噪声鲁棒性：分组箱线（用散点近似：不同噪声水平用颜色区分）
    colors = {0.0:'C0', 0.01:'C1', 0.02:'C2'}
    plt.figure(figsize=(5.2,3.4))
    for ns in np.unique(noise_c):
        mask = (noise_c==ns)
        plt.scatter(np.full(mask.sum(), ns*100), 100*err_c[mask], s=16, label=f'noise {int(ns*100)}%')
    plt.xlabel('added noise std (%)'); plt.ylabel('Rel. error (%)')
    plt.title('Noise robustness'); plt.legend(); plt.tight_layout(); plt.show()

# —— 运行 —— 
run_colab_block()

