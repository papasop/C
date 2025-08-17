import numpy as np
import matplotlib.pyplot as plt

# ================== Fraunhofer 双缝 ==================
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
    f = np.fft.fftshift(np.fft.fftfreq(Nfft, d=dx))   # cycles/m
    x_screen = lam * L * f
    I = np.abs(E_f)**2
    I /= I.max() + 1e-30

    meta = {"lam": lam, "L": L, "d": d, "a": a, "dx": dx, "delta_x_theory": lam*L/d}
    return x_screen, I, meta

# ================== 工具：抛物线插值细化频率峰 ==================
def _parabolic_peak_refine(f, mag, k):
    if k <= 0 or k >= len(mag)-1:
        return f[k]
    y1, y2, y3 = mag[k-1], mag[k], mag[k+1]
    denom = (y1 - 2*y2 + y3)
    delta = 0.5*(y1 - y3)/denom if abs(denom) > 1e-20 else 0.0
    return f[k] + delta*(f[k] - f[k-1])

def _peak_in_band_parabolic(f, mag, f_center, rel_bw):
    f_lo, f_hi = f_center*(1.0-rel_bw), f_center*(1.0+rel_bw)
    idx = np.where((f >= max(0.0, f_lo)) & (f <= f_hi))[0]
    if idx.size < 3:
        return None, None
    loc = idx[np.argmax(mag[idx])]
    loc = max(1, min(loc, len(mag)-2))
    f_ref = _parabolic_peak_refine(f, mag, loc)
    return f_ref, mag[loc]

# ================== 频域 + 谐波感知（按接近度选谐波） ==================
def measure_fringe_spacing_harmonic_aware(x, I, dx_hint, rel_bw=0.08, max_harmonic=16):
    x = np.asarray(x); I = np.asarray(I)
    dx = x[1]-x[0]; N = len(x)
    I0 = I - I.mean()
    win = 0.5 - 0.5*np.cos(2*np.pi*np.arange(N)/(N-1))
    Iw = I0 * win
    F  = np.fft.rfft(Iw)
    f  = np.fft.rfftfreq(N, d=dx)
    mag = np.abs(F)
    f1 = 1.0/(dx_hint + 1e-30)

    candidates = []
    for h in range(1, max_harmonic+1):
        f_dom, m = _peak_in_band_parabolic(f, mag, h*f1, rel_bw)
        if f_dom is None or f_dom <= 0:
            continue
        dx_h = h / f_dom
        err = abs(dx_h - dx_hint)
        candidates.append((err, -m, h, dx_h, f_dom))
    if not candidates:
        return np.nan
    candidates.sort()
    return float(candidates[0][3])

# ================== 自相关（带搜索窗约束） ==================
def measure_fringe_spacing_autocorr_windowed(x, I, dx_hint, win_ratio=(0.8, 1.25)):
    x = np.asarray(x); I = np.asarray(I)
    dx = x[1]-x[0]
    I0 = I - I.mean()
    F = np.fft.rfft(I0)
    S = np.abs(F)**2
    R = np.fft.irfft(S, n=len(I0))
    R = R/(R[0] + 1e-30)

    k_lo = max(2, int(win_ratio[0]*dx_hint/dx))
    k_hi = min(len(R)-2, int(win_ratio[1]*dx_hint/dx))
    if k_hi <= k_lo+2:
        return np.nan

    peaks = []
    for k in range(k_lo+1, k_hi):
        if R[k] >= R[k-1] and R[k] >= R[k+1]:
            y1, y2, y3 = R[k-1], R[k], R[k+1]
            denom = (y1 - 2*y2 + y3)
            delta = 0.5*(y1 - y3)/denom if abs(denom) > 1e-20 else 0.0
            peaks.append((abs((k+delta)*dx - dx_hint), -(y2), k+delta))
    if not peaks:
        return np.nan
    peaks.sort()
    return float(peaks[0][2]*dx)

# ================== 空间域峰检测（带带通约束） ==================
def measure_fringe_spacing_spatial_peaks(x, I, dx_hint, thr=0.2, band=(0.85,1.2)):
    x = np.asarray(x); I = np.asarray(I)
    dx = x[1]-x[0]
    kernel = np.array([0.25, 0.5, 0.25])
    I_s = np.convolve(I, kernel, mode='same')

    peaks = []
    last_ix = -10**9
    min_sep = max(1, int(0.35 * dx_hint / dx))
    thr_abs = thr * (I_s.max() + 1e-30)
    for i in range(1, len(I_s)-1):
        if I_s[i] >= I_s[i-1] and I_s[i] >= I_s[i+1] and I_s[i] >= thr_abs:
            if i - last_ix >= min_sep:
                peaks.append(i); last_ix = i
            else:
                if I_s[i] > I_s[peaks[-1]]:
                    peaks[-1] = i; last_ix = i
    if len(peaks) < 2:
        return np.nan

    xs = np.array([x[p] for p in peaks])
    spacings = np.diff(xs)
    mask = (spacings >= band[0]*dx_hint) & (spacings <= band[1]*dx_hint)
    spacings = spacings[mask]
    if spacings.size == 0:
        return np.nan
    k = len(spacings)
    if k >= 5:
        start, end = int(0.2*k), int(0.8*k)
        spacings = spacings[start:end] if end > start else spacings
    return float(np.median(np.abs(spacings)))

# ================== 中心余弦拟合精修 ==================
def measure_fringe_spacing_cosfit(x, I, dx_hint, window_mm=1.2e-3, f_scan_ppm=6000):
    x = np.asarray(x); I = np.asarray(I)
    mask = np.abs(x) <= window_mm/2
    xs, Is = x[mask], I[mask]
    if xs.size < 16:
        return np.nan
    f0 = 1.0/(dx_hint + 1e-30)
    df = f0 * (f_scan_ppm * 1e-6)
    fs = np.linspace(f0 - 3*df, f0 + 3*df, 121)

    def rss_for(freq):
        c = np.cos(2*np.pi*freq*xs); s = np.sin(2*np.pi*freq*xs)
        M = np.column_stack([np.ones_like(xs), c, s])
        p, *_ = np.linalg.lstsq(M, Is, rcond=None)
        resid = Is - M.dot(p)
        return float(np.dot(resid, resid))

    rss_vals = np.array([rss_for(f) for f in fs])
    j = int(np.argmin(rss_vals))
    if 0 < j < len(fs)-1:
        f1, f2, f3 = fs[j-1], fs[j], fs[j+1]
        y1, y2, y3 = rss_vals[j-1], rss_vals[j], rss_vals[j+1]
        denom = (y1 - 2*y2 + y3)
        delta = 0.5*(y1 - y3)/denom if abs(denom) > 1e-20 else 0.0
        f_ref = f2 + delta*(f2 - f1)
    else:
        f_ref = fs[j]
    if f_ref <= 0:
        return np.nan
    return float(1.0 / f_ref)

# ================== 融合策略 ==================
def combine_measurements(candidates, dx_ref, tol=0.12):
    vals = [v for v in candidates if np.isfinite(v) and v > 0]
    if not vals:
        return np.nan
    med = float(np.median(vals))
    devs = [abs(v - med) / (dx_ref + 1e-30) for v in vals]
    if len(vals) >= 2 and max(devs) > tol:
        idx = np.argsort(np.abs(np.array(vals) - med))
        v1, v2 = vals[idx[0]], vals[idx[1]]
        return float(np.sqrt(v1*v2))
    return med

# ================== Demo ==================
def demo(run_riemann=True, tol=0.02):
    lam, L, d, a = 633e-9, 1.2, 250e-6, 60e-6
    print("=== Double-Slit (Fraunhofer) — Baseline ===")
    x, I, meta = simulate_double_slit(lam=lam, L=L, d=d, a=a, Nx=8192, X=5e-3, pad=6, riemann_phase=False)
    dx_th = meta["delta_x_theory"]

    dx_f = measure_fringe_spacing_harmonic_aware(x, I, dx_hint=dx_th, rel_bw=0.08, max_harmonic=16)
    dx_c = measure_fringe_spacing_autocorr_windowed(x, I, dx_hint=dx_th, win_ratio=(0.8,1.25))
    dx_s = measure_fringe_spacing_spatial_peaks(x, I, dx_hint=dx_th, thr=0.2, band=(0.85,1.2))
    dx_fit = measure_fringe_spacing_cosfit(x, I, dx_hint=dx_th, window_mm=1.2e-3, f_scan_ppm=6000)

    dx_meas = combine_measurements([dx_f, dx_c, dx_s, dx_fit], dx_ref=dx_th, tol=0.12)
    rel_err = abs(dx_meas - dx_th)/(dx_th + 1e-30)
    print(f"Theory Δx = {dx_th:.6e} m")
    print(f"Meas.  Δx = {dx_meas:.6e} m")
    print(f"Rel. error ≈ {rel_err:.3%}")

    if run_riemann:
        print("\n=== With Riemann Phase Mask (structure modulation) ===")
        xr, Ir, _ = simulate_double_slit(lam=lam, L=L, d=d, a=a,
                                         Nx=8192, X=5e-3, pad=6,
                                         riemann_phase=True, phi_scale=0.25, sigma=1.2)
        dx_fr  = measure_fringe_spacing_harmonic_aware(xr, Ir, dx_hint=dx_th, rel_bw=0.08, max_harmonic=16)
        dx_cr  = measure_fringe_spacing_autocorr_windowed(xr, Ir, dx_hint=dx_th, win_ratio=(0.8,1.25))
        dx_sr  = measure_fringe_spacing_spatial_peaks(xr, Ir, dx_hint=dx_th, thr=0.2, band=(0.85,1.2))
        dx_fitr= measure_fringe_spacing_cosfit(xr, Ir, dx_hint=dx_th, window_mm=1.2e-3, f_scan_ppm=6000)

        dx_meas_r = combine_measurements([dx_fr, dx_cr, dx_sr, dx_fitr], dx_ref=dx_th, tol=0.12)
        rel_err_r = abs(dx_meas_r - dx_th)/(dx_th + 1e-30)
        print(f"Theory Δx = {dx_th:.6e} m")
        print(f"Meas.  Δx (with phase mask) = {dx_meas_r:.6e} m")
        print(f"Rel. error ≈ {rel_err_r:.3%}")

        I0 = (I - I.mean())/(I.std()+1e-15)
        I1 = (Ir - Ir.mean())/(Ir.std()+1e-15)
        corr = float(np.dot(I0, I1)/len(I0))
        print(f"Corr(baseline, riemann-mask) ≈ {corr:.3f}")

    print(f"[SELF-TEST] baseline spacing within {tol*100:.1f}%: {abs(dx_meas-dx_th)/(dx_th+1e-30) < tol}")
    if run_riemann:
        print(f"[SELF-TEST] riemann-mask spacing within {tol*100:.1f}%: {abs(dx_meas_r-dx_th)/(dx_th+1e-30) < tol}")

    # 扫描 d 做标度律
    ds = np.array([200e-6, 250e-6, 300e-6, 350e-6, 400e-6])
    dx_ths = lam*L/ds
    dx_meas_list = []
    for d_i in ds:
        xs, Is, _ = simulate_double_slit(lam=lam, L=L, d=d_i, a=a, Nx=8192, X=5e-3, pad=6, riemann_phase=False)
        dxt = lam*L/d_i
        dx_fi   = measure_fringe_spacing_harmonic_aware(xs, Is, dx_hint=dxt, rel_bw=0.08, max_harmonic=16)
        dx_ci   = measure_fringe_spacing_autocorr_windowed(xs, Is, dx_hint=dxt, win_ratio=(0.8,1.25))
        dx_si   = measure_fringe_spacing_spatial_peaks(xs, Is, dx_hint=dxt, thr=0.2, band=(0.85,1.2))
        dx_fiti = measure_fringe_spacing_cosfit(xs, Is, dx_hint=dxt, window_mm=1.2e-3, f_scan_ppm=6000)
        dxi     = combine_measurements([dx_fi, dx_ci, dx_si, dx_fiti], dx_ref=dxt, tol=0.12)
        dx_meas_list.append(dxi)
    dx_meas_list = np.array(dx_meas_list)

    # 线性拟合
    p = np.polyfit(dx_ths, dx_meas_list, 1)
    xx = np.linspace(dx_ths.min(), dx_ths.max(), 200)
    fit = np.polyval(p, xx)
    ss_res = np.sum((dx_meas_list - np.polyval(p, dx_ths))**2)
    ss_tot = np.sum((dx_meas_list - dx_meas_list.mean())**2) + 1e-30
    r2 = 1.0 - ss_res/ss_tot
    print(f"[SCALING] Δx_meas ≈ {p[0]:.4f} * Δx_theory + {p[1]:.2e},  R^2 ≈ {r2:.5f}")

    # 绘图
    plt.figure(figsize=(7,3.2))
    plt.plot(x*1e3, I, label='Baseline')
    if run_riemann:
        plt.plot(xr*1e3, Ir, alpha=0.85, label='With Riemann phase')
    plt.xlabel('Screen x (mm)'); plt.ylabel('Normalized intensity')
    plt.title('Double-slit Fraunhofer pattern')
    plt.legend(); plt.tight_layout()
    plt.show()

    plt.figure(figsize=(4.6,3.6))
    plt.plot(dx_ths*1e3, dx_meas_list*1e3, 'o', label='measured')
    plt.plot(xx*1e3, fit*1e3, '-', label=f'fit: R$^2$={r2:.4f}')
    plt.xlabel('Δx_theory (mm)')
    plt.ylabel('Δx_measured (mm)')
    plt.title('Fringe spacing scaling')
    plt.legend(); plt.tight_layout()
    plt.show()

# 直接运行
demo(run_riemann=True, tol=0.02)
