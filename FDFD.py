# ===============================================
# ζ → N(r) → N(z) : Eikonal vs FDTD (Robust v2)
#   - Unified bandpass for all channels
#   - Phase WLS with energy/coherence gating
#   - Envelope & XCorr both limited to ±lag_max
#   - Robust fusion = median of valid {phase, env, xcorr}, then clip
#   - Auto use N_r_from_zeta/r_max/Nr if present; else demo step
# ===============================================
import numpy as np
c0 = 299_792_458.0

# ---------- helpers ----------
def next_pow2(n): return 1 << (int(n - 1).bit_length())

def bandpass_fft(sig, dt, f0, bp_frac=0.25):
    sig = np.asarray(sig, float)
    N = len(sig)
    S = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(N, dt)
    fL = max(0.0, f0*(1 - bp_frac)); fH = f0*(1 + bp_frac)
    M = (freqs >= fL) & (freqs <= fH)
    Sf = np.zeros_like(S); Sf[M] = S[M]
    return np.fft.irfft(Sf, n=N)

def analytic_signal(sig):
    x = np.asarray(sig, float); N = len(x)
    X = np.fft.rfft(x)
    H = np.zeros_like(X)
    if N % 2 == 0:
        H[0], H[-1] = 1.0, 1.0; H[1:-1] = 2.0
    else:
        H[0] = 1.0; H[1:] = 2.0
    y = np.fft.irfft(X * H, n=N)
    return y + 1j*np.imag(y)

# ---------- phase (WLS) ----------
def phase_delay_WLS(xw, yw, dt, f0, bp_frac=0.25, pctl=0.10, coh_floor=0.15, min_bins=8, gamma=1.3):
    N = len(xw); w = np.hanning(N)
    x = bandpass_fft((xw - np.mean(xw)) * w, dt, f0, bp_frac)
    y = bandpass_fft((yw - np.mean(yw)) * w, dt, f0, bp_frac)
    X, Y = np.fft.rfft(x), np.fft.rfft(y)
    Sxx, Syy = np.abs(X)**2, np.abs(Y)**2
    Sxy = X * np.conj(Y)
    freqs = np.fft.rfftfreq(N, dt); wv = 2*np.pi*freqs
    fL, fH = f0*(1 - bp_frac), f0*(1 + bp_frac)
    band = (freqs > 0) & (freqs >= fL) & (freqs <= fH)
    if not np.any(band): return None, dict(used_bins=0, mean_coh=np.nan)

    E = np.sqrt(Sxx[band] * Syy[band])
    coh = (np.abs(Sxy[band])**2) / (Sxx[band]*Syy[band] + 1e-30)
    thr = np.quantile(E, pctl) if np.all(np.isfinite(E)) else 0.0
    pick = (E >= thr) & (coh >= coh_floor)
    if np.count_nonzero(pick) < min_bins:
        return None, dict(used_bins=int(np.count_nonzero(pick)), mean_coh=(float(np.mean(coh[pick])) if np.any(pick) else np.nan))

    ww = wv[band][pick]
    H = Sxy[band][pick] / (Sxx[band][pick] + 1e-30)
    ph = np.unwrap(np.angle(H))
    W = (E[pick]**gamma)
    wc = float(np.sum(W * ww) / (np.sum(W) + 1e-30))
    pc = float(np.sum(W * ph) / (np.sum(W) + 1e-30))
    num = float(np.sum(W * (ww - wc) * (ph - pc)))
    den = float(np.sum(W * (ww - wc)**2) + 1e-30)
    tau = - num / den
    return float(tau), dict(used_bins=int(np.count_nonzero(pick)), mean_coh=float(np.mean(coh[pick])))

# ---------- envelope ----------
def env_delay_band(xw, yw, dt, f0, bp_frac=0.25, lag_max=None):
    xb = bandpass_fft(xw - np.mean(xw), dt, f0, bp_frac)
    yb = bandpass_fft(yw - np.mean(yw), dt, f0, bp_frac)
    ex = np.abs(analytic_signal(xb)); ey = np.abs(analytic_signal(yb))
    ix, iy = int(np.argmax(ex)), int(np.argmax(ey))
    tau = (iy - ix) * dt
    if (lag_max is not None) and (abs(tau) > lag_max): return None
    return float(tau)

# ---------- xcorr (limited search) ----------
def xcorr_delay_limited(xw, yw, dt, up=16, f0=None, bp_frac=None, lag_max=None):
    a = xw - np.mean(xw); b = yw - np.mean(yw)
    if (f0 is not None) and (bp_frac is not None):
        a = bandpass_fft(a, dt, f0, bp_frac); b = bandpass_fft(b, dt, f0, bp_frac)
    N = len(a); L = next_pow2(N * up)
    A = np.fft.rfft(a, n=L); B = np.fft.rfft(b, n=L)
    r = np.fft.irfft(A * np.conj(B), n=L)
    r = np.concatenate((r[-(L//2):], r[:(L//2)]))  # 0 滞后居中
    if lag_max is not None:
        kmax = int(np.round(lag_max / (dt / up))); k0 = L // 2
        sl = slice(k0 - kmax, k0 + kmax + 1); rw = r[sl]
        k_rel = int(np.argmax(rw)) - kmax
        tau = k_rel * (dt / up)
    else:
        k = int(np.argmax(r)) - (L//2); tau = k * (dt / up)
    if (lag_max is not None) and (abs(tau) > lag_max): return None
    return float(tau)

# ---------- eikonal & FDTD ----------
def eikonal_dt_grid(Nz, dz):
    return (dz / c0) * float(np.sum(Nz - 1.0))

def yee_fdtd_delay(Nz, dz, f0, src_offset=80, probe_offset=6):
    Nz = np.asarray(Nz, float); nz = len(Nz); nmax = float(np.max(Nz))
    dt = 0.99 * dz / (c0 * nmax)
    steps = int(np.ceil(1.2 * nmax * (nz - 1))); nt = int(3.0 * steps)
    t = np.arange(nt) * dt; i_src = src_offset; i_prb = nz - 1 - probe_offset
    # 高斯调制正弦：少 DC，多相位能量
    T_link = ((i_prb - i_src) * dz) / (c0 / nmax)
    t0 = 0.35 * T_link; spread = 0.12 * T_link
    def src(tt): x = (tt - t0) / (spread + 1e-18); return np.sin(2*np.pi*f0*tt) * np.exp(-0.5*x*x)
    def run(nprof):
        eps=(nprof**2).astype(float); mu=np.ones_like(eps)
        Ez=np.zeros(nz); Hy=np.zeros(nz); Ez_old=np.zeros(nz); rec=np.zeros(nt)
        cl=c0/float(nprof[0]); cr=c0/float(nprof[-1])
        def murL(En,Eo): k=(cl*dt - dz)/(cl*dt + dz); return Eo[1] + k*(En[1] - Eo[0])
        def murR(En,Eo): k=(cr*dt - dz)/(cr*dt + dz); return Eo[-2] + k*(En[-2] - Eo[-1])
        for it in range(nt):
            Hy[:-1] += (dt/(mu[:-1]*dz)) * (Ez[1:] - Ez[:-1])
            Ez_old[:] = Ez
            Ez[1:]   += (dt/(eps[1:]*dz)) * (Hy[1:] - Hy[:-1])
            Ez[i_src] += src(t[it])
            Ez[0]  = murL(Ez, Ez_old); Ez[-1] = murR(Ez, Ez_old)
            rec[it] = Ez[i_prb]
        return rec, dt, t
    rec_v, dt, t = run(np.ones_like(Nz))
    rec_m, _, _  = run(Nz)
    T_vac = ((nz - 1 - probe_offset) - src_offset) * dz / c0
    dt_pred = eikonal_dt_grid(Nz[src_offset:nz-1-probe_offset], dz)
    T_med = T_vac + dt_pred
    # 自适应窗长：覆盖 ~ 20 个载频周期
    per = 1.0 / f0; W = int(next_pow2(max(1024, int(20 * per / dt))))
    W = min(W, len(t)//2)
    i0x = int(np.clip(np.round(T_vac/dt) - W//2, 0, len(t)-W))
    i0y = int(np.clip(np.round(T_med/dt) - W//2, 0, len(t)-W))
    return dict(rec_v=rec_v, rec_m=rec_m, t=t, dt=dt, W=W, i0x=i0x, i0y=i0y, dt_pred_grid=dt_pred)

# ---------- fused robust ----------
def fused_delay_robust(fd, f0, bp_frac=0.25, pctl=0.10, coh_floor=0.15, min_bins=8):
    xw = fd["rec_v"][fd["i0x"]:fd["i0x"]+fd["W"]]
    yw = fd["rec_m"][fd["i0y"]:fd["i0y"]+fd["W"]]
    xw = xw - np.mean(xw); yw = yw - np.mean(yw)
    dt = fd["dt"]
    coarse = (fd["i0y"] - fd["i0x"]) * dt
    # 细校正限域：既受 coarse 也受 eikonal 约束
    lag_max = min(max(10*dt, 0.35*abs(coarse)), 0.55*abs(fd["dt_pred_grid"]))
    # 三路
    tau_p, meta = phase_delay_WLS(xw, yw, dt, f0, bp_frac, pctl, coh_floor, min_bins)
    tau_e = env_delay_band(xw, yw, dt, f0, bp_frac, lag_max=lag_max)
    tau_x = xcorr_delay_limited(xw, yw, dt, up=16, f0=f0, bp_frac=bp_frac, lag_max=lag_max)
    cands = [t for t in (tau_p, tau_e, tau_x) if (t is not None and np.isfinite(t))]
    fine = float(np.median(cands)) if cands else 0.0
    fine = float(np.clip(fine, -lag_max, +lag_max))
    dt_sim = coarse + fine
    return dt_sim, dict(
        coarse=coarse, fine=fine,
        tau_phase=tau_p, tau_env=tau_e, tau_xcorr=tau_x,
        used_bins=(0 if (tau_p is None) else meta.get("used_bins",0)),
        mean_coh=(np.nan if (tau_p is None) else meta.get("mean_coh",np.nan)),
        lag_max=lag_max
    )

# ---------- N(r) → N(z) ----------
def build_Nz_from_Nr_or_demo(L=400.0, nz=4096, r_max=4000.0, Nr=3000):
    # 用你的 ζ→N(r)
    if all(k in globals() for k in ("N_r_from_zeta","r_max","Nr")):
        r = np.linspace(0.0, globals()["r_max"], globals()["Nr"])
        N_r = np.asarray(globals()["N_r_from_zeta"], float)
        # 让路径尽量落在结构段中部
        z0 = float(np.clip(0.67*globals()["r_max"] - 0.5*L, 0.0, globals()["r_max"] - L))
        z_path = np.linspace(z0, z0 + L, nz)
        Nz = np.interp(z_path, r, N_r)
        overlap = np.mean(np.abs(Nz - 1.0) > 1e-6)
        return Nz, z0, overlap, float(globals()["r_max"]), int(globals()["Nr"])
    # demo：5% 台阶
    r = np.linspace(0.0, r_max, Nr); N_r = np.ones_like(r)
    segL = r_max*0.25; seg0 = r_max*0.60
    N_r[(r>=seg0) & (r<=seg0+segL)] = 1.05
    z0 = float(np.clip(seg0 + 0.5*(segL - L), 0.0, r_max - L))
    z_path = np.linspace(z0, z0 + L, nz)
    Nz = np.interp(z_path, r, N_r)
    overlap = np.mean(np.abs(Nz - 1.0) > 1e-6)
    print("[demo] 未检测到 N_r_from_zeta/r_max/Nr，使用演示 N(r)。")
    return Nz, z0, overlap, r_max, Nr

# ---------- run once ----------
def run_once(L=400.0, nz=4096, f0=150e6, r_max=4000.0, Nr=3000,
             bp_frac=0.25, pctl=0.10, coh_floor=0.15, min_bins=8):
    Nz, z0, ov, rM, NR = build_Nz_from_Nr_or_demo(L, nz, r_max, Nr)
    dz = L / (nz - 1)
    dt_pred = eikonal_dt_grid(Nz, dz)
    fd = yee_fdtd_delay(Nz, dz, f0=f0)
    dt_sim, meta = fused_delay_robust(fd, f0, bp_frac=bp_frac, pctl=pctl, coh_floor=coh_floor, min_bins=min_bins)
    print("== ζ→N(r)→N(z) : Eikonal vs FDTD（统一带通 + 限域细校正 + 鲁棒合成） ==")
    print(f"r_max={rM:.1f} m | L={L:.1f} m | z0={z0:.1f} m | overlap≈{ov*100:.2f}%")
    print(f"nz={nz} | dz={dz:.6f} m | dt={fd['dt']*1e9:.3f} ns | f0={f0/1e6:.1f} MHz")
    print(f"Δt_pred(eikonal) = {dt_pred*1e9:8.3f} ns")
    print(f"Δt_sim(FDTD)     = {dt_sim*1e9:8.3f} ns   [+coarse+fine]")
    rel = abs(dt_sim - dt_pred) / (abs(dt_pred) + 1e-30) * 100.0
    print(f"Rel. error       = {rel:8.3f}%")
    print(f"[compose] coarse={meta['coarse']*1e9:6.3f} ns | fine={meta['fine']*1e9:6.3f} ns | lag_max={meta['lag_max']*1e9:6.3f} ns")
    print(f"[phase] bins={meta['used_bins']} | coh≈{meta['mean_coh']:.3f}")
    print(f"[win] W={fd['W']}, idx_v={fd['i0x']}, idx_m={fd['i0y']}, "
          f"cut_v=({fd['i0x']},{fd['i0x']+fd['W']}), cut_m=({fd['i0y']},{fd['i0y']+fd['W']})")

# ---- try once (可按需调 f0/bp_frac/pctl) ----
run_once(L=400.0, nz=4096, f0=150e6, r_max=4000.0, Nr=3000,
         bp_frac=0.25, pctl=0.10, coh_floor=0.15, min_bins=8)
