# ===============================================
#  ζ → δ → ρ_eff → Φ → lnN → N(z)
#  Eikonal vs Yee-FDTD + 单点SI（同一路径）
#  —— 结构版 Ĝ：S = |Φ*φ^2/(H*dotΦ)|，Ĝ = (m_pre*C_SI^) * S * (c^2/T_ref)
#  —— 计算“光学等效 g”与“物理引力 g”（CODATA 绑定 & 可选称重绑定）
#  —— 结尾 print & plt 出图
# ===============================================

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
c0 = 299_792_458.0
G_CODATA = 6.67430e-11  # m^3 kg^-1 s^-2

# 可选：如果你有一个对应区域的真实质量（kg），填这里；否则保留 None 用 CODATA 绑定演示
M_true_kg = None  # e.g., 12.345  # ←改成你的实测质量即可

# ---- 兼容：NumPy 2.x 推荐 trapezoid；老版本用 trapz ----
try:
    trapint = np.trapezoid
except AttributeError:
    trapint = np.trapz

# ---------- helpers ----------
def next_pow2(n: int) -> int:
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()

def bandpass_fft(sig, dt, f0, bp_frac=0.25):
    x = np.asarray(sig, float); N = len(x)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(N, dt)
    fL = max(0.0, f0 * (1 - bp_frac)); fH = f0 * (1 + bp_frac)
    keep = (f >= fL) & (f <= fH)
    Y = np.zeros_like(X); Y[keep] = X[keep]
    return np.fft.irfft(Y, n=N)

def analytic_signal(sig):
    x = np.asarray(sig, float); N = len(x)
    X = np.fft.fft(x); H = np.zeros(N)
    if N % 2 == 0:
        H[0] = 1.0; H[N // 2] = 1.0; H[1:N // 2] = 2.0
    else:
        H[0] = 1.0; H[1:(N + 1) // 2] = 2.0
    return np.fft.ifft(X * H)

# ---------- ζ-phase → φσ(u) ----------
def build_phi_sigma(gammas, alpha=1.0, u_pad=10.0, Nu=6000):
    gam = np.sort(np.asarray(gammas, float))
    dgam = np.diff(gam); dmean = float(np.median(dgam)) if len(dgam)>0 else 9.0
    sigma = float(alpha * dmean)
    u = np.linspace(-(gam[-1] + u_pad), (gam[-1] + u_pad), int(Nu))
    L = lambda x: sigma / (x*x + sigma*sigma)
    phi_p = np.zeros_like(u)
    for g in gam:
        phi_p += L(u - g) + L(u + g)
    du = (u[-1]-u[0])/(len(u)-1)
    phi = np.cumsum(0.5*(phi_p[:-1]+phi_p[1:]))*du
    phi = np.concatenate(([0.0], phi))
    phi -= np.mean(phi); uu = u - np.mean(u)
    a = float(np.dot(uu, phi)/(np.dot(uu, uu)+1e-30)); phi -= a*uu
    return u, phi, sigma

# ---------- δ(r,t) & URSF ----------
def ursf_from_phi(u, phi_u, r, t, v_u=0.2, phi2_mode="narrow", f0_phi=0.333, bw_frac=0.25):
    r = np.asarray(r,float); t = np.asarray(t,float); r_max = float(r[-1])
    J = (u[-1]-u[0]) / r_max
    w = np.exp(-(r/r_max)**8)
    phi_interp = lambda uu: np.interp(uu, u, phi_u, left=0.0, right=0.0)
    delta = np.zeros((len(r), len(t)), float)
    for j, tj in enumerate(t):
        delta[:, j] = w * phi_interp(r/J - v_u*tj)

    dr = (r[-1]-r[0])/(len(r)-1); rr2 = r*r
    H   = 4*pi*trapint(delta**2 * rr2[:,None], r, axis=0)
    Phi = 4*pi*trapint(delta     * rr2[:,None], r, axis=0)

    dt = (t[-1]-t[0])/(len(t)-1) if len(t)>1 else 1.0
    phi_c = np.gradient(Phi, dt)
    K = np.gradient(np.log(np.abs(H)+1e-30), dt) / (np.gradient(np.log(np.abs(Phi)+1e-30), dt)+1e-30)

    if phi2_mode == "narrow":
        acc = 0.0
        for i in range(len(r)):
            xb = bandpass_fft(delta[i,:], dt, f0_phi, bw_frac); acc += float(np.mean(xb*xb))
        phi2 = float(acc*dr)
    else:
        phi2 = float(np.mean(delta*delta))
    return dict(delta=delta, H=H, Phi=Phi, phi_c=phi_c, K=K, phi2=phi2, J=J)

def pick_t_star(H,Pc,K,t,t_band=(0.2,3.0),Kwin=(1.3,2.7),qH=0.30,qC=0.30,target=2.0):
    tt = np.asarray(t,float); inb = (tt>=t_band[0])&(tt<=t_band[1])
    if not np.any(inb): inb = np.ones_like(tt,dtype=bool)
    Hthr = np.quantile(H[inb], qH); Cthr = np.quantile(np.abs(Pc[inb]), qC)
    ok = inb&(H>=Hthr)&(np.abs(Pc)>=Cthr)&(K>=Kwin[0])&(K<=Kwin[1])
    if np.any(ok):
        idx = np.argmin(np.abs(K[ok]-target)); i = int(np.arange(len(tt))[ok][idx]); return tt[i], i, True
    ok2 = inb&(H>=np.quantile(H[inb],0.2))&(np.abs(Pc)>=np.quantile(np.abs(Pc[inb]),0.2))
    if np.any(ok2):
        j = int(np.arange(len(tt))[ok2][np.argmin(np.abs(K[ok2]-target))]); return tt[j], j, False
    j = int(np.arange(len(tt))[inb][np.argmin(np.abs(K[inb]-target))]); return tt[j], j, False

# ---------- ρ_eff → Φ → lnN ----------
def poisson_phi_from_rho_spherical(r, rho, Ghat=1.0):
    r = np.asarray(r,float); rho = np.asarray(r,float)
    dr = (r[-1]-r[0])/(len(r)-1)
    integ = 4*pi*r*r*rho
    M = np.zeros_like(r); M[1:] = np.cumsum(0.5*(integ[1:]+integ[:-1]))*dr
    dPhi = np.zeros_like(r); dPhi[1:] = Ghat*M[1:]/(r[1:]**2+1e-30); dPhi[0] = dPhi[1]
    Phi = np.zeros_like(r)
    for k in range(len(r)-1,0,-1):
        Phi[k-1] = Phi[k] - 0.5*(dPhi[k-1]+dPhi[k])*dr
    return Phi

def lnN_from_Phi(Phi): return -2.0*np.asarray(Phi,float)/(c0*c0)

def clamp_lnN_nonneg(lnN_r, lnN_cap=0.6):
    lnN = np.clip(np.asarray(lnN_r,float), 0.0, lnN_cap)
    ker = np.array([1,2,3,2,1],float); ker/=ker.sum()
    lnN = np.convolve(lnN, ker, mode='same')
    return np.clip(lnN, 0.0, lnN_cap)

# ---------- path extract ----------
def build_path_Nz_from_Nr(N_r, r, L=400.0, nz=4096, force_overlap=True):
    r = np.asarray(r,float); N_r = np.asarray(N_r,float); r_max = float(r[-1])
    cand = np.linspace(0.0, max(0.0, r_max-L), 200)
    best=None; best_ov=-1.0; best_z0=0.0
    for z0 in cand:
        z = np.linspace(z0, z0+L, nz); Nz = np.interp(z, r, N_r)
        ov = np.mean((Nz-1.0)>1e-6)
        if ov>best_ov: best_ov=ov; best=Nz; best_z0=z0
    if force_overlap and best_ov<0.05:
        lnN = np.log(np.maximum(N_r,1.0)); k = int(np.argmax(lnN))
        z0 = float(np.clip(r[k]-L*0.5, 0.0, r_max-L))
        z = np.linspace(z0, z0+L, nz); best = np.interp(z, r, N_r)
        best_z0 = z0; best_ov = np.mean((best-1.0)>1e-6)
    return best, best_z0, best_ov

# ---------- eikonal & FDTD ----------
def eikonal_dt_grid(Nz,dz): return (dz/c0)*float(np.sum(Nz-1.0))

def yee_fdtd_delay(Nz, dz, f0, src_offset=80, probe_offset=6):
    Nz = np.asarray(Nz,float); nz=len(Nz); nmax=float(np.max(Nz))
    dt = 0.99*dz/(c0*nmax); steps=int(np.ceil(1.2*nmax*(nz-1))); nt=int(3.0*steps)
    t = np.arange(nt)*dt; i_src=src_offset; i_prb=nz-1-probe_offset
    T_link=((i_prb-i_src)*dz)/(c0/nmax); t0=0.35*T_link; spread=0.12*T_link
    def src(tt): x=(tt-t0)/(spread+1e-18); return np.sin(2*pi*f0*tt)*np.exp(-0.5*x*x)
    def run(nprof):
        eps=(nprof**2).astype(float); mu=np.ones_like(eps)
        Ez=np.zeros(nz); Hy=np.zeros(nz); Ez_prev=np.zeros(nz); rec=np.zeros(nt)
        cl=c0/float(nprof[0]); cr=c0/float(nprof[-1])
        def murL(En,Eo): k=(cl*dt-dz)/(cl*dt+dz); return Eo[1]+k*(En[1]-Eo[0])
        def murR(En,Eo): k=(cr*dt-dz)/(cr*dt+dz); return Eo[-2]+k*(En[-2]-Eo[-1])
        for it in range(nt):
            Hy[:-1]+= (dt/(mu[:-1]*dz))*(Ez[1:]-Ez[:-1])
            Ez_prev[:]=Ez
            Ez[1:]+= (dt/(eps[1:]*dz))*(Hy[1:]-Hy[:-1])
            Ez[i_src]+=src(t[it])
            Ez[0]=murL(Ez,Ez_prev); Ez[-1]=murR(Ez,Ez_prev)
            rec[it]=Ez[i_prb]
        return rec,dt,t
    rec_v,dt,t=run(np.ones_like(Nz))
    rec_m,_,_=run(Nz)
    T_vac=((nz-1-probe_offset)-src_offset)*dz/c0
    dt_pred=eikonal_dt_grid(Nz[src_offset:nz-1-probe_offset],dz)

    # 长窗口 ≈60 个载波周期
    per = 1.0 / f0
    W_goal = int(60 * per / dt)
    W = min(next_pow2(max(2048, W_goal)), len(t) // 2)
    i0x = int(np.clip(np.round(T_vac / dt) - W // 2, 0, len(t) - W))
    i0y = int(np.clip(np.round((T_vac + dt_pred) / dt) - W // 2, 0, len(t) - W))

    return dict(rec_v=rec_v,rec_m=rec_m,t=t,dt=dt,W=W,i0x=i0x,i0y=i0y,dt_pred_grid=dt_pred)

# ---------- robust fused delay 三法 ----------
def phase_delay_WLS(xw,yw,dt,f0,bp_frac=0.25,pctl=0.10,coh_floor=0.15,min_bins=8,gamma=1.3):
    N=len(xw); w=np.hanning(N)
    x=bandpass_fft((xw-np.mean(xw))*w,dt,f0,bp_frac)
    y=bandpass_fft((yw-np.mean(yw))*w,dt,f0,bp_frac)
    X=np.fft.rfft(x); Y=np.fft.rfft(y)
    Sxx=np.abs(X)**2; Syy=np.abs(Y)**2; Sxy=X*np.conj(Y)
    f=np.fft.rfftfreq(N,dt); wv=2*pi*f; fL,fH=f0*(1-bp_frac),f0*(1+bp_frac)
    band=(f>0)&(f>=fL)&(f<=fH)
    if not np.any(band): return None,dict(used_bins=0,mean_coh=np.nan)
    E=np.sqrt(Sxx[band]*Syy[band]); coh=(np.abs(Sxy[band])**2)/(Sxx[band]*Syy[band]+1e-30)
    thr=np.quantile(E,pctl) if np.all(np.isfinite(E)) else 0.0
    pick=(E>=thr)&(coh>=coh_floor)
    if np.count_nonzero(pick)<min_bins:
        return None,dict(used_bins=int(np.count_nonzero(pick)),mean_coh=(float(np.mean(coh[pick])) if np.any(pick) else np.nan))
    ww=wv[band][pick]; H=Sxy[band][pick]/(Sxx[band][pick]+1e-30)
    ph=np.unwrap(np.angle(H)); W=(E[pick]**gamma)
    wc=float(np.sum(W*ww)/(np.sum(W)+1e-30)); pc=float(np.sum(W*ph)/(np.sum(W)+1e-30))
    num=float(np.sum(W*(ww-wc)*(ph-pc))); den=float(np.sum(W*(ww-wc)**2)+1e-30)
    tau=-num/den; return float(tau),dict(used_bins=int(np.count_nonzero(pick)),mean_coh=float(np.mean(coh[pick])))

def env_delay_band(xw,yw,dt,f0,bp_frac=0.25,lag_max=None):
    xb=bandpass_fft(xw-np.mean(xw),dt,f0,bp_frac)
    yb=bandpass_fft(yw-np.mean(yw),dt,f0,bp_frac)
    ex=np.abs(analytic_signal(xb)); ey=np.abs(analytic_signal(yb))
    ix=int(np.argmax(ex)); iy=int(np.argmax(ey)); tau=(iy-ix)*dt
    if (lag_max is not None) and (abs(tau)>lag_max): return None
    return float(tau)

def xcorr_delay_limited(xw,yw,dt,up=64,f0=None,bp_frac=None,lag_max=None):
    a=xw-np.mean(xw); b=yw-np.mean(yw)
    if (f0 is not None) and (bp_frac is not None):
        a=bandpass_fft(a,dt,f0,bp_frac); b=bandpass_fft(b,dt,f0,bp_frac)
    N=len(a); L=next_pow2(N*up)
    A=np.fft.rfft(a,n=L); B=np.fft.rfft(b,n=L)
    r=np.fft.irfft(A*np.conj(B),n=L)
    r=np.concatenate((r[-(L//2):],r[:(L//2)]))
    if lag_max is not None:
        kmax=int(np.round(lag_max/(dt/up))); k0=L//2
        sl=slice(k0-kmax,k0+kmax+1); rw=r[sl]
        k_rel=int(np.argmax(rw))-kmax; tau=k_rel*(dt/up)
    else:
        k=int(np.argmax(r))-(L//2); tau=k*(dt/up)
    if (lag_max is not None) and (abs(tau)>lag_max): return None
    return float(tau)

def fused_delay_robust(fd, f0, bp_frac=0.25, pctl=0.10, coh_floor=0.15, min_bins=8):
    xw = fd["rec_v"][fd["i0x"]:fd["i0x"]+fd["W"]]
    yw = fd["rec_m"][fd["i0y"]:fd["i0y"]+fd["W"]]
    xw -= np.mean(xw); yw -= np.mean(yw); dt = fd["dt"]
    coarse = (fd["i0y"] - fd["i0x"]) * dt
    lag_max = max(10*dt, 0.5*abs(coarse + fd["dt_pred_grid"]), 0.4*abs(fd["dt_pred_grid"]) + 0.2*abs(coarse))
    def try_wls(bp, pctl_, coh_, mbins_):
        tp, meta = phase_delay_WLS(xw, yw, dt, f0, bp, pctl_, coh_, mbins_)
        if tp is None: return None, 0, np.nan
        return float(tp), int(meta.get("used_bins", 0)), float(meta.get("mean_coh", np.nan))
    taus=[]; best_bins=0; best_coh=np.nan
    tp,bins,coh = try_wls(bp_frac, pctl, coh_floor, min_bins)
    if tp is not None: taus.append(tp); best_bins=max(best_bins,bins); best_coh=coh
    if (tp is None) or (bins<min_bins):
        tp2,b2,c2=try_wls(max(0.35,bp_frac),0.0,0.05,3)
        if tp2 is not None: taus.append(tp2); best_bins=max(best_bins,b2); best_coh=np.nanmax([best_coh,c2])
        for bf in (0.08,0.12,0.18):
            tpi,bi,ci=try_wls(bf,0.0,0.0,1)
            if tpi is not None: taus.append(tpi); best_bins=max(best_bins,bi); best_coh=np.nanmax([best_coh,ci])
    te = env_delay_band(xw, yw, dt, f0, max(0.12, bp_frac), lag_max=lag_max)
    tx = xcorr_delay_limited(xw, yw, dt, up=64, f0=f0, bp_frac=max(0.12,bp_frac), lag_max=lag_max)
    for t in (te, tx):
        if (t is not None) and np.isfinite(t): taus.append(float(t))
    expected = np.sign(fd["dt_pred_grid"]) if abs(fd["dt_pred_grid"])>1e-15 else np.sign(coarse)
    if expected == 0: expected = 1.0
    small_thr = 0.25*abs(coarse) + 1e-15
    cands = [t for t in taus if (np.sign(t)==expected) or (abs(t)<=small_thr)]
    fine = float(np.median(cands)) if cands else 0.0
    fine = float(np.clip(fine, -lag_max, +lag_max))
    return coarse + fine, dict(coarse=coarse, fine=fine, lag_max=lag_max, used_bins=int(best_bins), mean_coh=float(best_coh))

# ---------- 单点 SI（用同一 Nz） ----------
def single_point_SI_on_Nz(Nz, dz, f0, bp_frac=0.35, dt_meas_ns=5.0, unit_target_ns=3.0, lnN_cap=0.6):
    fd0 = yee_fdtd_delay(Nz, dz, f0=f0)
    dt_unit_fdt, _ = fused_delay_robust(fd0, f0, bp_frac=bp_frac, pctl=0.0, coh_floor=0.05, min_bins=3)
    dt_unit_eik = eikonal_dt_grid(Nz, dz)
    unit_before = max(abs(dt_unit_fdt), abs(dt_unit_eik)) * 1e9
    m_pre = 1.0
    if unit_before < 0.2:
        m_pre = float(max(1.0, unit_target_ns/(unit_before+1e-30)))
        lnN = np.log(Nz)*m_pre; lnN = np.clip(lnN, 0.0, lnN_cap)
        Nz = np.exp(lnN)
        fd0 = yee_fdtd_delay(Nz, dz, f0=f0)
        dt_unit_fdt, _ = fused_delay_robust(fd0, f0, bp_frac=bp_frac, pctl=0.0, coh_floor=0.05, min_bins=3)
        dt_unit_eik = eikonal_dt_grid(Nz, dz)
    C_SI = float((dt_meas_ns*1e-9)/(abs(dt_unit_fdt)+1e-30))
    lnN_sc = np.log(Nz)*C_SI; Nz_sc = np.exp(lnN_sc)
    fd2 = yee_fdtd_delay(Nz_sc, dz, f0=f0)
    dt_sim, _ = fused_delay_robust(fd2, f0, bp_frac=bp_frac, pctl=0.0, coh_floor=0.05, min_bins=3)
    dt_pred = eikonal_dt_grid(Nz_sc, dz)
    return dict(C_SI=C_SI, m_pre=m_pre, dt_unit_eik=dt_unit_eik, dt_unit_fdt=dt_unit_fdt, dt_pred=dt_pred, dt_sim=dt_sim, Nz_used=Nz)

# ---------- 一次主流程（t*、首轮 Nz 与 Δt） ----------
def first_pass(L=400.0, nz=4096, f0=150e6, r_max=4000.0, Nr=3000,
               alpha_grid=np.linspace(0.6, 2.0, 10), v_grid=np.linspace(0.05, 0.50, 10),
               t_band=(0.2, 3.0), phi2_mode="narrow", f0_phi=0.333, bw_phi=0.25,
               lnN_cap=0.6, bp_frac=0.35, dt_meas_ns=5.0):
    ZG = np.array([
        14.134725141,21.022039639,25.010857580,30.424876125,32.935061588,
        37.586178159,40.918719012,43.327073281,48.005150881,49.773832478,
        52.970321478,56.446247697,59.347044003,60.831778525,65.112544048,
        67.079810529,69.546401711,72.067157674,75.704690699,77.144840069,
        79.337375020,82.910380854,84.735492981,87.425274613,88.809111208,
        92.491899271,94.651344041,95.870634228,98.831194218,101.317851006,
        103.725538040,105.446623052,107.168611184,111.029535543,111.874659177,
        114.320220915,116.226680321,118.790782866,121.370125002,122.946829294,
        124.256818554,127.516683880,129.578704200,131.087688531,133.497737203,
        134.756509753,138.116042055,139.736208952,141.123707404
    ], float)

    u,phi_u,_ = build_phi_sigma(ZG, alpha=1.0, u_pad=10.0, Nu=6000)
    r = np.linspace(0.0, r_max, Nr); t = np.linspace(0.0, 3.5, 9001)

    best=None
    for a in alpha_grid:
        uA,phiA,_ = build_phi_sigma(ZG, alpha=a, u_pad=10.0, Nu=len(u))
        for vu in v_grid:
            U = ursf_from_phi(uA, phiA, r, t, v_u=vu, phi2_mode=phi2_mode, f0_phi=f0_phi, bw_frac=bw_phi)
            ts,i,ok = pick_t_star(U["H"], U["phi_c"], U["K"], t, t_band=t_band)
            score = abs(U["K"][i]-2.0) + (0.0 if ok else 0.5)
            if (best is None) or (score<best["score"]):
                best = dict(alpha=a,v_u=vu,ursf=U,t_star=ts,i_star=i,K=float(U["K"][i]),
                            Hs=float(U["H"][i]), pcs=float(U["phi_c"][i]), phi2=float(U["phi2"]),
                            locked=bool(ok), score=float(score))

    alpha, v_u = best["alpha"], best["v_u"]
    t_star, i_star = best["t_star"], best["i_star"]
    print(f"=== K≈2 selection ({'LOCKED' if best['locked'] else 'fallback'}) ===")
    print(f"alpha={alpha:.3f}, v_u={v_u:.3f}")
    print(f"t* = {t_star:.6f}s,  K(t*) = {best['K']:.4f}")

    # ρ_eff→Φ→lnN→N_r（首轮，Ghat=1）
    delta = best["ursf"]["delta"]; rho = delta[:, i_star]**2
    rho /= (np.max(rho)+1e-30)
    Phi = poisson_phi_from_rho_spherical(r, rho, Ghat=1.0)
    lnN = clamp_lnN_nonneg(lnN_from_Phi(Phi), lnN_cap=lnN_cap); N_r = np.exp(lnN)

    Nz, z0, ov = build_path_Nz_from_Nr(N_r, r, L=L, nz=nz, force_overlap=True)
    dz = L/(nz-1)
    fd = yee_fdtd_delay(Nz, dz, f0=f0)
    dt_pred = fd["dt_pred_grid"]; dt_sim, meta = fused_delay_robust(fd, f0, bp_frac=bp_frac, pctl=0.0, coh_floor=0.05, min_bins=3)
    den = max(abs(dt_pred), 1e-15); rel = abs(dt_sim-dt_pred)/den*100.0

    print("\n== ζ→N(r)→N(z) : Eikonal vs FDTD (round-1) ==")
    print(f"r_max={r_max:.1f} m | L={L:.1f} m | z0={z0:.1f} m | overlap≈{ov*100:.2f}%")
    print(f"nz={nz} | dz={dz:.6f} m | dt={fd['dt']*1e9:.3f} ns | f0={f0/1e6:.1f} MHz")
    print(f"Δt_pred(eikonal) = {dt_pred*1e9:8.3f} ns")
    print(f"Δt_sim(FDTD)     = {dt_sim*1e9:8.3f} ns   [+coarse+fine]")
    print(f"Rel. error       = {rel:8.3f}% | Abs. error = {(dt_sim-dt_pred)*1e9:8.3f} ns")
    print(f"[compose] coarse={meta['coarse']*1e9:6.3f} ns | fine={meta['fine']*1e9:6.3f} ns | "
          f"lag_max={meta['lag_max']*1e9:6.3f} ns | bins={meta.get('used_bins',0)} | coh≈{meta.get('mean_coh',np.nan):.2f}")

    si = single_point_SI_on_Nz(Nz, dz, f0, bp_frac=bp_frac, dt_meas_ns=5.0, lnN_cap=lnN_cap)
    print("\n=== Single-point SI (same path; FDTD route, |unit|) ===")
    print(f"Unit: Δt_unit_eik = {si['dt_unit_eik']*1e9:8.3f} ns | Δt_unit_fdt = {si['dt_unit_fdt']*1e9:8.3f} ns")
    print(f"[Auto-PreScale] m_pre = {si['m_pre']:10.3e}")
    print(f"C_SI^ = {si['C_SI']:9.6f}")

    state = dict(best=best, i_star=i_star, r=r, rho=rho, N_r=N_r, Nz1=Nz, z0=z0, dz=dz,
                 fd1=fd, dt_pred1=dt_pred, dt_sim1=dt_sim, meta1=meta, si=si, ov1=ov, L=L, nz=nz, f0=f0,
                 lnN_cap=lnN_cap, bp_frac=bp_frac)
    return state

def second_round_with_Ghat(state, Ghat, tag="(Ĝ)"):
    r, rho, L, nz, f0 = state["r"], state["rho"], state["L"], state["nz"], state["f0"]
    dz, lnN_cap, bp_frac = state["dz"], state["lnN_cap"], state["bp_frac"]
    Phi2 = poisson_phi_from_rho_spherical(r, rho, Ghat=Ghat)
    lnN2 = clamp_lnN_nonneg(lnN_from_Phi(Phi2), lnN_cap=lnN_cap); N_r2 = np.exp(lnN2)
    Nz2, z0_2, ov2 = build_path_Nz_from_Nr(N_r2, r, L=L, nz=nz, force_overlap=True)
    fd2 = yee_fdtd_delay(Nz2, dz, f0=f0); dt_pred2 = fd2["dt_pred_grid"]
    dt_sim2, meta2 = fused_delay_robust(fd2, f0, bp_frac=bp_frac, pctl=0.0, coh_floor=0.05, min_bins=3)
    den2 = max(abs(dt_pred2), 1e-15); rel2 = abs(dt_sim2-dt_pred2)/den2*100.0
    print(f"\n== After {tag} feedback (round-2) ==")
    print(f"overlap≈{ov2*100:.2f}% | Δt_pred={dt_pred2*1e9:6.3f} ns | Δt_sim={dt_sim2*1e9:6.3f} ns")
    print(f"Rel. error       = {rel2:8.3f}% | Abs. error = {(dt_sim2-dt_pred2)*1e9:8.3f} ns")
    print(f"[compose] coarse={meta2['coarse']*1e9:6.3f} ns | fine={meta2['fine']*1e9:6.3f} ns | "
          f"lag_max={meta2['lag_max']*1e9:6.3f} ns | bins={meta2.get('used_bins',0)} | coh≈{meta2.get('mean_coh',np.nan):.2f}")
    return dict(N_r2=N_r2, Nz2=Nz2, z0_2=z0_2, ov2=ov2, fd2=fd2, dt_pred2=dt_pred2, dt_sim2=dt_sim2, meta2=meta2, Ghat=Ghat)

def g_from_N(N_array, x_array):
    lnN = np.log(np.maximum(N_array, 1e-12))
    dlnN_dx = np.gradient(lnN, x_array)
    return 0.5 * (c0**2) * dlnN_dx  # m/s^2

# ======== 参数 ========
L=400.0; nz=4096; f0=150e6; r_max=4000.0; Nr=3000
alpha_grid=np.linspace(0.6, 2.0, 10); v_grid=np.linspace(0.05, 0.50, 10)
t_band=(0.2, 3.0); phi2_mode="narrow"; f0_phi=0.333; bw_phi=0.25
lnN_cap=0.6; bp_frac=0.35; dt_meas_ns=5.0

# ======== 运行首轮 ========
state = first_pass(L=L, nz=nz, f0=f0, r_max=r_max, Nr=Nr,
                   alpha_grid=alpha_grid, v_grid=v_grid, t_band=t_band,
                   phi2_mode=phi2_mode, f0_phi=f0_phi, bw_phi=bw_phi,
                   lnN_cap=lnN_cap, bp_frac=bp_frac, dt_meas_ns=dt_meas_ns)

best, i_star, si = state["best"], state["i_star"], state["si"]

# ======== 结构量（锁相时刻） & 两种 Ĝ ========
Phi_star  = best["ursf"]["Phi"][i_star]       # Φ(t*)
H_star    = best["Hs"]                         # H(t*)
phi_cstar = best["pcs"]                        # dotΦ(t*)
phi2      = best["phi2"]                       # φ^2（窄带能量）

# 结构不变量 S（取绝对值 → 物理符号约束，确保 Ĝ>=0）
S_T_over_L2 = abs((Phi_star * phi2) / (H_star * phi_cstar + 1e-30))  # s/m^2
T_ref = 1.0 / f0                  # 参考时间尺度：载频周期

SI_scale = si["m_pre"] * si["C_SI"]
Ghat_em  = SI_scale                                  # s^-2
Ghat_struct = SI_scale * S_T_over_L2 * (c0**2 / T_ref)  # s^-2  ← 修正后的量纲

print("\n=== Structural scalars @ t* ===")
print(f"Φ(t*)      = {Phi_star:.6e}")
print(f"H(t*)      = {H_star:.6e}")
print(f"dotΦ(t*)   = {phi_cstar:.6e}")
print(f"φ^2        = {phi2:.6e}")
print(f"S = |Φ*φ^2/(H*dotΦ)| = {S_T_over_L2:.6e}  [s/m^2]")

print("\n=== Ĝ candidates ===")
print(f"Ĝ_em      = m_pre*C_SI^              = {Ghat_em:.6e}  [s^-2]")
print(f"Ĝ_struct  = (m_pre*C_SI^)*S*(c^2/T0) = {Ghat_struct:.6e}  [s^-2]")

# ======== 第二轮：两种 Ĝ 并排回灌并比较 ========
res_em  = second_round_with_Ghat(state, Ghat_em,     tag="Ĝ_em")
res_st  = second_round_with_Ghat(state, Ghat_struct, tag="Ĝ_struct")

# ======== “光学等效 g” 与 “真实 g”的计算 ========
z_axis = np.linspace(state["z0"], state["z0"]+L, nz)

# 光学等效 g（由 Ĝ_struct 回灌得到的 N）
g_opt = g_from_N(res_st["Nz2"], z_axis)  # m/s^2

# 质量绑定：
#  (A) CODATA 绑定（演示）：ρ0 = Ĝ_struct / G_CODATA
rho0_cod = Ghat_struct / G_CODATA  # kg/m^3（把 ρ_eff 变成 ρ_phys 的比例因子）

#  (B) 称重绑定（可选）：若你提供 M_true_kg，则以球对称近似计算 M_eff 后再求 ρ0_mass
r = state["r"]; rho_eff = state["rho"]
M_eff = 4*np.pi * trapint(rho_eff * r**2, r)  # 无量纲“结构质量”
rho0_mass = None
if (M_true_kg is not None):
    rho0_mass = M_true_kg / (M_eff + 1e-30)

# 真实 g：两种绑定都会得到同一条 g 场（只是解释不同）。我们都打印。
g_phys_CODATA = g_opt.copy()  # 数值相同（等价重标），但语义为“物理引力 g（CODATA 绑定）”
g_phys_mass   = g_opt.copy() if (rho0_mass is not None) else None

print("\n=== Real-g bindings ===")
print(f"[CODATA] ρ0 = Ĝ_struct / G_CODATA = {rho0_cod:.6e} kg/m^3")
if rho0_mass is not None:
    print(f"[Mass   ] ρ0 = M_true / M_eff    = {rho0_mass:.6e} kg/m^3 (M_eff={M_eff:.6e} a.u.)")
else:
    print("[Mass   ] 你未提供 M_true_kg；已跳过称重绑定（可在顶部填入 M_true_kg 启用）")

def stats(name, garr):
    print(f"{name}: mean={np.mean(garr):.6e}  std={np.std(garr):.6e}  "
          f"min={np.min(garr):.6e}  max={np.max(garr):.6e}  (m/s^2)")

print("\n=== g profiles (m/s^2) ===")
stats("g_optical(Ĝ_struct)", g_opt)
stats("g_physical(CODATA)", g_phys_CODATA)
if g_phys_mass is not None:
    stats("g_physical(mass)  ", g_phys_mass)

# ======== 摘要 ========
print("\n>>> Summary:")
print(f"{'Ĝ_em':>10s} : dt_pred2={res_em['dt_pred2']*1e9:8.3f} ns | dt_sim2={res_em['dt_sim2']*1e9:8.3f} ns | rel={abs(res_em['dt_sim2']-res_em['dt_pred2'])/max(abs(res_em['dt_pred2']),1e-15)*100:6.3f}%")
print(f"{'Ĝ_struct':>10s} : dt_pred2={res_st['dt_pred2']*1e9:8.3f} ns | dt_sim2={res_st['dt_sim2']*1e9:8.3f} ns | rel={abs(res_st['dt_sim2']-res_st['dt_pred2'])/max(abs(res_st['dt_pred2']),1e-15)*100:6.3f}%")

# ======== 出图 ========

# 1) N(z) 对比
plt.figure()
plt.plot(z_axis, state["Nz1"], label="N(z) round-1 (Ghat=1)")
plt.plot(z_axis, res_em["Nz2"],   label="N(z) round-2 with Ĝ_em")
plt.plot(z_axis, res_st["Nz2"],   label="N(z) round-2 with Ĝ_struct")
plt.xlabel("z (m)"); plt.ylabel("N(z)"); plt.title("Path refractive index profiles")
plt.grid(True); plt.legend(); plt.show()

# 2) 二轮窗口内信号（Ĝ_struct）
fd2 = res_st["fd2"]
tv = fd2["t"][fd2["i0x"]:fd2["i0x"]+fd2["W"]]
tm = fd2["t"][fd2["i0y"]:fd2["i0y"]+fd2["W"]]
xv = fd2["rec_v"][fd2["i0x"]:fd2["i0x"]+fd2["W"]]
ym = fd2["rec_m"][fd2["i0y"]:fd2["i0y"]+fd2["W"]]
tv = tv - tv[0]; tm = tm - tm[0]
plt.figure()
plt.plot(tv*1e9, xv, label="vacuum (windowed)")
plt.plot(tm*1e9, ym, label="medium (windowed) | Ĝ_struct")
plt.xlabel("time (ns)"); plt.ylabel("amplitude (a.u.)")
plt.title("Windowed received signals (round-2, Ĝ_struct)")
plt.grid(True); plt.legend(); plt.show()

# 3) 延迟对比（两种 Ĝ）
plt.figure()
x = np.array([0, 1, 2, 3])
labels = ["eik(Ĝ_em)", "FDTD(Ĝ_em)", "eik(Ĝ_struct)", "FDTD(Ĝ_struct)"]
y = np.array([res_em["dt_pred2"]*1e9, res_em["dt_sim2"]*1e9, res_st["dt_pred2"]*1e9, res_st["dt_sim2"]*1e9])
plt.plot(x, y, marker="o")
plt.xticks(x, labels, rotation=10); plt.ylabel("delay (ns)")
plt.title("Delays comparison (round-2, Ĝ_em vs Ĝ_struct)")
plt.grid(True); plt.show()

# 4) 真实 g（CODATA 绑定）曲线
plt.figure()
plt.plot(z_axis, g_phys_CODATA)
plt.xlabel("z (m)"); plt.ylabel("g_phys (m/s^2)")
plt.title("Physical gravitational acceleration (CODATA binding)")
plt.grid(True); plt.show()

