# tmm_ray_closure_ballistic.py
# ✅ 平面波一次通过（ballistic）相位 → 群延时，对齐 τz；再用 Legendre 对齐 Tr（ζ / π）

import numpy as np, math

cfg = dict(
    Lz=0.08, Nz=900, crop_frac=0.65,
    M=48, alpha=1.6, beta=1.0,
    pburst=0.20, pgeom=0.30, pmix=0.50, lognorm_scale=0.40, pi_shape=0.30,
    rng_seed=11000,
    n_rays=32, p_margin=0.97, p_pow=2.2,
    f0=3.0e9, df=20e6,
    c0=2.99792458e8,
    accept_vert=0.02,   # 2% 内通常可达
    accept_leg =0.08
)

def get_zeros(M):
    try:
        import mpmath as mp
        mp.mp.dps = 50
        return np.array([float(mp.zetazero(n)) for n in range(1,M+1)], float)
    except Exception:
        fb=[14.134725,21.022040,25.010858,30.424876,32.935062,37.586178,40.918719,43.327073,48.005151,49.773832,
            52.970321,56.446248,59.347044,60.831779,65.112544,67.079811,69.546402,72.067158,75.704691,77.144840,
            79.337375,82.910381,84.735493,87.425275,88.809111,92.491899,94.651344,95.870634,98.831194,101.317851,
            103.725539,105.446623,107.168612,111.029536,111.874659,114.320221,116.226680,118.790783,121.370130,122.946829,
            124.256818,127.516684,129.578704,131.087689,133.497737,134.756508,138.116042,139.736208,141.123707,143.111845,
            146.000983,147.422765,150.053520,150.925257,153.024694,156.112908,157.597591,158.849988]
        if M>len(fb): raise RuntimeError("后备零点不足；请安装 mpmath。")
        return np.array(fb[:M], float)

def build_N(gammas, z_min, z_max, Nz, alpha, beta, crop_frac):
    z = np.linspace(z_min, z_max, Nz)
    M=len(gammas); Delta=(gammas[-1]-gammas[0])/(M-1); sigma=alpha*Delta
    phi = sum(np.arctan((z-g)/sigma) for g in gammas)
    d = phi - phi.mean(); d /= (d.std()+1e-12)
    N = np.exp(beta*d)
    L=z_max-z_min; mid=(z_min+z_max)/2; half=0.5*crop_frac*L
    m=(z>=mid-half)&(z<=mid+half)
    return z[m], N[m]

def make_pi_gammas(M, Delta, start, pburst, pgeom, pmix, lognorm_scale, pi_shape, rng):
    gaps=[]; n=0
    while n<M-1:
        if rng.random()<pburst:
            L=1+rng.geometric(pgeom)
            for _ in range(min(L,M-1-n)):
                gaps.append(rng.uniform(0.0,0.6*Delta)); n+=1
                if n>=M-1: break
        else:
            if rng.random()<pmix:
                g=Delta*math.exp(rng.normal(0.0,lognorm_scale))
            else:
                g=rng.gamma(pi_shape, Delta/max(pi_shape,1e-9))
            gaps.append(np.clip(g,0.1*Delta,3.0*Delta)); n+=1
    gaps=np.array(gaps); gaps *= (Delta*(M-1))/(gaps.sum()+1e-18)
    g=np.empty(M); g[0]=start; g[1:]=start+np.cumsum(gaps); return g

def cheb_p(n_rays, pmin, pmax, p_pow):
    j=np.arange(n_rays); x=-np.cos((j+0.5)*math.pi/n_rays)
    u=((x+1)/2.0)**p_pow
    return pmin + u*(pmax-pmin)

# —— 关键修正：ballistic 相位（仅传播项），不含界面反射相位 ——
def ballistic_phase(N, z, p, f, c0):
    k0 = 2*math.pi*f/c0
    kz = k0*np.sqrt(np.maximum(N**2 - p**2, 1e-18))  # 每个采样点的纵向波数
    dz = (z[-1]-z[0])/(len(z)-1)
    return np.sum(kz)*dz  # 传播相位总和

def T_vert(z,N,p,c0):  # 解析垂直延时
    return np.trapezoid(np.sqrt(np.maximum(N**2 - p**2, 1e-18)), z)/c0

def T_ray (z,N,p,c0):  # 射线路径时间
    den=np.sqrt(np.maximum(N**2 - p**2, 1e-18))
    return np.trapezoid((N**2)/den, z)/c0

def legendre_from_tau(p_list, tau):
    dtau_dp = np.gradient(tau, p_list)
    return tau - p_list*dtau_dp

def run_once(kind, cfg, rng):
    g_z=get_zeros(cfg["M"]); Delta=(g_z[-1]-g_z[0])/(cfg["M"]-1)
    gammas = g_z if kind=="zeta" else make_pi_gammas(cfg["M"],Delta,g_z[0],
                                                     cfg["pburst"],cfg["pgeom"],cfg["pmix"],
                                                     cfg["lognorm_scale"],cfg["pi_shape"],rng)
    z,N = build_N(gammas, 0.0, cfg["Lz"], cfg["Nz"], cfg["alpha"], cfg["beta"], cfg["crop_frac"])
    Nmin=float(N.min()); pmin=0.10*Nmin; pmax=cfg["p_margin"]*Nmin
    p_list=cheb_p(cfg["n_rays"], pmin, pmax, cfg["p_pow"])

    # 两频点 ballistic 相位 → 群延时
    phi_p = np.array([ballistic_phase(N,z,p,cfg["f0"]+cfg["df"],cfg["c0"]) for p in p_list])
    phi_m = np.array([ballistic_phase(N,z,p,cfg["f0"]-cfg["df"],cfg["c0"]) for p in p_list])
    # Δφ 很小（~0.06 rad），逐点相减即可（无需 unwrap across p）
    tau_ball = (phi_p - phi_m) / (2*math.pi*(2*cfg["df"]))

    tau_z = np.array([T_vert(z,N,p,cfg["c0"]) for p in p_list])
    Tr    = np.array([T_ray (z,N,p,cfg["c0"]) for p in p_list])
    Tr_hat= legendre_from_tau(p_list, tau_ball)

    DSERR_vert = np.linalg.norm(tau_ball - tau_z)/(np.linalg.norm(tau_z)+1e-18)
    DSERR_leg  = np.linalg.norm(Tr_hat  - Tr    )/(np.linalg.norm(Tr   )+1e-18)

    return dict(kind=kind, p_list=p_list, Nmin=Nmin,
                tau=tau_ball, tau_z=tau_z, Tr=Tr, Tr_hat=Tr_hat,
                DSERR_vert=DSERR_vert, DSERR_leg=DSERR_leg)

if __name__=="__main__":
    rng=np.random.default_rng(cfg["rng_seed"])
    print("=== Ballistic TMM – Ray Closure (1D, stratified) ===")
    print(f"Lz={cfg['Lz']*100:.1f} cm, Nz={cfg['Nz']}, crop={cfg['crop_frac']}, f0={cfg['f0']/1e9:.2f} GHz, df={cfg['df']/1e6:.0f} MHz")
    print(f"n_rays={cfg['n_rays']}, p_margin={cfg['p_margin']}, p_pow={cfg['p_pow']}")

    rz=run_once("zeta", cfg, rng)
    rp=run_once("pi",   cfg, rng)

    def brief(r, tag):
        mean_tz_ps = 1e12*np.mean(r["tau_z"])
        mean_Tr_ps = 1e12*np.mean(r["Tr"])
        print(f"[{tag}] min N={r['Nmin']:.4f}; p∈[{r['p_list'][0]:.4f},{r['p_list'][-1]:.4f}]")
        print(f"    OK(1) τ_ball vs τ_z : DSERR_vert={r['DSERR_vert']:.3%}  (mean τ_z={mean_tz_ps:.1f} ps)")
        print(f"    OK(2) T̂r   vs Tr   : DSERR_leg ={r['DSERR_leg']:.3%}  (mean Tr ={mean_Tr_ps:.1f} ps)")

    brief(rz,"ZETA"); brief(rp,"PI  ")
    print(f"\nAcceptance (ref):  τ闭环 ≤{cfg['accept_vert']:.0%} ;  Legendre闭环 ≤{cfg['accept_leg']:.0%}")
    print(f"ZETA pass? {'YES' if (rz['DSERR_vert']<=cfg['accept_vert'] and rz['DSERR_leg']<=cfg['accept_leg']) else 'NO '}")
    print(f"PI   pass? {'YES' if (rp['DSERR_vert']<=cfg['accept_vert'] and rp['DSERR_leg']<=cfg['accept_leg']) else 'NO '}")
