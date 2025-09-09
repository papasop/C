# Colab-ready, single-cell notebook (REAL φ_ζ version)
# - Builds φ_zeta from the *actual* ζ phase: φ_zeta(x,y) := cos( ν(t(U)) ), where ν(t) = unwrap(arg ζ(1/2 + i t))
# - Constructs n(x)=exp{g φ(x)} / <exp{g φ}>
# - Integrates rays (geometrical optics) under the ζ model
# - Evaluates windowed path-delay errors and *ray-certificate* energies for: zeta (truth), pi (warped), phase (random-phase alt)
# - Uses both normal-component certificate E_perp and full *vector* certificate E_vec (recommended)
# - Pure NumPy + Matplotlib + mpmath, no SciPy / internet

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# -------------------------
# Helpers & config
# -------------------------
trapz = getattr(np, "trapezoid", np.trapz)

def set_seed(seed=0):
    return np.random.default_rng(seed)

# -------------------------
# REAL ζ PHASE: ν(t) = unwrap(arg ζ(1/2 + i t))
# -------------------------
def build_phi_zeta_true_grid(L=100.0, N=480, t0=35.0, t_span=120.0, Nt=1200, theta_deg=28.0, seed=0):
    """
    Returns:
      xs, ys: grid coordinates
      phi_zeta_true: φ_ζ(x,y) = cos( ν( t(U(x,y)) ) )
      map info dict
    """
    rng = set_seed(seed)
    xs = np.linspace(0, L, N); ys = np.linspace(0, L, N)
    X, Y = np.meshgrid(xs, ys)

    # Direction for the "optical axis" U = e·(x,y)
    th = np.deg2rad(theta_deg)
    e = np.array([np.cos(th), np.sin(th)])
    U = e[0]*X + e[1]*Y
    Umin, Umax = U.min(), U.max()

    # t-grid along the optical axis
    t_lo = t0 - 0.5*t_span
    t_hi = t0 + 0.5*t_span
    t = np.linspace(t_lo, t_hi, Nt)

    # compute ν(t) = unwrap(arg ζ(1/2 + i t))
    mp.mp.dps = 40
    def arg_zeta_on_line(tt):
        s = 0.5 + 1j*tt
        z = mp.zeta(s)
        return float(mp.arg(z))

    nu_raw = np.array([arg_zeta_on_line(tt) for tt in t], dtype=float)
    nu = np.unwrap(nu_raw)  # continuous phase

    # simple smoothing optional (commented out to keep true structure)
    # from numpy import convolve
    # nu = convolve(nu, np.ones(5)/5.0, mode='same')

    # map U -> t affinely
    # we choose a mapping so that [Umin, Umax] ~ [t_lo, t_hi]
    alpha = (t_hi - t_lo) / max(Umax - Umin, 1e-12)
    beta  = t_lo - alpha * Umin
    Tgrid = alpha * U + beta

    # interpolate ν at Tgrid
    nu_on_grid = np.interp(Tgrid.ravel(), t, nu).reshape(U.shape)

    # build φ_ζ(x,y) = cos( ν )
    phi_zeta_true = np.cos(nu_on_grid)
    info = {'t': t, 'nu': nu, 'nu_raw': nu_raw, 'Umin': Umin, 'Umax': Umax,
            't_lo': t_lo, 't_hi': t_hi, 'alpha': alpha, 'beta': beta, 'e': e}
    return xs, ys, phi_zeta_true, info

def n_from_phi(phi, g):
    n = np.exp(g * phi)
    return n / np.mean(n)

# -------------------------
# Finite-diff tools (grid)
# -------------------------
def grad_centered(F, L):
    N = F.shape[0]; scale = (N-1)/L
    Fx = np.zeros_like(F); Fy = np.zeros_like(F)
    Fx[:,1:-1] = (F[:,2:] - F[:,:-2]) * (0.5*scale)
    Fx[:, 0]   = (F[:,1]  - F[:,0])  * scale
    Fx[:,-1]   = (F[:,-1] - F[:,-2]) * scale
    Fy[1:-1,:] = (F[2:,:] - F[:-2,:]) * (0.5*scale)
    Fy[ 0 ,:]  = (F[1 ,:] - F[ 0 ,:]) * scale
    Fy[-1 ,:]  = (F[-1,:] - F[-2,:]) * scale
    return Fx, Fy

def bilinear(grid, xq, yq, L):
    N = grid.shape[0]
    dx = L/(N-1); dy = L/(N-1)
    ix = np.clip((xq/dx).astype(int), 0, N-2)
    iy = np.clip((yq/dy).astype(int), 0, N-2)
    x0 = ix*dx; y0 = iy*dy
    tx = np.clip((xq - x0)/dx, 0, 1)
    ty = np.clip((yq - y0)/dy, 0, 1)
    g00 = grid[iy,   ix  ]
    g10 = grid[iy,   ix+1]
    g01 = grid[iy+1, ix  ]
    g11 = grid[iy+1, ix+1]
    return (1-tx)*(1-ty)*g00 + tx*(1-ty)*g10 + (1-tx)*ty*g01 + tx*ty*g11

# -------------------------
# Ray integrator (arclength RK4) under Fermat metric n^2 δ_ij
# -------------------------
def ray_integrate(n_grid, L, x0, y0, tau0, s_max, ds, bounce=False):
    nx_grid, ny_grid = grad_centered(n_grid, L)
    xs = [x0]; ys = [y0]
    tx, ty = tau0 / np.linalg.norm(tau0)
    txs = [tx]; tys = [ty]
    s = 0.0
    def gradn_at(xq,yq):
        gx = bilinear(nx_grid, np.array([xq]), np.array([yq]), L)[0]
        gy = bilinear(ny_grid, np.array([xq]), np.array([yq]), L)[0]
        return gx, gy
    def n_at(xq,yq):
        return bilinear(n_grid, np.array([xq]), np.array([yq]), L)[0]
    def f_state(x,y,tx,ty):
        nval = n_at(x,y)
        gx, gy = gradn_at(x,y)
        dot = gx*tx + gy*ty
        dtx = (gx - dot*tx)/max(nval,1e-12)
        dty = (gy - dot*ty)/max(nval,1e-12)
        return tx, ty, dtx, dty
    for _ in range(int(np.ceil(s_max/ds))):
        x, y = xs[-1], ys[-1]
        if not (0 <= x <= L and 0 <= y <= L): break
        k1 = f_state(x, y, tx, ty)
        k2 = f_state(x + 0.5*ds*k1[0], y + 0.5*ds*k1[1], tx + 0.5*ds*k1[2], ty + 0.5*ds*k1[3])
        k3 = f_state(x + 0.5*ds*k2[0], y + 0.5*ds*k2[1], tx + 0.5*ds*k2[2], ty + 0.5*ds*k2[3])
        k4 = f_state(x + ds*k3[0], y + ds*k3[1], tx + ds*k3[2], ty + ds*k3[3])
        dxds = (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6.0
        dyds = (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6.0
        dtxds= (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6.0
        dtyds= (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])/6.0
        x_new = x + ds*dxds; y_new = y + ds*dyds
        tx_new = tx + ds*dtxds; ty_new = ty + ds*dtyds
        norm = max(np.hypot(tx_new, ty_new), 1e-12)
        tx_new /= norm; ty_new /= norm
        if bounce:
            if x_new < 0 or x_new > L: tx_new *= -1
            if y_new < 0 or y_new > L: ty_new *= -1
            x_new = np.clip(x_new, 0, L); y_new = np.clip(y_new, 0, L)
        xs.append(x_new); ys.append(y_new)
        txs.append(tx_new); tys.append(ty_new)
        tx, ty = tx_new, ty_new
        s += ds
    return np.array(xs), np.array(ys), np.array(txs), np.array(tys)

def arclength(x, y):
    return np.hstack([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])

def curvature_and_normal(x, y):
    s = arclength(x, y)
    if s[-1] < 1e-9:
        Z = np.zeros_like(s); return Z, s, Z, Z
    xp  = np.gradient(x, s);  yp  = np.gradient(y, s)
    xpp = np.gradient(xp, s); ypp = np.gradient(yp, s)
    num = np.abs(xp*ypp - yp*xpp)
    den = (xp*xp + yp*yp)**1.5 + 1e-15
    kappa = num/den
    tau_norm = np.maximum(np.hypot(xp, yp), 1e-15)
    nx = -yp/tau_norm; ny = xp/tau_norm
    return kappa, s, nx, ny

# -------------------------
# Ray-certificate energies
# -------------------------
def E_ray_window_perp(x, y, n_grid, L, u1_idx, u2_idx):
    i1 = max(int(np.floor(u1_idx)), 0); i2 = min(int(np.ceil(u2_idx)), len(x)-1)
    if i2 - i1 < 3: return np.nan
    xx = x[i1:i2+1]; yy = y[i1:i2+1]
    nx_grid, ny_grid = grad_centered(n_grid, L)
    kappa, s, nx_hat, ny_hat = curvature_and_normal(xx, yy)
    dn_x = bilinear(nx_grid, xx, yy, L); dn_y = bilinear(ny_grid, xx, yy, L)
    dperp_n = dn_x*nx_hat + dn_y*ny_hat
    n_on = bilinear(n_grid, xx, yy, L)
    R_perp = n_on * kappa - dperp_n
    E = trapz(R_perp**2, s) / max(s[-1], 1e-12)
    return float(E)

def E_ray_window_vec(x, y, n_grid, L, u1_idx, u2_idx):
    """Vector residual: R_vec = d/ds(n τ) - ∇n, along the ray segment."""
    i1 = max(int(np.floor(u1_idx)), 0); i2 = min(int(np.ceil(u2_idx)), len(x)-1)
    if i2 - i1 < 3: return np.nan
    xx = x[i1:i2+1]; yy = y[i1:i2+1]
    s = arclength(xx, yy)
    xp = np.gradient(xx, s); yp = np.gradient(yy, s)
    tau_norm = np.maximum(np.hypot(xp, yp), 1e-12)
    tx = xp/tau_norm; ty = yp/tau_norm
    nx_grid, ny_grid = grad_centered(n_grid, L)
    n_on = bilinear(n_grid, xx, yy, L)
    gx = bilinear(nx_grid, xx, yy, L); gy = bilinear(ny_grid, xx, yy, L)
    a = n_on * tx; b = n_on * ty
    da_ds = np.gradient(a, s); db_ds = np.gradient(b, s)
    Rx = da_ds - gx; Ry = db_ds - gy
    R2 = Rx*Rx + Ry*Ry
    E = trapz(R2, s) / max(s[-1], 1e-12)
    return float(E)

# -------------------------
# Models & evaluation
# -------------------------
def build_models_real_zeta(L=100.0, N=480, g=0.12, seed=11000, t0=35.0, t_span=120.0, Nt=1200, theta_deg=28.0):
    xs, ys, phi_zeta_true, info = build_phi_zeta_true_grid(L=L, N=N, t0=t0, t_span=t_span, Nt=Nt, theta_deg=theta_deg, seed=seed)
    n_zeta  = n_from_phi(phi_zeta_true, g=g)

    # Alternative models (mildly mismatched): π-warp + random-phase warp
    phi_pi  = np.cos(1.1 * np.arccos(np.clip(phi_zeta_true, -1, 1)))
    n_pi    = n_from_phi(phi_pi, g=g)

    rng = set_seed(seed+1)
    rand_phase = rng.uniform(0, 2*np.pi, size=phi_zeta_true.shape)
    phi_phase = np.cos(np.arccos(np.clip(phi_zeta_true, -1, 1)) + 0.28*rand_phase)
    n_phase  = n_from_phi(phi_phase, g=g)

    models = {'zeta': {'phi':phi_zeta_true, 'n':n_zeta},
              'pi':   {'phi':phi_pi,        'n':n_pi},
              'phase':{'phi':phi_phase,     'n':n_phase},
              'info': info}
    return xs, ys, models

def generate_rays(n_grid, L, n_rays=18, s_max=260.0, ds=0.25, theta_in_deg=8.0, seed=0):
    rng = set_seed(seed)
    starts_y = np.linspace(0.1*L, 0.9*L, n_rays)
    theta = np.deg2rad(theta_in_deg); tau0 = np.array([np.cos(theta), np.sin(theta)])
    rays = []
    for y0 in starts_y:
        x, y, tx, ty = ray_integrate(n_grid, L, x0=0.0, y0=y0, tau0=tau0, s_max=s_max, ds=ds)
        rays.append({'x':x, 'y':y, 'tx':tx, 'ty':ty})
    return rays

def travel_time_on_segment(x, y, n_grid, L, i1, i2):
    xx = x[i1:i2+1]; yy = y[i1:i2+1]
    s = arclength(xx, yy)
    n_on = bilinear(n_grid, xx, yy, L)
    T = trapz(n_on, s); T_vac = s[-1]
    return T - T_vac

def window_stats(models, rays, L, w_idx=120, stride_idx=60):
    rows = []
    for ridx, ray in enumerate(rays):
        x = ray['x']; y = ray['y']; Np = len(x); s_all = arclength(x, y)
        for i1 in range(0, Np-3, stride_idx):
            i2 = min(i1 + w_idx, Np-1)
            if i2 - i1 < 3: break
            dt_meas = travel_time_on_segment(x, y, models['zeta']['n'], L, i1, i2)
            dt_preds = {k: travel_time_on_segment(x, y, models[k]['n'], L, i1, i2) for k in ['zeta','pi','phase']}
            errs = {k: abs(dt_preds[k] - dt_meas) for k in dt_preds}
            Eperp = {k: E_ray_window_perp(x, y, models[k]['n'], L, i1, i2) for k in ['zeta','pi','phase']}
            Evec  = {k: E_ray_window_vec (x, y, models[k]['n'], L, i1, i2) for k in ['zeta','pi','phase']}
            rows.append({
                'ray': ridx+1, 'u1': s_all[i1], 'u2': s_all[i2],
                'err_zeta': errs['zeta'], 'err_pi': errs['pi'], 'err_phase': errs['phase'],
                'dt_meas': dt_meas, 'dt_pred_zeta': dt_preds['zeta'], 'dt_pred_pi': dt_preds['pi'], 'dt_pred_phase': dt_preds['phase'],
                'Eperp_zeta': Eperp['zeta'], 'Eperp_pi': Eperp['pi'], 'Eperp_phase': Eperp['phase'],
                'Evec_zeta':  Evec['zeta'],  'Evec_pi':  Evec['pi'],  'Evec_phase':  Evec['phase'],
            })
    return rows

def summarize_errors(rows, keys):
    out = {}
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[key] = {'MAE': float(np.mean(np.abs(vals))) if vals.size else np.nan,
                    'RMSE': float(np.sqrt(np.mean(vals**2))) if vals.size else np.nan,
                    'N': int(vals.size)}
    return out

def paired_sign_test(a, b):
    a = np.asarray(a); b = np.asarray(b)
    diff = a - b
    wins = int(np.sum(diff < 0)); losses = int(np.sum(diff > 0)); ties = int(np.sum(diff == 0))
    n_eff = wins + losses
    from math import comb
    p = sum(comb(n_eff, k) for k in range(wins, n_eff+1)) * (0.5**n_eff) if n_eff>0 else 1.0
    return wins, losses, ties, p

# -------------------------
# Run (REAL ζ φ)
# -------------------------
L = 100.0; N = 480; g = 0.12; seed = 11000
t0 = 35.0; t_span = 120.0; Nt = 1200; theta_axis = 28.0
N_RAYS = 18; s_max = 260.0; ds = 0.25; theta_in = 8.0
w_idx = 120; stride_idx = 60

print("[SETUP] Building models from REAL ζ phase ...")
xs, ys, models = build_models_real_zeta(L=L, N=N, g=g, seed=seed, t0=t0, t_span=t_span, Nt=Nt, theta_deg=theta_axis)
print("  models: zeta(TRUE φ_ζ), pi(warp), phase(rand-phase warp); all DC-normalized <n>=1")

# sanity: show ν(t) basic info
nu = models['info']['nu']; t = models['info']['t']
dnu = np.gradient(nu, t)
print(f"[PHASE] t∈[{t[0]:.1f},{t[-1]:.1f}], mean dν/dt ≈ {np.mean(dnu):.3f}, std ≈ {np.std(dnu):.3f}")

print("[TRACE] Generating rays under TRUTH_MODEL=ζ ...")
rays = generate_rays(models['zeta']['n'], L, n_rays=N_RAYS, s_max=s_max, ds=ds, theta_in_deg=theta_in, seed=seed)

print("[WINDOW] Computing windowed stats (errors and ray-certificate energies) ...")
rows = window_stats(models, rays, L, w_idx=w_idx, stride_idx=stride_idx)

print("\nwin,ray,u1,u2,err_zeta,err_pi,err_phase,dt_meas,dt_pred_zeta,dt_pred_pi,dt_pred_phase,Eperp_zeta,Eperp_pi,Eperp_phase,Evec_zeta,Evec_pi,Evec_phase")
for i, r in enumerate(rows[:12]):
    print(f"{i+1:02d},{r['ray']:02d},{r['u1']:.2f},{r['u2']:.2f},"
          f"{r['err_zeta']:.4e},{r['err_pi']:.4e},{r['err_phase']:.4e},"
          f"{r['dt_meas']:.4f},{r['dt_pred_zeta']:.4f},{r['dt_pred_pi']:.4f},{r['dt_pred_phase']:.4f},"
          f"{r['Eperp_zeta']:.3e},{r['Eperp_pi']:.3e},{r['Eperp_phase']:.3e},"
          f"{r['Evec_zeta']:.3e},{r['Evec_pi']:.3e},{r['Evec_phase']:.3e}")

# summaries
err_keys   = ['err_zeta','err_pi','err_phase']
Eperp_keys = ['Eperp_zeta','Eperp_pi','Eperp_phase']
Evec_keys  = ['Evec_zeta','Evec_pi','Evec_phase']

err_summary   = summarize_errors(rows, err_keys)
Eperp_summary = summarize_errors(rows, Eperp_keys)
Evec_summary  = summarize_errors(rows, Evec_keys)

print("\n[ERROR SUMMARY  (lower is better)]")
for k in err_keys:
    meta = err_summary[k]; print(f"{k:12s}  MAE={meta['MAE']:.6f}  RMSE={meta['RMSE']:.6f}  (N={meta['N']})")

print("\n[RAY CERTIFICATE  normal-component E_perp SUMMARY]")
for k in Eperp_keys:
    meta = Eperp_summary[k]; print(f"{k:12s}  MAE={meta['MAE']:.6e}  RMSE={meta['RMSE']:.6e}  (N={meta['N']})")

print("\n[RAY CERTIFICATE  vector E_vec SUMMARY  (recommended)]")
for k in Evec_keys:
    meta = Evec_summary[k]; print(f"{k:12s}  MAE={meta['MAE']:.6e}  RMSE={meta['RMSE']:.6e}  (N={meta['N']})")

# paired tests
def collect(rows, name): return np.array([r[name] for r in rows], dtype=float)
ez, ep, ef = collect(rows,'err_zeta'), collect(rows,'err_pi'), collect(rows,'err_phase')
Eperp_z, Eperp_p, Eperp_f = collect(rows,'Eperp_zeta'), collect(rows,'Eperp_pi'), collect(rows,'Eperp_phase')
Evec_z,  Evec_p,  Evec_f  = collect(rows,'Evec_zeta'),  collect(rows,'Evec_pi'),  collect(rows,'Evec_phase')

wins, losses, ties, p = paired_sign_test(ez, ep)
print(f"\n[Sign test] |err_zeta| < |err_pi|      : wins={wins}, losses={losses}, ties={ties}, p≈{p:.3e}")
wins, losses, ties, p = paired_sign_test(ez, ef)
print(f"[Sign test] |err_zeta| < |err_phase|   : wins={wins}, losses={losses}, ties={ties}, p≈{p:.3e}")

wins, losses, ties, p = paired_sign_test(Eperp_z, Eperp_p)
print(f"[Cert⊥ ]    Eperp_zeta  < Eperp_pi     : wins={wins}, losses={losses}, ties={ties}, p≈{p:.3e}")
wins, losses, ties, p = paired_sign_test(Eperp_z, Eperp_f)
print(f"[Cert⊥ ]    Eperp_zeta  < Eperp_phase  : wins={wins}, losses={losses}, ties={ties}, p≈{p:.3e}")

wins, losses, ties, p = paired_sign_test(Evec_z, Evec_p)
print(f"[Cert→ ]    Evec_zeta   < Evec_pi      : wins={wins}, losses={losses}, ties={ties}, p≈{p:.3e}")
wins, losses, ties, p = paired_sign_test(Evec_z, Evec_f)
print(f"[Cert→ ]    Evec_zeta   < Evec_phase   : wins={wins}, losses={losses}, ties={ties}, p≈{p:.3e}")

# lock-rate by E_vec threshold from ζ
def mad(a):
    a = a[np.isfinite(a)]
    m = np.median(a) if a.size>0 else np.nan
    return (np.median(np.abs(a-m)) if a.size>0 else np.nan) + 1e-18

tau = (np.median(Evec_z[np.isfinite(Evec_z)]) + 3.0*mad(Evec_z))
lock = lambda arr: 100.0*np.mean(arr[np.isfinite(arr)] <= tau) if np.isfinite(tau) else np.nan
print(f"\n[LOCK RATE by E_vec threshold τ={tau:.3e}]  ζ={lock(Evec_z):.1f}%   π={lock(Evec_p):.1f}%   phase={lock(Evec_f):.1f}%")

# diagnostic plots
plt.figure(figsize=(6,4))
plt.plot(t, nu, label='unwrap arg ζ(1/2+it)')
plt.xlabel('t'); plt.ylabel('ν(t)'); plt.title('REAL ζ phase ν(t)')
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(6,5))
plt.hist(np.log10(Evec_z+1e-30), bins=30, alpha=0.6, label='log10 E_vec(zeta)')
plt.hist(np.log10(Evec_p+1e-30), bins=30, alpha=0.6, label='log10 E_vec(pi)')
plt.hist(np.log10(Evec_f+1e-30), bins=30, alpha=0.6, label='log10 E_vec(phase)')
plt.title("Ray-certificate (vector) energies")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(6,5))
plt.scatter(ez+1e-30, Evec_z+1e-30, s=10, alpha=0.7, label='zeta')
plt.scatter(ep+1e-30, Evec_p+1e-30, s=10, alpha=0.7, label='pi')
plt.scatter(ef+1e-30, Evec_f+1e-30, s=10, alpha=0.7, label='phase')
plt.xscale('log'); plt.yscale('log')
plt.xlabel("|err|"); plt.ylabel("E_vec")
plt.legend(); plt.tight_layout(); plt.show()

print("\nDone. This is the REAL-φ_ζ pipeline: ν(t) comes directly from arg ζ(1/2+it), no synthetic gaps.\n"
      "Swap (t0,t_span,Nt,theta_deg) to probe other spectral windows. The source detection uses delay errors + vector certificate E_vec.")
