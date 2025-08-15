# --- Robust Colab cell: Riemann-phase N(x) -> Δt_pred vs Δt_meas (windowed xcorr) ---

import numpy as np
import matplotlib.pyplot as plt

# constants
c0  = 299792458.0
mu0 = 4e-7*np.pi
eps0 = 1.0/(mu0*c0**2)

# 20 Riemann zero imag parts
gammas = np.array([
    14.134725141, 21.022039639, 25.010857580, 30.424876125, 32.935061588,
    37.586178159, 40.918719012, 43.327073281, 48.005150881, 49.773832478,
    52.970321478, 56.446247697, 59.347044003, 60.831778525, 65.112544048,
    67.079810529, 69.546401711, 72.067157674, 75.704690699, 77.144840069
])

def phi_kernel(u, gammas, sigma):
    U = u[:, None] - gammas[None, :]
    return np.arctan(U / sigma).sum(axis=1)

# grid & medium
L   = 1.0
Nx  = 4000                   # 更细网格
x   = np.linspace(0.0, L, Nx)
xm, xM = 0.10, 0.90          # 更长介质段（0.8 m）
mask = (x >= xm) & (x <= xM)

# map x->u in medium
Umax = 60.0
u = np.zeros_like(x)
u[mask] = (x[mask]-xm)/(xM-xm) * Umax

# Riemann phase -> tau
sigma, b = 1.0, 1.0
phi_u = phi_kernel(u, gammas, sigma)
tau_u = b * (phi_u - phi_u[mask].mean())

# positive-biased mapping to ensure N>=1
alpha, beta = 0.8, 0.9
N = np.ones_like(x)
N_med = np.exp(alpha * (np.tanh(beta*tau_u)**2))
N[mask] = N_med[mask]

# predicted delay (WKB group delay)
delta_t_pred = (1.0/c0) * np.trapezoid(N[mask]-1.0, x[mask])
print(f"[Predict] <N>_med={N[mask].mean():.6f}, Δt_pred={delta_t_pred:.3e} s")

# Yee 1D FDTD (Mur), run twice: baseline (N=1) & medium N(x)
def run_fdtd_1d(Nprofile, S=0.35, steps=None, src_idx=40, det_idx=None,
                t0=6e-9, tau_src=1.2e-9, amp=2e-5):
    if det_idx is None:
        det_idx = int(0.95*Nx)
    dx = x[1]-x[0]
    # N>=1 => c(x)<=c0 -> use c0 for CFL
    dt = S * dx / c0
    # 步数：确保覆盖基线飞行时间 + 预测延迟 + 裕量
    if steps is None:
        base_time = (det_idx - src_idx)*dx / c0
        T = base_time + abs(delta_t_pred) + 6*tau_src
        steps = int(T/dt) + 400
    t = np.arange(steps)*dt

    eps = eps0 * (Nprofile**2)
    mu  = mu0 * np.ones(Nx)

    Ez = np.zeros(Nx)
    Hy = np.zeros(Nx-1)

    Ez_l_prev = 0.0
    Ez_r_prev = 0.0
    rec = []

    for n in range(steps):
        # H update
        Hy += (dt/(mu[:-1]*dx)) * (Ez[1:] - Ez[:-1])
        # E update
        curlH = np.zeros_like(Ez)
        curlH[1:-1] = Hy[1:] - Hy[:-1]
        Ez += (dt/(eps*dx)) * curlH
        # Gaussian source
        Ez[src_idx] += amp*np.exp(-0.5*((n*dt - t0)/tau_src)**2)
        # Mur-1 boundaries (outside medium N=1)
        Ez[0]  = Ez_l_prev  + (c0*dt - dx)/(c0*dt + dx) * (Ez[1]  - Ez[0])
        Ez_l_prev  = Ez[1]
        Ez[-1] = Ez_r_prev + (c0*dt - dx)/(c0*dt + dx) * (Ez[-2] - Ez[-1])
        Ez_r_prev = Ez[-2]
        rec.append(Ez[det_idx])

    return np.array(rec), t, dt

det_idx = int(0.95*Nx)
E_base, t, dt = run_fdtd_1d(np.ones_like(x), det_idx=det_idx)
E_med,  _, _  = run_fdtd_1d(N,               det_idx=det_idx)

# windowed cross-correlation around expected arrival
def window_xcorr(s_ref, s_sig, t, t_center, win_half_width, dt):
    # 取 [t_center - W, t_center + W] 的窗口做互相关
    lo = max(0, np.searchsorted(t, t_center - win_half_width))
    hi = min(len(t), np.searchsorted(t, t_center + win_half_width))
    a = s_ref[lo:hi] - np.mean(s_ref[lo:hi])
    b = s_sig[lo:hi] - np.mean(s_sig[lo:hi])
    corr = np.correlate(b, a, mode="full")
    lag = np.argmax(corr) - (len(a)-1)
    return lag*dt, corr, (lo, hi)

# 估算基线到达时刻（粗）
t_arr_base_est = t[np.argmax(np.abs(E_base))]
# 以这个时刻为中心开窗口，半宽取 5*tau_src + |Δt_pred|
win_half = 5*1.2e-9 + abs(delta_t_pred)
dt_xc, corr_win, (lo, hi) = window_xcorr(E_base, E_med, t, t_arr_base_est, win_half, dt)

# also coarse peak method
t_arr_base = t[np.argmax(np.abs(E_base))]
t_arr_med  = t[np.argmax(np.abs(E_med))]
dt_peak = t_arr_med - t_arr_base

# diagnostics
sample_delay_pred = delta_t_pred/dt
print(f"[Diag] dt={dt:.3e}s,  predicted ~ {sample_delay_pred:.1f} samples")
print(f"[Measure] Δt_xcorr={dt_xc:.3e} s,  Δt_peak={dt_peak:.3e} s")

# plots
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(x, N)
plt.xlabel("x [m]"); plt.ylabel("N(x)")
plt.title("N(x) (Riemann-phase, >=1)")

plt.subplot(1,3,2)
plt.plot(t, E_base, label="Baseline")
plt.plot(t, E_med,  label="Medium")
plt.axvline(t_arr_base_est, ls="--")
plt.xlabel("time [s]"); plt.ylabel("E(det)")
plt.legend(); plt.title("Detector waveforms")

plt.subplot(1,3,3)
lags = (np.arange(len(corr_win)) - ( (hi-lo)-1 ))*dt
plt.plot(lags, corr_win)
plt.axvline(delta_t_pred, ls="--", label="Pred")
plt.axvline(dt_xc, ls=":", label="Xcorr")
plt.xlabel("lag [s]"); plt.ylabel("xcorr (windowed)")
plt.title("Windowed cross-correlation")
plt.legend()
plt.tight_layout()
plt.show()

# compare
rel_err = abs(dt_xc - delta_t_pred)/(abs(delta_t_pred) + 1e-20)
print(f"[Compare] Pred: {delta_t_pred:.3e} s | Xcorr: {dt_xc:.3e} s | Rel.err ≈ {rel_err:.2f}")
