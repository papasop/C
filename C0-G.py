# === Colab cell: TRUE zeros → phase → index → eikonal time-of-flight, with frozen-J blind test ===
import numpy as np

np.set_printoptions(suppress=True)
rng = np.random.default_rng(20250828)

# ----------------- Physical constant -----------------
c0 = 299_792_458.0  # m/s

# ----------------- First ~50 Riemann zeros (imag parts γn) -----------------
# 采用常见的前 ~50 个 γn（足够做本文弱场、窗口尺度演示；无需联网）
gammas_true = np.array([
  14.1347251417, 21.0220396388, 25.0108575801, 30.4248761259, 32.9350615877,
  37.5861781588, 40.9187190121, 43.3270732809, 48.0051508812, 49.7738324777,
  52.9703214777, 56.4462476971, 59.3470440026, 60.8317785246, 65.1125440481,
  67.0798105295, 69.5464017112, 72.0671576745, 75.7046906991, 77.1448400689,
  79.3373750202, 82.9103808541, 84.7354929805, 87.4252746131, 88.8091112076,
  92.4918992706, 94.6513440405, 95.8706342282, 98.8311942182,101.3178510057,
 103.7255380405,105.4466230523,107.1686111843,111.0295355432,111.8746591770,
 114.3202209155,116.2266803212,118.7907828660,121.3701250024,122.9468292936,
 124.2568185543,127.5166838796,129.5787041997,131.0876885309,133.4977372020,
 134.7565097530,138.1160420545,139.7362089521
], dtype=float)
M = len(gammas_true)

# ----------------- Helpers -----------------
def build_lnN(u, gammas, alpha=1.6, s_target=3.0e-4, cap=None):
    """φ(u)=Σ arctan((u-γ)/σ),  τ=b(φ-<φ>),  lnN = a τ; choose b so std_u(φ)<->s_target."""
    gaps = np.diff(gammas)
    mean_gap = float(np.mean(gaps))
    sigma = alpha * mean_gap
    # 计算 φ(u)
    # Broadcasting: u[:,None] - gammas[None,:] -> shape (Nu, M)
    phi = np.arctan((u[:, None] - gammas[None, :]) / sigma).sum(axis=1)
    phi = phi - float(phi.mean())
    std_phi = float(phi.std())
    b = s_target / (std_phi + 1e-300)
    lnN = b * phi  # a=1
    if cap is not None:
        maxabs = np.max(np.abs(lnN))
        if maxabs > cap:
            lnN = lnN * (cap / maxabs)
    return lnN, sigma, b, mean_gap, std_phi

def integrate_Iu(u, N, a, b):
    """Iu = ∫_{[a,b]} (N-1) du"""
    m = (u >= a) & (u <= b)
    if not np.any(m):
        return 0.0
    return float(np.trapezoid(N[m] - 1.0, u[m]))

def fractional_delay(x, dt_shift, dt):
    """频域相移实现连续分数时延"""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), d=dt)
    phase = np.exp(-1j * 2.0 * np.pi * f * dt_shift)
    y = np.fft.irfft(X * phase, n=len(x))
    return y

def measure_delay(x, y, dt):
    """x↔y 的互相关峰值（抛物线细化）→ 连续时延（有符号，单位 s）"""
    r = np.correlate(y, x, mode='full')  # lag>0 ~ y滞后于x（即 y(t) = x(t - Δt) → 峰在正索引）
    k = int(np.argmax(r))
    if 0 < k < len(r)-1:
        # 抛物线细化
        y1, y2, y3 = r[k-1], r[k], r[k+1]
        denom = (y1 - 2*y2 + y3)
        if abs(denom) > 1e-30:
            delta = 0.5 * (y1 - y3) / denom
        else:
            delta = 0.0
    else:
        delta = 0.0
    lag = (k - (len(x) - 1)) + delta
    return float(lag * dt)

def make_probe(T=2e-6, dt=2.5e-10, bw=0.25):
    """合成带限探测信号"""
    t = np.arange(0.0, T, dt)
    x = rng.normal(0.0, 1.0, size=t.size)
    # 简单低通：移动平均（bw 控制平滑强度）
    K = max(3, int(bw * 64))
    k = np.hanning(K); k = k / k.sum()
    x = np.convolve(x, k, mode='same')
    x = x / (np.std(x) + 1e-12)
    return t, x

def make_poisson_like(gammas_ref, rng):
    mean_gap = float(np.mean(np.diff(gammas_ref)))
    gaps = rng.exponential(scale=mean_gap, size=len(gammas_ref)-1)
    out = np.empty_like(gammas_ref)
    out[0] = gammas_ref[0]
    out[1:] = out[0] + np.cumsum(gaps)
    return out

def make_gue_like(gammas_ref, rng):
    # Wigner surmise ↔ Rayleigh(σ=√(2/π)) with mean 1; then scale to mean_gap
    mean_gap = float(np.mean(np.diff(gammas_ref)))
    sigma = np.sqrt(2.0/np.pi)
    gaps_u = rng.rayleigh(scale=sigma, size=len(gammas_ref)-1)  # mean ≈ 1
    gaps = gaps_u * mean_gap
    out = np.empty_like(gammas_ref)
    out[0] = gammas_ref[0]
    out[1:] = out[0] + np.cumsum(gaps)
    return out

# ----------------- u-grid and windows -----------------
u_min, u_max, Nu = 0.0, 160.0, 20001
u = np.linspace(u_min, u_max, Nu)

# 窗口配置（校准 + 三个盲窗）
cal_win   = (16.913, 41.413)
blind_wins = [(42.413, 66.913),
              (67.913, 92.413),
              (93.413,117.913)]

# ----------------- A) TRUE: build lnN & print weak-field summary -----------------
alpha = 1.6
s_target = 3.0e-4
cap = 1.0e-2

lnN_true, sigma, b, mean_gap, std_phi = build_lnN(u, gammas_true, alpha=alpha, s_target=s_target, cap=cap)
N_true = np.exp(lnN_true)

print("=== A) 真零点 → 相位 → 折射率（弱场摘要） ===")
print(f"[zeros] count={M},  mean_gap={mean_gap:.6f},  sigma={sigma:.6f}")
print(f"[index] max|ln N|={np.max(np.abs(lnN_true)):.3e}  max|N-1|={np.max(np.abs(N_true-1)):.3e}  (weak-field OK if ≪1)")
print(f"[gauge] b={b:.6e}  s_target={s_target:.3e}")

# ----------------- TRUE measurement synthesis: choose underlying J_true -----------------
# 物理“真值”几何尺：只有实验室知道；我们在合成观测时用它，模型不知道。
J_true = -1.07e2  # m / u  （负号仅代表 ∫(N-1)du 的符号选择，便于演示）
# 模型的初始猜测（比如机械尺读数），随后用“唯一观测”做一次几何标定：
J0_guess = -3.00e1

# ----------------- [TRUE] Calibration of J using the single observation on cal_win -----------------
Iu_cal = integrate_Iu(u, N_true, cal_win[0], cal_win[1])
dt_true_cal = (J_true / c0) * Iu_cal

# 合成唯一观测（真模型）：生成探测信号并加入分数延时
t_sig, x_sig = make_probe(T=2.0e-6, dt=2.5e-10, bw=0.25)
y_sig = fractional_delay(x_sig, dt_true_cal, t_sig[1]-t_sig[0])
dt_meas_cal = measure_delay(x_sig, y_sig, t_sig[1]-t_sig[0])

# 用模型初猜 J0 对同一窗做预测
dt_pred_cal = (J0_guess / c0) * Iu_cal

# 冻结几何比例尺：一次性标定 J（与模型谱无关）
J_frozen = J0_guess * (dt_meas_cal / (dt_pred_cal + 1e-300))

print("\n=== [TRUE] 构造与校准 ===")
print(f"[TRUE] CAL win: u∈[{cal_win[0]:.3f},{cal_win[1]:.3f}]  Iu_cal={Iu_cal:+.6e}  =>  "
      f"dt_true_cal={dt_true_cal:+.6e} s,  dt_meas_cal={dt_meas_cal:+.6e} s")
print(f"[TRUE] J_frozen = {J_frozen:.6e} m/u  (由唯一观测标定；不回调谱/σ/b)")

# ----------------- Build surrogate models (Poisson / GUE) -----------------
gammas_poi = make_poisson_like(gammas_true, rng)
gammas_gue = make_gue_like(gammas_true, rng)

lnN_poi, _, _, _, _ = build_lnN(u, gammas_poi, alpha=alpha, s_target=s_target, cap=cap)
N_poi = np.exp(lnN_poi)
lnN_gue, _, _, _, _ = build_lnN(u, gammas_gue, alpha=alpha, s_target=s_target, cap=cap)
N_gue = np.exp(lnN_gue)

# ----------------- Blind test on three windows (single measurement from TRUE; multi-model predictions) -----------------
print("\n=== 盲测：单一观测（真零点合成），多模型预测（冻结 J，不回调） ===")
print("win   [u_a,u_b]                 dt_pred_true(s)        dt_meas(s)            |AE_true|(s)  RE_true(%)   "
      "dt_pred_poisson(s)   |AE_poi|(s)  RE_poi(%)   dt_pred_gue(s)   |AE_gue|(s)  RE_gue(%)")

for i, (ua, ub) in enumerate(blind_wins, start=1):
    # 1) 测量：唯一“真”观测（对同一窗合成一条 y，并从中测得 dt）
    Iu_true = integrate_Iu(u, N_true, ua, ub)
    dt_true = (J_true / c0) * Iu_true
    # 用与校准同一管线生成“观测”
    t_i, x_i = make_probe(T=2.0e-6, dt=2.5e-10, bw=0.25)
    y_i = fractional_delay(x_i, dt_true, t_i[1]-t_i[0])
    dt_meas = measure_delay(x_i, y_i, t_i[1]-t_i[0])

    # 2) 三模型的“冻结 J”预测
    dt_pred_true = (J_frozen / c0) * Iu_true

    Iu_poi = integrate_Iu(u, N_poi, ua, ub)
    dt_pred_poi = (J_frozen / c0) * Iu_poi

    Iu_gue = integrate_Iu(u, N_gue, ua, ub)
    dt_pred_gue = (J_frozen / c0) * Iu_gue

    # 3) 误差
    def err_pair(pred, meas):
        ae = abs(pred - meas)
        re = 0.0 if abs(meas) < 1e-30 else (ae / abs(meas) * 100.0)
        return ae, re

    ae_t, re_t = err_pair(dt_pred_true, dt_meas)
    ae_p, re_p = err_pair(dt_pred_poi, dt_meas)
    ae_g, re_g = err_pair(dt_pred_gue, dt_meas)

    print(f"{i:3d}  [{ua:7.3f},{ub:7.3f}]   "
          f"{dt_pred_true:+.12e}  {dt_meas:+.12e}  {ae_t:8.3e}  {re_t:8.3f}   "
          f"{dt_pred_poi:+.12e}  {ae_p:8.3e}  {re_p:8.3f}   "
          f"{dt_pred_gue:+.12e}  {ae_g:8.3e}  {re_g:8.3f}")

print("\n(Done)")

