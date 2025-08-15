# 验证实验：Riemann零点诱导的几何相位是否调制有效光速 c_eff 和内部时间 T_int？
# 测试弱耦合(a=0.02)、过度平滑(sigma=12)、非Riemann谱(均匀晶格)对调制的影响。
# 添加 SCI K(tau) 计算，验证 K=1 吸引子。
# 适配 Google Colab，包含统计表和可视化。

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi

# ----------------------- 数据：前30个Riemann零点虚部 gamma_n -----------------------
gammas = np.array([
    14.13472514, 21.02203964, 25.01085758, 30.42487613, 32.93506159,
    37.58617816, 40.91871901, 43.32707328, 48.00515088, 49.77383248,
    52.97032148, 56.44624770, 59.34704400, 60.83177953, 65.11254405,
    67.07981053, 69.54640171, 72.06715767, 75.70469070, 77.14484007,
    79.33737502, 82.91038085, 84.73549298, 87.42527461, 88.80911121,
    92.49189927, 94.65134404, 95.87063423, 98.83119422, 101.3178510
])

# ----------------------- 辅助函数 -----------------------
def geometric_phase(u, gammas, sigma):
    """计算几何相位 phi(u) = sum_n arctan((u - gamma_n) / sigma)"""
    x = (u[:, None] - gammas[None, :]) / sigma
    return np.arctan(x).sum(axis=1)

def compute_fields(u, gammas, sigma=2.0, a=0.25, b=0.03, c0=299792458.0):
    """计算相位、tau、折射率、有效光速、内部时间和 SCI"""
    phi = geometric_phase(u, gammas, sigma)
    tau = b * (phi - phi.mean())  # 零均值 tau
    N = np.exp(a * tau)  # 折射率
    c_eff = c0 / N  # 有效光速
    T_int = np.cumsum(N) * (u[1] - u[0]) / c0  # 内部时间
    # SCI: K = Phi / H
    Phi = np.exp(tau)  # 结构密度
    H = np.exp(tau) / (np.abs(tau) + 1 + 1e-6)  # 熵代理，确保正值
    K = np.where((H != 0) & (~np.isnan(H)), Phi / H, 0)  # 避免除零
    return phi, tau, N, c_eff, T_int, K

def summarize(u, tau, N, c_eff, T_int, K):
    """统计调制深度、时间膨胀和相关性"""
    N_min, N_max = N.min(), N.max()
    c_min, c_max = c_eff.min(), c_eff.max()
    mod_N = (N_max - N_min) / N.mean()
    mod_c = (c_max - c_min) / c_eff.mean()
    T_baseline = (u[-1] - u[0]) / c0
    T_total = T_int[-1]
    dilation = (T_total - T_baseline) / T_baseline
    corr = np.corrcoef(tau, np.log(c_eff))[0, 1]
    return {
        "N_mean": N.mean(),
        "N_mod_rel": mod_N,
        "c_eff_mean": c_eff.mean(),
        "c_eff_mod_rel": mod_c,
        "T_total_s": T_total,
        "T_rel_dilation": dilation,
        "corr_tau_logc": corr,
        "K_mean": np.mean(K),
        "K_min": np.min(K[K > 0]),  # 排除零值
        "K_max": np.max(K)
    }

# ----------------------- 网格和场景 -----------------------
u = np.linspace(10.0, 110.0, 4000)  # 覆盖 gamma_n 范围
c0 = 299792458.0  # 真空光速

# 场景 A: 基线 (Riemann零点，适中耦合)
phi_A, tau_A, N_A, c_A, T_A, K_A = compute_fields(u, gammas, sigma=2.0, a=0.25, b=0.03)
sum_A = summarize(u, tau_A, N_A, c_A, T_A, K_A)

# 场景 B: 弱耦合 (a=0.02)
phi_B, tau_B, N_B, c_B, T_B, K_B = compute_fields(u, gammas, sigma=2.0, a=0.02, b=0.03)
sum_B = summarize(u, tau_B, N_B, c_B, T_B, K_B)

# 场景 C: 过度平滑 (sigma=12)
phi_C, tau_C, N_C, c_C, T_C, K_C = compute_fields(u, gammas, sigma=12.0, a=0.25, b=0.03)
sum_C = summarize(u, tau_C, N_C, c_C, T_C, K_C)

# 场景 D: 均匀晶格谱
gammas_uniform = np.linspace(gammas.min(), gammas.max(), len(gammas))
phi_D, tau_D, N_D, c_D, T_D, K_D = compute_fields(u, gammas_uniform, sigma=2.0, a=0.25, b=0.03)
sum_D = summarize(u, tau_D, N_D, c_D, T_D, K_D)

# ----------------------- 统计表 -----------------------
df = pd.DataFrame.from_records([
    dict(Scenario="A: Riemann, a=0.25, σ=2", **sum_A),
    dict(Scenario="B: Riemann, a=0.02, σ=2", **sum_B),
    dict(Scenario="C: Riemann, a=0.25, σ=12", **sum_C),
    dict(Scenario="D: Uniform lattice, a=0.25, σ=2", **sum_D),
]).round(6)

print("验证统计表：")
print(df[['Scenario', 'N_mod_rel', 'c_eff_mod_rel', 'T_rel_dilation', 'corr_tau_logc', 'K_mean']])

# ----------------------- 可视化 -----------------------
# 图1：c_eff(u) 叠加
plt.figure(figsize=(10, 6))
plt.plot(u, c_A, label="A: Riemann, a=0.25, σ=2")
plt.plot(u, c_B, label="B: Riemann, a=0.02, σ=2")
plt.plot(u, c_C, label="C: Riemann, a=0.25, σ=12")
plt.plot(u, c_D, label="D: Uniform, a=0.25, σ=2")
plt.title("Effective Light Speed c_eff(u) Across Scenarios")
plt.xlabel("u")
plt.ylabel("c_eff(u) [m/s]")
plt.legend()
plt.grid(True)
plt.show()

# 图2：T_int(u) 叠加
plt.figure(figsize=(10, 6))
plt.plot(u, T_A, label="A: Riemann, a=0.25, σ=2")
plt.plot(u, T_B, label="B: Riemann, a=0.02, σ=2")
plt.plot(u, T_C, label="C: Riemann, a=0.25, σ=12")
plt.plot(u, T_D, label="D: Uniform, a=0.25, σ=2")
plt.title("Internal Time Accumulation T_int(u) Across Scenarios")
plt.xlabel("u")
plt.ylabel("T_int(u) [s]")
plt.legend()
plt.grid(True)
plt.show()

# 图3：基线 phi(u)
plt.figure(figsize=(10, 6))
plt.plot(u, phi_A, label="phi(u) baseline")
plt.title("Geometric Phase phi(u) (Baseline)")
plt.xlabel("u")
plt.ylabel("phi(u) [radians]")
plt.grid(True)
plt.show()

# 图4：基线 log c_eff(u)
plt.figure(figsize=(10, 6))
plt.plot(u, np.log(c_A), label="log c_eff(u) baseline")
plt.title("Log Effective Light Speed log c_eff(u) (Baseline)")
plt.xlabel("u")
plt.ylabel("log c_eff(u)")
plt.grid(True)
plt.show()

# 图5：基线 K(u)
plt.figure(figsize=(10, 6))
plt.plot(u, K_A, label="K(u) baseline")
plt.title("Structural Conservation Index K(u) (Baseline)")
plt.xlabel("u")
plt.ylabel("K(u)")
plt.grid(True)
plt.show()

# 输出基线统计
print("\n基线统计 (Scenario A):")
for key, value in sum_A.items():
    print(f"{key}: {value:.6f}")
