import numpy as np

# ---------- 1) 构造“黎曼相位格子”诱导的 N(z) ----------
gamma = np.array([
    14.134725, 21.022039, 25.010858, 30.424876, 32.935062, 37.586178, 40.918719, 43.327073,
    48.005150, 49.773832, 52.970321, 56.446247, 59.347044, 60.831779, 65.112545, 67.079811,
    69.546401, 72.067158, 75.704690, 77.144840
])

def phi_from_riemann(u, sigma=1.5):
    du = (u[:, None] - gamma[None, :]) / sigma
    return np.arctan(du).sum(axis=1)

def N_from_phi(phi, a=0.20):
    phi_c = phi - phi.mean()
    return np.exp(a * phi_c)

z = np.linspace(-1.0, 1.0, 2001)
u0, alpha = 60.0, 5.0
u = u0 + alpha * z
phi = phi_from_riemann(u, sigma=1.5)
N = N_from_phi(phi, a=0.20)
mask1 = z < 0
mask2 = z >= 0
n1 = N[mask1].mean()
n2 = N[mask2].mean()

# ---------- 2) 两段路径几何 ----------
A = np.array([-1.0, -1.0])
B = np.array([1.0, 1.0])

def path_lengths(x):
    P = np.array([x, 0.0])
    L1 = np.linalg.norm(P - A)
    L2 = np.linalg.norm(B - P)
    return L1, L2

def optical_action(x):
    L1, L2 = path_lengths(x)
    return n1 * L1 + n2 * L2

xs = np.linspace(-1.5, 1.5, 30001)  # 扩展范围，增加网格点
Axs = np.array([optical_action(x) for x in xs])

# ---------- 3) 驻点主导检验 ----------
x_star = 0.7047  # 驻点
sigma = 0.08     # 局域宽度
sigma_win = 0.015  # 更窄的总积分窗
n = 5            # 局域范围 n * sigma

def compute_R(k0, xs, Axs, x_star, sigma, sigma_win, n):
    mask = np.abs(xs - x_star) < n * sigma
    integrand = np.exp(1j * k0 * Axs)
    local_integral = np.trapz(integrand[mask], xs[mask])
    window = np.exp(-(xs - x_star)**2 / (2 * sigma_win**2))
    total_integral = np.trapz(integrand * window, xs)
    R = np.abs(local_integral) / np.abs(total_integral) if np.abs(total_integral) > 1e-10 else np.nan
    return R

def compute_SPA_and_window(k0, xs, Axs, x_star, sigma):
    ix_star = np.argmin(np.abs(xs - x_star))
    A_star = Axs[ix_star]
    dx = xs[1] - xs[0]
    x = x_star
    Spp = n1 / (1 + (x + 1)**2)**(1.5) + n2 / (1 + (x - 1)**2)**(1.5)
    K_SPA = np.exp(1j * k0 * A_star) * np.sqrt(2 * np.pi / (k0 * np.abs(Spp))) * np.exp(1j * np.pi / 4 * np.sign(Spp))
    window = np.exp(-(xs - x_star)**2 / (2 * sigma**2))
    K_win = np.trapz(np.exp(1j * k0 * Axs) * window, xs)
    rel_error = np.abs(K_SPA - K_win) / np.abs(K_win) if np.abs(K_win) > 1e-10 else np.nan
    phase_error = np.abs(np.angle(K_SPA) - np.angle(K_win)) if np.abs(K_win) > 1e-10 else np.nan
    return rel_error, phase_error

# 测试多个 λ
lambdas = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
k0s = [2.0 * np.pi / lam for lam in lambdas]
results = []

for k0, lam in zip(k0s, lambdas):
    R = compute_R(k0, xs, Axs, x_star, sigma, sigma_win, n)
    rel_error, phase_error = compute_SPA_and_window(k0, xs, Axs, x_star, sigma)
    results.append((lam, k0, R, rel_error, phase_error))
    print(f"λ={lam:.4f}, k0={k0:.2f}: R={R:.3f}, 相对误差={rel_error:.3%}, 相位误差={phase_error:.3f} rad")

# 保存结果用于可视化
lambdas, k0s, Rs, rel_errors, phase_errors = zip(*results)