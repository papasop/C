import numpy as np
import mpmath
import matplotlib.pyplot as plt

# Parameters
M = 50  # Number of nontrivial zeros
sigma = 1.0
b = 1.0
a = 0.1
u0, u1 = 0, 246  # Approximate based on max(gamma_n) + 10
L = u1 - u0
window_size = 0.1 * L  # 0.1L for sliding window
num_points = 10000
u = np.linspace(u0, u1, num_points)
du = u[1] - u[0]

# Get Riemann nontrivial zeros
gamma_n = [float(mpmath.zetazero(n).imag) for n in range(1, M+1)]
gamma_n_random = np.sort(np.random.uniform(0, max(gamma_n), M))  # Random control

# Compute phase phi(u) for Riemann and random
def compute_phi(u, gamma, sigma):
    phi = np.zeros_like(u)
    for gamma_i in gamma:
        phi += np.arctan((u - gamma_i) / sigma)
    return phi

phi_riemann = compute_phi(u, gamma_n, sigma)
phi_random = compute_phi(u, gamma_n_random, sigma)

# Compute tau(u) = b * (phi(u) - <phi>)
def compute_tau(phi, u, du):
    phi_mean = np.trapz(phi, dx=du) / L
    return b * (phi - phi_mean)

tau_riemann = compute_tau(phi_riemann, u, du)
tau_random = compute_tau(phi_random, u, du)

# Compute N(u) = exp(a * tau(u)) and normalization
def compute_N(tau, u, du):
    N = np.exp(a * tau)
    Z_int = np.trapz(N, dx=du) / L
    N_tilde = N / Z_int  # Normalized to <N_tilde> = 1
    return N_tilde

N_riemann = compute_N(tau_riemann, u, du)
N_random = compute_N(tau_random, u, du)

# Sliding window analysis
def sliding_window_K(N, u, window_size, du):
    window_points = int(window_size / du)
    Kw = []
    u_centers = []
    for i in range(0, len(u) - window_points + 1, 1):  # Unit stride
        window = slice(i, i + window_points)
        u_w = u[window]
        N_w = N[window]
        mean_N = np.trapz(N_w, dx=du) / window_size
        mean_N_inv = np.trapz(1/N_w, dx=du) / window_size
        Kw.append(mean_N * mean_N_inv)
        u_centers.append(u_w[len(u_w)//2])
    # Mirror endpoints (extend symmetrically)
    left_extend = int(window_points // 2)
    Kw_left = Kw[:left_extend][::-1]
    Kw_right = Kw[-left_extend:][::-1]
    Kw = Kw_left + Kw + Kw_right
    u_centers = [u0 - (u_centers[0] - u0) + i * du for i in range(-left_extend, 0)] + u_centers + [u1 + (u1 - u_centers[-1]) + i * du for i in range(1, left_extend + 1)]
    return np.array(u_centers), np.array(Kw)

u_centers_riemann, Kw_riemann = sliding_window_K(N_riemann, u, window_size, du)
u_centers_random, Kw_random = sliding_window_K(N_random, u, window_size, du)

# Compute statistics
Kmin_riemann, Kmax_riemann = np.min(Kw_riemann), np.max(Kw_riemann)
Kmin_random, Kmax_random = np.min(Kw_random), np.max(Kw_random)
var_Kw_riemann = np.var(Kw_riemann)
var_Kw_random = np.var(Kw_random)

# Print results
print("Riemann Zeros:")
print(f"  K_min: {Kmin_riemann:.2f}, K_max: {Kmax_riemann:.2f}, Var(K_w): {var_Kw_riemann:.2e}")
print("Random Control:")
print(f"  K_min: {Kmin_random:.2f}, K_max: {Kmax_random:.2f}, Var(K_w): {var_Kw_random:.2e}")

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(u_centers_riemann, Kw_riemann, label='Riemann Zeros', color='blue')
plt.plot(u_centers_random, Kw_random, label='Random Control', color='red', linestyle='--')
plt.xlabel('u')
plt.ylabel('K_w')
plt.title('Sliding Window Structural Ratio K_w')
plt.legend()
plt.grid(True)
plt.show()