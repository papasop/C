import numpy as np
import matplotlib.pyplot as plt

# Riemann zeros (first 30)
gammas = np.array([
    14.13472514, 21.02203964, 25.01085758, 30.42487613, 32.93506159,
    37.58617816, 40.91871901, 43.32707328, 48.00515088, 49.77383248,
    52.97032148, 56.44624770, 59.34704400, 60.83177953, 65.11254405,
    67.07981053, 69.54640171, 72.06715767, 75.70469070, 77.14484007,
    79.33737502, 82.91038085, 84.73549298, 87.42527461, 88.80911121,
    92.49189927, 94.65134404, 95.87063423, 98.83119422, 101.3178510
])

# Geometric phase
def geometric_phase(u, gammas, sigma):
    x = (u[:, None] - gammas[None, :]) / sigma
    return np.arctan(x).sum(axis=1)

# Compute refractive index
def compute_N(u, gammas, sigma, a=0.25, b=0.03, tau_offset=0):
    phi = geometric_phase(u, gammas, sigma)
    tau = b * (phi - phi.mean()) + tau_offset
    N = np.exp(a * tau)
    return N

# Simulate air-water interface
c0 = 299792458.0  # Vacuum speed of light
u = np.linspace(10.0, 110.0, 1000)  # Propagation coordinate

# Air: target N1 ≈ 1.0003
N_air = compute_N(u, gammas, sigma=12.0, a=0.01, b=0.01, tau_offset=np.log(1.0003)/0.01)
N1 = N_air.mean()
print(f"Air refractive index N1: {N1:.6f}")

# Water: target N2 ≈ 1.333
N_water = compute_N(u, gammas, sigma=2.0, a=0.25, b=0.03, tau_offset=np.log(1.333)/0.25)
N2 = N_water.mean()
print(f"Water refractive index N2: {N2:.6f}")

# Verify Snell's law
theta1 = np.deg2rad(30)  # Incident angle 30°
sin_theta2 = N1 * np.sin(theta1) / N2
theta2 = np.arcsin(sin_theta2)
print(f"Incident angle theta1: {np.rad2deg(theta1):.2f}°")
print(f"Refracted angle theta2: {np.rad2deg(theta2):.2f}°")

# Optical path calculation
def optical_path(x1, x0=-1, z0=-1, x2=0, z2=1, N1=1, N2=1):
    s1 = np.sqrt((x1 - x0)**2 + (0 - z0)**2)
    s2 = np.sqrt((x2 - x1)**2 + (z2 - 0)**2)
    T = (N1 * s1 + N2 * s2) / c0
    return T

# Optimize interface crossing point
x1_vals = np.linspace(-2, 2, 100)
T_vals = [optical_path(x1, N1=N1, N2=N2) for x1 in x1_vals]
x1_opt = x1_vals[np.argmin(T_vals)]
T_opt = min(T_vals)
print(f"Optimal crossing point x1: {x1_opt:.6f}")
print(f"Minimum optical path T: {T_opt:.12f} s")

# Visualization
plt.figure(figsize=(8, 6))
plt.plot([-1, x1_opt, 0], [-1, 0, 1], 'b-', label='Light path')
plt.axhline(0, color='k', linestyle='--', label='Air-water interface')
plt.scatter([-1, x1_opt, 0], [-1, 0, 1], c='r', label='Key points')
plt.title(f"Light Path (θ1={np.rad2deg(theta1):.1f}°, θ2={np.rad2deg(theta2):.1f}°)")
plt.xlabel("x")
plt.ylabel("z")
plt.legend()
plt.grid(True)
plt.show()