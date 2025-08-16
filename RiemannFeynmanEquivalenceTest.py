import numpy as np
import matplotlib.pyplot as plt

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

# z ∈ [-1,1]，把 z 映射到 u 区间（覆盖 55~65，包含多个零点）
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
B = np.array([ 1.0,  1.0])

def path_lengths(x):
    P = np.array([x, 0.0])
    L1 = np.linalg.norm(P - A)
    L2 = np.linalg.norm(B - P)
    return L1, L2

def optical_action(x):
    L1, L2 = path_lengths(x)
    return n1 * L1 + n2 * L2

def optical_phase(x, lambda0):
    k0 = 2.0 * np.pi / lambda0
    return k0 * optical_action(x)

xs = np.linspace(-0.9, 0.9, 2001)
Axs = np.array([optical_action(x) for x in xs])

lam_small = 0.01
lam_large = 0.10
Phi_small = np.array([optical_phase(x, lam_small) for x in xs])
Phi_large = np.array([optical_phase(x, lam_large) for x in xs])

k0_small = 2.0 * np.pi / lam_small
k0_large = 2.0 * np.pi / lam_large
A_from_small = Phi_small / k0_small
A_from_large = Phi_large / k0_large

ix_min_A = int(np.argmin(Axs))
x_min_A = xs[ix_min_A]

ix_min_small = int(np.argmin(A_from_small))
ix_min_large = int(np.argmin(A_from_large))
x_min_small = xs[ix_min_small]
x_min_large = xs[ix_min_large]

corr_small = np.corrcoef(Axs, A_from_small)[0,1]
corr_large = np.corrcoef(Axs, A_from_large)[0,1]
max_abs_err_small = float(np.max(np.abs(Axs - A_from_small)))
max_abs_err_large = float(np.max(np.abs(Axs - A_from_large)))

print("=== Riemann-induced N(z) summary ===")
print(f"n1 (z<0) mean = {n1:.6f},  n2 (z>=0) mean = {n2:.6f}")
print()
print("=== Equivalence check: A(x) vs Φ(x)/k0 ===")
print(f"corr(A, Φ/k0)  @ λ={lam_small:.3f} : {corr_small:.12f},  max|Δ|={max_abs_err_small:.3e}")
print(f"corr(A, Φ/k0)  @ λ={lam_large:.3f} : {corr_large:.12f},  max|Δ|={max_abs_err_large:.3e}")
print()
print("=== Minimizer (stationary point) ===")
print(f"x* from A(x)           : {x_min_A:.6f}")
print(f"x* from Φ(x)/k0 (0.01) : {x_min_small:.6f}")
print(f"x* from Φ(x)/k0 (0.10) : {x_min_large:.6f}")

# ---------- 4) 作图 ----------
plt.figure(figsize=(8,3))
plt.plot(z, N)
plt.xlabel("z"); plt.ylabel("N(z) from Riemann phase lattice")
plt.title("Riemann-induced index N(z) across the interface (z=0)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,3))
plt.plot(xs, Axs)
plt.axvline(x_min_A, linestyle="--")
plt.xlabel("x (crossing on z=0)"); plt.ylabel("Optical action  A(x)=n1 L1 + n2 L2")
plt.title("Action landscape and stationary crossing point")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,3))
plt.plot(xs, Phi_small)
plt.xlabel("x (crossing on z=0)"); plt.ylabel("Optical phase Φ(x)  [λ=0.01]")
plt.title("Optical phase vs crossing (semiclassical: rapid oscillations)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,3))
plt.plot(xs, Phi_large)
plt.xlabel("x (crossing on z=0)"); plt.ylabel("Optical phase Φ(x)  [λ=0.10]")
plt.title("Optical phase vs crossing (longer wavelength)")
plt.tight_layout()
plt.show()
