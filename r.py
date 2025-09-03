# @title True-zero transverse GRIN: ray–curvature closure (copy-paste ready)
# -*- coding: utf-8 -*-
# Colab 单细胞：真实零点 -> 平滑 ln N(x) -> 2D 射线追踪 -> 曲率测量 vs 预测
# 验证公式：kappa = ||(I - t t^T) ∇ ln N|| = (2/c0^2) ||(I - t t^T) g||，g=(c0^2/2) ∇ ln N

import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

# ============== 配置 ==============
CFG = {
    "RNG_SEED": 123,
    "M_ZEROS": 120,          # 零点个数
    "ALPHA": 1.6,            # σ_u = ALPHA * mean_gap
    "LN_CAP": 0.02,          # 限幅：max |ln N|
    "J_U2X": 1.0,            # x = J * u （单位换算，m/u）
    "MARGIN_SIGMA": 2.0,     # u 边界外扩倍数（×σ_u）
    "NX": 4000,              # x 方向采样点数
    "Z_MAX": 300.0,          # 传播 z 上限（m）
    "DS": 0.02,              # 弧长步长（m）
    "S_MAX": 600.0,          # 最长积分弧长（m）
    "RAY_X0_LIST": [-5.0, 0.0, 20.0],     # 射线初始 x（m）
    "RAY_THETA0_DEG": [5.0, 10.0, 20.0],  # 射线初始仰角（deg），0°=沿 +z
    "AUTO_CLAMP": True,      # 起点自动夹紧到域内（留 0.5% 边距）
    "REL_EPS": 1e-12,        # 相对误差防零
}

# 物理常数
c0 = 299_792_458.0

# ========= 工具 =========
def get_riemann_zeros(M):
    # 使用 mpmath.zetazero(n) 取前 M 个临界线零点的虚部 γ_n
    gam = [float(mp.im(mp.zetazero(n))) for n in range(1, M+1)]
    gam = np.array(gam, dtype=float)
    gam.sort()
    return gam

def build_lnN_from_zeros(gamma, alpha, ln_cap, J=1.0, nx=4000, margin_sigma=2.0):
    # u 轴窗口： [γ1 - margin*σ, γM + margin*σ]
    gaps = np.diff(gamma)
    mean_gap = gaps.mean()
    sigma_u = alpha * mean_gap

    u_min = gamma[0] - margin_sigma * sigma_u
    u_max = gamma[-1] + margin_sigma * sigma_u

    # 物理坐标 x = J * (u - gamma[0])，令左侧出现少量负数方便测试
    u_grid = np.linspace(u_min, u_max, nx)
    x_grid = J * (u_grid - gamma[0])

    # 计算平滑相位 φσ(u) = sum arctan((u-γ)/σ)
    # 向量化：对每个 u，累加所有零点贡献
    U = u_grid[:, None]
    G = gamma[None, :]
    phi = np.arctan((U - G) / sigma_u).sum(axis=1)

    # 去均值；按最大绝对值缩放到 ln_cap
    phi0 = phi - phi.mean()
    beta_scale = ln_cap / (np.max(np.abs(phi0)) + 1e-30)
    lnN = beta_scale * phi0

    return x_grid, lnN, mean_gap, sigma_u, beta_scale

def finite_diff_1d(f, dx):
    # 二阶中心差分（边界采用一阶）
    df = np.empty_like(f)
    df[1:-1] = (f[2:] - f[:-2]) / (2.0 * dx)
    df[0] = (f[1] - f[0]) / dx
    df[-1] = (f[-1] - f[-2]) / dx
    return df

def interp1(x, y, xq):
    # 简单线性插值（域外返回 nan）
    yq = np.interp(xq, x, y, left=np.nan, right=np.nan)
    return yq

def trace_ray(x0, theta0_deg, dlnNdx_func, x_min, x_max, z_max, ds=0.02, s_max=600.0):
    # 射线追踪：r=(x,z), t=(tx,tz)，|t|=1
    # 演化律： dt/ds = (I - t t^T) ∇ ln N，且 ∇ ln N = (dlnNdx, 0)
    th = np.deg2rad(theta0_deg)
    r = np.array([x0, 0.0], dtype=float)
    t = np.array([np.sin(th), np.cos(th)], dtype=float)
    t /= np.linalg.norm(t)

    X, Z = [r[0]], [r[1]]
    k_meas = []
    k_pred = []

    s = 0.0
    while s < s_max and 0.0 <= r[1] <= z_max and (x_min <= r[0] <= x_max):
        grad_lnN = np.array([dlnNdx_func(r[0]), 0.0], dtype=float)
        if not np.isfinite(grad_lnN[0]):
            break

        P = np.eye(2) - np.outer(t, t)           # 垂直投影
        k_vec = P @ grad_lnN                     # 预测曲率向量
        k_pred.append(np.linalg.norm(k_vec))

        t_new = t + k_vec * ds
        nrm = np.linalg.norm(t_new)
        if nrm < 1e-14:
            break
        t_new /= nrm

        r_new = r + t_new * ds

        # 测量曲率（数值）：||Δt|| / ds
        k_meas.append(np.linalg.norm(t_new - t) / ds)

        # 推进一步
        r, t = r_new, t_new
        s += ds
        X.append(r[0]); Z.append(r[1])

        # 轻微防护：过界就停
        if r[0] < x_min or r[0] > x_max or r[1] < 0.0 or r[1] > z_max:
            break

    return {
        "x": np.array(X), "z": np.array(Z),
        "k_meas": np.array(k_meas, dtype=float),
        "k_pred": np.array(k_pred, dtype=float)
    }

# ============== 主流程 ==============
np.random.seed(CFG["RNG_SEED"])

# 1) 真实零点
gamma = get_riemann_zeros(CFG["M_ZEROS"])
mean_gap = float(np.mean(np.diff(gamma)))
sigma_u = CFG["ALPHA"] * mean_gap

# 2) 构造 ln N(x)
x, lnN, mean_gap, sigma_u, beta = build_lnN_from_zeros(
    gamma, CFG["ALPHA"], CFG["LN_CAP"], CFG["J_U2X"], CFG["NX"], CFG["MARGIN_SIGMA"]
)
dx = float(np.mean(np.diff(x)))
x_min, x_max = x[0], x[-1]

# 3) 梯度、等效引力 & 两种预测的一致性
dlnNdx = finite_diff_1d(lnN, dx)

def dlnNdx_func(xq): return interp1(x, dlnNdx, xq)

# 用 ∇lnN 与 用 g = (c0^2/2)∇lnN 的预测一致性（应当机器精度相同）
g_x = 0.5 * c0**2 * dlnNdx
k_pred_grad = np.abs(dlnNdx)                      # 初始 t=(0,1) 情况下 P_perp ∇lnN 的范数就是 |dlnNdx|
k_pred_g    = (2.0/c0**2) * np.abs(g_x)          # 完全相同
mean_kappa_delta = np.nanmean(np.abs(k_pred_grad - k_pred_g))

# 4) 射线追踪（2D，x-横向，z-传播）
rays = []
x0_list = CFG["RAY_X0_LIST"]
th_list = CFG["RAY_THETA0_DEG"]

# 自动夹紧起点到域内（留 0.5% 边距）
if CFG["AUTO_CLAMP"]:
    span = x_max - x_min
    x0_list = [min(max(xx, x_min + 0.005*span), x_max - 0.005*span) for xx in x0_list]

for x0, th in zip(x0_list, th_list):
    R = trace_ray(
        x0, th, dlnNdx_func,
        x_min, x_max,
        CFG["Z_MAX"],
        ds=CFG["DS"], s_max=CFG["S_MAX"]
    )
    rays.append(R)

# 5) 统计误差（聚合所有射线；忽略空样本）
all_meas = np.concatenate([R["k_meas"] for R in rays if R["k_meas"].size > 0]) if any(R["k_meas"].size>0 for R in rays) else np.array([])
all_pred = np.concatenate([R["k_pred"] for R in rays if R["k_pred"].size > 0]) if any(R["k_pred"].size>0 for R in rays) else np.array([])

if all_meas.size > 0:
    mean_abs_err = float(np.nanmean(np.abs(all_meas - all_pred)))
    median_abs_err = float(np.nanmedian(np.abs(all_meas - all_pred)))
    mean_rel_err = float(np.nanmean(np.abs(all_meas - all_pred) / (np.abs(all_pred) + CFG["REL_EPS"])))
else:
    mean_abs_err = median_abs_err = mean_rel_err = np.nan

# 6) 打印汇总
print(f"Computed {CFG['M_ZEROS']} zeros. last γ={gamma[-1]:.6f},  mean gap={mean_gap:.6f},  σ_u={sigma_u:.6f}\n")
print("=== True-zero curvature test (2D transverse GRIN) ===")
print(f"M_zeros={CFG['M_ZEROS']},  α={CFG['ALPHA']:.2f},  lnN_cap={CFG['LN_CAP']}")
print(f"x-range: [{x_min:.2f}, {x_max:.2f}] m,  J={CFG['J_U2X']:.3f} m/u,  dx={dx:.4f} m")
print(f"beta (scale) = {beta:.9e},  max|lnN|={np.max(np.abs(lnN)):.3e},  std(lnN)={np.std(lnN):.3e}")
print(f"mean |κ_meas-κ_pred|      : {mean_abs_err:.9e} m^-1")
print(f"median |κ_meas-κ_pred|    : {median_abs_err:.9e} m^-1")
print(f"mean relative error       : {mean_rel_err:.3e}")
print(f"κ_pred via ∇lnN vs via g  : mean |Δ| = {mean_kappa_delta:.3e} m^-1")

print("Summary]")
for i, (x0, th) in enumerate(zip(CFG["RAY_X0_LIST"], CFG["RAY_THETA0_DEG"])):
    R = rays[i]
    if R["k_meas"].size == 0:
        print(f"ray#{i}: x0={x0:6.1f} m, θ0={th:5.1f}° | (no samples inside window; skipped)")
        continue
    m = float(np.nanmean(np.abs(R["k_meas"] - R["k_pred"])))
    r = float(np.nanmean(np.abs(R["k_meas"] - R["k_pred"]) / (np.abs(R["k_pred"]) + CFG["REL_EPS"])))
    print(f"ray#{i}: x0={x0:6.1f} m, θ0={th:5.1f}° | mean|Δκ|={m:.3e} m^-1, mean rel.err={r:.3e}")

# 7) 可视化
plt.figure(figsize=(8,3))
plt.plot(x, lnN, lw=1.5)
plt.title("ln N(x) from true Riemann zeros (smoothed)")
plt.xlabel("x (m)")
plt.ylabel("ln N")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 背景显示 lnN(x)（沿 z 复制） + 射线路径
Nz_bg = 300
bg = np.tile(lnN[np.newaxis, :], (Nz_bg, 1))
extent = [x_min, x_max, 0.0, CFG["Z_MAX"]]

plt.figure(figsize=(8,5))
plt.imshow(bg, aspect="auto", extent=extent, origin="lower")
for R in rays:
    if R["x"].size > 1:
        plt.plot(R["x"], R["z"], lw=2)
plt.title("Ray trajectories in transverse GRIN (color = ln N(x))")
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.colorbar(label="ln N")
plt.tight_layout()
plt.show()

# 选一条射线画 κ(s) 的测量 vs 预测（取第二条，如存在）
pick = 1 if len(rays) > 1 else 0
R = rays[pick]
if R["k_meas"].size > 5:
    s_axis = np.arange(R["k_meas"].size) * CFG["DS"]
    plt.figure(figsize=(8,3))
    plt.plot(s_axis, R["k_meas"], label="measured κ(s)", lw=1.8)
    plt.plot(s_axis, R["k_pred"], label="predicted κ(s) = ||(I-tt^T)∇lnN||", lw=1.2, ls="--")
    plt.title(f"Curvature along ray #{pick}  (x0={CFG['RAY_X0_LIST'][pick]:.1f} m, θ0={CFG['RAY_THETA0_DEG'][pick]:.1f}°)")
    plt.xlabel("arc length s (m)")
    plt.ylabel("κ (1/m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print(f"[info] ray #{pick} has too few samples to plot κ(s).")

print("\n[Done] Ray–curvature closure finished. 图已输出。")
