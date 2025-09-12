# ================== One-Click Colab: Abell 2744 (CATS v4) — Σ-form & κ-form (comp+DC) ==================
# 如遇 import 报错，可解除下一行安装（Colab 通常已自带）
# !pip -q install astropy scipy numpy matplotlib

import os, math, pathlib, urllib.request
import numpy as np
import numpy.fft as nft
import matplotlib.pyplot as plt
from numpy import trapezoid
from scipy.ndimage import gaussian_filter, map_coordinates

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.constants import G, c
from astropy.cosmology import Planck18 as cosmo

import pandas as pd

# -------------------- 参数（可调） --------------------
# 数据与保存目录
URL_FITS  = "https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits"
FITS_PATH = "/content/hlsp_frontier_model_abell2744_cats_v4_kappa.fits"
OUT_DIR   = "/content/out"

# 几何（Abell 2744 常用）
z_d = 0.308       # 簇红移
z_s = 2.0         # 源红移（可替换为具体像系或加权有效 redshift）

# 数值与证书参数（两口径共用）
use_smoothing   = True
sigma_pix       = 1.5     # 高斯平滑 σ（像素）
apod_frac       = 0.10    # 加窗宽度（0.04~0.12）
ring_count      = 24      # 同心环数
ring_inner_frac = 0.10    # 相对 min(nx,ny)
ring_outer_frac = 0.35
M_samples       = 4096    # 圆周采样点数

# κ-口径的环形背景厚度与盘/环间隙（补偿）
bg_annulus_frac = 0.08    # 背景环厚度（相对 min(nx,ny)；可 0.05~0.10）
gap_frac        = 0.01    # 盘与背景环之间的缝隙
win_thr         = 1e-3    # 仅在 win>阈值 的像素参与 κ 背景估计/面积积分

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------- 工具函数 --------------------
def apodize(shape, frac=0.06):
    ny, nx = shape
    win = np.ones((ny, nx), dtype=np.float64)
    ax = int(nx*frac); ay = int(ny*frac)
    if ax>0:
        wx = 0.5*(1-np.cos(np.linspace(0,np.pi,ax)))
        for i in range(ax):
            win[:, i]    *= wx[i]
            win[:, -i-1] *= wx[i]
    if ay>0:
        wy = 0.5*(1-np.cos(np.linspace(0,np.pi,ay)))
        for j in range(ay):
            win[j, :]    *= wy[j]
            win[-j-1, :] *= wy[j]
    return win

def solve_poisson_2d(Rw, pix_rad):
    """解 ∇^2 φ = Rw （2D, FFT，DC=0 规约）"""
    ny, nx = Rw.shape
    kx = nft.fftfreq(nx, d=pix_rad) * 2*np.pi
    ky = nft.fftfreq(ny, d=pix_rad) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    k2 = KX**2 + KY**2
    Rk = nft.fft2(Rw)
    Phi_k = np.zeros_like(Rk, dtype=np.complex128)
    mask = (k2 > 0)
    Phi_k[mask] = Rk[mask] / (-k2[mask])     # DC=0 规约
    Phi = nft.ifft2(Phi_k).real
    LapPhi = nft.ifft2(-k2 * nft.fft2(Phi)).real
    return Phi, Phi_k, LapPhi, (KX, KY, k2)

def ring_integral_normal(Phi_k, grids, Rpix, pix_rad, M_samples=4096):
    """外法向线积分 F_line = ∮ ∇φ·n̂ dl"""
    KX, KY, _ = grids
    ny, nx = Phi_k.shape
    cy, cx = (ny-1)/2.0, (nx-1)/2.0
    ang = np.linspace(0, 2*np.pi, M_samples, endpoint=False)
    xx = cx + Rpix*np.cos(ang)
    yy = cy + Rpix*np.sin(ang)
    nxu, nyu = np.cos(ang), np.sin(ang)      # 外法向
    dphidx = nft.ifft2((1j*KX)*Phi_k).real
    dphidy = nft.ifft2((1j*KY)*Phi_k).real
    gx = map_coordinates(dphidx, [yy, xx], order=3, mode='nearest')
    gy = map_coordinates(dphidy, [yy, xx], order=3, mode='nearest')
    grad_n = gx*nxu + gy*nyu
    return trapezoid(grad_n, ang) * (Rpix*pix_rad)

def disk_mask(ny, nx, Rpix):
    cy, cx = (ny-1)/2.0, (nx-1)/2.0
    YY, XX = np.indices((ny, nx))
    rr = np.hypot(YY - cy, XX - cx)
    return (rr <= Rpix), rr, (YY,XX), (cy,cx)

def save_summary(prefix, summary_dict, per_ring_dict, out_dir=OUT_DIR):
    summ = pd.DataFrame({"Metric": list(summary_dict.keys()), "Value": list(summary_dict.values())})
    pr   = pd.DataFrame(per_ring_dict)
    sp   = os.path.join(out_dir, f"{prefix}_summary.csv")
    prp  = os.path.join(out_dir, f"{prefix}_per_ring.csv")
    summ.to_csv(sp, index=False); pr.to_csv(prp, index=False)
    print("Saved CSV:", sp)
    print("Saved per-ring CSV:", prp)

# -------------------- 下载并读取 κ-FITS --------------------
if not pathlib.Path(FITS_PATH).exists():
    print("Downloading:", URL_FITS)
    urllib.request.urlretrieve(URL_FITS, FITS_PATH)
else:
    print("File exists:", FITS_PATH)

hdul = fits.open(FITS_PATH)
kappa = hdul[0].data.astype(np.float64)
hdr   = hdul[0].header
wcs   = WCS(hdr)
ny, nx = kappa.shape
deg2rad = np.pi/180.0

# 像素弧度尺度
if all(k in hdr for k in ("CD1_1","CD1_2","CD2_1","CD2_2")):
    CD = np.array([[hdr["CD1_1"], hdr["CD1_2"]],[hdr["CD2_1"], hdr["CD2_2"]]])
    sx = math.hypot(CD[0,0], CD[1,0])  # deg/pix
    sy = math.hypot(CD[0,1], CD[1,1])  # deg/pix
    pix_rad = math.sqrt(sx*sy) * deg2rad
elif all(k in hdr for k in ("CDELT1","CDELT2")):
    pix_rad = np.mean([abs(hdr["CDELT1"]), abs(hdr["CDELT2"])]) * deg2rad
else:
    raise RuntimeError("No CD/CDELT in FITS header.")

print(f"FITS path: {FITS_PATH}")
print(f"kappa shape: {kappa.shape}")
print(f"Pixel scale ≈ {pix_rad/deg2rad*3600:.3f} arcsec/pix")

# 加窗
win = apodize(kappa.shape, frac=apod_frac)
valid = (win > 1e-3)

# ==================== Σ-form（原始 & DC-consistent） ====================
print("\n=== Σ-form ===")
Dd  = cosmo.angular_diameter_distance(z_d).to(u.m)
Ds  = cosmo.angular_diameter_distance(z_s).to(u.m)
Dds = cosmo.angular_diameter_distance_z1z2(z_d, z_s).to(u.m)
Sigma_crit = (c**2/(4*np.pi*G) * (Ds/(Dd*Dds))).to(u.kg/(u.m*u.m)).value
print("Σcrit [kg/m^2]:", f"{Sigma_crit:.3e}")

Sigma   = kappa * Sigma_crit
Sigma_s = gaussian_filter(Sigma, sigma=sigma_pix) if use_smoothing else Sigma
k_th    = -(8*np.pi*G.value/c.value**2)

# ---- 原始（与求解使用同一 RHS = Σ_s * win） ----
Rw_sigma = k_th * Sigma_s * win
Phi_n, Phi_n_k, LapPhi_n, grids_n = solve_poisson_2d(Rw_sigma, pix_rad)

lhs_sigma = LapPhi_n[valid]; rhs_sigma = Rw_sigma[valid]
relL2_sigma = np.linalg.norm(lhs_sigma - rhs_sigma) / (np.linalg.norm(rhs_sigma) + 1e-30)
corr_sigma  = np.corrcoef(lhs_sigma, rhs_sigma)[0,1]
print(f"[Σ] Pixel closure  Rel-L2 = {relL2_sigma:.3e}   Corr = {corr_sigma:.6f}")

Rpix_min = int(min(nx,ny) * ring_inner_frac)
Rpix_max = int(min(nx,ny) * ring_outer_frac)
radii    = np.linspace(Rpix_min, Rpix_max, ring_count)

Fs_sigma, Ms_sigma, rels_sigma, Fvols_sigma = [], [], [], []
for Rpix in radii:
    F_line = ring_integral_normal(Phi_n_k, grids_n, Rpix, pix_rad, M_samples)
    inside, _, _, _ = disk_mask(ny, nx, Rpix)
    dΩ = pix_rad**2
    Marea = (Sigma_s[inside] * dΩ).sum()                   # 面积端用 Σ_s（标准）
    F_vol = (LapPhi_n[inside] * dΩ).sum()                  # 体积分核验
    rel_close = np.abs(F_line + k_th*Marea) / (np.abs(F_line)+1e-30)
    Fs_sigma.append(F_line); Ms_sigma.append(Marea); rels_sigma.append(rel_close); Fvols_sigma.append(F_vol)

Fs_sigma, Ms_sigma, rels_sigma, Fvols_sigma = map(np.array, (Fs_sigma, Ms_sigma, rels_sigma, Fvols_sigma))
rels_sigma_pct = 100*rels_sigma
q1s, q3s = np.percentile(rels_sigma_pct, [25,75])
k_ls     = -np.sum(Fs_sigma*Ms_sigma) / (np.sum(Ms_sigma**2) + 1e-30)
ratio_k  = k_ls / k_th if k_th != 0 else np.nan
rel_diffs_sigma = np.abs(Fs_sigma - Fvols_sigma) / (np.abs(Fs_sigma)+1e-30)

sigma_summary = {
    "Rel-L2": relL2_sigma, "Corr(LHS,RHS)": corr_sigma,
    "Gauss_median(%)": np.median(rels_sigma_pct), "Gauss_q1(%)": q1s, "Gauss_q3(%)": q3s, "Gauss_IQR(%)": q3s-q1s,
    "k_LS": k_ls, "k_theory": k_th, "k_ratio": ratio_k,
    "Flux_line_vs_vol_medianΔ(%)": 100*np.median(rel_diffs_sigma), "Flux_line_vs_vol_maxΔ(%)": 100*np.max(rel_diffs_sigma)
}
sigma_per_ring = {"Rpix": radii, "rel_close_%": rels_sigma_pct, "F_line": Fs_sigma, "F_vol": Fvols_sigma, "Marea": Ms_sigma}
save_summary("abell2744_sigma", sigma_summary, sigma_per_ring, OUT_DIR)

# ---- DC-consistent（面积端减去与 FFT 去除相同的 DC 常数）----
Sigma_eff   = Sigma_s * win
Rbar_sigma  = Rw_sigma.mean()            # FFT 解去掉的 DC
Sigma0_dc   = - Rbar_sigma / k_th        # 对应面密度常数

Fs_dc, Ms_dc, rels_dc = [], [], []
for Rpix in radii:
    F_line = ring_integral_normal(Phi_n_k, grids_n, Rpix, pix_rad, M_samples)
    inside, _, _, _ = disk_mask(ny, nx, Rpix)
    dΩ = pix_rad**2
    M_comp = ((Sigma_eff[inside] - Sigma0_dc) * dΩ).sum()
    rel_close = np.abs(F_line + k_th*M_comp) / (np.abs(F_line)+1e-30)
    Fs_dc.append(F_line); Ms_dc.append(M_comp); rels_dc.append(rel_close)

rels_sigma_dc_pct = 100*np.array(rels_dc)
q1sd, q3sd = np.percentile(rels_sigma_dc_pct, [25,75])
k_ls_dc    = -np.sum(np.array(Fs_dc)*np.array(Ms_dc)) / (np.sum(np.array(Ms_dc)**2)+1e-30)
ratio_k_dc = k_ls_dc / k_th

sigma_dc_summary = {
    "Gauss_median(%)": np.median(rels_sigma_dc_pct), "Gauss_q1(%)": q1sd, "Gauss_q3(%)": q3sd, "Gauss_IQR(%)": q3sd-q1sd,
    "k_LS(DC)": k_ls_dc, "k_theory": k_th, "k_ratio(DC)": ratio_k_dc
}
sigma_dc_ring = {"Rpix": radii, "rel_close_DC_%": rels_sigma_dc_pct, "F_line": Fs_dc, "Marea_comp(DC)": Ms_dc}
save_summary("abell2744_sigma_DC", sigma_dc_summary, sigma_dc_ring, OUT_DIR)

# 图（Σ）
fig, axs = plt.subplots(1, 3, figsize=(15,5))
im0 = axs[0].imshow(kappa, origin='lower'); axs[0].set_title("κ (convergence)")
im1 = axs[1].imshow(Phi_n, origin='lower'); axs[1].set_title("Φ_n (Σ-form)")
im2 = axs[2].imshow((LapPhi_n - Rw_sigma), origin='lower'); axs[2].set_title("Residual: ∇²Φ_n - Rw")
plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.02)
plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.02)
plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.02)
plt.tight_layout()
png_sigma = os.path.join(OUT_DIR, "abell2744_sigma.png")
plt.savefig(png_sigma, dpi=150); plt.show()
print("Saved Figure:", png_sigma)

# ==================== κ-form（补偿 + 窗一致 & DC-consistent） ====================
print("\n=== κ-form (compensated, window-consistent) ===")
kappa_s   = gaussian_filter(kappa, sigma=sigma_pix) if use_smoothing else kappa
kappa_eff = kappa_s * win        # 与求解/面积/背景估计保持同一口径
R_kappa_w = 2.0 * kappa_eff

# 解 ∇^2 ψ = R_kappa_w
Psi, Psi_k, LapPsi, grids_k = solve_poisson_2d(R_kappa_w, pix_rad)

# 像素级闭合（与 R_kappa_w 一致）
lhs_k = LapPsi[valid]; rhs_k = R_kappa_w[valid]
relL2_kappa = np.linalg.norm(lhs_k - rhs_k) / (np.linalg.norm(rhs_k) + 1e-30)
corr_kappa  = np.corrcoef(lhs_k, rhs_k)[0,1]
print(f"[κ*win] Pixel closure  Rel-L2 = {relL2_kappa:.3e}   Corr = {corr_kappa:.6f}")

# 环参数
Rpix_min = int(min(nx,ny) * ring_inner_frac)
Rpix_max = int(min(nx,ny) * ring_outer_frac)
radii    = np.linspace(Rpix_min, Rpix_max, ring_count)
dR_bg    = int(min(nx,ny) * bg_annulus_frac)
gap_px   = max(2, int(min(nx,ny) * gap_frac))

# ---- 背景补偿（mass-sheet 局部补偿） ----
Fs_k, Ms_comp, Ms_raw, rels_k, Fvols_k = [], [], [], [], []
for Rpix in radii:
    F_line = ring_integral_normal(Psi_k, grids_k, Rpix, pix_rad, M_samples)

    inside, rr, (YY,XX), _ = disk_mask(ny, nx, Rpix)
    annulus = (rr > (Rpix + gap_px)) & (rr <= min(Rpix + gap_px + dR_bg, int(min(nx,ny)*ring_outer_frac + dR_bg)))

    inside  = inside  & (win > win_thr)
    annulus = annulus & (win > win_thr)

    w_ann = win[annulus]; k_ann = kappa_eff[annulus]
    k_bg  = (np.sum(k_ann*w_ann) / np.sum(w_ann)) if np.sum(w_ann)>0 else (np.mean(k_ann) if k_ann.size>0 else 0.0)

    dΩ = pix_rad**2
    M_raw  = (kappa_eff[inside] * dΩ).sum()
    M_comp = ((kappa_eff[inside] - k_bg) * dΩ).sum()

    F_vol  = (LapPsi[inside] * dΩ).sum()
    rel_close = np.abs(F_line - 2.0*M_comp) / (np.abs(F_line)+1e-30)

    Fs_k.append(F_line); Ms_comp.append(M_comp); Ms_raw.append(M_raw)
    rels_k.append(rel_close); Fvols_k.append(F_vol)

Fs_k, Ms_comp, Ms_raw, rels_k, Fvols_k = map(np.array, (Fs_k, Ms_comp, Ms_raw, rels_k, Fvols_k))
rels_k_pct = 100*rels_k
q1k, q3k = np.percentile(rels_k_pct, [25,75])
a_hat  = np.sum(Fs_k*Ms_comp) / (np.sum(Ms_comp**2) + 1e-30)
ratio2 = a_hat/2.0
rel_diffs_k = np.abs(Fs_k - Fvols_k) / (np.abs(Fs_k)+1e-30)

kappa_summary = {
    "Rel-L2": relL2_kappa, "Corr(LHS,RHS)": corr_kappa,
    "Gauss_median(comp)(%)": np.median(rels_k_pct), "Gauss_q1(comp)(%)": q1k, "Gauss_q3(comp)(%)": q3k, "Gauss_IQR(comp)(%)": q3k-q1k,
    "slope_a_hat": a_hat, "a_theory": 2.0, "a_ratio": ratio2,
    "Flux_line_vs_vol_medianΔ(%)": 100*np.median(rel_diffs_k), "Flux_line_vs_vol_maxΔ(%)": 100*np.max(rel_diffs_k)
}
kappa_per_ring = {"Rpix": radii, "rel_close_comp_%": rels_k_pct, "F_line": Fs_k, "F_vol": Fvols_k, "Marea_comp": Ms_comp, "Marea_raw": Ms_raw}
save_summary("abell2744_kappa_comp", kappa_summary, kappa_per_ring, OUT_DIR)

# ---- DC-consistent（面积端减去与 FFT 去除相同的 DC 常数）----
Rbar_k   = R_kappa_w.mean()         # 被 FFT 去掉的 DC
kappa0_dc= Rbar_k / 2.0             # 等效 κ 常数

Fs_dc, Ms_dc, rels_dc = [], [], []
for Rpix in radii:
    F_line = ring_integral_normal(Psi_k, grids_k, Rpix, pix_rad, M_samples)
    inside, _, _, _ = disk_mask(ny, nx, Rpix)
    inside = inside & (win > win_thr)
    dΩ = pix_rad**2
    M_comp_dc = ((kappa_eff[inside] - kappa0_dc) * dΩ).sum()
    rel_close = np.abs(F_line - 2.0*M_comp_dc) / (np.abs(F_line)+1e-30)
    Fs_dc.append(F_line); Ms_dc.append(M_comp_dc); rels_dc.append(rel_close)

rels_k_dc_pct = 100*np.array(rels_dc)
q1kd, q3kd = np.percentile(rels_k_dc_pct, [25,75])
a_hat_dc   = np.sum(np.array(Fs_dc)*np.array(Ms_dc)) / (np.sum(np.array(Ms_dc)**2) + 1e-30)
ratio2_dc  = a_hat_dc/2.0

kappa_dc_summary = {
    "Gauss_median(DC)(%)": np.median(rels_k_dc_pct), "Gauss_q1(DC)(%)": q1kd, "Gauss_q3(DC)(%)": q3kd, "Gauss_IQR(DC)(%)": q3kd-q1kd,
    "slope_a_hat(DC)": a_hat_dc, "a_theory": 2.0, "a_ratio(DC)": ratio2_dc
}
kappa_dc_ring = {"Rpix": radii, "rel_close_DC_%": rels_k_dc_pct, "F_line": Fs_dc, "Marea_comp_DC": Ms_dc}
save_summary("abell2744_kappa_DC", kappa_dc_summary, kappa_dc_ring, OUT_DIR)

# 图（κ）
fig, axs = plt.subplots(1, 3, figsize=(15,5))
im0 = axs[0].imshow(kappa, origin='lower'); axs[0].set_title("κ (convergence)")
im1 = axs[1].imshow(Psi, origin='lower');   axs[1].set_title("ψ (κ-form)")
im2 = axs[2].imshow((LapPsi - R_kappa_w), origin='lower'); axs[2].set_title("Residual: ∇²ψ - (2κ_eff)")
plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.02)
plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.02)
plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.02)
plt.tight_layout()
png_kappa = os.path.join(OUT_DIR, "abell2744_kappa_comp.png")
plt.savefig(png_kappa, dpi=150); plt.show()
print("Saved Figure:", png_kappa)

print("\nDone. Outputs in:", OUT_DIR)
# ==================================================================================================

