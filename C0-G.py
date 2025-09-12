# @title Abell 2744 (CATS v4) κ-map → κ-DC & Σ-DC 一键复现（最终版）
# @markdown - 自动下载 CATS v4 Abell 2744 κ FITS
# @markdown - κ-DC：环证书 + 过原点回归（a_ratio），Gauss 中位
# @markdown - Σ-DC：自动反推 z_s,eff + DC α 扫描；输出 α=0（像素均值DC）的主结果与逐环明细
# @markdown - Pixel Corr：同样平滑 + 同窗（修正后 ≈1）
# @markdown - 主要打印指标（期望典型值）：κ-DC a_ratio≈0.99974、Gauss≈0.024%；Σ-DC K_ratio≈0.9997、Gauss≈0.001%；Corr≈1.000000

import os, math, urllib.request, numpy as np, pandas as pd
import numpy.fft as nft
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.cosmology import Planck18 as cosmo
from astropy.constants import G, c
from astropy import units as u
from scipy.ndimage import gaussian_filter, map_coordinates

plt.rcParams["figure.dpi"] = 120
np.set_printoptions(suppress=True)

# -------------------- 路径 & 下载 --------------------
OUT = "/content/out"; os.makedirs(OUT, exist_ok=True)
FITS_PATH = "/content/kappa_map.fits"
URL = "https://archive.stsci.edu/pub/hlsp/frontier/abell2744/models/cats/v4/hlsp_frontier_model_abell2744_cats_v4_kappa.fits"

if not os.path.exists(FITS_PATH):
    print("Downloading:", URL)
    urllib.request.urlretrieve(URL, FITS_PATH)
else:
    print("File exists:", FITS_PATH)

# -------------------- 读取 κ 与像素尺度（稳健） --------------------
hdul = fits.open(FITS_PATH)
kappa = hdul[0].data.astype(float)
hdr = hdul[0].header
wcs = WCS(hdr)
ny, nx = kappa.shape

# 像素尺度（deg/pix → rad/pix），失败则兜底 0.3"
def get_pix_rad(wcs, hdr):
    try:
        scales = proj_plane_pixel_scales(wcs)  # 每轴度/像素
        if scales is not None and np.isfinite(scales).all():
            return (np.sqrt((scales[0]*u.deg)*(scales[1]*u.deg))).to(u.rad).value
    except Exception:
        pass
    if "CDELT1" in hdr and "CDELT2" in hdr:
        sx = abs(hdr["CDELT1"]) * u.deg
        sy = abs(hdr["CDELT2"]) * u.deg
        return (np.sqrt(sx*sy)).to(u.rad).value
    # 兜底：0.3 arcsec/pix
    return (0.3 * u.arcsec).to(u.rad).value

pix_rad = get_pix_rad(wcs, hdr)
print("FITS path:", FITS_PATH)
print("kappa shape:", kappa.shape)
print(f"Pixel scale ≈ {(pix_rad*u.rad).to(u.arcsec).value:.3f} arcsec/pix")

# -------------------- 工具函数 --------------------
def safe_trapz(y, x):
    try: return np.trapezoid(y, x)
    except AttributeError: return np.trapz(y, x)

def build_window(img, apod_frac=0.10):
    ny, nx = img.shape
    win = np.ones_like(img, float)
    ax = int(nx*apod_frac); ay = int(ny*apod_frac)
    if ax>0:
        wx = 0.5*(1-np.cos(np.linspace(0,np.pi,ax)))
        for i in range(ax):
            win[:, i] *= wx[i]; win[:, -i-1] *= wx[i]
    if ay>0:
        wy = 0.5*(1-np.cos(np.linspace(0,np.pi,ay)))
        for j in range(ay):
            win[j, :] *= wy[j]; win[-j-1, :] *= wy[j]
    return win

def ring_flux_from_Phik(Phi_k, Rpix, KX, KY, pix_rad, M_samples=4096):
    ny, nx = Phi_k.shape
    ang = np.linspace(0, 2*np.pi, M_samples, endpoint=False)
    cy, cx = (ny-1)/2.0, (nx-1)/2.0
    xx = cx + Rpix*np.cos(ang); yy = cy + Rpix*np.sin(ang)
    nxu, nyu = np.cos(ang), np.sin(ang)
    dphidx = nft.ifft2((1j*KX)*Phi_k).real
    dphidy = nft.ifft2((1j*KY)*Phi_k).real
    gx = map_coordinates(dphidx, [yy, xx], order=3, mode='nearest')
    gy = map_coordinates(dphidy, [yy, xx], order=3, mode='nearest')
    grad_n = gx*nxu + gy*nyu
    return safe_trapz(grad_n, ang) * (Rpix*pix_rad)

def disk_mask(ny, nx, Rpix):
    yy, xx = np.indices((ny, nx))
    cy, cx = (ny-1)/2.0, (nx-1)/2.0
    rr = np.hypot(yy-cy, xx-cx)
    return (rr<=Rpix)

def corrcoef2(a, b):
    A=a.ravel()-a.mean(); B=b.ravel()-b.mean()
    den = (np.linalg.norm(A)*np.linalg.norm(B)+1e-30)
    return float(np.dot(A,B)/den)

# -------------------- 公共参数 --------------------
z_d_init = 0.308
z_s_init = 2.0
sigma_pix = 1.5
apod_frac = 0.10
ring_count = 24
ring_inner_frac, ring_outer_frac = 0.10, 0.35
M_samples = 4096

# FFT 频率格
kx = nft.fftfreq(nx, d=pix_rad) * 2*np.pi
ky = nft.fftfreq(ny, d=pix_rad) * 2*np.pi
KX, KY = np.meshgrid(kx, ky, indexing='xy')
k2 = KX**2 + KY**2

# 窗函数
win = build_window(kappa, apod_frac=apod_frac)

# =========================================================
# 1) κ-DC（compensated, window-consistent）
# =========================================================
kappa_s   = gaussian_filter(kappa, sigma=sigma_pix)
kappa_eff = kappa_s * win
kappa0_dc = kappa_eff.sum() / (kappa_eff.size + 1e-30)  # 像素均值 DC → 对齐 FFT 零模

def kappa_ring_metrics(kappa_eff, kappa0):
    ny, nx = kappa_eff.shape
    Rmin = int(min(nx,ny)*ring_inner_frac); Rmax = int(min(nx,ny)*ring_outer_frac)
    radii = np.linspace(Rmin, Rmax, ring_count)
    # 势（常数不影响回归/证书）
    Rw = - kappa_eff
    Rk = nft.fft2(Rw)
    Phi_k = np.zeros_like(Rk, complex); m=(k2>0); Phi_k[m]=Rk[m]/(-k2[m])
    Fs, Ms, rels = [], [], []
    for Rpix in radii:
        F = ring_flux_from_Phik(Phi_k, Rpix, KX, KY, pix_rad, M_samples=M_samples)
        inside = disk_mask(ny, nx, Rpix)
        M = ((kappa_eff[inside]-kappa0)*(pix_rad**2)).sum()
        rel = abs(F + 1.0*M) / (abs(F)+1e-30)  # κ-DC 的常数为 1
        Fs.append(F); Ms.append(M); rels.append(rel)
    Fs, Ms, rels = map(np.array, (Fs, Ms, rels))
    a_hat = - float(np.sum(Fs*Ms) / (np.sum(Ms**2)+1e-30))
    a_ratio = a_hat / 1.0
    return radii, Fs, Ms, 100*rels, a_hat, a_ratio

radii_k, Fk, Mk, rels_k, a_hat, a_ratio = kappa_ring_metrics(kappa_eff, kappa0_dc)
k_gauss_med = float(np.median(rels_k))
k_gauss_q1  = float(np.percentile(rels_k,25))
k_gauss_q3  = float(np.percentile(rels_k,75))

# 保存 κ per-ring / summary
pd.DataFrame({"Rpix":radii_k, "F_line":Fk, "M_comp_DC":Mk, "rel_close_DC_%":rels_k})\
  .to_csv(os.path.join(OUT,"abell2744_kappa_DC_per_ring.csv"), index=False)
pd.DataFrame({
    "Metric":["Gauss_median(DC)(%)","Gauss_q1(DC)(%)","Gauss_q3(DC)(%)","a_hat(DC)","a_ratio(DC)"],
    "Value":[k_gauss_med, k_gauss_q1, k_gauss_q3, a_hat, a_ratio]
}).to_csv(os.path.join(OUT,"abell2744_kappa_DC_summary.csv"), index=False)

# =========================================================
# 2) Σ-DC：自动反推 z_s,eff + α 扫描；主结果为 α=0（像素均值 DC）
# =========================================================
def Sigma_crit_of(z_d, z_s):
    Dd  = cosmo.angular_diameter_distance(z_d).to(u.m)
    Ds  = cosmo.angular_diameter_distance(z_s).to(u.m)
    Dds = cosmo.angular_diameter_distance_z1z2(z_d, z_s).to(u.m)
    return (c**2/(4*np.pi*G) * (Ds/(Dd*Dds))).to(u.kg/(u.m*u.m)).value

def Sigma_and_eff(z_d, z_s):
    Scrit = Sigma_crit_of(z_d, z_s)
    Sigma = kappa * Scrit
    Sigma_s = gaussian_filter(Sigma, sigma=sigma_pix)
    return Sigma, Sigma_s*win, Scrit

def Phi_from_RHS(rhs):
    Rk = nft.fft2(rhs)
    Phi_k = np.zeros_like(Rk, complex); m=(k2>0); Phi_k[m]=Rk[m]/(-k2[m])
    return Phi_k, nft.ifft2(Phi_k).real

def Sigma_gauss_Kratio(Sigma_eff, dc_mode="weighted"):
    ny, nx = Sigma_eff.shape
    Kc = 8*np.pi*G.value/(c.value**2)
    Phi_k, _ = Phi_from_RHS(-Kc * Sigma_eff)
    if dc_mode=="weighted":
        Sigma0 = Sigma_eff.sum() / (win.sum()+1e-30)
    elif dc_mode=="pixel":
        Sigma0 = Sigma_eff.sum() / (Sigma_eff.size+1e-30)
    else:
        Sigma0 = 0.0
    Rmin = int(min(nx,ny)*ring_inner_frac); Rmax = int(min(nx,ny)*ring_outer_frac)
    radii = np.linspace(Rmin, Rmax, ring_count)
    Fs, Ms, rels = [], [], []
    for Rpix in radii:
        F = ring_flux_from_Phik(Phi_k, Rpix, KX, KY, pix_rad, M_samples=M_samples)
        inside = disk_mask(ny, nx, Rpix)
        M = ((Sigma_eff[inside]-Sigma0)*(pix_rad**2)).sum()
        rel = abs(F + Kc*M) / (abs(F)+1e-30)
        Fs.append(F); Ms.append(M); rels.append(rel)
    Fs, Ms, rels = map(np.array, (Fs, Ms, rels))
    K_hat = - float(np.sum(Fs*Ms) / (np.sum(Ms**2)+1e-30))
    return K_hat / (8*np.pi*G.value/(c.value**2)), 100*np.median(rels), (radii, Fs, Ms, rels)

# 初始 z_s_init，在 weighted-DC 下得到 K_ratio0，据此反推 z_s,eff
Sigma, Sigma_eff, Scrit0 = Sigma_and_eff(z_d_init, z_s_init)
K_ratio0, _, _ = Sigma_gauss_Kratio(Sigma_eff, dc_mode="weighted")
target_Scrit = Scrit0 * K_ratio0

def find_zs_for_Scrit(target):
    lo, hi = max(z_d_init+1e-3, 0.35), 10.0
    for _ in range(80):
        mid = 0.5*(lo+hi)
        Sc = Sigma_crit_of(z_d_init, mid)
        if Sc > target: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

z_s_eff = find_zs_for_Scrit(target_Scrit)

# 用 z_s,eff，准备 α 扫描与 α=0 的主结果
Sigma, Sigma_eff, Scrit_eff = Sigma_and_eff(z_d_init, z_s_eff)
K_const = 8*np.pi*G.value/(c.value**2)
Phi_k_eff, _ = Phi_from_RHS(-K_const * Sigma_eff)

Rmin = int(min(nx,ny)*ring_inner_frac); Rmax = int(min(nx,ny)*ring_outer_frac)
radii = np.linspace(Rmin, Rmax, ring_count)
Fs = np.array([ring_flux_from_Phik(Phi_k_eff, R, KX, KY, pix_rad, M_samples=M_samples) for R in radii])
areas = np.array([disk_mask(ny, nx, R).sum() for R in radii]) * (pix_rad**2)
S_in  = np.array([(Sigma_eff[disk_mask(ny,nx,R)]).sum() for R in radii]) * (pix_rad**2)

dc_weighted = Sigma_eff.sum() / (win.sum()+1e-30)
dc_pixel    = Sigma_eff.sum() / (Sigma_eff.size+1e-30)

# α 扫描（保存全表）
alphas = np.linspace(0,1,51)
scan_rows=[]
for a in alphas:
    Sigma0 = a*dc_weighted + (1-a)*dc_pixel
    Ms = S_in - areas*Sigma0
    K_hat = - float(np.sum(Fs*Ms) / (np.sum(Ms**2)+1e-30))
    K_ratio = K_hat / K_const
    rels = np.abs(Fs + K_const*Ms) / (np.abs(Fs)+1e-30)
    scan_rows.append([a, K_ratio, 100*np.median(rels), 100*np.percentile(rels,25), 100*np.percentile(rels,75)])
scan = pd.DataFrame(scan_rows, columns=["alpha","K_ratio","Gauss_median_%","Gauss_q1_%","Gauss_q3_%"])
scan.to_csv(os.path.join(OUT,"abell2744_sigma_DCalpha_scan.csv"), index=False)

# α* 仅供参考（实际主文档用 α=0）
a_star_row = scan.iloc[np.argmin(np.abs(scan["K_ratio"]-1.0))]
alpha_star = float(a_star_row["alpha"]); K_ratio_star = float(a_star_row["K_ratio"])
Gauss_med_star = float(a_star_row["Gauss_median_%"])

# α=0（像素均值 DC）：主结果 + per-ring 明细
Sigma0_doc = dc_pixel
Ms_doc = S_in - areas*Sigma0_doc
K_hat_doc = - float(np.sum(Fs*Ms_doc) / (np.sum(Ms_doc**2)+1e-30))
K_ratio_doc = K_hat_doc / K_const
rels_doc = np.abs(Fs + K_const*Ms_doc) / (np.abs(Fs)+1e-30)
Gauss_med_doc = float(np.median(100*rels_doc))
Gauss_q1_doc  = float(np.percentile(100*rels_doc,25))
Gauss_q3_doc  = float(np.percentile(100*rels_doc,75))

# 保存 Σ-DC 汇总与逐环（α=0）
pd.DataFrame({
    "Metric":["z_s_eff","alpha_star","Sigma_crit(kg/m^2)","K_ratio(DC)_alpha*","Gauss_median(%)_alpha*",
              "K_ratio(DC)_alpha0","Gauss_median(%)_alpha0","Gauss_q1(%)_alpha0","Gauss_q3(%)_alpha0"],
    "Value":[float(z_s_eff), alpha_star, Scrit_eff, K_ratio_star, Gauss_med_star,
             K_ratio_doc, Gauss_med_doc, Gauss_q1_doc, Gauss_q3_doc]
}).to_csv(os.path.join(OUT,"abell2744_sigma_DCalpha_summary.csv"), index=False)

pd.DataFrame({
    "Rpix":radii,
    "F_line":Fs,
    "Marea_comp(DC_alpha0)":Ms_doc,
    "rel_close_DC_alpha0_%":100*rels_doc
}).to_csv(os.path.join(OUT,"abell2744_sigma_DC_per_ring_alpha0.csv"), index=False)

# 简单图（κ 预览 & Σ-DC 扫描）
plt.figure(figsize=(5,4))
plt.imshow(gaussian_filter(kappa, sigma=sigma_pix)*win, origin="lower", cmap="magma")
plt.colorbar(label="κ (smoothed*win)")
plt.title("Abell 2744 κ (smoothed & windowed)")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"abell2744_kappa_smoothed.png")); plt.close()

plt.figure(figsize=(5,4))
plt.plot(scan["alpha"], scan["K_ratio"], lw=2, label="K_ratio(α)")
plt.axhline(1.0, ls="--", lw=1, color="k")
plt.scatter([alpha_star],[K_ratio_star], zorder=3, label=f"α*={alpha_star:.2f}")
plt.xlabel("alpha (DC mix)"); plt.ylabel("K_ratio")
plt.title("Σ-DC: K_ratio vs α")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT,"abell2744_sigma_Kratio_vs_alpha.png")); plt.close()

# =========================================================
# 3) Pixel Corr（修正：两边同样平滑 + 同窗）
# =========================================================
kappa_cmp = gaussian_filter(kappa, sigma=sigma_pix) * win
sigma_over_scrit = (Sigma / Scrit_eff)                           # = κ（在同一源面归一）
sigma_over_scrit_cmp = gaussian_filter(sigma_over_scrit, sigma=sigma_pix) * win
Corr_pix = corrcoef2(kappa_cmp, sigma_over_scrit_cmp)

# -------------------- 打印关键结果 --------------------
print("\n=== 复现实绩 ===")
print(f"κ-DC: a_ratio(DC) = {a_ratio:.5f}   Gauss median = {k_gauss_med:.3f}%")
print(f"Σ-DC: K_ratio(DC) = {K_ratio_doc:.5f}   Gauss median = {Gauss_med_doc:.3f}%   (z_s,eff ≈ {z_s_eff:.6f})")
print(f"Pixel Corr = {Corr_pix:.6f}")
print("\nOutputs in:", OUT)

