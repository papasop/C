# ===== Riemann Fingerprint Evidence Matrix (Colab single cell, fixed factorial) =====
# deps: numpy, pandas, matplotlib（Colab自带）

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math  # 修正：使用 math.factorial

# ---------- 1) 用你的数值（可替换） ----------
metrics = {
    "macro_corr": 0.999437,   # 结构形状相关 (越大越好)
    "macro_rmse": 0.008129,   # RMSE (越小越好)
    "p_plv": 0.007,           # 相位锁定（PLV）置换检验 p 值 (越小越好)
    "p_mse_U": 0.0005,        # 模板MSE: Riemann优于 Uniform 的 p 值 (越小越好)
    "p_pc_U": 0.0005,         # 偏相关: corr(resid, Riemann | Uniform) 的 p 值 (越小越好)
    "p_jacc_U": 1.0,          # 频谱主峰Jaccard: Riemann优于 Uniform 的 p 值 (越小越好; 你可替换为实际值)
}

# ---------- 2) 指标 → 归一化得分（0~1, 高=好） ----------
# corr 直接用；RMSE 用 exp(-rmse/scale)；p 值用 1 - p（截断到[0,1]）
rmse_scale = 0.02
score = {}
score["Shape corr (R)"]        = float(np.clip(metrics["macro_corr"], 0, 1))
score["RMSE"]                  = float(np.exp(-metrics["macro_rmse"]/rmse_scale))
score["Phase locking (PLV)"]   = float(np.clip(1.0 - metrics["p_plv"], 0, 1))
score["Template MSE vs U"]     = float(np.clip(1.0 - metrics["p_mse_U"], 0, 1))
score["Partial corr | U"]      = float(np.clip(1.0 - metrics["p_pc_U"], 0, 1))
score["Spectral Jaccard vs U"] = float(np.clip(1.0 - metrics["p_jacc_U"], 0, 1))

# ---------- 3) Fisher 合并 p 值（只合并“检验类”p） ----------
use_pvals = [metrics["p_plv"], metrics["p_mse_U"], metrics["p_pc_U"]]
use_pvals = np.array(use_pvals, dtype=float)
X = -2.0 * np.sum(np.log(np.clip(use_pvals, 1e-300, 1.0)))
k = len(use_pvals)
nu = 2 * k  # 自由度

# 右尾 p 值：P(Chi2_{nu} >= X) ≈ exp(-X/2) * sum_{j=0}^{nu/2 - 1} (X/2)^j / j!
terms = [ (X/2)**j / math.factorial(j) for j in range(nu//2) ]
p_fisher = float(math.exp(-X/2) * np.sum(terms))

# ---------- 4) 证据矩阵表 ----------
rows = [
    ["Shape corr (R)",        metrics["macro_corr"],        score["Shape corr (R)"],        "结构匹配强度（形状）"],
    ["RMSE",                  metrics["macro_rmse"],        score["RMSE"],                  "误差小→得分高（指数压缩）"],
    ["Phase locking (PLV)",   metrics["p_plv"],             score["Phase locking (PLV)"],   "相位锁定显著性（1-p）"],
    ["Template MSE vs U",     metrics["p_mse_U"],           score["Template MSE vs U"],     "模板MSE检验显著性（1-p）"],
    ["Partial corr | U",      metrics["p_pc_U"],            score["Partial corr | U"],      "偏相关显著性（1-p）"],
    ["Spectral Jaccard vs U", metrics["p_jacc_U"],          score["Spectral Jaccard vs U"], "谱主峰重合显著性（1-p）"],
]
df = pd.DataFrame(rows, columns=["Evidence axis", "Raw metric", "Normalized score (0-1, ↑=better)", "Meaning"])
print("Riemann fingerprint — evidence matrix")
print(df.to_string(index=False))
print(f"\nFisher combined p (PLV + MSEvsU + pcorr|U): p ≈ {p_fisher:.4g}  (k={k}, X={X:.3f})")

# ---------- 5) 雷达图 ----------
labels = [r[0] for r in rows]
values = [r[2] for r in rows]
angles = np.linspace(0, 2*np.pi, len(values), endpoint=False)
values_c = np.asarray(values + values[:1])
angles_c = np.asarray(list(angles) + [angles[0]])

fig = plt.figure(figsize=(7,7))
ax = plt.subplot(111, polar=True)
ax.plot(angles_c, values_c, linewidth=2)
ax.fill(angles_c, values_c, alpha=0.2)
ax.set_thetagrids(angles * 180/np.pi, labels)
ax.set_title("Riemann Fingerprint — Radar")
ax.set_rlabel_position(0)
ax.set_ylim(0, 1.05)
plt.show()

# ---------- 6) 热力图 ----------
H = np.array(values).reshape(-1,1)
fig, ax = plt.subplots(figsize=(5, 3 + 0.35*len(values)))
im = ax.imshow(H, aspect="auto")
ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
ax.set_xticks([0]); ax.set_xticklabels(["Score"])
for i,v in enumerate(values):
    ax.text(0, i, f"{v:.3f}", ha="center", va="center", color="white" if v>0.5 else "black")
ax.set_title("Evidence heatmap (0–1)")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.show()

# ---------- 7) 论文引用提示 ----------
print("\nSuggested paper phrasing:")
print(f"- Macro shape: corr = {metrics['macro_corr']:.6f}, RMSE ≈ {metrics['macro_rmse']:.4f}.")
print(f"- Residual fingerprint: PLV permutation p = {metrics['p_plv']:.3g}, "
      f"MSE vs uniform p = {metrics['p_mse_U']:.3g}, partial-corr|uniform p = {metrics['p_pc_U']:.3g}.")
print(f"- Combined significance (Fisher): p ≈ {p_fisher:.3g}.")
print("Note: Replace p_jacc_U when a robust Jaccard test is available; current value is a placeholder for plotting.")
