# pd_demo.py —— trefoil -> 点云 -> VR(H1) -> 持久图 + I(K)
import os
os.environ["TOPOLY_PARALLEL"] = "0"  # 在 import topoly 前关闭并行，避免卡死

import numpy as np
from topoly.params import Closure
import topoly as tp

from ripser import ripser
from persim import plot_diagrams
try:
    import miniball
except Exception:
    miniball = None

from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use("Agg")  # 无交互后端，脚本模式
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图支持

# -------- 可调参数 --------
DELTA_ARC = 0.2   # 每隔 0.1 弧长取一个点
PTS_PER_SEG = 10  # densify 每条边插值点数
THRESH_FRAC = 1.0 # 持久同调算到 RS
EPS_FILTER = 1e-3 # 过滤掉寿命过短的 bar
# --------------------------

# 🔹 按固定弧长间隔采样 trefoil
def sample_trefoil_fixed_step(delta=0.1):
    # 高分辨率参数化
    t = np.linspace(0, 2*np.pi, 20000)
    x = (2 + np.cos(3*t)) * np.cos(2*t)
    y = (2 + np.cos(3*t)) * np.sin(2*t)
    z = np.sin(3*t)
    P = np.column_stack([x, y, z])

    # 计算弧长
    diffs = np.diff(P, axis=0)
    seglens = np.linalg.norm(diffs, axis=1)
    arclen = np.concatenate([[0], np.cumsum(seglens)])
    total_len = arclen[-1]

    # 按固定间隔采样 (每隔 delta 弧长一个点)
    new_s = np.arange(0, total_len, delta)
    Px = np.interp(new_s, arclen, P[:,0])
    Py = np.interp(new_s, arclen, P[:,1])
    Pz = np.interp(new_s, arclen, P[:,2])
    P_new = np.column_stack([Px, Py, Pz])

    return total_len, P, P_new

def densify_polyline(P0, pts_per_seg=10):
    P0 = np.asarray(P0, float)
    out = []
    for i in range(len(P0)):
        a, b = P0[i], P0[(i+1) % len(P0)]  # 闭合
        ts = np.linspace(0, 1, pts_per_seg, endpoint=False)
        seg = (1-ts)[:, None]*a + ts[:, None]*b
        out.append(seg)
    return np.vstack(out)

def enclosing_sphere_diameter(X):
    X = np.asarray(X, float)
    if miniball is not None:
        _, r2 = miniball.get_bounding_ball(X)
        return 2*np.sqrt(r2)
    return pdist(X).max()

def betti1_integral(H1):
    """Integral of the Betti curve = sum of all finite bar lengths."""
    lengths = []
    for b, d in H1:
        if np.isinf(d):
            continue  # 丢掉无穷 bar
        lengths.append(d - b)
    return float(np.sum(lengths))



# 🔹 Betti 曲线（过滤短条形）
def betti_curve(H, tmax, eps=1e-3):
    events = []
    for b, d in H:
        if np.isinf(d):
            d = tmax
        if d - b > eps:
            events.append((b, +1))
            events.append((d, -1))
    events.sort()

    ts, vals = [0], [0]
    cur = 0
    for t, e in events:
        ts.append(t)
        vals.append(cur)
        cur += e
        ts.append(t)
        vals.append(cur)

    ts.append(tmax)
    vals.append(cur)
    return np.array(ts), np.array(vals)

def main():
    # --- 0) 输出路径
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_png_trefoil = os.path.join(OUT_DIR, "trefoil.png")
    out_png_pd = os.path.join(OUT_DIR, "pd_trefoil.png")
    out_png_betti = os.path.join(OUT_DIR, "betti_curve.png")

    # --- 1) 生成 trefoil 并识别 Alexander
    total_len, P_highres, P0 = sample_trefoil_fixed_step(delta=DELTA_ARC)
    print(f"Trefoil total length ≈ {total_len:.4f}")
    print(f"Sampled {len(P0)} points (arc-length step = {DELTA_ARC})")
    print("Knot ID (Alexander):", tp.alexander(P0.tolist(), closure=Closure.CLOSED))

    # --- 绘制 trefoil 3D 图 (蓝线=高分辨率，红点=固定间隔采样)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(P_highres[:,0], P_highres[:,1], P_highres[:,2], color="blue", lw=0.5, alpha=0.6, label="High-res curve")
    ax.scatter(P0[:,0], P0[:,1], P0[:,2], color="red", s=8, label="Arc-length samples (Δ=0.1)")
    ax.set_title("Trefoil Knot in R^3")
    ax.legend()
    plt.savefig(out_png_trefoil, dpi=150, bbox_inches="tight")
    print("Saved ->", out_png_trefoil)

    # --- 2) densify 点云
    PK = densify_polyline(P0, pts_per_seg=PTS_PER_SEG)
    print(f"Densified to {len(PK)} points (each edge {PTS_PER_SEG} segments)")

    # --- 3) VR(H1) + I(K)
    RS = enclosing_sphere_diameter(PK)
    res = ripser(PK, maxdim=1, thresh=RS*THRESH_FRAC)
    H1 = res["dgms"][1]
    I = betti1_integral(H1)

    # --- 打印信息
    num_inf = np.sum(np.isinf(H1[:, 1]))
    print(f"H1 bars: {len(H1)} (inf bars: {num_inf})")
    print(f"RS(K) ≈ {RS:.4f}")
    print(f"I(K)  = {I:.4f}")

    # --- 4) 保存持久图
    plt.figure(figsize=(4, 4))
    plot_diagrams([np.empty((0, 2)), H1], show=False)
    plt.title("H1 persistence diagram")
    plt.savefig(out_png_pd, dpi=150, bbox_inches="tight")
    print("Saved ->", out_png_pd)

    # --- 5) 保存 Betti 曲线
    ts, vals = betti_curve(H1, RS, eps=EPS_FILTER)
    plt.figure(figsize=(5, 3))
    plt.step(ts, vals, where="post", color="red")
    plt.xlabel("t")
    plt.ylabel(r"$\beta_1(K)$")
    plt.title("First Betti number curve")
    plt.savefig(out_png_betti, dpi=150, bbox_inches="tight")
    print("Saved ->", out_png_betti)

if __name__ == "__main__":
    main()
