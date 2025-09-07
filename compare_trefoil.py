# compare_trefoil.py —— trefoil 特征 + IR(K) + RS(K) + 持久图 + β1曲线（含过滤）
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 只保存图片
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from ripser import ripser
from persim import plot_diagrams

from github_python_functions import (
    compute_curvature, compute_torsion,
    compute_max_distance, radius_of_gyration,
    create_necklace_around_segment, betti_curve_features
)
from pd_demo_modified import sample_trefoil_fixed_step, enclosing_sphere_diameter

# ============== 可调参数 ==============
DELTA_ARC = 0.2     # trefoil 折线弧长采样间隔（越小越精细但更慢）
NECKLACE_K = 10     # 每条边生成的“项链点”数量
TAU = 0.05          # 过滤阈值：只保留持久度 >= TAU 的条形（单位与 t 一致）
# =====================================

# ---------- 工具：把作者函数返回的“点”压平为 [x,y,z] ----------
def _flatten_point(p):
    arr = np.asarray(p, dtype=float).squeeze()
    return [float(arr[0]), float(arr[1]), float(arr[2])]

def _necklace_points(a, b, k):
    return [_flatten_point(q) for q in create_necklace_around_segment(a, b, k)]

# ---------- 几何 IR(K)：Rawdon polygonal thickness ----------
def _segment_segment_distance(a0, a1, b0, b1):
    a0 = np.asarray(a0, float); a1 = np.asarray(a1, float)
    b0 = np.asarray(b0, float); b1 = np.asarray(b1, float)
    u = a1 - a0; v = b1 - b0; w0 = a0 - b0
    a = np.dot(u,u); b = np.dot(u,v); c = np.dot(v,v)
    d = np.dot(u,w0); e = np.dot(v,w0); D = a*c - b*b
    SMALL = 1e-12
    if D < SMALL:
        sc = 0.0
        tc = np.clip(e/c if c>SMALL else 0.0, 0.0, 1.0)
    else:
        sN = b*e - c*d; tN = a*e - b*d
        if sN < 0.0: sN, tN = 0.0, e
        elif sN > D: sN, tN = D, e + b
        if tN < 0.0:
            tN = 0.0; sN = np.clip(-d, 0.0, D)
        elif tN > D:
            tN = D; sN = np.clip(-d + b, 0.0, D)
        sc = 0.0 if abs(D)<SMALL else sN / D
        tc = 0.0 if abs(D)<SMALL else tN / D
    pA = a0 + sc*u; pB = b0 + tc*v
    return float(np.linalg.norm(pA - pB))

def _min_radius_vertex(P, i):
    n = len(P)
    a = np.asarray(P[(i-1)%n], float)
    b = np.asarray(P[i], float)
    c = np.asarray(P[(i+1)%n], float)
    e1 = b - a; e2 = c - b
    L1 = np.linalg.norm(e1); L2 = np.linalg.norm(e2)
    if L1 < 1e-12 or L2 < 1e-12: return np.inf
    u1 = e1 / L1; u2 = e2 / L2
    cosang = np.clip(np.dot(u1,u2), -1.0, 1.0)
    theta = np.arccos(cosang)
    if theta < 1e-12: return np.inf
    return float(min(L1, L2) / (2.0 * np.tan(theta/2.0)))

def injectivity_radius_geometric(P):
    """IR(K) = min( min_i MinRad(v_i), 0.5 * min_{非相邻边} dist(ei,ej) )"""
    P = np.asarray(P, float); n = len(P)
    # 局部项
    minrad = min(_min_radius_vertex(P, i) for i in range(n))
    # 全局项（跳过共享端点的相邻边）
    mindist = np.inf
    for i in range(n):
        a0, a1 = P[i], P[(i+1)%n]
        for j in range(n):
            if j in {i, (i-1)%n, (i+1)%n}:
                continue
            b0, b1 = P[j], P[(j+1)%n]
            d = _segment_segment_distance(a0, a1, b0, b1)
            if d < mindist: mindist = d
    return float(min(minrad, 0.5*mindist))

# ---------- β1 曲线 / 过滤 ----------
def filter_bars_by_persistence(H1, tau):
    """仅保留持久度 >= tau 的有限条形"""
    H1 = np.asarray(H1, float)
    if H1.size == 0:
        return H1
    finite = ~np.isinf(H1[:, 1])
    L = H1[:, 1] - H1[:, 0]
    keep = finite & (L >= tau)
    return H1[keep]

def betti_step_from_bars(H1):
    """由条形构造 β1 阶梯曲线 (ts, vals)"""
    H1 = np.asarray(H1, float)
    events = []
    for b, d in H1:
        events.append((float(b), +1))
        events.append((float(d), -1))
    if not events:
        return np.array([0.0]), np.array([0.0])
    events.sort()
    ts, vals, cur = [0.0], [0], 0
    last = 0.0
    for t, e in events:
        if t != last:
            ts.append(t); vals.append(cur); last = t
        cur += e
        ts.append(t); vals.append(cur)
    ts.append(ts[-1]); vals.append(cur)
    return np.array(ts), np.array(vals)

def betti_integral_strict(H1):
    """∫β1 = Σ(d-b)（调用前应已过滤 ∞ 与短条形）"""
    H1 = np.asarray(H1, float)
    if H1.size == 0:
        return 0.0
    return float(np.sum(H1[:, 1] - H1[:, 0]))

# ----------------- 主流程 -----------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "outputs"); os.makedirs(outdir, exist_ok=True)
    png_trefoil  = os.path.join(outdir, "trefoil.png")
    png_pd       = os.path.join(outdir, "pd_trefoil.png")
    png_betti_R  = os.path.join(outdir, "betti_curve_raw.png")
    png_betti_F  = os.path.join(outdir, "betti_curve_filtered.png")

    # 1) Trefoil 折线
    total_len, P_highres, P0 = sample_trefoil_fixed_step(delta=DELTA_ARC)

    # 几何量
    curv = compute_curvature(P0)
    tors = compute_torsion(P0)
    diam = compute_max_distance(P0)
    gyr  = radius_of_gyration(P0)

    # 2) trefoil 3D 图
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(P_highres[:,0], P_highres[:,1], P_highres[:,2], lw=0.5, alpha=0.6, label="High-res")
    ax.scatter(P0[:,0], P0[:,1], P0[:,2], s=8, label="Arc-length samples")
    ax.legend(); ax.set_title("Trefoil Knot in R^3")
    plt.savefig(png_trefoil, dpi=150, bbox_inches="tight")

    # 3) “项链”点云 + RS(K)
    cloud = []
    for i in range(len(P0)):
        cloud += _necklace_points(P0[i], P0[(i+1)%len(P0)], NECKLACE_K)
    cloud = np.array(cloud, dtype=float)
    RS = enclosing_sphere_diameter(cloud)

    # 4) ripser 条形（H1）
    res = ripser(cloud, maxdim=1)
    H1 = res["dgms"][1]

    # 5) β1（原始 & 过滤）+ 积分
    H1_finite = H1[~np.isinf(H1[:,1])]
    H1_filt   = filter_bars_by_persistence(H1_finite, TAU)

    ts_raw,  vals_raw  = betti_step_from_bars(H1_finite)
    ts_filt, vals_filt = betti_step_from_bars(H1_filt)

    I_raw  = betti_integral_strict(H1_finite)
    I_filt = betti_integral_strict(H1_filt)

    # 6) 几何 IR(K) 与 L/D
    IR = injectivity_radius_geometric(P0)
    edges = P0[(np.arange(len(P0)) + 1) % len(P0)] - P0
    L = float(np.linalg.norm(edges, axis=1).sum())
    LD = L / (2.0 * IR) if IR > 0 else np.inf

    # 7) 持久图
    plt.figure(figsize=(4,4))
    plot_diagrams([np.empty((0,2)), H1], show=False)
    plt.title("H1 persistence diagram")
    plt.savefig(png_pd, dpi=150, bbox_inches="tight")

    # 8) β1 原始曲线
    plt.figure(figsize=(6,3.6))
    plt.step(ts_raw, vals_raw, where="post", linewidth=2)
    plt.xlabel("t"); plt.ylabel(r"$\beta_1(K)$"); plt.title("First Betti number curve (raw)")
    plt.axvline(IR, linestyle="--"); plt.text(IR, max(1, vals_raw.max())*0.9, f" IR≈{IR:.3f}", rotation=90, va="top", ha="right")
    plt.axvline(RS, linestyle=":");  plt.text(RS, max(1, vals_raw.max())*0.8, f" RS≈{RS:.3f}", rotation=90, va="top", ha="right")
    plt.savefig(png_betti_R, dpi=150, bbox_inches="tight")

    # 9) β1 过滤后曲线
    plt.figure(figsize=(6,3.6))
    plt.step(ts_filt, vals_filt, where="post", linewidth=2)
    plt.xlabel("t"); plt.ylabel(r"$\beta_1(K)$"); plt.title(f"First Betti number curve (filtered, τ={TAU})")
    plt.axvline(IR, linestyle="--"); plt.text(IR, max(1, vals_filt.max())*0.9, f" IR≈{IR:.3f}", rotation=90, va="top", ha="right")
    plt.axvline(RS, linestyle=":");  plt.text(RS, max(1, vals_filt.max())*0.8, f" RS≈{RS:.3f}", rotation=90, va="top", ha="right")
    plt.savefig(png_betti_F, dpi=150, bbox_inches="tight")

    # 10) 打印
    print("=== Trefoil Features ===")
    print(f"Total length      ≈ {total_len:.4f}")
    print(f"Curvature         = {curv:.4f}")
    print(f"Torsion           = {tors:.4f}")
    print(f"Diameter (maxdist)= {diam:.4f}")
    print(f"Gyration          = {gyr:.4f}")
    print(f"RS(K) (minball)   ≈ {RS:.4f}")
    print(f"IR(K) (geometric) ≈ {IR:.4f}")
    print(f"L/D               ≈ {LD:.4f}")
    print(f"Bars (finite)     = {len(H1_finite)}")
    print(f"Bars (filtered)   = {len(H1_filt)}  (τ={TAU})")
    print(f"Integral raw      = {I_raw:.6f}")
    print(f"Integral filtered = {I_filt:.6f}")
    print("Saved ->", png_trefoil)
    print("Saved ->", png_pd)
    print("Saved ->", png_betti_R)
    print("Saved ->", png_betti_F)

if __name__ == "__main__":
    main()
