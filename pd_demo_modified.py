# pd_demo.py — trefoil -> point cloud -> VR(H1) -> persistence diagram + I(K)
import os
os.environ["TOPOLY_PARALLEL"] = "0"  # disable parallelism before importing topoly to avoid hangs

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
matplotlib.use("Agg")  # non-interactive backend for script mode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D plotting support)

# -------- Tunable parameters --------
DELTA_ARC = 0.2   # arc-length step for sampling along the trefoil
PTS_PER_SEG = 10  # number of interpolated points per edge when densifying the polyline
THRESH_FRAC = 1.0 # compute PH up to RS (min enclosing sphere diameter)
EPS_FILTER = 1e-3 # drop bars whose persistence is shorter than this epsilon
# ------------------------------------


def sample_trefoil_fixed_step(delta: float = 0.1):
    """
    Sample a trefoil curve at (approximately) constant arc-length spacing.

    Parameters
    ----------
    delta : float
        Target arc-length step between consecutive sampled points.

    Returns
    -------
    total_len : float
        Total arclength of the high-resolution parametric curve.
    P_highres : (m, 3) ndarray
        High-resolution polyline obtained from dense parameter sampling.
    P_new : (n, 3) ndarray
        Downsampled polyline at roughly constant arc-length step 'delta'.
    """
    # High-resolution parametrization of a trefoil
    t = np.linspace(0, 2 * np.pi, 20000)
    x = (2 + np.cos(3 * t)) * np.cos(2 * t)
    y = (2 + np.cos(3 * t)) * np.sin(2 * t)
    z = np.sin(3 * t)
    P = np.column_stack([x, y, z])

    # Arclength computation
    diffs = np.diff(P, axis=0)
    seglens = np.linalg.norm(diffs, axis=1)
    arclen = np.concatenate([[0], np.cumsum(seglens)])
    total_len = arclen[-1]

    # Resample at fixed arc-length intervals (every 'delta' units)
    new_s = np.arange(0, total_len, delta)
    Px = np.interp(new_s, arclen, P[:, 0])
    Py = np.interp(new_s, arclen, P[:, 1])
    Pz = np.interp(new_s, arclen, P[:, 2])
    P_new = np.column_stack([Px, Py, Pz])

    return total_len, P, P_new


def densify_polyline(P0, pts_per_seg: int = 10):
    """
    Insert evenly spaced points on each edge of a closed polyline.

    Parameters
    ----------
    P0 : (n, 3) array-like
        Vertices of a closed polygonal curve.
    pts_per_seg : int
        Number of points to insert per edge (excluding the endpoint).

    Returns
    -------
    (N, 3) ndarray
        Densified point cloud along the polyline.
    """
    P0 = np.asarray(P0, float)
    out = []
    for i in range(len(P0)):
        a, b = P0[i], P0[(i + 1) % len(P0)]  # closed
        ts = np.linspace(0, 1, pts_per_seg, endpoint=False)
        seg = (1 - ts)[:, None] * a + ts[:, None] * b
        out.append(seg)
    return np.vstack(out)


def enclosing_sphere_diameter(X):
    """
    Diameter of the minimum enclosing ball of X.

    Uses 'miniball' if available; falls back to pairwise max distance.

    Parameters
    ----------
    X : (N, d) array-like

    Returns
    -------
    float
        2 * radius of the minimum enclosing ball (approximate if fallback).
    """
    X = np.asarray(X, float)
    if miniball is not None:
        _, r2 = miniball.get_bounding_ball(X)
        return 2 * np.sqrt(r2)
    return pdist(X).max()


def betti1_integral(H1):
    """
    Integral of the first Betti curve, equal to the sum of lengths of all finite bars.

    Parameters
    ----------
    H1 : (k, 2) ndarray
        Birth/death pairs for H1.

    Returns
    -------
    float
    """
    lengths = []
    for b, d in H1:
        if np.isinf(d):
            continue  # drop infinite bars
        lengths.append(d - b)
    return float(np.sum(lengths))


def betti_curve(H, tmax: float, eps: float = 1e-3):
    """
    Build a step-function approximation of the first Betti curve from bars.

    Infinite bars are truncated at tmax. Bars shorter than 'eps' are ignored.

    Parameters
    ----------
    H : (k, 2) ndarray
        Birth/death pairs.
    tmax : float
        Right endpoint for truncating infinite bars and closing the step curve.
    eps : float
        Minimum persistence to include a bar.

    Returns
    -------
    ts : (m,) ndarray
        Monotone time grid for the step curve.
    vals : (m,) ndarray
        Corresponding β1 values (right-continuous, 'post' convention).
    """
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
    # --- 0) Output paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_png_trefoil = os.path.join(OUT_DIR, "trefoil.png")
    out_png_pd = os.path.join(OUT_DIR, "pd_trefoil.png")
    out_png_betti = os.path.join(OUT_DIR, "betti_curve.png")

    # --- 1) Generate trefoil and identify via Alexander polynomial
    total_len, P_highres, P0 = sample_trefoil_fixed_step(delta=DELTA_ARC)
    print(f"Trefoil total length ≈ {total_len:.4f}")
    print(f"Sampled {len(P0)} points (arc-length step = {DELTA_ARC})")
    print("Knot ID (Alexander):", tp.alexander(P0.tolist(), closure=Closure.CLOSED))

    # --- Plot 3D trefoil (blue = high-res curve, red = constant-arc samples)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(P_highres[:, 0], P_highres[:, 1], P_highres[:, 2],
            color="blue", lw=0.5, alpha=0.6, label="High-res curve")
    ax.scatter(P0[:, 0], P0[:, 1], P0[:, 2],
               color="red", s=8, label=f"Arc-length samples (Δ={DELTA_ARC})")
    ax.set_title("Trefoil Knot in R^3")
    ax.legend()
    plt.savefig(out_png_trefoil, dpi=150, bbox_inches="tight")
    print("Saved ->", out_png_trefoil)

    # --- 2) Densify along the polyline to obtain a point cloud
    PK = densify_polyline(P0, pts_per_seg=PTS_PER_SEG)
    print(f"Densified to {len(PK)} points (each edge {PTS_PER_SEG} segments)")

    # --- 3) VR(H1) + I(K)
    RS = enclosing_sphere_diameter(PK)
    res = ripser(PK, maxdim=1, thresh=RS * THRESH_FRAC)
    H1 = res["dgms"][1]
    I = betti1_integral(H1)

    # --- Report
    num_inf = np.sum(np.isinf(H1[:, 1]))
    print(f"H1 bars: {len(H1)} (inf bars: {num_inf})")
    print(f"RS(K) ≈ {RS:.4f}")
    print(f"I(K)  = {I:.4f}")

    # --- 4) Save persistence diagram
    plt.figure(figsize=(4, 4))
    plot_diagrams([np.empty((0, 2)), H1], show=False)
    plt.title("H1 persistence diagram")
    plt.savefig(out_png_pd, dpi=150, bbox_inches="tight")
    print("Saved ->", out_png_pd)

    # --- 5) Save Betti curve
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
