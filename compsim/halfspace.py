from __future__ import annotations

import numpy as np


def normalize_halfspaces(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    nrm = np.linalg.norm(A, axis=1)
    nrm = np.maximum(nrm, 1e-12)
    return (A / nrm[:, None]).astype(np.float64), (b / nrm).astype(np.float64)


def reduce_coplanar_halfspaces(
    A: np.ndarray,
    b: np.ndarray,
    ang_tol: float = 1e-6,
    off_tol: float = 5e-6,
) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep: list[int] = []
    for i in range(A.shape[0]):
        ni = A[i]
        di = b[i]
        dup = False
        for j in keep:
            nj = A[j]
            dj = b[j]
            if float(np.dot(ni, nj)) >= 1.0 - ang_tol and abs(float(di - dj)) <= off_tol:
                dup = True
                break
        if not dup:
            keep.append(i)
    return A[keep].astype(np.float64), b[keep].astype(np.float64)


# ============================================================
# 2D convex hull (Andrew monotone chain)
# ============================================================

def convex_hull_2d_indices(points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    M = pts.shape[0]
    if M <= 1:
        return np.arange(M, dtype=np.int32)

    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts_s = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[int] = []
    for i in range(M):
        while len(lower) >= 2 and cross(pts_s[lower[-2]], pts_s[lower[-1]], pts_s[i]) <= 0:
            lower.pop()
        lower.append(i)

    upper: list[int] = []
    for i in range(M - 1, -1, -1):
        while len(upper) >= 2 and cross(pts_s[upper[-2]], pts_s[upper[-1]], pts_s[i]) <= 0:
            upper.pop()
        upper.append(i)

    hull_s = lower[:-1] + upper[:-1]
    hull_orig = [int(order[i]) for i in hull_s]

    seen: set[int] = set()
    out: list[int] = []
    for idx in hull_orig:
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return np.asarray(out, dtype=np.int32)


def build_patch_vertices_fixedK(Vw_all: np.ndarray, support_eps: float, Kmax: int) -> tuple[np.ndarray, float]:
    Vw_all = np.asarray(Vw_all, dtype=np.float64)
    z = Vw_all[:, 2]
    zmin = float(np.min(z))
    N = Vw_all.shape[0]

    cand_idx = np.where(z <= (zmin + support_eps))[0]
    if cand_idx.size < 3:
        take = min(max(6, Kmax), N)
        cand_idx = np.argsort(z)[:take]

    cand = Vw_all[cand_idx]

    if cand.shape[0] >= 3:
        h_rel = convex_hull_2d_indices(cand[:, :2])
        hull_idx = cand_idx[h_rel]
    else:
        hull_idx = cand_idx.copy()

    hull_idx = list(map(int, hull_idx))
    H = len(hull_idx)
    if H == 0:
        hull_idx = [int(np.argmin(z))]
        H = 1

    if H > Kmax:
        pick = np.linspace(0, H, Kmax, endpoint=False).astype(np.int32)
        sel_idx = [hull_idx[int(i)] for i in pick]
        return Vw_all[sel_idx].astype(np.float64), zmin

    sel_set = set(hull_idx)
    sel_idx = hull_idx.copy()

    rest = [int(i) for i in cand_idx if int(i) not in sel_set]
    if rest:
        rest = sorted(rest, key=lambda i: (Vw_all[i, 2], Vw_all[i, 0] ** 2 + Vw_all[i, 1] ** 2))
    for i in rest:
        if len(sel_idx) >= Kmax:
            break
        sel_idx.append(i)
        sel_set.add(i)

    if len(sel_idx) < Kmax:
        for i in np.argsort(z):
            i = int(i)
            if i in sel_set:
                continue
            sel_idx.append(i)
            sel_set.add(i)
            if len(sel_idx) >= Kmax:
                break

    sel_idx = sel_idx[:Kmax]
    return Vw_all[sel_idx].astype(np.float64), zmin


def polygon_area_centroid_xy(poly_xy: np.ndarray) -> np.ndarray:
    P = np.asarray(poly_xy, dtype=np.float64)
    M = P.shape[0]
    if M == 0:
        return np.zeros(2, dtype=np.float64)
    if M < 3:
        return P.mean(axis=0)

    x = P[:, 0]
    y = P[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    cross = x * y2 - x2 * y
    A2 = np.sum(cross)
    if abs(float(A2)) < 1e-12:
        return P.mean(axis=0)

    cx = np.sum((x + x2) * cross) / (3.0 * A2)
    cy = np.sum((y + y2) * cross) / (3.0 * A2)
    return np.array([cx, cy], dtype=np.float64)


def support_polygon_center_xy(Vw_all: np.ndarray, support_eps: float) -> np.ndarray:
    V = np.asarray(Vw_all, dtype=np.float64)
    z = V[:, 2]
    zmin = float(np.min(z))
    cand_idx = np.where(z <= (zmin + float(support_eps)))[0]
    if cand_idx.size < 3:
        take = min(max(6, 12), V.shape[0])
        cand_idx = np.argsort(z)[:take]

    cand = V[cand_idx]
    if cand.shape[0] == 0:
        return np.zeros(2, dtype=np.float64)

    if cand.shape[0] >= 3:
        h_rel = convex_hull_2d_indices(cand[:, :2])
        poly = cand[h_rel, :2]
        if poly.shape[0] >= 3:
            return polygon_area_centroid_xy(poly)
        return cand[:, :2].mean(axis=0)

    return cand[:, :2].mean(axis=0)

# -----------------------------------------------------------------------------
# Backwards-compatible aliases (from original v081).
# -----------------------------------------------------------------------------
# Original v081 used private helper names with leading underscores.
# Keep them as aliases so existing code that imported them directly won't break.
_normalize_halfspaces = normalize_halfspaces
_reduce_coplanar_halfspaces = reduce_coplanar_halfspaces
