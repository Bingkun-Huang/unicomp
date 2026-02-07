#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
2 MCP : MCP for grounding contact + MCP for tool contact
Fixed Errors with unstable ECPs using NEW freeze-by-velocity logic
'''
import time
import argparse
import numpy as np
from scipy.spatial import ConvexHull

import mujoco
import mujoco.viewer
from mujoco.glfw import glfw

import jax
import jax.numpy as jnp

import siconos.numerics as sn
from siconos.numerics import MCP, mcp_newton_FB_FBLSA
from recording_helpers import RealtimeDataLogger

# ============================================================
# =====================  TUNABLE PARAMETERS  ==================
# ============================================================

# -----------------------------
# JAX config
# -----------------------------
JAX_PLATFORM_NAME = "cpu"
JAX_ENABLE_X64 = True

# -----------------------------
# ECP Stabilization (NEW)
# -----------------------------
# -----------------------------
# ECP freeze-by-velocity (NEW)
# -----------------------------
ECP_FREEZE_ENABLE = True

# dwell steps before entering FROZEN (dt=0.01 -> 10 steps ~ 0.1s)
ECP_FREEZE_DWELL_N = 100

# reference length to convert angular speed to linear-equivalent speed
ECP_FREEZE_L_REF = 0.6  # meters

# speed thresholds (with hysteresis)
ECP_FREEZE_V_SLEEP = 1   # enter candidate if v_eff < this
ECP_FREEZE_V_WAKE  = 1.5   # leave frozen immediately if v_eff > this

# "no external wrench except gravity" gate
# use weight-scaled thresholds to be model-independent
ECP_FREEZE_F_SLEEP_FACTOR  = 0.01   # 1% * m*g
ECP_FREEZE_F_WAKE_FACTOR   = 0.03   # 3% * m*g
ECP_FREEZE_TAU_SLEEP_FACTOR = 0.01  # 1% * m*g*L
ECP_FREEZE_TAU_WAKE_FACTOR  = 0.03  # 3% * m*g*L
ECP_FREEZE_Z_PLANE = 0.0
ECP_FREEZE_DEBUG = False

_CP_LAST_GID = None
_CP_TIE_EPS = 2e-5



jax.config.update("jax_platform_name", JAX_PLATFORM_NAME)
jax.config.update("jax_enable_x64", JAX_ENABLE_X64)


# -----------------------------
# Halfspace utilities
# -----------------------------
def _normalize_halfspaces(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    nrm = np.linalg.norm(A, axis=1)
    nrm = np.maximum(nrm, 1e-12)
    return (A / nrm[:, None]).astype(np.float64), (b / nrm).astype(np.float64)


def com_to_body_origin_qpos(q_com_wxyz: np.ndarray, ipos_body: np.ndarray) -> np.ndarray:
    """
    q_com_wxyz: [com_world(3), quat_wxyz(4)]
    ipos_body: model.body_ipos[body_id] (COM in body frame)
    returns qpos for MuJoCo freejoint: [body_origin_world(3), quat_wxyz(4)]
    """
    q = np.asarray(q_com_wxyz, dtype=np.float64).copy()
    com = q[:3]
    quat = q[3:7]
    R = np.asarray(quat_to_R_wxyz(jnp.array(quat, dtype=jnp.float64)))  # body->world
    body_origin = com - R @ np.asarray(ipos_body, dtype=np.float64)
    return np.hstack([body_origin, quat])


def body_origin_qpos_to_com(qpos_wxyz: np.ndarray, ipos_body: np.ndarray) -> np.ndarray:
    """
    qpos_wxyz: [body_origin_world(3), quat_wxyz(4)] from MuJoCo
    returns q_com: [com_world(3), quat_wxyz(4)]
    """
    qpos = np.asarray(qpos_wxyz, dtype=np.float64).copy()
    body_origin = qpos[:3]
    quat = qpos[3:7]
    R = np.asarray(quat_to_R_wxyz(jnp.array(quat, dtype=jnp.float64)))  # body->world
    com = body_origin + R @ np.asarray(ipos_body, dtype=np.float64)
    return np.hstack([com, quat])


def quat_to_R_np_wxyz(q):
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return np.array([
        [ww + xx - yy - zz, 2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz]
    ], dtype=np.float64)


def place_body_on_ground_qpos_from_quat(
    quat_wxyz,
    ipos_body,
    local_pts_com,      # local_points_ref (COM frame)
    origin_xy=(0.0, 0.0),
    clearance=1e-6
):
    """
    Return qpos0 (body origin state) such that the body (given quat) is placed on ground z=0.
    - quat_wxyz: initial orientation
    - ipos_body: model.body_ipos[body_id] (COM in body frame)
    - local_pts_com: corners already shifted to COM frame
    """
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    quat = quat / (np.linalg.norm(quat) + 1e-12)

    R = quat_to_R_np_wxyz(quat)

    pts = np.asarray(np.array(local_pts_com), dtype=np.float64)   # (N,3) in COM frame
    zmin_rot = float(np.min((pts @ R.T)[:, 2]))  # since world = R*pts + COM, take z component

    com = np.array([origin_xy[0], origin_xy[1], clearance - zmin_rot], dtype=np.float64)
    origin = com - R @ np.asarray(ipos_body, dtype=np.float64)

    qpos0 = np.zeros(7, dtype=np.float64)
    qpos0[0:3] = origin
    qpos0[3:7] = quat
    return qpos0, com


def _reduce_coplanar_halfspaces(
    A: np.ndarray,
    b: np.ndarray,
    ang_tol: float = 1e-6,
    off_tol: float = 5e-6,
) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = []
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


# ===========================
# Load MuJoCo model
# ===========================
XML_PATH = "81_t_block_optimized.xml"  # or "table_big.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "T_siconos")
total_mass = float(model.body_mass[body_id])
inertia_body_diag = model.body_inertia[body_id].astype(np.float64)

geom_start = model.body_geomadr[body_id]
geom_num = model.body_geomnum[body_id]
T_GEOM_IDS = np.arange(geom_start, geom_start + geom_num, dtype=np.int32)
T_GEOM_TYPE = np.array(model.geom_type[T_GEOM_IDS], dtype=np.int32)
T_GEOM_POS = np.array(model.geom_pos[T_GEOM_IDS], dtype=np.float64)
T_GEOM_QUAT = np.array(model.geom_quat[T_GEOM_IDS], dtype=np.float64)  # wxyz
T_GEOM_SIZE = np.array(model.geom_size[T_GEOM_IDS], dtype=np.float64)
T_IPOS_BODY = np.array(model.body_ipos[body_id], dtype=np.float64)  # COM in body frame
T_GEOM_POS = (T_GEOM_POS - T_IPOS_BODY).astype(np.float64)


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

    lower = []
    for i in range(M):
        while len(lower) >= 2 and cross(pts_s[lower[-2]], pts_s[lower[-1]], pts_s[i]) <= 0:
            lower.pop()
        lower.append(i)

    upper = []
    for i in range(M - 1, -1, -1):
        while len(upper) >= 2 and cross(pts_s[upper[-2]], pts_s[upper[-1]], pts_s[i]) <= 0:
            upper.pop()
        upper.append(i)

    hull_s = lower[:-1] + upper[:-1]
    hull_orig = [int(order[i]) for i in hull_s]

    seen = set()
    out = []
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


# ============================================================
# NEW: Support polygon centroid helpers (ECP stabilization)
# ============================================================
def polygon_area_centroid_xy(poly_xy: np.ndarray) -> np.ndarray:
    """
    Compute area centroid of a simple polygon in 2D.
    poly_xy: (M,2) in CCW or CW order; returns (2,)
    Falls back to vertex mean if area is degenerate.
    """
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
    """
    Build support polygon from world vertices close to lowest z, then return its centroid in XY.
    """
    V = np.asarray(Vw_all, dtype=np.float64)
    z = V[:, 2]
    zmin = float(np.min(z))
    cand_idx = np.where(z <= (zmin + float(support_eps)))[0]
    if cand_idx.size < 3:
        # take few lowest points
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


# ===========================
# Quaternion helpers (NumPy)
# ===========================
def quat_conj_np(q_wxyz):
    return np.array([q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3]], dtype=np.float64)


def quat_mul_wxyz_np(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def rotate_vector_by_quaternion_np(v, q_wxyz):
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + w * v)


def rotate_world_to_local_np(v_world, q_body_wxyz):
    return rotate_vector_by_quaternion_np(v_world, quat_conj_np(q_body_wxyz))


def quat_from_omega_world_np(omega_world, dt):
    omega = np.asarray(omega_world, dtype=np.float64)
    ang = float(np.linalg.norm(omega) * dt)
    if ang < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = omega / (np.linalg.norm(omega) + 1e-12)
    half = 0.5 * ang
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


# ===========================
# Sample points (corners) in BODY frame
# ===========================
def generate_corners_from_model(model_):
    pts = []
    gstart = model_.body_geomadr[body_id]
    gnum = model_.body_geomnum[body_id]
    for i in range(gstart, gstart + gnum):
        if model_.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        gp = np.array(model_.geom_pos[i], dtype=np.float64)
        gq = np.array(model_.geom_quat[i], dtype=np.float64)  # wxyz
        gs = np.array(model_.geom_size[i], dtype=np.float64)  # half extents
        for dx in (-1, 1):
            for dy in (-1, 1):
                for dz in (-1, 1):
                    corner_local = np.array([dx * gs[0], dy * gs[1], dz * gs[2]], dtype=np.float64)
                    corner_body = rotate_vector_by_quaternion_np(corner_local, gq) + gp
                    corner_com = corner_body - T_IPOS_BODY  # shift to COM frame
                    pts.append(corner_com)
    return jnp.array(np.asarray(pts, dtype=np.float64), dtype=jnp.float64)


local_points_ref = generate_corners_from_model(model)
N_ALL = int(local_points_ref.shape[0])
print(f"[STEP3B-LS-PROX] total sampled points = {N_ALL}")


# ===========================
# JAX helpers
# ===========================
@jax.jit
def quaternion_rotate(q, v):
    w, x, y, z = q
    return v + 2.0 * jnp.cross(q[1:], jnp.cross(q[1:], v) + w * v)


@jax.jit
def get_world_points(q_pos, local_pts):
    pos = q_pos[0:3]
    quat = q_pos[3:7]
    return jax.vmap(lambda p: quaternion_rotate(quat, p))(local_pts) + pos


@jax.jit
def build_jacobian_single(r_world):
    r = r_world
    skew = jnp.array(
        [[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]],
        dtype=jnp.float64,
    )
    return jnp.concatenate([jnp.eye(3, dtype=jnp.float64), -skew], axis=1)


@jax.jit
def quat_to_R_wxyz(q):
    w, x, y, z = q
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return jnp.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=jnp.float64,
    )


@jax.jit
def mass_matrix_inv_6x6(mass, inertia_body_diag, quat_wxyz):
    R = quat_to_R_wxyz(quat_wxyz)
    I_body = jnp.diag(inertia_body_diag)
    I_world = R @ I_body @ R.T
    I_world_inv = jnp.linalg.inv(I_world)
    Minv = jnp.zeros((6, 6), dtype=jnp.float64)
    Minv = Minv.at[0:3, 0:3].set((1.0 / mass) * jnp.eye(3, dtype=jnp.float64))
    Minv = Minv.at[3:6, 3:6].set(I_world_inv)
    return Minv


@jax.jit
def project_ellipsoid3(pt, po, pr, limit, e_t, e_o, e_r):
    limit = jnp.maximum(limit, 0.0)
    ut = pt / (e_t + 1e-18)
    uo = po / (e_o + 1e-18)
    ur = pr / (e_r + 1e-18)

    nrm = jnp.sqrt(ut * ut + uo * uo + ur * ur + 1e-18)
    scale = jnp.where(nrm > limit, limit / nrm, 1.0)

    utp = ut * scale
    uop = uo * scale
    urp = ur * scale

    return utp * e_t, uop * e_o, urp * e_r


# ===========================
# External wrench projection (optional)
# ===========================
def _box_sdf_and_closest_point(p_g, half_ext):
    p = np.asarray(p_g, dtype=np.float64)
    b = np.asarray(half_ext, dtype=np.float64)
    p_closest = np.clip(p, -b, b)
    q = np.abs(p) - b
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(max(q[0], q[1], q[2]), 0.0)
    sdf = outside + inside
    return sdf, p_closest


def project_point_to_tblock_surface_world(p_world, q_curr):
    pos_com = q_curr[:3]
    q_body = q_curr[3:7]
    p_body = rotate_world_to_local_np(p_world - pos_com, q_body)

    best_sdf = 1e9
    best_p_body = None
    for gt, gp, gq, gs in zip(T_GEOM_TYPE, T_GEOM_POS, T_GEOM_QUAT, T_GEOM_SIZE):
        if gt != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        p_g = rotate_vector_by_quaternion_np(p_body - gp, quat_conj_np(gq))
        sdf, p_g_closest = _box_sdf_and_closest_point(p_g, gs[:3])
        if sdf < best_sdf:
            best_sdf = sdf
            p_body_closest = rotate_vector_by_quaternion_np(p_g_closest, gq) + gp
            best_p_body = p_body_closest

    if best_p_body is None:
        return p_world.copy(), 1e9

    p_proj_world = rotate_vector_by_quaternion_np(best_p_body, q_body) + pos_com
    return p_proj_world, best_sdf


def add_world_wrench_projected(f_ext_acc, p_world, F_world, tau_world, q_curr):
    p_proj, _ = project_point_to_tblock_surface_world(p_world, q_curr)
    pos_com = q_curr[:3]
    r = p_proj - pos_com
    f_ext_acc[0:3] += F_world
    f_ext_acc[3:6] += (tau_world + np.cross(r, F_world))
    return p_proj


# ===========================
# Tool (sphere) contact helpers
# ===========================
def _box_sdf_and_surface_point(p_g, half_ext):
    p = np.asarray(p_g, dtype=np.float64)
    b = np.asarray(half_ext, dtype=np.float64)

    p_clip = np.clip(p, -b, b)
    q = np.abs(p) - b
    outside_vec = np.maximum(q, 0.0)
    outside_dist = float(np.linalg.norm(outside_vec))

    if outside_dist > 0.0:
        return outside_dist, p_clip

    d = b - np.abs(p)
    ax = int(np.argmin(d))
    p_surf = p.copy()
    s = 1.0 if p[ax] >= 0.0 else -1.0
    p_surf[ax] = s * b[ax]
    sdf = -float(d[ax])
    return sdf, p_surf


def closest_point_on_tblock_surface_world(p_world, q_curr):
    global _CP_LAST_GID

    pos_com = q_curr[:3]
    q_body = q_curr[3:7]
    p_body = rotate_world_to_local_np(p_world - pos_com, q_body)

    best_abs = 1e18
    candidates = []  # (geom_id, sdf, p_body_surf)

    # 给每个 geom 一个稳定 id（按遍历顺序）
    gid_counter = 0

    for gt, gp, gq, gs in zip(T_GEOM_TYPE, T_GEOM_POS, T_GEOM_QUAT, T_GEOM_SIZE):
        if gt != mujoco.mjtGeom.mjGEOM_BOX:
            continue

        # p_body -> geom local
        p_g = rotate_vector_by_quaternion_np(p_body - gp, quat_conj_np(gq))
        sdf, p_g_surf = _box_sdf_and_surface_point(p_g, gs[:3])

        abs_sdf = abs(float(sdf))
        p_body_surf = rotate_vector_by_quaternion_np(p_g_surf, gq) + gp

        if abs_sdf < best_abs - _CP_TIE_EPS:
            best_abs = abs_sdf
            candidates = [(gid_counter, float(sdf), p_body_surf)]
        elif abs_sdf <= best_abs + _CP_TIE_EPS:
            candidates.append((gid_counter, float(sdf), p_body_surf))

        gid_counter += 1

    if not candidates:
        return np.asarray(p_world, dtype=np.float64).copy(), 1e18

    # tie-break：如果上一次 geom 在候选集合里，就继续用它（减少 stem/top 跳变）
    chosen = None
    if _CP_LAST_GID is not None:
        for (gid, sdf, p_body_surf) in candidates:
            if gid == _CP_LAST_GID:
                chosen = (gid, sdf, p_body_surf)
                break

    if chosen is None:
        # 默认取 candidates[0]（它就是最先达到 best_abs 的）
        chosen = candidates[0]

    _CP_LAST_GID = chosen[0]

    best_sdf = float(chosen[1])
    best_p_body = chosen[2]

    p_surf_world = rotate_vector_by_quaternion_np(best_p_body, q_body) + pos_com
    return p_surf_world, best_sdf


def _orthonormal_tangent_basis(n):
    n = np.asarray(n, dtype=np.float64)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(n, ref))) > 0.95:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t1 = np.cross(ref, n)
    t1n = float(np.linalg.norm(t1))
    if t1n < 1e-12:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        t1 = np.cross(ref, n)
        t1n = float(np.linalg.norm(t1)) + 1e-12
    t1 /= t1n
    t2 = np.cross(n, t1)
    t2 /= (float(np.linalg.norm(t2)) + 1e-12)
    return t1, t2


def _skew(r):
    r = np.asarray(r, dtype=np.float64).reshape(3,)
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]], dtype=np.float64)


def inertia_world_from_body_diag(inertia_body_diag_, quat_wxyz):
    w, x, y, z = quat_wxyz
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    R = np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float64,
    )
    I_body = np.diag(np.asarray(inertia_body_diag_, dtype=np.float64))
    return R @ I_body @ R.T


def compute_tool_block_impulse(
    q_block, v_block6,
    tool_pos, tool_vel,
    tool_radius,
    tool_mu=0.6,
    dt=0.002,
    contact_eps=1e-6,
    restitution=0.0,
    enable_margin=1e-4,
):
    """Compute one-step Coulomb impulse between a dynamic sphere tool and the block.

    Tool is a *dynamic point mass* at its center (no rotation). The contact point is
    the closest point on the block surface to the sphere center.

    **Minimal MCP variant (1D):**


    Tangential impulse is computed by a Coulomb projection (closed-form) using the
    updated relative velocity after applying p_n.
    """
    tool_pos = np.asarray(tool_pos, dtype=np.float64).reshape(3,)
    tool_vel = np.asarray(tool_vel, dtype=np.float64).reshape(3,)
    v_block6 = np.asarray(v_block6, dtype=np.float64).reshape(6,)

    # closest surface point on block
    a, sdf = closest_point_on_tblock_surface_world(tool_pos, q_block)
    a = np.asarray(a, dtype=np.float64).reshape(3,)
    g = float(sdf) - float(tool_radius)

    # gating: far away -> no impulse
    if g > float(enable_margin):
        return np.zeros(3, dtype=np.float64), a, np.array([0.0, 0.0, 1.0], dtype=np.float64), g

    # contact normal points from tool center to closest point on block
    d = a - tool_pos
    dn = float(np.linalg.norm(d))
    if dn < 1e-12:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n = d / dn

    # block point velocity at contact
    com = np.asarray(q_block[:3], dtype=np.float64)
    r = a - com
    v_lin = v_block6[0:3]
    w = v_block6[3:6]
    v_block_c = v_lin + np.cross(w, r)

    # relative velocity (block point wrt tool center)
    v_rel = v_block_c - tool_vel
    vn = float(np.dot(n, v_rel))

    # effective relative velocity update matrix K_rel = K_block + (1/m_tool)I
    # We compute K_block = (1/m)I - [r]x I^{-1} [r]x
    m_b = float(total_mass)
    Iw = inertia_world_from_body_diag(inertia_body_diag, q_block[3:7])
    Iw_inv = np.linalg.inv(Iw + 1e-12*np.eye(3))
    rx = _skew(r)
    K_block = (1.0/m_b) * np.eye(3) - rx @ Iw_inv @ rx

    # tool treated as point mass with mass stored as attribute on this function
    m_t = float(getattr(compute_tool_block_impulse, "tool_mass", 1.0))
    K_rel = K_block + (1.0/m_t) * np.eye(3)

    nKn = float(n.T @ K_rel @ n)

    # ---------------------------
    # Minimal MCP solve for p_n
    # ---------------------------
    # w(pn) = g + contact_eps*pn + dt*(vn + nKn*pn + restitution*min(vn,0))
    #      = b + a*pn
    vn_minus_neg = min(vn, 0.0)
    b = g + float(dt) * (vn + float(restitution) * vn_minus_neg)
    a_lin = float(contact_eps) + float(dt) * float(nKn)

    # default analytic fallback (also a great warm-start)
    pn_fallback = 0.0
    if a_lin > 0.0:
        pn_fallback = max(0.0, -b / a_lin)

    # If predicted next-step gap is nonnegative with pn=0, skip solve (reduces chatter)
    if b >= 0.0:
        pn = 0.0
    else:
        # cached solver options / work vectors
        opts = getattr(compute_tool_block_impulse, "_mcp_options", None)
        if opts is None:
            opts = sn.SolverOptions(sn.SICONOS_MCP_NEWTON_FB_FBLSA)
            opts.iparam[sn.SICONOS_IPARAM_MAX_ITER] = 10
            opts.dparam[sn.SICONOS_DPARAM_TOL] = 1e-7
            compute_tool_block_impulse._mcp_options = opts

        z_guess = getattr(compute_tool_block_impulse, "_z_guess", None)
        if z_guess is None:
            z_guess = np.array([pn_fallback], dtype=np.float64)
            compute_tool_block_impulse._z_guess = z_guess

        w_sol = getattr(compute_tool_block_impulse, "_w_sol", None)
        if w_sol is None:
            w_sol = np.zeros((1,), dtype=np.float64)
            compute_tool_block_impulse._w_sol = w_sol

        # warm-start from previous pn if available
        pn_prev = float(getattr(compute_tool_block_impulse, "pn_guess", pn_fallback))
        if not np.isfinite(pn_prev) or pn_prev < 0.0:
            pn_prev = pn_fallback
        z_guess[0] = pn_prev

        def call_F(n_dim, z, w_out):
            # MCP expects w_out >= 0 complementary to z >= 0
            w_out[0] = b + a_lin * float(z[0])

        def call_Jac(n_dim, z, J_out):
            J_out[0, 0] = a_lin

        problem = MCP(0, 1, call_F, call_Jac)
        info = mcp_newton_FB_FBLSA(problem, z_guess, w_sol, opts)

        if info == 0 and np.isfinite(z_guess[0]) and z_guess[0] >= 0.0:
            pn = float(z_guess[0])
        else:
            pn = float(pn_fallback)

    # store for next warmstart
    compute_tool_block_impulse.pn_guess = float(0.75 * pn + 0.25 * pn_fallback)

    # apply normal impulse then compute tangential via Coulomb projection
    pN = n * pn
    v_rel2 = v_rel + K_rel @ pN

    t1, t2 = _orthonormal_tangent_basis(n)
    vt1 = float(np.dot(t1, v_rel2))
    vt2 = float(np.dot(t2, v_rel2))

    k1 = float(t1.T @ K_rel @ t1)
    k2 = float(t2.T @ K_rel @ t2)
    m1 = 1.0 / (k1 + 1e-12)
    m2 = 1.0 / (k2 + 1e-12)

    pt1 = -m1 * vt1
    pt2 = -m2 * vt2

    # project to Coulomb disk
    lim = float(tool_mu) * pn
    nrm = float(np.hypot(pt1, pt2))
    if nrm > lim and nrm > 1e-12:
        s = lim / nrm
        pt1 *= s
        pt2 *= s

    pT = t1 * pt1 + t2 * pt2
    p_lin = pN + pT
    # pt = p_lin - np.dot(p_lin,n)*n

    # print("[Tool-Block Impulse] pn = %.6f, |pt| = %.6f" % (pn, float(np.linalg.norm(pt))))

    return p_lin.astype(np.float64), a, n.astype(np.float64), g


# ===========================
# MCP residual + Jacobian (方案A：proximal 椭球投影)
# ===========================
@jax.jit
def mcp_residual_step3B_prox_hull(
    z, A_w, b_w, com, v_free, M_inv, dt, restitution, contact_eps,
    mu_fric, e_t, e_o, e_r,
    ecp_xy_reg, a0_xy
):
    v_next = z[0:6]
    p_tx = z[6]
    p_ty = z[7]
    p_r = z[8]
    a = z[9:12]
    p_n = z[12]
    l = z[13:]  # (m,)

    r = a - com
    J = build_jacobian_single(r)

    J_tx = J[0:1, :]
    J_ty = J[1:2, :]
    J_n = J[2:3, :]

    p_contact = (J_n.T * p_n).reshape((6,)) + (J_tx.T * p_tx).reshape((6,)) + (J_ty.T * p_ty).reshape((6,))
    p_contact = p_contact + jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, p_r], dtype=jnp.float64)

    G_dyn = v_next - (v_free + (M_inv @ p_contact))

    vn = (J_n @ v_next)[0]
    vn_minus = (J_n @ v_free)[0]
    vn_minus_neg = jnp.minimum(vn_minus, 0.0)

    vtx = (J_tx @ v_next)[0]
    vty = (J_ty @ v_next)[0]
    v_r = v_next[5]  # omega_z

    gamma = 0.05
    reg = 1e-6
    limit = mu_fric * p_n

    pt0 = p_tx - gamma * vtx
    po0 = p_ty - gamma * vty
    pr0 = p_r - gamma * v_r

    ppt, ppo, ppr = project_ellipsoid3(pt0, po0, pr0, limit, e_t, e_o, e_r)

    G_fric_x = (p_tx - ppt) + reg * p_tx
    G_fric_y = (p_ty - ppo) + reg * p_ty
    G_fric_r = (p_r - ppr) + reg * p_r

    # KKT stationarity with tie-break around a0_xy (NOT com_xy)
    a0_xy = jnp.asarray(a0_xy, dtype=jnp.float64)
    grad = jnp.array(
        [
            ecp_xy_reg * (a[0] - a0_xy[0]),
            ecp_xy_reg * (a[1] - a0_xy[1]),
            1.0,
        ],
        dtype=jnp.float64,
    )
    G_kkt = grad + (A_w.T @ l)

    slack = b_w - (A_w @ a)

    gap_curr = a[2]
    G_gap = gap_curr + contact_eps * p_n + dt * (vn + restitution * vn_minus_neg)

    free = jnp.concatenate([G_dyn, jnp.array([G_fric_x, G_fric_y, G_fric_r], dtype=jnp.float64), G_kkt])
    comp = jnp.concatenate([jnp.array([G_gap], dtype=jnp.float64), slack])
    return jnp.concatenate([free, comp])


mcp_jacobian_step3B_hull = jax.jit(jax.jacfwd(mcp_residual_step3B_prox_hull, argnums=0))


# ============================================================
# Frozen MCP: a is FIXED (no hull KKT), only solve contact+friction
# z_frozen = [v_next(6), p_tx, p_ty, p_r, p_n]  -> 10 dims
# MCP split: n1_frozen=9 (v_next + p_t(3)), n2_frozen=1 (gap/p_n)
# ============================================================
@jax.jit
def mcp_residual_step3B_prox_fixed_a(
    z,
    com, v_free, M_inv,
    dt, restitution, contact_eps,
    mu_fric, e_t, e_o, e_r,
    a_fixed_world, gap_curr
):
    v_next = z[0:6]
    p_tx = z[6]
    p_ty = z[7]
    p_r  = z[8]
    p_n  = z[9]

    r = a_fixed_world - com
    J = build_jacobian_single(r)

    J_tx = J[0:1, :]
    J_ty = J[1:2, :]
    J_n  = J[2:3, :]

    # contact impulse in generalized coords (same as full model)
    p_contact = (J_n.T * p_n).reshape((6,)) + (J_tx.T * p_tx).reshape((6,)) + (J_ty.T * p_ty).reshape((6,))
    p_contact = p_contact + jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, p_r], dtype=jnp.float64)

    # dynamics
    G_dyn = v_next - (v_free + (M_inv @ p_contact))

    # normal relative velocity
    vn = (J_n @ v_next)[0]
    vn_minus = (J_n @ v_free)[0]
    vn_minus_neg = jnp.minimum(vn_minus, 0.0)

    # tangential + spin velocities
    vtx = (J_tx @ v_next)[0]
    vty = (J_ty @ v_next)[0]
    v_r = v_next[5]  # omega_z

    # friction prox projection
    gamma = 0.05
    reg = 1e-6
    limit = mu_fric * p_n

    pt0 = p_tx - gamma * vtx
    po0 = p_ty - gamma * vty
    pr0 = p_r  - gamma * v_r

    ppt, ppo, ppr = project_ellipsoid3(pt0, po0, pr0, limit, e_t, e_o, e_r)

    G_fric_x = (p_tx - ppt) + reg * p_tx
    G_fric_y = (p_ty - ppo) + reg * p_ty
    G_fric_r = (p_r  - ppr) + reg * p_r

    # complementarity for normal contact
    G_gap = gap_curr + contact_eps * p_n + dt * (vn + restitution * vn_minus_neg)

    free = jnp.concatenate([G_dyn, jnp.array([G_fric_x, G_fric_y, G_fric_r], dtype=jnp.float64)])
    comp = jnp.array([G_gap], dtype=jnp.float64)
    return jnp.concatenate([free, comp])


mcp_jacobian_step3B_fixed_a = jax.jit(jax.jacfwd(mcp_residual_step3B_prox_fixed_a, argnums=0))



# ===========================
# Solver class
# ===========================
class TBlockSimulator_Step_NoBounce:
    def __init__(
        self,
        dt,
        mass,
        inertia_diag_body,
        local_pts_all,
        Kmax=12,
        support_eps=1.5e-3,
        alpha_sigma=0.06,
        alpha_rho=1e-3,
        alpha_com_blend=0.25,
        mu_fric=0.5,
        e_t=1.0,
        e_o=1.0,
        e_r=None,
        e_r_factor=0.10,
        restitution=0.10,
        contact_eps=1e-6,
        ground_enable_margin=2e-3,
        proj_tol=1e-6,
    ):
        self.dt = float(dt)
        self.restitution = float(restitution)
        self.contact_eps = float(contact_eps)
        self.proj_tol = float(proj_tol)
        self.ground_enable_margin = float(ground_enable_margin)

        self.mass = jnp.array(float(mass), dtype=jnp.float64)
        self.mass_scalar = float(mass)
        self.inertia_body_diag = jnp.array(np.asarray(inertia_diag_body, dtype=np.float64), dtype=jnp.float64)

        self.local_pts_all = local_pts_all
        self.N_all = int(local_pts_all.shape[0])

        self.K = int(min(Kmax, self.N_all))
        self.support_eps = float(support_eps)
        self.alpha_sigma = float(alpha_sigma)

        self.mu_fric = float(mu_fric)
        self.e_t = float(e_t)
        self.e_o = float(e_o)

        # ---- convex hull halfspaces in BODY frame (for full MCP only)
        pts_hull = np.asarray(np.array(local_pts_all), dtype=np.float64)
        try:
            hull = ConvexHull(pts_hull)
            eq = hull.equations
            self.A_body = eq[:, :3].astype(np.float64)
            self.b_body = (-eq[:, 3]).astype(np.float64)
        except Exception:
            mn = pts_hull.min(axis=0)
            mx = pts_hull.max(axis=0)
            self.A_body = np.array(
                [
                    [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
                ],
                dtype=np.float64,
            )
            self.b_body = np.array([mx[0], -mn[0], mx[1], -mn[1], mx[2], -mn[2]], dtype=np.float64)

        self.A_body, self.b_body = _normalize_halfspaces(self.A_body, self.b_body)
        self.A_body, self.b_body = _reduce_coplanar_halfspaces(self.A_body, self.b_body)
        self.m_hull = int(self.A_body.shape[0])

        # tie-break in full MCP
        self.ecp_xy_reg = 1e-2

        # ellipsoid spin radius
        if e_r is None:
            bbox = pts_hull.max(axis=0) - pts_hull.min(axis=0)
            L = float(np.linalg.norm(bbox[0:2])) + 1e-12
            self.e_r = float(e_r_factor * L)
        else:
            self.e_r = float(e_r)

        self.alpha_rho = float(alpha_rho)
        self.alpha_com_blend = float(alpha_com_blend)

        # ---- full MCP sizes
        self.n1 = 12
        self.n2 = 1 + self.m_hull
        self.z_dim = self.n1 + self.n2

        self.options = sn.SolverOptions(sn.SICONOS_MCP_NEWTON_FB_FBLSA)
        self.options.iparam[sn.SICONOS_IPARAM_MAX_ITER] = 240
        self.options.dparam[sn.SICONOS_DPARAM_TOL] = 1e-6

        self.z_guess = np.zeros(self.z_dim, dtype=np.float64)
        self.z_guess[12] = 1e-6
        self.w_sol = np.zeros(self.z_dim, dtype=np.float64)

        self.jac_reg = 1e-8
        self.ecp_prev = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # ============================================================
        # ECP FREEZE (your final logic)
        #   - Freeze condition: (low speed) AND (no external wrench except gravity)
        #   - Wake condition: (speed high) OR (external wrench present) OR (tool impulse)
        # ============================================================
        self.ecp_freeze_enable = bool(ECP_FREEZE_ENABLE)
        self.ecp_freeze_dwell_n = int(ECP_FREEZE_DWELL_N)
        self.ecp_freeze_L = float(ECP_FREEZE_L_REF)

        self.ecp_freeze_v_sleep = float(ECP_FREEZE_V_SLEEP)
        self.ecp_freeze_v_wake  = float(ECP_FREEZE_V_WAKE)

        self.ecp_freeze_f_sleep = float(ECP_FREEZE_F_SLEEP_FACTOR  * self.mass_scalar * 9.81)
        self.ecp_freeze_f_wake  = float(ECP_FREEZE_F_WAKE_FACTOR   * self.mass_scalar * 9.81)
        self.ecp_freeze_tau_sleep = float(ECP_FREEZE_TAU_SLEEP_FACTOR * self.mass_scalar * 9.81 * self.ecp_freeze_L)
        self.ecp_freeze_tau_wake  = float(ECP_FREEZE_TAU_WAKE_FACTOR  * self.mass_scalar * 9.81 * self.ecp_freeze_L)

        self.ecp_freeze_z_plane = float(ECP_FREEZE_Z_PLANE)
        self.ecp_freeze_debug = bool(ECP_FREEZE_DEBUG)

        self._freeze_count = 0
        self._frozen = False

        # ---- frozen MCP sizes
        self.n1_frozen = 9
        self.n2_frozen = 1
        self.z_dim_frozen = self.n1_frozen + self.n2_frozen

        self.z_guess_frozen = np.zeros(self.z_dim_frozen, dtype=np.float64)
        self.z_guess_frozen[9] = 1e-6  # p_n
        self.w_sol_frozen = np.zeros(self.z_dim_frozen, dtype=np.float64)

        self._warmup()

    # -----------------------------
    # helpers
    # -----------------------------
    def _wrench_without_gravity(self, f_applied_np: np.ndarray) -> np.ndarray:
        """
        Your f_applied includes gravity already (f[2] += -m*g).
        Remove gravity so we can detect "external" wrench.
        """
        f = np.asarray(f_applied_np, dtype=np.float64).copy()
        f[2] += self.mass_scalar * 9.81
        return f

    def _v_eff(self, v_curr_np: np.ndarray) -> float:
        v_curr_np = np.asarray(v_curr_np, dtype=np.float64).reshape(6,)
        v_lin = float(np.linalg.norm(v_curr_np[0:3]))
        w = float(np.linalg.norm(v_curr_np[3:6]))
        return v_lin + self.ecp_freeze_L * w

    def _compute_a_frozen(self, q_curr_np: np.ndarray, Vw_all: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Frozen a: COM projection in XY + current zmin as gap surrogate.
        When supported, zmin ~ 0 -> you effectively get COM projection on plane.
        """
        com = np.asarray(q_curr_np[0:3], dtype=np.float64)
        zmin = float(np.min(np.asarray(Vw_all[:, 2], dtype=np.float64)))

        a = np.array([com[0], com[1], zmin], dtype=np.float64)

        # If you want to hard snap to a plane once very close:
        if abs(a[2] - self.ecp_freeze_z_plane) <= (5.0 * self.proj_tol):
            a[2] = self.ecp_freeze_z_plane

        return a, zmin

    def _update_freeze_state(
        self,
        v_curr_np: np.ndarray,
        f_applied_np: np.ndarray,
        zmin: float,
        z_pred_min: float,
        step_idx: int,
        tool_impulse_norm: float = 0.0,
    ) -> bool:
        """
        Returns current frozen flag.
        Only consider freezing when contact is possible (otherwise do not freeze).
        """
        if not self.ecp_freeze_enable:
            self._frozen = False
            self._freeze_count = 0
            return False

        # If clearly airborne -> do not freeze
        airborne = (zmin > self.ground_enable_margin) and (z_pred_min > 0.0)
        if airborne:
            self._frozen = False
            self._freeze_count = 0
            return False

        # External wrench (remove gravity)
        wng = self._wrench_without_gravity(f_applied_np)
        fn = float(np.linalg.norm(wng[0:3]))
        taun = float(np.linalg.norm(wng[3:6]))

        v_eff = self._v_eff(v_curr_np)
        tool_hit = float(tool_impulse_norm) > 1e-12

        if self._frozen:
            # Wake hysteresis: any strong motion OR any external wrench OR tool impulse
            if (v_eff > self.ecp_freeze_v_wake) or (fn > self.ecp_freeze_f_wake) or (taun > self.ecp_freeze_tau_wake) or tool_hit:
                if self.ecp_freeze_debug:
                    print(f"[ECP-FREEZE] step {step_idx}: WAKE  v_eff={v_eff:.3e}, fn={fn:.3e}, tau={taun:.3e}, tool={tool_impulse_norm:.3e}")
                self._frozen = False
                self._freeze_count = 0
        else:
            # Sleep dwell: must be slow AND no external wrench AND no tool impulse
            ok_sleep = (v_eff < self.ecp_freeze_v_sleep) and (fn < self.ecp_freeze_f_sleep) and (taun < self.ecp_freeze_tau_sleep) and (not tool_hit)
            if ok_sleep:
                self._freeze_count += 1
                if self._freeze_count >= self.ecp_freeze_dwell_n:
                    self._frozen = True
                    if self.ecp_freeze_debug:
                        print(f"[ECP-FREEZE] step {step_idx}: ENTER frozen  v_eff={v_eff:.3e}, fn={fn:.3e}, tau={taun:.3e}")
            else:
                self._freeze_count = 0

        return self._frozen

    # -----------------------------
    # warmup
    # -----------------------------
    def _warmup(self):
        q0 = jnp.array([0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
        v0 = jnp.zeros((6,), dtype=jnp.float64)
        f0 = jnp.zeros((6,), dtype=jnp.float64)

        M_inv0 = mass_matrix_inv_6x6(self.mass, self.inertia_body_diag, q0[3:7])
        v_free0 = v0 + (M_inv0 @ (self.dt * f0))

        R0 = np.asarray(quat_to_R_wxyz(q0[3:7]))
        A_w0 = jnp.array(self.A_body @ R0.T, dtype=jnp.float64)
        b_w0 = jnp.array(self.b_body + (self.A_body @ R0.T) @ np.asarray(q0[0:3]), dtype=jnp.float64)

        Vw_all0 = np.asarray(get_world_points(q0, self.local_pts_all))
        a0 = Vw_all0[int(np.argmin(Vw_all0[:, 2]))].astype(np.float64)
        a0_xy = jnp.array(a0[0:2], dtype=jnp.float64)

        self.ecp_prev = a0.copy()

        # full guess init
        self.z_guess[:] = 0.0
        self.z_guess[0:6] = np.asarray(v0)
        self.z_guess[9:12] = a0
        self.z_guess[12] = 1e-6

        # frozen guess init
        self.z_guess_frozen[:] = 0.0
        self.z_guess_frozen[0:6] = np.asarray(v0)
        self.z_guess_frozen[9] = 1e-6

        # jit warmup full residual/jac
        mcp_residual_step3B_prox_hull(
            self.z_guess,
            A_w0,
            b_w0,
            q0[0:3],
            v_free0,
            M_inv0,
            self.dt,
            self.restitution,
            self.contact_eps,
            self.mu_fric,
            self.e_t,
            self.e_o,
            self.e_r,
            self.ecp_xy_reg,
            a0_xy,
        ).block_until_ready()

        mcp_jacobian_step3B_hull(
            self.z_guess,
            A_w0,
            b_w0,
            q0[0:3],
            v_free0,
            M_inv0,
            self.dt,
            self.restitution,
            self.contact_eps,
            self.mu_fric,
            self.e_t,
            self.e_o,
            self.e_r,
            self.ecp_xy_reg,
            a0_xy,
        ).block_until_ready()

        # jit warmup frozen residual/jac
        a_fix0 = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)
        gap0 = jnp.array(0.0, dtype=jnp.float64)

        mcp_residual_step3B_prox_fixed_a(
            self.z_guess_frozen,
            q0[0:3],
            v_free0,
            M_inv0,
            self.dt,
            self.restitution,
            self.contact_eps,
            self.mu_fric,
            self.e_t,
            self.e_o,
            self.e_r,
            a_fix0,
            gap0,
        ).block_until_ready()

        mcp_jacobian_step3B_fixed_a(
            self.z_guess_frozen,
            q0[0:3],
            v_free0,
            M_inv0,
            self.dt,
            self.restitution,
            self.contact_eps,
            self.mu_fric,
            self.e_t,
            self.e_o,
            self.e_r,
            a_fix0,
            gap0,
        ).block_until_ready()

    # -----------------------------
    # main solve
    # -----------------------------
    def solve_step(self, q_curr_np, v_curr_np, f_applied_np, step_idx, return_ecp=False, tool_impulse_norm: float = 0.0):
        """
        tool_impulse_norm:
          - offline (no tool): keep default 0.0
          - realtime tool mode: pass np.linalg.norm(p_lin) from previous tool solve step to wake immediately
        """

        
        q = jnp.array(q_curr_np, dtype=jnp.float64)
        v = jnp.array(v_curr_np, dtype=jnp.float64)
        f = jnp.array(f_applied_np, dtype=jnp.float64)

        M_inv = mass_matrix_inv_6x6(self.mass, self.inertia_body_diag, q[3:7])
        v_free = v + (M_inv @ (self.dt * f))

        R = np.asarray(quat_to_R_wxyz(q[3:7]))
        A_w_np = (self.A_body @ R.T).astype(np.float64)
        b_w_np = (self.b_body + A_w_np @ np.asarray(q_curr_np[0:3], dtype=np.float64)).astype(np.float64)

        A_w = jnp.array(A_w_np, dtype=jnp.float64)
        b_w = jnp.array(b_w_np, dtype=jnp.float64)
        com = q[0:3]

        Vw_all = np.asarray(get_world_points(q, self.local_pts_all))
        zmin = float(Vw_all[:, 2].min())

        v_free_np = np.asarray(v_free, dtype=np.float64)
        v_lin = v_free_np[0:3]
        w_np = v_free_np[3:6]
        com_np = np.asarray(q_curr_np[0:3], dtype=np.float64)

        r_all = Vw_all - com_np[None, :]
        v_all = v_lin[None, :] + np.cross(w_np[None, :], r_all)
        z_pred_min = float(np.min(Vw_all[:, 2] + self.dt * v_all[:, 2]))

        # Early exit: clearly no contact
        if (zmin > self.ground_enable_margin) and (z_pred_min > 0.0):
            v_next = np.asarray(v_free, dtype=np.float64)
            self.z_guess[6:9] = 0.0
            self.z_guess[12] = 0.0
            self.z_guess[13:] = 0.0
            a_guess = Vw_all[int(np.argmin(Vw_all[:, 2]))].astype(np.float64)
            self.z_guess[9:12] = a_guess
            self.ecp_prev = a_guess.copy()

            # airborne -> definitely not frozen
            self._frozen = False
            self._freeze_count = 0

            info = 0
            if return_ecp:
                return v_next, self.ecp_prev.copy(), info
            return v_next, None, info

        # decide freeze state (your final logic)
        frozen_now = self._update_freeze_state(
            v_curr_np=v_curr_np,
            f_applied_np=f_applied_np,
            zmin=zmin,
            z_pred_min=z_pred_min,
            step_idx=step_idx,
            tool_impulse_norm=float(tool_impulse_norm),
        )

        # ============================================================
        # FROZEN branch: solve reduced MCP with fixed a
        # ============================================================
        if frozen_now:
            a_fix_np, gap_curr = self._compute_a_frozen(q_curr_np, Vw_all)
            a_fix = jnp.array(a_fix_np, dtype=jnp.float64)
            gap_j = jnp.array(float(gap_curr), dtype=jnp.float64)

            def call_F_frozen(n, z, w_out):
                w_out[:] = np.asarray(
                    mcp_residual_step3B_prox_fixed_a(
                        z,
                        com,
                        v_free,
                        M_inv,
                        self.dt,
                        self.restitution,
                        self.contact_eps,
                        self.mu_fric,
                        self.e_t,
                        self.e_o,
                        self.e_r,
                        a_fix,
                        gap_j,
                    )
                )

            def call_Jac_frozen(n, z, J_out):
                J_out[:] = np.asarray(
                    mcp_jacobian_step3B_fixed_a(
                        z,
                        com,
                        v_free,
                        M_inv,
                        self.dt,
                        self.restitution,
                        self.contact_eps,
                        self.mu_fric,
                        self.e_t,
                        self.e_o,
                        self.e_r,
                        a_fix,
                        gap_j,
                    )
                )
                if self.jac_reg > 0.0:
                    np.fill_diagonal(J_out, np.diag(J_out) + self.jac_reg)

            problem_frozen = MCP(self.n1_frozen, self.n2_frozen, call_F_frozen, call_Jac_frozen)

            # warm-start frozen
            self.z_guess_frozen[0:6] = np.asarray(v_curr_np, dtype=np.float64)
            self.z_guess_frozen[6:9] *= 0.9
            self.z_guess_frozen[9] = max(float(self.z_guess_frozen[9]), 1e-6)

            info = mcp_newton_FB_FBLSA(problem_frozen, self.z_guess_frozen, self.w_sol_frozen, self.options)

            if info != 0:
                # fallback: if frozen solve fails, wake and do full solve next step
                if self.ecp_freeze_debug:
                    print(f"[ECP-FREEZE] step {step_idx}: frozen solve FAIL -> wake")
                self._frozen = False
                self._freeze_count = 0
                # simple fallback velocity
                v_next = np.asarray(v_free, dtype=np.float64)
                self.ecp_prev = a_fix_np.copy()
                if return_ecp:
                    return v_next, self.ecp_prev.copy(), info
                return v_next, None, info

            v_next = self.z_guess_frozen[0:6].copy()
            if frozen_now and (self._v_eff(v_next) < self.ecp_freeze_v_sleep):
                v_next[3] *= 0.0
                v_next[4] *= 0.0
            # pn_ground is at index 9 in frozen vector
            # pn_ground = float(self.z_guess_frozen[9])

            # ECP output in frozen mode = fixed a (COM proj)
            ecp_used = a_fix_np.copy()
            self.ecp_prev = ecp_used.copy()

            if return_ecp:
                return v_next, self.ecp_prev.copy(), 0
            return v_next, None, 0

        # ============================================================
        # ACTIVE branch: solve full MCP (original)
        # ============================================================
        a_guess = Vw_all[int(np.argmin(Vw_all[:, 2]))].astype(np.float64)
        a0_xy = a_guess[0:2].copy()

        # keep your warm-start blending
        if self.ecp_prev is not None:
            a_guess[0:2] = 0.85 * a_guess[0:2] + 0.15 * self.ecp_prev[0:2]

        a0_xy_j = jnp.array(a0_xy, dtype=jnp.float64)

        def call_F(n, z, w_out):
            w_out[:] = np.asarray(
                mcp_residual_step3B_prox_hull(
                    z,
                    A_w,
                    b_w,
                    com,
                    v_free,
                    M_inv,
                    self.dt,
                    self.restitution,
                    self.contact_eps,
                    self.mu_fric,
                    self.e_t,
                    self.e_o,
                    self.e_r,
                    self.ecp_xy_reg,
                    a0_xy_j,
                )
            )

        def call_Jac(n, z, J_out):
            J_out[:] = np.asarray(
                mcp_jacobian_step3B_hull(
                    z,
                    A_w,
                    b_w,
                    com,
                    v_free,
                    M_inv,
                    self.dt,
                    self.restitution,
                    self.contact_eps,
                    self.mu_fric,
                    self.e_t,
                    self.e_o,
                    self.e_r,
                    self.ecp_xy_reg,
                    a0_xy_j,
                )
            )
            if self.jac_reg > 0.0:
                np.fill_diagonal(J_out, np.diag(J_out) + self.jac_reg)

        problem = MCP(self.n1, self.n2, call_F, call_Jac)

        # warm-start full
        self.z_guess[0:6] = np.asarray(v_curr_np, dtype=np.float64)
        self.z_guess[6:9] *= 0.9
        self.z_guess[9:12] = a_guess
        self.z_guess[12] = max(float(self.z_guess[12]), 1e-6)
        self.z_guess[13:] *= 0.5

        info = mcp_newton_FB_FBLSA(problem, self.z_guess, self.w_sol, self.options)

        if info != 0:
            v_next = np.array(v_curr_np, dtype=np.float64)
            v_next[0:3] = np.array(v_free[0:3])
            if v_next[2] < 0.0:
                v_next[2] = 0.0
            if return_ecp:
                return v_next, self.ecp_prev.copy(), info
            return v_next, None, info

        v_next = self.z_guess[0:6].copy()
        ecp_raw = self.z_guess[9:12].copy()

        self.ecp_prev = np.asarray(ecp_raw, dtype=np.float64).copy()

        if return_ecp:
            return v_next, self.ecp_prev.copy(), 0
        return v_next, None, 0



# ===========================
# Anti-bounce projection / damping (FIXED)
# ===========================
def project_to_ground_and_damp(
    q_np,
    v_np,
    dt,
    contact_tol=5e-5,
    lin_damp_contact=0.02,
    ang_damp_contact=0.02,
    vz_sleep=1e-3,
    pn_ground=0.0,
    pn_support_thresh=0.0,
):
    """
    关键修复：
      - penetration correction: min_z < 0 -> lift and enforce vz >= 0
      - supported(contact) 时：
          * 只 damp v_xy
          * 只 damp omega_z
        绝不 damp omega_x/omega_y，否则会表现为“扶正/回弹到初始姿态”
    """
    qj = jnp.array(q_np, dtype=jnp.float64)
    Vw_all = np.asarray(get_world_points(qj, local_points_ref))
    min_z = float(np.min(Vw_all[:, 2]))

    supported = (min_z <= float(contact_tol)) and (float(pn_ground) > float(pn_support_thresh))

    if min_z < 0.0:
        q_np[2] -= min_z
        if v_np[2] < 0.0:
            v_np[2] = 0.0

    if supported:
        v_np[0:2] *= (1.0 - float(lin_damp_contact))
        v_np[5] *= (1.0 - float(ang_damp_contact))
        if abs(v_np[2]) < float(vz_sleep):
            v_np[2] = 0.0

    return q_np, v_np, min_z, supported


# ===========================
# Simulation
# ===========================

def run_simulation(
    dt=0.01,
    steps=1200,
    view=False,
    restitution=0.10,
    proj_tol=1e-6,
    ground_enable_margin=2e-3,
    contact_eps=1e-6,
    ecp_xy_reg=1e-2,
    jac_reg=1e-8,
    use_tool=False,
    tool_mass=1.0,
    tool_radius=0.03,
    tool_mu=0.6,
    tool_k=800.0,
    tool_d=80.0,
    tool_fmax=200.0,
    tool_pos0=(0.08, -0.35, 0.08),
    tool_des0=(0.08, -0.35, 0.08),
    tool_des_vel=(0.0, 0.25, 0.0),
    tool_tstart=1.2,
    tool_restitution=0.0,
    tool_contact_eps=1e-6,
    tool_enable_margin=2e-4,
    tool_mouse=False,
    mouse_sensitivity=0.002,
    mouse_z_step=0.01,
    tool_mocap=False,
    mocap_body="target_mocap",
):
    sim = TBlockSimulator_Step_NoBounce(
        dt=dt,
        mass=total_mass,
        inertia_diag_body=inertia_body_diag,
        local_pts_all=local_points_ref,
        Kmax=12,
        support_eps=1.5e-3,
        alpha_sigma=0.06,
        alpha_rho=1e-3,
        alpha_com_blend=0.25,
        mu_fric=0.5,
        e_t=1.0,
        e_o=1.0,
        e_r=None,
        e_r_factor=0.10,
        restitution=restitution,
        contact_eps=contact_eps,
        proj_tol=proj_tol,
        ground_enable_margin=ground_enable_margin,
    )
    sim.ecp_xy_reg = float(ecp_xy_reg)
    sim.jac_reg = float(jac_reg)

    # ------------------------------------------------------------
    # NEW: only for project_to_ground_and_damp() "supported" gate
    # (this is NOT your freeze logic; it's just damping gating)
    # ------------------------------------------------------------
    CONTACT_PN_SUPPORT_FACTOR = 0.05  # tune if needed
    pn_support_thresh = float(CONTACT_PN_SUPPORT_FACTOR * total_mass * 9.81 * dt)

    print(f"total mass = {total_mass}, inertia_body_diag = {inertia_body_diag}")
    print(f"[STEP3B-LS-PROX + TOOL-IMP] Start: steps={steps}, dt={dt}, use_tool={use_tool}")
    try:
        print(
            f"[ECP-FREEZE] enable={bool(ECP_FREEZE_ENABLE)}, dwell_n={int(ECP_FREEZE_DWELL_N)}, "
            f"v_sleep={float(ECP_FREEZE_V_SLEEP)}, v_wake={float(ECP_FREEZE_V_WAKE)}"
        )
    except Exception:
        pass

    q = np.array([0, 0, 0.4, 0.9239, 0.3827, 0, 0], dtype=np.float64)
    q[3:] /= np.linalg.norm(q[3:])
    v = np.zeros(6, dtype=np.float64)

    tool_pos = np.array(tool_pos0, dtype=np.float64).reshape(3,)
    tool_vel = np.zeros(3, dtype=np.float64)
    compute_tool_block_impulse.tool_mass = float(tool_mass)

    if view and (tool_mouse or tool_mocap):
        return run_simulation_realtime(
            dt=dt,
            steps=steps,
            restitution=restitution,
            proj_tol=proj_tol,
            ground_enable_margin=ground_enable_margin,
            contact_eps=contact_eps,
            ecp_xy_reg=ecp_xy_reg,
            jac_reg=jac_reg,
            use_tool=use_tool,
            tool_mass=tool_mass,
            tool_radius=tool_radius,
            tool_mu=tool_mu,
            tool_k=tool_k,
            tool_d=tool_d,
            tool_fmax=tool_fmax,
            tool_pos0=tool_pos0,
            tool_des0=tool_des0,
            tool_restitution=tool_restitution,
            tool_contact_eps=tool_contact_eps,
            tool_enable_margin=tool_enable_margin,
            mouse_sensitivity=mouse_sensitivity,
            mouse_z_step=mouse_z_step,
            tool_mocap=tool_mocap,
            mocap_body=mocap_body,
        )

    traj_q = np.zeros((steps, 7), dtype=np.float64)
    ecp_hist = np.full((steps, 3), np.nan, dtype=np.float64)
    info_hist = np.zeros((steps,), dtype=np.int32)
    minz_hist = np.zeros((steps,), dtype=np.float64)
    contact_hist = np.zeros((steps,), dtype=np.int32)

    tool_pos_hist = np.zeros((steps, 3), dtype=np.float64)
    tool_vel_hist = np.zeros((steps, 3), dtype=np.float64)
    tool_contact_hist = np.zeros((steps,), dtype=np.int32)
    tool_contact_pt_hist = np.full((steps, 3), np.nan, dtype=np.float64)

    force_history = []

    # NEW: tool impulse memory for wake gate (one-step delay, same as realtime)
    last_tool_impulse_norm = 0.0

    t0 = time.perf_counter()

    for i in range(steps):
        t = i * dt

        f_ext = np.zeros(6, dtype=np.float64)
        f_ext[2] += -9.81 * total_mass

        # Optional user wrench schedule (default is 0 force; keep torque 0 for pure drop)
        t_start = 1.0
        duration = 5.0
        period = 0.2

        p0 = np.array([0.08, -0.3, 0.01], dtype=np.float64)
        dp = np.array([0.0, 0.002, 0.0], dtype=np.float64)
        F_const = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        tau_const = np.array([2000.0, 0.0, 0.0], dtype=np.float64)

        attempted = False
        p_world = None
        F_world = None

        if t_start <= t < t_start + duration:
            k = max(int(round(period / dt)), 1)
            if ((i - int(round(t_start / dt))) % k) == 0:
                attempted = True
                n = (i - int(round(t_start / dt))) // k
                p_world = p0 + n * dp
                F_world = F_const
                p_world = add_world_wrench_projected(f_ext, p_world, F_world, tau_const, q)

        if attempted:
            force_history.append({"step": i, "pos": np.array(p_world), "vec": np.array(F_world)})

        # 1) Ground solve (your NEW freeze logic lives inside solve_step)
        v_ground, ecp, info = sim.solve_step(
            q, v, f_ext, i,
            return_ecp=True,
            tool_impulse_norm=float(last_tool_impulse_norm),
        )

        # pn_ground: frozen vs active MCP
        if getattr(sim, "_frozen", False) and hasattr(sim, "z_guess_frozen"):
            pn_ground = float(sim.z_guess_frozen[9])
        else:
            pn_ground = float(sim.z_guess[12]) if hasattr(sim, "z_guess") else 0.0

        info_hist[i] = int(info)
        ecp_hist[i] = ecp

        # 2) Tool impedance free motion
        if use_tool:
            x_des0 = np.array(tool_des0, dtype=np.float64).reshape(3,)
            v_des0 = np.array(tool_des_vel, dtype=np.float64).reshape(3,)

            if t < tool_tstart:
                x_des = x_des0
                v_des = np.zeros(3, dtype=np.float64)
            else:
                x_des = x_des0 + v_des0 * (t - tool_tstart)
                v_des = v_des0

            K = float(tool_k)
            D = float(tool_d)
            F_cmd = K * (x_des - tool_pos) + D * (v_des - tool_vel)

            fmax = float(tool_fmax)
            fn = float(np.linalg.norm(F_cmd))
            if fn > fmax and fn > 1e-12:
                F_cmd *= (fmax / fn)

            tool_vel_free = tool_vel + dt * (F_cmd / float(tool_mass))
        else:
            tool_vel_free = tool_vel.copy()

        # 3) Tool-block impulse (split)
        v_next = v_ground.copy()
        tool_vel_next = tool_vel_free.copy()

        p_lin = np.zeros(3, dtype=np.float64)
        a_c = None

        if use_tool:
            p_lin, a_c, n_c, gap = compute_tool_block_impulse(
                q_block=q,
                v_block6=v_ground,
                tool_pos=tool_pos,
                tool_vel=tool_vel_free,
                tool_radius=float(tool_radius),
                tool_mu=float(tool_mu),
                dt=float(dt),
                contact_eps=float(tool_contact_eps),
                restitution=float(tool_restitution),
                enable_margin=float(tool_enable_margin),
            )

            pnorm = float(np.linalg.norm(p_lin))
            last_tool_impulse_norm = pnorm  # for NEXT step wake gate

            if pnorm > 0.0:
                com = q[:3]
                r = a_c - com
                Iw = inertia_world_from_body_diag(inertia_body_diag, q[3:7])
                Iw_inv = np.linalg.inv(Iw + 1e-12 * np.eye(3))

                v_next[0:3] = v_ground[0:3] + p_lin / float(total_mass)
                v_next[3:6] = v_ground[3:6] + (Iw_inv @ np.cross(r, p_lin))

                tool_vel_next = tool_vel_free - p_lin / float(tool_mass)

                tool_contact_hist[i] = 1
                tool_contact_pt_hist[i] = a_c
        else:
            last_tool_impulse_norm = 0.0

        # 4) Integrate block & tool
        pos_next = q[:3] + v_next[:3] * dt
        omega_world = v_next[3:6]
        dq = quat_from_omega_world_np(omega_world, dt)
        quat_next = quat_mul_wxyz_np(dq, q[3:7])
        quat_next /= (np.linalg.norm(quat_next) + 1e-12)

        q = np.concatenate([pos_next, quat_next])
        v = v_next

        tool_pos = tool_pos + tool_vel_next * dt
        tool_vel = tool_vel_next

        # 5) Post-fix (anti-penetration + contact damping WITHOUT self-righting)
        q, v, min_z, in_contact = project_to_ground_and_damp(
            q,
            v,
            dt,
            contact_tol=proj_tol,
            lin_damp_contact=0.02,
            ang_damp_contact=0.02,
            vz_sleep=1e-3,
            pn_ground=pn_ground,
            pn_support_thresh=pn_support_thresh,
        )
        minz_hist[i] = min_z
        contact_hist[i] = 1 if in_contact else 0

        traj_q[i] = q
        tool_pos_hist[i] = tool_pos
        tool_vel_hist[i] = tool_vel

        if i % 80 == 0:
            fails = int(np.sum(info_hist[: i + 1] != 0))
            print(
                f"step {i:4d}: info={info}, fails={fails}, min_z={min_z: .6e}, "
                f"tool_contact={int(tool_contact_hist[i])}, "
                f"ecp=({ecp_hist[i,0]: .3f},{ecp_hist[i,1]: .3f},{ecp_hist[i,2]: .6f}), "
                f"pn={pn_ground: .3e}, tool_imp={last_tool_impulse_norm:.3e}, frozen={int(getattr(sim,'_frozen',False))}"
            )

    t1 = time.perf_counter()
    print(f"[STEP3B-LS-PROX + TOOL-IMP] Done in {t1 - t0:.3f}s, total FAIL steps={int(np.sum(info_hist!=0))}")

    if view:
        visualize(
            traj_q,
            ecp_hist,
            force_history,
            dt,
            tool_traj=tool_pos_hist if use_tool else None,
            tool_radius=float(tool_radius),
            tool_contact_pt_hist=tool_contact_pt_hist,
        )

    return traj_q, ecp_hist, info_hist, minz_hist, contact_hist, tool_pos_hist, tool_vel_hist, tool_contact_hist


# ===========================
# Real-time simulation (viewer)
# ===========================
def run_simulation_realtime(
    dt=0.002,
    steps=1200,
    restitution=0.10,
    proj_tol=1e-6,
    ground_enable_margin=2e-3,
    contact_eps=1e-6,
    ecp_xy_reg=1e-2,
    jac_reg=1e-8,
    use_tool=True,
    tool_mass=1.0,
    tool_radius=0.03,
    tool_mu=0.6,
    tool_k=800.0,
    tool_d=80.0,
    tool_fmax=200.0,
    tool_pos0=(0.08, -0.35, 0.08),
    tool_des0=(0.08, -0.35, 0.08),
    tool_restitution=0.0,
    tool_contact_eps=1e-6,
    tool_enable_margin=2e-4,
    mouse_sensitivity=0.002,
    mouse_z_step=0.01,
    tool_mocap=False,
    mocap_body="target_mocap",
):
    sim = TBlockSimulator_Step_NoBounce(
        dt=dt,
        mass=total_mass,
        inertia_diag_body=inertia_body_diag,
        local_pts_all=local_points_ref,
        Kmax=12,
        support_eps=1.5e-3,
        alpha_sigma=0.06,
        alpha_rho=1e-3,
        alpha_com_blend=0.25,
        mu_fric=1,
        e_t=1.0,
        e_o=1.0,
        e_r=None,
        e_r_factor=0.10,
        restitution=restitution,
        contact_eps=contact_eps,
        ground_enable_margin=ground_enable_margin,
        proj_tol=proj_tol,
    )
    sim.ecp_xy_reg = float(ecp_xy_reg)
    sim.jac_reg = float(jac_reg)

    quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # wxyz, upright
    qpos0, com0 = place_body_on_ground_qpos_from_quat(
        quat_wxyz=quat0,
        ipos_body=T_IPOS_BODY,
        local_pts_com=local_points_ref,
        origin_xy=(0.0, 0.0),
        clearance=1e-6,
    )
    q = body_origin_qpos_to_com(qpos0, T_IPOS_BODY)
    q[3:] /= (np.linalg.norm(q[3:]) + 1e-12)
    v = np.zeros(6, dtype=np.float64)

    tool_pos = np.array(tool_pos0, dtype=np.float64).reshape(3,)
    tool_vel = np.zeros(3, dtype=np.float64)
    tool_des = np.array(tool_des0, dtype=np.float64).reshape(3,)
    tool_des_prev = tool_des.copy()

    compute_tool_block_impulse.tool_mass = float(tool_mass)

    model_vis = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model_vis)

    mocap_id = -1
    if tool_mocap:
        try:
            bid = mujoco.mj_name2id(model_vis, mujoco.mjtObj.mjOBJ_BODY, str(mocap_body))
            mocap_id = int(model_vis.body_mocapid[bid])
        except Exception:
            mocap_id = -1
        if mocap_id >= 0:
            data.mocap_pos[mocap_id] = np.array(tool_des0, dtype=np.float64)
            data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            mujoco.mj_forward(model_vis, data)
        else:
            tool_mocap = False

    body_id_vis = mujoco.mj_name2id(model_vis, mujoco.mjtObj.mjOBJ_BODY, "T_siconos")
    jnt_id = mujoco.mj_name2id(model_vis, mujoco.mjtObj.mjOBJ_JOINT, "root_siconos")
    q_adr = int(model_vis.jnt_qposadr[jnt_id])
    ipos_vis = np.array(model_vis.body_ipos[body_id_vis], dtype=np.float64)

    print("T_siconos body_quat =", np.array(model_vis.body_quat[body_id_vis], dtype=np.float64))
    print("T_siconos body_pos  =", np.array(model_vis.body_pos[body_id_vis], dtype=np.float64))

    # ---- FREEZE debug print (instead of old STAB)
    try:
        print(
            f"[ECP-FREEZE] enable={bool(ECP_FREEZE_ENABLE)}, dwell_n={int(ECP_FREEZE_DWELL_N)}, "
            f"v_sleep={float(ECP_FREEZE_V_SLEEP)}, v_wake={float(ECP_FREEZE_V_WAKE)}"
        )
    except Exception:
        pass

    st = {"drag": False, "last": None, "mouse_enabled": True, "paused": False}
    viewer_box = {"viewer": None}

    # -----------------------------
    # Data logger (NEW)
    # -----------------------------
    logger = RealtimeDataLogger(
        out_dir="logs",
        energy_csv="block_energy.csv",
        impulse_csv="tool_impulse.csv",
        flush_every=200,  # optional: flush to disk every N steps
    )

    def _camera_right_up(cam):
        az = np.deg2rad(float(cam.azimuth))
        el = np.deg2rad(float(cam.elevation))
        vec = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], dtype=np.float64)
        forward = -vec
        forward /= (np.linalg.norm(forward) + 1e-12)

        right = np.array([-np.sin(az), np.cos(az), 0.0], dtype=np.float64)
        right /= (np.linalg.norm(right) + 1e-12)

        up = np.cross(right, forward)
        up /= (np.linalg.norm(up) + 1e-12)
        return right, up, forward

    def poll_input_no_callbacks(viewer):
        window = getattr(viewer, "window", None)
        if window is None:
            return
        glfw.poll_events()

        ks = st.setdefault("_key_prev", {})

        def edge(k):
            pressed = (glfw.get_key(window, k) == glfw.PRESS)
            prev = ks.get(k, False)
            ks[k] = pressed
            return pressed and (not prev)

        if edge(glfw.KEY_SPACE):
            st["paused"] = not st["paused"]
        if edge(glfw.KEY_M):
            st["mouse_enabled"] = not st["mouse_enabled"]
            st["last"] = None
        if edge(glfw.KEY_R):
            tool_des[:] = tool_pos
            st["last"] = None

        if not st.get("mouse_enabled", True):
            st["last"] = None
            return

        x, y = glfw.get_cursor_pos(window)
        left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        right_btn = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

        if left or right_btn:
            if st["last"] is None:
                st["last"] = (x, y)
                return
            dx = x - st["last"][0]
            dy = y - st["last"][1]
            st["last"] = (x, y)

            right_vec, up_vec, _ = _camera_right_up(viewer.cam)
            scale = float(mouse_sensitivity) * float(viewer.cam.distance)

            if left:
                delta = scale * (dx * right_vec - dy * up_vec)
                delta[2] = 0.0
                tool_des[:] = tool_des + delta

            if right_btn:
                tool_des[2] = float(tool_des[2]) + (-dy) * (0.5 * scale)

            tool_des[2] = max(0.02, float(tool_des[2]))
        else:
            st["last"] = None

    arrow_width = 0.01
    arrow_length = 0.9
    ecp_radius = 0.012
    des_radius = 0.010

    # ---- IMPORTANT: keep a pn threshold for "supported" damping
    # You can tune the factor (0.05) if needed; this is similar to your old idea.
    pn_support_thresh = float(0.05 * total_mass * 9.81 * dt)

    # ---- NEW: tool impulse memory (wake gate comes from previous step)
    last_tool_impulse_norm = 0.0

    try:
        try:
            callbacks_ok = True
            viewer_ctx = mujoco.viewer.launch_passive(model_vis, data)
        except TypeError:
            callbacks_ok = False
            viewer_ctx = mujoco.viewer.launch_passive(model_vis, data)

        with viewer_ctx as viewer:
            viewer_box["viewer"] = viewer
            scn = viewer.user_scn
            scn.ngeom = 4

            qpos_vis = com_to_body_origin_qpos(q, ipos_vis)
            data.qpos[q_adr: q_adr + 7] = qpos_vis
            mujoco.mj_forward(model_vis, data)
            viewer.sync()

            mujoco.mjv_initGeom(
                scn.geoms[0],
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.array([arrow_width, 1e-6, arrow_width], dtype=np.float64),
                np.array([100.0, 100.0, 100.0], dtype=np.float64),
                np.eye(3, dtype=np.float64).ravel(),
                np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )
            mujoco.mjv_initGeom(
                scn.geoms[1],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([ecp_radius, 0, 0], dtype=np.float64),
                np.array([100.0, 100.0, 100.0], dtype=np.float64),
                np.eye(3, dtype=np.float64).ravel(),
                np.array([0.2, 0.4, 1.0, 1.0], dtype=np.float32),
            )
            mujoco.mjv_initGeom(
                scn.geoms[2],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([float(tool_radius), 0, 0], dtype=np.float64),
                np.array([100.0, 100.0, 100.0], dtype=np.float64),
                np.eye(3, dtype=np.float64).ravel(),
                np.array([0.2, 0.9, 0.2, 1.0], dtype=np.float32),
            )
            mujoco.mjv_initGeom(
                scn.geoms[3],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([des_radius, 0, 0], dtype=np.float64),
                np.array([100.0, 100.0, 100.0], dtype=np.float64),
                np.eye(3, dtype=np.float64).ravel(),
                np.array([1.0, 0.6, 0.1, 1.0], dtype=np.float32),
            )

            i = 0
            last_wall = time.perf_counter()



            while True:
                if hasattr(viewer, "is_running") and (not viewer.is_running()):
                    break
                if i >= int(steps):
                    break

                if not callbacks_ok and (not tool_mocap):
                    poll_input_no_callbacks(viewer)

                if st["paused"]:
                    qpos_vis = com_to_body_origin_qpos(q, ipos_vis)
                    data.qpos[q_adr: q_adr + 7] = qpos_vis
                    mujoco.mj_forward(model_vis, data)
                    scn.geoms[2].pos[:] = tool_pos
                    scn.geoms[3].pos[:] = tool_des
                    viewer.sync()
                    time.sleep(0.01)
                    continue

                # only gravity in f_ext
                f_ext = np.zeros(6, dtype=np.float64)
                f_ext[2] += -9.81 * total_mass

                # ---- CHANGED: pass last_tool_impulse_norm into solve_step
                v_ground, ecp, info = sim.solve_step(
                    q, v, f_ext, i,
                    return_ecp=True,
                    tool_impulse_norm=float(last_tool_impulse_norm),
                )

                # ---- CHANGED: pn_ground depends on frozen/active branch
                if getattr(sim, "_frozen", False) and hasattr(sim, "z_guess_frozen"):
                    pn_ground = float(sim.z_guess_frozen[9])
                else:
                    pn_ground = float(sim.z_guess[12]) if hasattr(sim, "z_guess") else 0.0

                # desired tool velocity (mouse/mocap)
                if tool_mocap and mocap_id >= 0:
                    tool_des[:] = data.mocap_pos[mocap_id]
                    v_des = (tool_des - tool_des_prev) / float(dt)
                    tool_des_prev[:] = tool_des
                else:
                    v_des = np.zeros(3, dtype=np.float64)

                x_des = tool_des.copy()
                K = float(tool_k)
                D = float(tool_d)
                F_cmd = K * (x_des - tool_pos) + D * (v_des - tool_vel)

                fmax = float(tool_fmax)
                fn = float(np.linalg.norm(F_cmd))
                if fn > fmax and fn > 1e-12:
                    F_cmd *= (fmax / fn)

                tool_vel_free = tool_vel + dt * (F_cmd / float(tool_mass))

                v_next = v_ground.copy()
                tool_vel_next = tool_vel_free.copy()
                p_lin = np.zeros(3, dtype=np.float64)
                a_c = None

                # tool-block impulse
                p_lin, a_c, n_c, gap = compute_tool_block_impulse(
                    q_block=q,
                    v_block6=v_ground,
                    tool_pos=tool_pos,
                    tool_vel=tool_vel_free,
                    tool_radius=float(tool_radius),
                    tool_mu=float(tool_mu),
                    dt=float(dt),
                    contact_eps=float(tool_contact_eps),
                    restitution=float(tool_restitution),
                    enable_margin=float(tool_enable_margin),
                )

                # ---- CHANGED: update last_tool_impulse_norm for NEXT step wake gate
                last_tool_impulse_norm = float(np.linalg.norm(p_lin))

                # -----------------------------
                # Log tool impulse (NEW)
                # -----------------------------
                t_sim = float(i) * float(dt)
                logger.log_tool_impulse(
                    step=i,
                    t=t_sim,
                    p_lin=p_lin,
                    contact_pt=a_c,
                    normal=n_c,
                    gap=gap,
                    tool_pos=tool_pos,
                    tool_vel=tool_vel_free,  # or tool_vel (depending what you want)
                )
                # -----------------------------

                if last_tool_impulse_norm > 0.0:
                    com = q[:3]
                    r = a_c - com
                    Iw = inertia_world_from_body_diag(inertia_body_diag, q[3:7])
                    Iw_inv = np.linalg.inv(Iw + 1e-12 * np.eye(3))

                    v_next[0:3] = v_ground[0:3] + p_lin / float(total_mass)
                    v_next[3:6] = v_ground[3:6] + (Iw_inv @ np.cross(r, p_lin))
                    tool_vel_next = tool_vel_free - p_lin / float(tool_mass)

                # integrate
                pos_next = q[:3] + v_next[:3] * dt
                omega_world = v_next[3:6]
                dq = quat_from_omega_world_np(omega_world, dt)
                quat_next = quat_mul_wxyz_np(dq, q[3:7])
                quat_next /= (np.linalg.norm(quat_next) + 1e-12)

                q_next = np.hstack([pos_next, quat_next])
                v_next2 = v_next.copy()

                tool_pos_next = tool_pos + tool_vel_next * dt
                tool_vel_next2 = tool_vel_next.copy()

                # post-fix
                q_next, v_next2, min_z, in_contact = project_to_ground_and_damp(
                    q_next,
                    v_next2,
                    dt,
                    contact_tol=float(proj_tol),
                    lin_damp_contact=0.02,
                    ang_damp_contact=0.02,
                    vz_sleep=1e-3,
                    pn_ground=pn_ground,
                    pn_support_thresh=pn_support_thresh,
                )

                # commit
                q = q_next
                v = v_next2
                tool_pos = tool_pos_next
                tool_vel = tool_vel_next2

                # -----------------------------
                # Log block energies (NEW)
                # -----------------------------
                t_sim = float(i) * float(dt)
                logger.log_block_energy(
                    step=i,
                    t=t_sim,
                    q_block=q,             # COM state after commit
                    v_block6=v,            # 6D vel after commit
                    mass=float(total_mass),
                    inertia_body_diag=inertia_body_diag,
                    frozen_flag=int(getattr(sim, "_frozen", False)),
                    g=9.81,
                    z0=0.0,
                )


                # render
                qpos_vis = com_to_body_origin_qpos(q, ipos_vis)
                data.qpos[q_adr: q_adr + 7] = qpos_vis
                mujoco.mj_forward(model_vis, data)

                if np.all(np.isfinite(ecp)):
                    scn.geoms[1].pos[:] = np.asarray(ecp, dtype=np.float64)
                else:
                    scn.geoms[1].pos[:] = np.array([100.0, 100.0, 100.0], dtype=np.float64)

                scn.geoms[2].pos[:] = tool_pos
                scn.geoms[3].pos[:] = tool_des

                if a_c is not None and last_tool_impulse_norm > 0.0:
                    vec = (p_lin / float(dt))
                    mag = float(np.linalg.norm(vec))
                    if mag < 1e-10:
                        scn.geoms[0].pos[:] = np.array([100.0, 100.0, 100.0], dtype=np.float64)
                    else:
                        dvec = vec / mag
                        frm = np.asarray(a_c, dtype=np.float64)
                        to = frm + dvec * arrow_length
                        mujoco.mjv_connector(scn.geoms[0], mujoco.mjtGeom.mjGEOM_ARROW, arrow_width, frm, to)
                else:
                    scn.geoms[0].pos[:] = np.array([100.0, 100.0, 100.0], dtype=np.float64)

                viewer.sync()

                # realtime pacing
                now = time.perf_counter()
                elapsed = now - last_wall
                sleep_t = dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
                last_wall = time.perf_counter()

                i += 1

    except Exception as e:
        print("[Realtime] Viewer error:", e)

    finally:
        # -----------------------------
        # Ensure logs are written (NEW)
        # -----------------------------
        try:
            logger.close()
            print(f"[Logger] wrote: {str(logger.energy_path)}")
            print(f"[Logger] wrote: {str(logger.impulse_path)}")
        except Exception as _e:
            print("[Logger] close failed:", _e)


    return



# ===========================
# Visualization (offline)
# ===========================
def visualize(traj_q, ecp_hist, force_history, dt, tool_traj=None, tool_radius=0.03, tool_contact_pt_hist=None):
    model_vis = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model_vis)

    body_id_vis = mujoco.mj_name2id(model_vis, mujoco.mjtObj.mjOBJ_BODY, "T_siconos")
    q_adr = model_vis.jnt_qposadr[model_vis.body_jntadr[body_id_vis]]
    ipos_vis = np.array(model_vis.body_ipos[body_id_vis], dtype=np.float64)
    qpos_vis0 = com_to_body_origin_qpos(traj_q[0], ipos_vis)
    data.qpos[q_adr: q_adr + 7] = qpos_vis0
    mujoco.mj_forward(model_vis, data)

    force_map = {item["step"]: item for item in sorted(force_history, key=lambda d: d["step"])}

    arrow_width = 0.01
    arrow_length = 0.9
    ecp_radius = 0.012

    try:
        with mujoco.viewer.launch_passive(model_vis, data) as viewer:
            scn = viewer.user_scn
            scn.ngeom = 3

            mujoco.mjv_initGeom(
                scn.geoms[0],
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.array([arrow_width, 1e-6, arrow_width], dtype=np.float64),
                np.array([100.0, 100.0, 100.0], dtype=np.float64),
                np.eye(3, dtype=np.float64).ravel(),
                np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )
            mujoco.mjv_initGeom(
                scn.geoms[1],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([ecp_radius, 0, 0], dtype=np.float64),
                np.array([100.0, 100.0, 100.0], dtype=np.float64),
                np.eye(3, dtype=np.float64).ravel(),
                np.array([0.2, 0.4, 1.0, 1.0], dtype=np.float32),
            )
            mujoco.mjv_initGeom(
                scn.geoms[2],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([float(tool_radius), 0, 0], dtype=np.float64),
                np.array([100.0, 100.0, 100.0], dtype=np.float64),
                np.eye(3, dtype=np.float64).ravel(),
                np.array([0.2, 0.9, 0.2, 1.0], dtype=np.float32),
            )

            for i, qpos in enumerate(traj_q):
                qpos_vis = com_to_body_origin_qpos(qpos, np.array(model_vis.body_ipos[body_id_vis], dtype=np.float64))
                data.qpos[q_adr: q_adr + 7] = qpos_vis
                mujoco.mj_forward(model_vis, data)

                if np.all(np.isfinite(ecp_hist[i])):
                    scn.geoms[1].pos[:] = ecp_hist[i]
                else:
                    scn.geoms[1].pos[:] = np.array([100.0, 100.0, 100.0], dtype=np.float64)

                if tool_traj is not None:
                    scn.geoms[2].pos[:] = np.asarray(tool_traj[i], dtype=np.float64)
                else:
                    scn.geoms[2].pos[:] = np.array([100.0, 100.0, 100.0], dtype=np.float64)

                if i in force_map:
                    pos = np.asarray(force_map[i]["pos"], dtype=np.float64)
                    vec = np.asarray(force_map[i]["vec"], dtype=np.float64)
                    mag = float(np.linalg.norm(vec))
                    if mag < 1e-10:
                        scn.geoms[0].pos[:] = np.array([100.0, 100.0, 100.0], dtype=np.float64)
                    else:
                        d = vec / mag
                        frm = pos
                        to = pos + d * arrow_length
                        mujoco.mjv_connector(scn.geoms[0], mujoco.mjtGeom.mjGEOM_ARROW, arrow_width, frm, to)
                else:
                    scn.geoms[0].pos[:] = np.array([100.0, 100.0, 100.0], dtype=np.float64)

                viewer.sync()
                time.sleep(dt * 4)
    except Exception as e:
        print("[Viewer] failed:", e)
        print("Tip: try `export MUJOCO_GL=egl` or `export MUJOCO_GL=osmesa`, or run without --view.")


def _parse_vec3(s: str):
    parts = [p.strip() for p in str(s).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated floats, got: {s}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=500000)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--restitution", type=float, default=0.10)
    ap.add_argument("--proj_tol", type=float, default=1e-6)
    ap.add_argument("--ground_enable_margin", type=float, default=2e-3)
    ap.add_argument("--contact_eps", type=float, default=1e-6)
    ap.add_argument("--ecp_xy_reg", type=float, default=1e-2)
    ap.add_argument("--jac_reg", type=float, default=1e-8)

    ap.add_argument("--use_tool", action="store_true")
    ap.add_argument("--tool_mass", type=float, default=1.0)
    ap.add_argument("--tool_radius", type=float, default=0.05)
    ap.add_argument("--tool_mu", type=float, default=0.6)
    ap.add_argument("--tool_k", type=float, default=800.0)
    ap.add_argument("--tool_d", type=float, default=80.0)
    ap.add_argument("--tool_fmax", type=float, default=10.0)
    ap.add_argument("--tool_pos0", type=str, default="0.08,-0.35,0.08")
    ap.add_argument("--tool_des0", type=str, default="0.08,-0.35,0.08")
    ap.add_argument("--tool_des_vel", type=str, default="0.0,0.25,0.0")
    ap.add_argument("--tool_tstart", type=float, default=1.2)
    ap.add_argument("--tool_restitution", type=float, default=0.0)
    ap.add_argument("--tool_contact_eps", type=float, default=1e-6)
    ap.add_argument("--tool_enable_margin", type=float, default=2e-4)

    ap.add_argument("--tool_mouse", action="store_true")
    ap.add_argument("--mouse_sensitivity", type=float, default=0.002)
    ap.add_argument("--mouse_z_step", type=float, default=0.01)

    ap.add_argument("--tool_mocap", action="store_true")
    ap.add_argument("--mocap_body", type=str, default="target_mocap")

    args = ap.parse_args()

    tool_pos0 = _parse_vec3(args.tool_pos0)
    tool_des0 = _parse_vec3(args.tool_des0)
    tool_des_vel = _parse_vec3(args.tool_des_vel)

    run_simulation(
        dt=args.dt,
        steps=args.steps,
        view=args.view,
        restitution=args.restitution,
        proj_tol=args.proj_tol,
        ground_enable_margin=args.ground_enable_margin,
        contact_eps=args.contact_eps,
        ecp_xy_reg=args.ecp_xy_reg,
        jac_reg=args.jac_reg,
        use_tool=args.use_tool,
        tool_mass=args.tool_mass,
        tool_radius=args.tool_radius,
        tool_mu=args.tool_mu,
        tool_k=args.tool_k,
        tool_d=args.tool_d,
        tool_fmax=args.tool_fmax,
        tool_pos0=tool_pos0,
        tool_des0=tool_des0,
        tool_des_vel=tool_des_vel,
        tool_tstart=args.tool_tstart,
        tool_restitution=args.tool_restitution,
        tool_contact_eps=args.tool_contact_eps,
        tool_enable_margin=args.tool_enable_margin,
        tool_mouse=args.tool_mouse,
        mouse_sensitivity=args.mouse_sensitivity,
        mouse_z_step=args.mouse_z_step,
        tool_mocap=args.tool_mocap,
        mocap_body=args.mocap_body,
    )