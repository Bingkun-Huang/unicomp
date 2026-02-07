from __future__ import annotations

import numpy as np
import mujoco

import siconos.numerics as sn
from siconos.numerics import MCP, mcp_newton_FB_FBLSA

from . import state
from .math3d import (
    rotate_world_to_local_np,
    rotate_vector_by_quaternion_np,
    quat_conj_np,
    quat_to_R_np_wxyz,
    skew_np,
)

_CP_LAST_GID: int | None = None
_CP_TIE_EPS: float = 2e-5  


def _box_sdf_and_surface_point(p_g: np.ndarray, half_ext: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Signed distance for axis-aligned box centered at origin with half extents b.
    Returns (sdf, p_surf) in the SAME frame as p_g (box-local).
      sdf > 0 : outside
      sdf < 0 : inside (penetrating)
    p_surf is closest point on the box surface (clamped/outside or nearest face inside).
    """
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


def closest_point_on_tblock_surface_world(p_world: np.ndarray, q_curr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    v081-style closest point on the union of T-block box geoms (in state.*).
    Returns:
      p_surf_world : closest surface point (world)
      sdf          : signed distance to surface in geom-local sense (same as v081 usage)
    Key addition vs naive:
      - tie-break: if multiple geoms yield similar |sdf| (within _CP_TIE_EPS),
        reuse last chosen geom id to avoid jitter near edges/seams.
    """
    global _CP_LAST_GID

    state.ensure_initialized()
    pos_com = np.asarray(q_curr[:3], dtype=np.float64)
    q_body = np.asarray(q_curr[3:7], dtype=np.float64)
    p_body = rotate_world_to_local_np(np.asarray(p_world, dtype=np.float64) - pos_com, q_body)

    best_abs = 1e18
    candidates: list[tuple[int, float, np.ndarray]] = []  # (gid, sdf, p_body_surf)

    # Enumerate geoms in state (gid is index into those arrays)
    for gid, (gt, gp, gq, gs) in enumerate(zip(state.T_GEOM_TYPE, state.T_GEOM_POS, state.T_GEOM_QUAT, state.T_GEOM_SIZE)):
        if int(gt) != mujoco.mjtGeom.mjGEOM_BOX:
            continue

        # point into this geom local
        p_g = rotate_vector_by_quaternion_np(p_body - gp, quat_conj_np(gq))
        sdf, p_g_surf = _box_sdf_and_surface_point(p_g, gs[:3])
        abs_sdf = abs(float(sdf))

        # Update candidate set with tie-break epsilon
        if abs_sdf < best_abs - _CP_TIE_EPS:
            best_abs = abs_sdf
            p_body_surf = rotate_vector_by_quaternion_np(p_g_surf, gq) + gp
            candidates = [(gid, float(sdf), p_body_surf)]
        elif abs(abs_sdf - best_abs) <= _CP_TIE_EPS:
            p_body_surf = rotate_vector_by_quaternion_np(p_g_surf, gq) + gp
            candidates.append((gid, float(sdf), p_body_surf))

    if not candidates:
        return np.asarray(p_world, dtype=np.float64).copy(), 1e18

    # Choose with v081-style preference: reuse last gid if still in candidates
    chosen = candidates[0]
    if _CP_LAST_GID is not None:
        for c in candidates:
            if c[0] == _CP_LAST_GID:
                chosen = c
                break

    _CP_LAST_GID = chosen[0]
    best_sdf = float(chosen[1])
    best_p_body = np.asarray(chosen[2], dtype=np.float64)

    p_surf_world = rotate_vector_by_quaternion_np(best_p_body, q_body) + pos_com
    return p_surf_world, best_sdf


def _orthonormal_tangent_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def inertia_world_from_body_diag(inertia_body_diag_: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    R = quat_to_R_np_wxyz(quat_wxyz)
    I_body = np.diag(np.asarray(inertia_body_diag_, dtype=np.float64))
    return R @ I_body @ R.T


def compute_tool_block_impulse(
    q_block: np.ndarray,
    v_block6: np.ndarray,
    tool_pos: np.ndarray,
    tool_vel: np.ndarray,
    tool_radius: float,
    tool_mu: float = 0.6,
    dt: float = 0.002,
    contact_eps: float = 1e-6,
    restitution: float = 0.0,
    enable_margin: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute one-step Coulomb impulse between a dynamic sphere tool and the block.

    Same algorithm as v081 (normal 1D MCP + tangential Coulomb projection).
    The only behavior-sensitive part is the closest-point selection, now with v081 tie-break.
    """
    state.ensure_initialized()

    tool_pos = np.asarray(tool_pos, dtype=np.float64).reshape(3,)
    tool_vel = np.asarray(tool_vel, dtype=np.float64).reshape(3,)
    v_block6 = np.asarray(v_block6, dtype=np.float64).reshape(6,)

    a, sdf = closest_point_on_tblock_surface_world(tool_pos, q_block)
    a = np.asarray(a, dtype=np.float64).reshape(3,)
    g = float(sdf) - float(tool_radius)

    if g > float(enable_margin):
        return np.zeros(3, dtype=np.float64), a, np.array([0.0, 0.0, 1.0], dtype=np.float64), g

    d = a - tool_pos
    dn = float(np.linalg.norm(d))
    if dn < 1e-12:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n = d / dn

    com = np.asarray(q_block[:3], dtype=np.float64)
    r = a - com
    v_lin = v_block6[0:3]
    w = v_block6[3:6]
    v_block_c = v_lin + np.cross(w, r)

    v_rel = v_block_c - tool_vel
    vn = float(np.dot(n, v_rel))

    m_b = float(state.total_mass)
    Iw = inertia_world_from_body_diag(
        np.asarray(state.inertia_body_diag, dtype=np.float64),
        np.asarray(q_block[3:7], dtype=np.float64),
    )
    Iw_inv = np.linalg.inv(Iw + 1e-12 * np.eye(3))
    rx = skew_np(r)
    K_block = (1.0 / m_b) * np.eye(3) - rx @ Iw_inv @ rx

    m_t = float(getattr(compute_tool_block_impulse, "tool_mass", 1.0))
    K_rel = K_block + (1.0 / m_t) * np.eye(3)

    nKn = float(n.T @ K_rel @ n)

    vn_minus_neg = min(vn, 0.0)
    b = g + float(dt) * (vn + float(restitution) * vn_minus_neg)
    a_lin = float(contact_eps) + float(dt) * float(nKn)

    pn_fallback = 0.0
    if a_lin > 0.0:
        pn_fallback = max(0.0, -b / a_lin)

    if b >= 0.0:
        pn = 0.0
    else:
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

        pn_prev = float(getattr(compute_tool_block_impulse, "pn_guess", pn_fallback))
        if not np.isfinite(pn_prev) or pn_prev < 0.0:
            pn_prev = pn_fallback
        z_guess[0] = pn_prev

        def call_F(n_dim, z, w_out):
            w_out[0] = b + a_lin * float(z[0])

        def call_Jac(n_dim, z, J_out):
            J_out[0, 0] = a_lin

        problem = MCP(0, 1, call_F, call_Jac)
        info = mcp_newton_FB_FBLSA(problem, z_guess, w_sol, opts)

        if info == 0 and np.isfinite(z_guess[0]) and z_guess[0] >= 0.0:
            pn = float(z_guess[0])
        else:
            pn = float(pn_fallback)

    compute_tool_block_impulse.pn_guess = float(0.75 * pn + 0.25 * pn_fallback)

    # normal impulse
    pN = n * pn
    v_rel2 = v_rel + K_rel @ pN

    # tangential impulse via Coulomb projection
    t1, t2 = _orthonormal_tangent_basis(n)
    vt1 = float(np.dot(t1, v_rel2))
    vt2 = float(np.dot(t2, v_rel2))

    k1 = float(t1.T @ K_rel @ t1)
    k2 = float(t2.T @ K_rel @ t2)
    m1 = 1.0 / (k1 + 1e-12)
    m2 = 1.0 / (k2 + 1e-12)

    pt1 = -m1 * vt1
    pt2 = -m2 * vt2

    lim = float(tool_mu) * pn
    nrm = float(np.hypot(pt1, pt2))
    if nrm > lim and nrm > 1e-12:
        s = lim / nrm
        pt1 *= s
        pt2 *= s

    pT = t1 * pt1 + t2 * pt2
    p_lin = pN + pT

    return p_lin.astype(np.float64), a, n.astype(np.float64), float(g)
