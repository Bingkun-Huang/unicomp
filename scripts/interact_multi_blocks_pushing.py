#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""push_two_blocks_compsim_live.py

Two dynamic T-blocks (ground MCP per block) + dynamic tool sphere (impedance) + block-block contact,
implemented using the **compsim** library (no legacy v08x monolith).

What this gives you
-------------------
- Two free rigid bodies: `T_siconos` and `T_siconos2`.
- Each block-ground contact solved by compsim's v081-style Step3B MCP solver.
- Tool is a dynamic sphere controlled by a simple impedance law tracking `target_mocap`.
- Block-block contact uses a *single best* contact candidate each step (probe-points -> closest surface ->
  two-body Coulomb impulse with 1D MCP for normal impulse).
- MuJoCo is used ONLY for visualization + mocap dragging; all MuJoCo contacts are disabled.

Run
---
  export JAX_PLATFORM_NAME=cpu
  cd ~/compsim
  PYTHONPATH=. python3 -m scripts.push_two_blocks_compsim_live --view \
      --xml ./model/table_two_blocks_tool_compsim.xml

Mouse
-----
- Drag the **red** mocap (target_mocap) in the viewer; the **green** tool will follow it.
- If your viewer drag only moves XY, use your viewer's Z-drag modifier (varies by MuJoCo version).

Notes
-----
- This is intentionally minimal / robust. If you want richer block-block contact (multi-point GS),
  we can extend this to multiple candidates/iterations.
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import time
import argparse
from dataclasses import dataclass

import numpy as np

import mujoco
import mujoco.viewer

import siconos.numerics as sn
from siconos.numerics import MCP, mcp_newton_FB_FBLSA

import compsim
from compsim import state
from compsim.samples import get_local_points_ref
from compsim.pose import body_origin_qpos_to_com, com_to_body_origin_qpos
from compsim.tool_contact import closest_point_on_tblock_surface_world, _orthonormal_tangent_basis
from compsim.integration import project_to_ground_and_damp
from compsim.math3d import (
    quat_to_R_np_wxyz,
    skew_np,
    quat_from_omega_world_np,
    quat_mul_wxyz_np,
    rotate_vector_by_quaternion_np,
)


# -----------------------------
# Params
# -----------------------------

@dataclass
class ToolParams:
    radius: float
    mu: float
    mass: float
    restitution: float
    contact_eps: float
    enable_margin: float
    k: float
    d: float
    fmax: float
    vcap: float


@dataclass
class BBParams:
    mu: float
    restitution: float
    contact_eps: float
    enable_margin: float


# -----------------------------
# MuJoCo helpers
# -----------------------------

def _abs_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _disable_all_contacts(m: mujoco.MjModel) -> None:
    """Make MuJoCo purely visual (compsim handles all contact)."""
    try:
        m.geom_contype[:] = 0
        m.geom_conaffinity[:] = 0
    except Exception:
        pass
    try:
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    except Exception:
        pass


def _find_freejoint_qposadr(m: mujoco.MjModel, body_name: str) -> int:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"body '{body_name}' not found")
    jadr = int(m.body_jntadr[bid])
    jnum = int(m.body_jntnum[bid])
    for k in range(jnum):
        jid = jadr + k
        if int(m.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(m.jnt_qposadr[jid])
    raise ValueError(f"body '{body_name}' has no freejoint")


def _find_mocap_id(m: mujoco.MjModel, body_name: str) -> int:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return -1
    return int(m.body_mocapid[bid])


def _viewer_running(viewer) -> bool:
    if hasattr(viewer, "is_running"):
        try:
            return bool(viewer.is_running())
        except Exception:
            return True
    if hasattr(viewer, "is_alive"):
        try:
            return bool(viewer.is_alive())
        except Exception:
            return True
    return True


def _norm_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _cap_norm(x: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n <= max_norm or n < 1e-12:
        return x
    return x * (max_norm / (n + 1e-12))


# -----------------------------
# Viewer overlay (user_scn)
# -----------------------------

def _set_geom_identity_mat(g) -> None:
    I = np.eye(3, dtype=np.float32)
    try:
        if hasattr(g, "mat") and getattr(g.mat, "shape", None) == (3, 3):
            g.mat[:] = I
        else:
            g.mat[:] = I.reshape(-1)
    except Exception:
        pass


def _set_sphere(g, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size[:] = 0
    g.size[0] = float(radius)
    g.pos[:] = np.asarray(pos, dtype=np.float32)
    _set_geom_identity_mat(g)
    g.rgba[:] = np.asarray(rgba, dtype=np.float32)


def _rot_from_x_axis(x_dir: np.ndarray) -> np.ndarray:
    x = np.asarray(x_dir, dtype=np.float64)
    xn = float(np.linalg.norm(x))
    if xn < 1e-12:
        return np.eye(3)
    x = x / xn
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(x, ref))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    y = np.cross(ref, x)
    yn = float(np.linalg.norm(y))
    if yn < 1e-12:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        y = np.cross(ref, x)
        yn = float(np.linalg.norm(y)) + 1e-12
    y = y / yn
    z = np.cross(x, y)
    z = z / (float(np.linalg.norm(z)) + 1e-12)
    return np.stack([x, y, z], axis=1)


def _set_arrow(g, origin: np.ndarray, vec: np.ndarray, radius: float, length: float, rgba: np.ndarray) -> None:
    g.type = mujoco.mjtGeom.mjGEOM_ARROW
    g.size[:] = 0
    g.size[0] = float(radius)
    g.size[1] = float(max(0.0, length))
    g.pos[:] = np.asarray(origin, dtype=np.float32)
    R = _rot_from_x_axis(vec)
    try:
        if hasattr(g, "mat") and getattr(g.mat, "shape", None) == (3, 3):
            g.mat[:] = R.astype(np.float32)
        else:
            g.mat[:] = R.astype(np.float32).reshape(-1)
    except Exception:
        pass
    g.rgba[:] = np.asarray(rgba, dtype=np.float32)


# -----------------------------
# Rigid-body math
# -----------------------------

def _inertia_world_inv_from_body_diag(inertia_body_diag: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    R = quat_to_R_np_wxyz(np.asarray(quat_wxyz, dtype=np.float64))
    I_body = np.diag(np.asarray(inertia_body_diag, dtype=np.float64))
    Iw = R @ I_body @ R.T
    return np.linalg.inv(Iw + 1e-12 * np.eye(3))


def _apply_point_impulse(v6: np.ndarray, q_com: np.ndarray, a_world: np.ndarray, p_world: np.ndarray,
                         mass: float, inertia_body_diag: np.ndarray, sign: float) -> np.ndarray:
    """Apply impulse (+p) at point a to body twist v6. sign=+1 applies +p, sign=-1 applies -p."""
    v6 = np.asarray(v6, dtype=np.float64).copy()
    p = sign * np.asarray(p_world, dtype=np.float64).reshape(3,)
    if float(np.linalg.norm(p)) < 1e-14:
        return v6
    r = np.asarray(a_world, dtype=np.float64).reshape(3,) - np.asarray(q_com[:3], dtype=np.float64)
    v6[:3] += p / max(1e-12, float(mass))
    Iw_inv = _inertia_world_inv_from_body_diag(inertia_body_diag, q_com[3:7])
    v6[3:] += Iw_inv @ np.cross(r, p)
    return v6


# -----------------------------
# Block-block contact (single best candidate)
# -----------------------------

def _probe_points_body_from_boxes(max_per_box: int = 24) -> np.ndarray:
    """Cheap probe points on the union-of-boxes surface, in BODY frame.

    We generate corners + face-centers for each box geom, then deduplicate.
    """
    state.ensure_initialized()
    pts = []
    for gt, gp, gq, gs in zip(state.T_GEOM_TYPE, state.T_GEOM_POS, state.T_GEOM_QUAT, state.T_GEOM_SIZE):
        if int(gt) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        b = np.asarray(gs[:3], dtype=np.float64)
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.append([sx * b[0], sy * b[1], sz * b[2]])
        faces = [
            [ b[0], 0.0, 0.0], [-b[0], 0.0, 0.0],
            [0.0,  b[1], 0.0], [0.0, -b[1], 0.0],
            [0.0, 0.0,  b[2]], [0.0, 0.0, -b[2]],
        ]
        local = np.array(corners + faces, dtype=np.float64)
        # transform: geom local -> body
        for p in local:
            pb = rotate_vector_by_quaternion_np(p, np.asarray(gq, dtype=np.float64)) + np.asarray(gp, dtype=np.float64)
            pts.append(pb)

    if not pts:
        return np.zeros((0, 3), dtype=np.float64)

    P = np.asarray(pts, dtype=np.float64)

    # Deduplicate (grid hash)
    key = np.round(P / 1e-4).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    P = P[np.sort(idx)]

    # Optional subsample
    if P.shape[0] > max_per_box * max(1, int(len(state.T_GEOM_TYPE))):
        P = P[::2]
    return P


def _world_points_from_body(P_body: np.ndarray, q_com: np.ndarray) -> np.ndarray:
    pos = np.asarray(q_com[:3], dtype=np.float64)
    quat = np.asarray(q_com[3:7], dtype=np.float64)
    # rotate each point
    return np.asarray([rotate_vector_by_quaternion_np(p, quat) + pos for p in P_body], dtype=np.float64)


def _best_penetration(p_world_list: np.ndarray, q_other: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (sdf_min, p_penetrating_world, a_surface_world_on_other)"""
    best_sdf = 1e18
    best_p = None
    best_a = None
    for p in p_world_list:
        a, sdf = closest_point_on_tblock_surface_world(p, q_other)
        if float(sdf) < best_sdf:
            best_sdf = float(sdf)
            best_p = np.asarray(p, dtype=np.float64)
            best_a = np.asarray(a, dtype=np.float64)
    if best_p is None:
        best_p = np.zeros(3, dtype=np.float64)
        best_a = np.zeros(3, dtype=np.float64)
    return float(best_sdf), best_p, best_a


def _solve_scalar_mcp(b: float, a_lin: float, pn_guess: float) -> float:
    """Solve 0 <= pn ⟂ (b + a_lin*pn) >= 0 via Siconos MCP."""
    pn_fallback = 0.0
    if a_lin > 0:
        pn_fallback = max(0.0, -b / a_lin)
    if b >= 0.0:
        return 0.0

    opts = getattr(_solve_scalar_mcp, "_opts", None)
    if opts is None:
        opts = sn.SolverOptions(sn.SICONOS_MCP_NEWTON_FB_FBLSA)
        opts.iparam[sn.SICONOS_IPARAM_MAX_ITER] = 10
        opts.dparam[sn.SICONOS_DPARAM_TOL] = 1e-7
        _solve_scalar_mcp._opts = opts

    z = np.array([max(0.0, float(pn_guess))], dtype=np.float64)
    w = np.zeros((1,), dtype=np.float64)

    def call_F(n_dim, z_in, w_out):
        w_out[0] = float(b) + float(a_lin) * float(z_in[0])

    def call_J(n_dim, z_in, J_out):
        J_out[0, 0] = float(a_lin)

    prob = MCP(0, 1, call_F, call_J)
    info = mcp_newton_FB_FBLSA(prob, z, w, opts)

    if info == 0 and np.isfinite(z[0]) and z[0] >= 0.0:
        return float(z[0])
    return float(pn_fallback)


def _two_body_coulomb_impulse(
    q1: np.ndarray,
    v1: np.ndarray,
    m1: float,
    I1_diag: np.ndarray,
    q2: np.ndarray,
    v2: np.ndarray,
    m2: float,
    I2_diag: np.ndarray,
    c: np.ndarray,
    n: np.ndarray,
    g: float,
    mu: float,
    dt: float,
    contact_eps: float,
    restitution: float,
    pn_guess: float,
) -> tuple[np.ndarray, float]:
    """Return impulse p applied to body1 at contact point c (body2 gets -p), and pn."""
    c = np.asarray(c, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    n = n / (float(np.linalg.norm(n)) + 1e-12)

    com1 = np.asarray(q1[:3], dtype=np.float64)
    com2 = np.asarray(q2[:3], dtype=np.float64)
    r1 = c - com1
    r2 = c - com2

    v1_lin = np.asarray(v1[:3], dtype=np.float64)
    w1 = np.asarray(v1[3:], dtype=np.float64)
    v2_lin = np.asarray(v2[:3], dtype=np.float64)
    w2 = np.asarray(v2[3:], dtype=np.float64)

    v1c = v1_lin + np.cross(w1, r1)
    v2c = v2_lin + np.cross(w2, r2)
    v_rel = v1c - v2c
    vn = float(np.dot(n, v_rel))

    I1_inv = _inertia_world_inv_from_body_diag(I1_diag, q1[3:7])
    I2_inv = _inertia_world_inv_from_body_diag(I2_diag, q2[3:7])

    rx1 = skew_np(r1)
    rx2 = skew_np(r2)
    K1 = (1.0 / max(1e-12, float(m1))) * np.eye(3) - rx1 @ I1_inv @ rx1
    K2 = (1.0 / max(1e-12, float(m2))) * np.eye(3) - rx2 @ I2_inv @ rx2
    K_rel = K1 + K2

    nKn = float(n.T @ K_rel @ n)

    vn_minus_neg = min(vn, 0.0)
    b = float(g) + float(dt) * (vn + float(restitution) * vn_minus_neg)
    a_lin = float(contact_eps) + float(dt) * float(nKn)

    pn = _solve_scalar_mcp(b, a_lin, pn_guess)
    pN = n * pn

    # friction: projected impulse (same as tool_contact)
    v_rel2 = v_rel + K_rel @ pN

    t1, t2 = _orthonormal_tangent_basis(n)
    vt1 = float(np.dot(t1, v_rel2))
    vt2 = float(np.dot(t2, v_rel2))

    k1 = float(t1.T @ K_rel @ t1)
    k2 = float(t2.T @ K_rel @ t2)
    m_eff1 = 1.0 / (k1 + 1e-12)
    m_eff2 = 1.0 / (k2 + 1e-12)

    pt1 = -m_eff1 * vt1
    pt2 = -m_eff2 * vt2

    lim = float(mu) * pn
    nrm = float(np.hypot(pt1, pt2))
    if nrm > lim and nrm > 1e-12:
        s = lim / nrm
        pt1 *= s
        pt2 *= s

    pT = t1 * pt1 + t2 * pt2
    return (pN + pT).astype(np.float64), float(pn)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, default="model/table_two_blocks_tool_compsim.xml")
    ap.add_argument("--view", action="store_true")

    ap.add_argument("--body1", type=str, default="T_siconos")
    ap.add_argument("--body2", type=str, default="T_siconos2")
    ap.add_argument("--target_mocap", type=str, default="target_mocap")
    ap.add_argument("--tool_mocap", type=str, default="tool_mocap")

    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--realtime", action="store_true")

    # Ground (per block)
    ap.add_argument("--mu_ground", type=float, default=0.7)
    ap.add_argument("--restitution", type=float, default=0.10)
    ap.add_argument("--contact_eps", type=float, default=1e-6)

    # Tool
    ap.add_argument("--tool_radius", type=float, default=0.05)
    ap.add_argument("--tool_mu", type=float, default=0.6)
    ap.add_argument("--tool_mass", type=float, default=1.0)
    ap.add_argument("--tool_restitution", type=float, default=0.0)
    ap.add_argument("--tool_enable_margin", type=float, default=1e-4)

    ap.add_argument("--tool_k", type=float, default=800.0)
    ap.add_argument("--tool_d", type=float, default=80.0)
    ap.add_argument("--tool_fmax", type=float, default=50.0)
    ap.add_argument("--tool_vcap", type=float, default=3.0)

    # Block-block
    ap.add_argument("--bb_mu", type=float, default=0.6)
    ap.add_argument("--bb_restitution", type=float, default=0.0)
    ap.add_argument("--bb_contact_eps", type=float, default=1e-6)
    ap.add_argument("--bb_enable_margin", type=float, default=2e-4)

    # Visualization sizes
    ap.add_argument("--ecp_radius", type=float, default=0.02)
    ap.add_argument("--arrow_radius", type=float, default=0.006)
    ap.add_argument("--force_scale", type=float, default=0.02)
    ap.add_argument("--force_max_len", type=float, default=0.25)

    # Performance guards
    ap.add_argument("--ground_max_iter", type=int, default=180)
    ap.add_argument("--ground_max_iter_fallback", type=int, default=80)
    ap.add_argument("--ground_budget_ms", type=float, default=6.0)
    ap.add_argument("--spike_ms", type=float, default=25.0)

    args = ap.parse_args()
    xml_path = _abs_path(args.xml)

    # Init compsim cache from XML (for geometry used in closest-point queries)
    compsim.init_from_xml(xml_path, body=args.body1)

    # Load MuJoCo model for visualization + inertial props
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    _disable_all_contacts(m)

    # Body ids and freejoint qpos addresses
    bid1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, args.body1)
    bid2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, args.body2)
    if bid1 < 0 or bid2 < 0:
        raise RuntimeError("body1/body2 not found in XML")

    qadr1 = _find_freejoint_qposadr(m, args.body1)
    qadr2 = _find_freejoint_qposadr(m, args.body2)

    ipos1 = np.asarray(m.body_ipos[bid1], dtype=np.float64)
    ipos2 = np.asarray(m.body_ipos[bid2], dtype=np.float64)
    mass1 = float(m.body_mass[bid1])
    mass2 = float(m.body_mass[bid2])
    I1_diag = np.asarray(m.body_inertia[bid1], dtype=np.float64)
    I2_diag = np.asarray(m.body_inertia[bid2], dtype=np.float64)

    print(f"[two_blocks] xml={xml_path}")
    print(f"  body1={args.body1} mass={mass1:.6f} inertia_diag={I1_diag}")
    print(f"  body2={args.body2} mass={mass2:.6f} inertia_diag={I2_diag}")

    # Mocap ids
    target_mocap_id = _find_mocap_id(m, args.target_mocap)
    tool_mocap_id = _find_mocap_id(m, args.tool_mocap)
    if target_mocap_id < 0 or tool_mocap_id < 0:
        raise RuntimeError("target_mocap/tool_mocap not found or not mocap=true")

    # Precompute local ground contact points
    local_pts_all = get_local_points_ref(verbose=True)

    # Build two independent ground solvers
    solver1 = compsim.TBlockSimulator_Step_NoBounce(
        dt=float(args.dt),
        mass=float(mass1),
        inertia_diag_body=np.asarray(I1_diag, dtype=np.float64),
        local_pts_all=local_pts_all,
        mu_fric=float(args.mu_ground),
        restitution=float(args.restitution),
        contact_eps=float(args.contact_eps),
    )
    solver2 = compsim.TBlockSimulator_Step_NoBounce(
        dt=float(args.dt),
        mass=float(mass2),
        inertia_diag_body=np.asarray(I2_diag, dtype=np.float64),
        local_pts_all=local_pts_all,
        mu_fric=float(args.mu_ground),
        restitution=float(args.restitution),
        contact_eps=float(args.contact_eps),
    )

    try:
        solver1.options.iparam[sn.SICONOS_IPARAM_MAX_ITER] = int(args.ground_max_iter)
        solver2.options.iparam[sn.SICONOS_IPARAM_MAX_ITER] = int(args.ground_max_iter)
    except Exception:
        pass

    tool = ToolParams(
        radius=float(args.tool_radius),
        mu=float(args.tool_mu),
        mass=float(args.tool_mass),
        restitution=float(args.tool_restitution),
        contact_eps=float(args.contact_eps),
        enable_margin=float(args.tool_enable_margin),
        k=float(args.tool_k),
        d=float(args.tool_d),
        fmax=float(args.tool_fmax),
        vcap=float(args.tool_vcap),
    )

    bb = BBParams(
        mu=float(args.bb_mu),
        restitution=float(args.bb_restitution),
        contact_eps=float(args.bb_contact_eps),
        enable_margin=float(args.bb_enable_margin),
    )

    # Force initial overlap: green tool at red target
    d.mocap_pos[tool_mocap_id][:] = d.mocap_pos[target_mocap_id][:]

    # Tool state
    tool_pos = np.asarray(d.mocap_pos[tool_mocap_id], dtype=np.float64).copy()
    tool_vel = np.zeros(3, dtype=np.float64)
    tool_des_prev = np.asarray(d.mocap_pos[target_mocap_id], dtype=np.float64).copy()

    # Block states (COM representation)
    qpos1_0 = np.asarray(d.qpos[qadr1 : qadr1 + 7], dtype=np.float64).copy()
    qpos2_0 = np.asarray(d.qpos[qadr2 : qadr2 + 7], dtype=np.float64).copy()
    q1 = body_origin_qpos_to_com(qpos1_0, ipos1)
    q2 = body_origin_qpos_to_com(qpos2_0, ipos2)
    q1[3:7] = _norm_quat(q1[3:7])
    q2[3:7] = _norm_quat(q2[3:7])
    v1 = np.zeros(6, dtype=np.float64)
    v2 = np.zeros(6, dtype=np.float64)

    # Gravity wrenches (force on COM)
    f1 = np.zeros(6, dtype=np.float64)
    f2 = np.zeros(6, dtype=np.float64)
    f1[2] = -mass1 * 9.81
    f2[2] = -mass2 * 9.81

    # Block-block probe points in BODY frame
    P_probe_body = _probe_points_body_from_boxes(max_per_box=64)

    # Warm-start for block-block pn
    pn_bb_guess = 0.0

    # Visualization colors
    rgba_ecp1 = np.array([0.2, 0.4, 1.0, 1.0], dtype=np.float32)  # blue
    rgba_ecp2 = np.array([0.0, 0.8, 1.0, 1.0], dtype=np.float32)  # cyan
    rgba_force = np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32)  # red
    rgba_bb = np.array([1.0, 0.6, 0.0, 1.0], dtype=np.float32)  # orange

    last_tool_imp_norm = 0.0
    last_good_v1 = None
    last_good_v2 = None
    last_good_ecp1 = None
    last_good_ecp2 = None

    def _pn_ground(solver) -> float:
        if getattr(solver, "_frozen", False):
            z = getattr(solver, "z_guess_frozen", np.zeros(10))
            return float(z[9])
        z = getattr(solver, "z_guess", np.zeros(13))
        return float(z[12])

    def step_once(i: int):
        nonlocal q1, q2, v1, v2
        nonlocal tool_pos, tool_vel, tool_des_prev
        nonlocal pn_bb_guess
        nonlocal last_tool_imp_norm, last_good_v1, last_good_v2, last_good_ecp1, last_good_ecp2

        # --- tool desired (red mocap) ---
        tool_des = np.asarray(d.mocap_pos[target_mocap_id], dtype=np.float64).copy()
        v_des = (tool_des - tool_des_prev) / float(args.dt)
        v_des = _cap_norm(v_des, tool.vcap)
        tool_des_prev = tool_des.copy()

        # Impedance command -> tool free velocity
        F_cmd = tool.k * (tool_des - tool_pos) + tool.d * (v_des - tool_vel)
        F_cmd = _cap_norm(F_cmd, tool.fmax)

        tool_vel_free = tool_vel + float(args.dt) * (F_cmd / max(1e-9, tool.mass))
        tool_vel_free = _cap_norm(tool_vel_free, tool.vcap)

        # --- 1) ground solve per block ---
        t0 = time.perf_counter()
        v1_g, ecp1, info1 = solver1.solve_step(
            q_curr_np=q1,
            v_curr_np=v1,
            f_applied_np=f1,
            step_idx=int(i),
            return_ecp=True,
            tool_impulse_norm=float(last_tool_imp_norm),
        )
        t1_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        v2_g, ecp2, info2 = solver2.solve_step(
            q_curr_np=q2,
            v_curr_np=v2,
            f_applied_np=f2,
            step_idx=int(i),
            return_ecp=True,
            tool_impulse_norm=float(last_tool_imp_norm),
        )
        t2_ms = (time.perf_counter() - t0) * 1000.0

        # basic stall guards
        if int(info1) == 0 and t1_ms <= float(args.ground_budget_ms):
            last_good_v1 = np.asarray(v1_g, dtype=np.float64).copy()
            last_good_ecp1 = None if ecp1 is None else np.asarray(ecp1, dtype=np.float64).copy()
        elif last_good_v1 is not None:
            v1_g = last_good_v1
            ecp1 = last_good_ecp1
            try:
                solver1.options.iparam[sn.SICONOS_IPARAM_MAX_ITER] = int(args.ground_max_iter_fallback)
            except Exception:
                pass

        if int(info2) == 0 and t2_ms <= float(args.ground_budget_ms):
            last_good_v2 = np.asarray(v2_g, dtype=np.float64).copy()
            last_good_ecp2 = None if ecp2 is None else np.asarray(ecp2, dtype=np.float64).copy()
        elif last_good_v2 is not None:
            v2_g = last_good_v2
            ecp2 = last_good_ecp2
            try:
                solver2.options.iparam[sn.SICONOS_IPARAM_MAX_ITER] = int(args.ground_max_iter_fallback)
            except Exception:
                pass

        # --- 2) tool-block impulses (with both blocks) ---
        p_tool_1 = np.zeros(3, dtype=np.float64)
        p_tool_2 = np.zeros(3, dtype=np.float64)
        a1 = np.zeros(3, dtype=np.float64)
        a2 = np.zeros(3, dtype=np.float64)
        g1 = 1e9
        g2 = 1e9

        # We re-use the closest-point + impulse logic from compsim via block surface queries
        # Here we implement the impulse in-place (same math as tool_contact.py but with per-block inertia/mass).
        def sphere_block_impulse(qb, vb, mass_b, I_diag_b):
            nonlocal tool_pos, tool_vel_free
            a, sdf = closest_point_on_tblock_surface_world(tool_pos, qb)
            g = float(sdf) - float(tool.radius)
            if g > float(tool.enable_margin):
                return np.zeros(3, dtype=np.float64), np.asarray(a, dtype=np.float64), g

            dvec = np.asarray(a, dtype=np.float64) - tool_pos
            dn = float(np.linalg.norm(dvec))
            if dn < 1e-12:
                n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                n = dvec / dn

            com = np.asarray(qb[:3], dtype=np.float64)
            r = np.asarray(a, dtype=np.float64) - com
            v_lin = np.asarray(vb[:3], dtype=np.float64)
            w = np.asarray(vb[3:], dtype=np.float64)
            v_block_c = v_lin + np.cross(w, r)

            v_rel = v_block_c - tool_vel_free
            vn = float(np.dot(n, v_rel))

            Iw_inv = _inertia_world_inv_from_body_diag(I_diag_b, qb[3:7])
            rx = skew_np(r)
            K_block = (1.0 / max(1e-12, float(mass_b))) * np.eye(3) - rx @ Iw_inv @ rx
            K_rel = K_block + (1.0 / max(1e-12, float(tool.mass))) * np.eye(3)

            nKn = float(n.T @ K_rel @ n)

            vn_minus_neg = min(vn, 0.0)
            b = g + float(args.dt) * (vn + float(tool.restitution) * vn_minus_neg)
            a_lin = float(tool.contact_eps) + float(args.dt) * float(nKn)

            pn = _solve_scalar_mcp(b, a_lin, getattr(sphere_block_impulse, "pn_guess", 0.0))
            sphere_block_impulse.pn_guess = float(0.75 * pn + 0.25 * max(0.0, -b / max(1e-12, a_lin)))

            pN = n * pn
            v_rel2 = v_rel + K_rel @ pN

            t1, t2 = _orthonormal_tangent_basis(n)
            vt1 = float(np.dot(t1, v_rel2))
            vt2 = float(np.dot(t2, v_rel2))

            k1 = float(t1.T @ K_rel @ t1)
            k2 = float(t2.T @ K_rel @ t2)
            m1_eff = 1.0 / (k1 + 1e-12)
            m2_eff = 1.0 / (k2 + 1e-12)

            pt1 = -m1_eff * vt1
            pt2 = -m2_eff * vt2

            lim = float(tool.mu) * pn
            nrm = float(np.hypot(pt1, pt2))
            if nrm > lim and nrm > 1e-12:
                s = lim / nrm
                pt1 *= s
                pt2 *= s

            pT = t1 * pt1 + t2 * pt2
            return (pN + pT).astype(np.float64), np.asarray(a, dtype=np.float64), g

        p_tool_1, a1, g1 = sphere_block_impulse(q1, v1_g, mass1, I1_diag)
        p_tool_2, a2, g2 = sphere_block_impulse(q2, v2_g, mass2, I2_diag)

        p_tool_total = p_tool_1 + p_tool_2
        last_tool_imp_norm = float(np.linalg.norm(p_tool_total))

        # apply to blocks (tool impulse is applied to blocks)
        v1_t = _apply_point_impulse(v1_g, q1, a1, p_tool_1, mass1, I1_diag, sign=+1.0)
        v2_t = _apply_point_impulse(v2_g, q2, a2, p_tool_2, mass2, I2_diag, sign=+1.0)

        # tool reaction (tool gets -p)
        tool_vel = tool_vel_free - (p_tool_total / max(1e-9, tool.mass))
        tool_vel = _cap_norm(tool_vel, tool.vcap)
        tool_pos = tool_pos + float(args.dt) * tool_vel
        d.mocap_pos[tool_mocap_id] = tool_pos

        # --- 3) block-block impulse (single best candidate) ---
        # Find deepest penetration in both directions
        P1_world = _world_points_from_body(P_probe_body, q1)
        P2_world = _world_points_from_body(P_probe_body, q2)

        sdf12, p1_pen, a_on_2 = _best_penetration(P1_world, q2)
        sdf21, p2_pen, a_on_1 = _best_penetration(P2_world, q1)

        bb_contact = False
        p_bb = np.zeros(3, dtype=np.float64)
        c_bb = np.zeros(3, dtype=np.float64)

        if sdf12 < sdf21:
            # block1 point into block2
            g = sdf12
            if g < bb.enable_margin:
                bb_contact = True
                n = (a_on_2 - p1_pen)
                nn = float(np.linalg.norm(n))
                if nn < 1e-12:
                    n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                else:
                    n = n / nn
                c = 0.5 * (a_on_2 + p1_pen)
                p_bb, pn_bb = _two_body_coulomb_impulse(
                    q1, v1_t, mass1, I1_diag,
                    q2, v2_t, mass2, I2_diag,
                    c=c,
                    n=n,
                    g=float(g),
                    mu=bb.mu,
                    dt=float(args.dt),
                    contact_eps=bb.contact_eps,
                    restitution=bb.restitution,
                    pn_guess=float(pn_bb_guess),
                )
                pn_bb_guess = 0.8 * pn_bb_guess + 0.2 * pn_bb
                c_bb = c
        else:
            # block2 point into block1 -> impulse computed for body2 as "body1" in helper
            g = sdf21
            if g < bb.enable_margin:
                bb_contact = True
                n = (a_on_1 - p2_pen)
                nn = float(np.linalg.norm(n))
                if nn < 1e-12:
                    n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                else:
                    n = n / nn
                c = 0.5 * (a_on_1 + p2_pen)
                p_on_2, pn_bb = _two_body_coulomb_impulse(
                    q2, v2_t, mass2, I2_diag,
                    q1, v1_t, mass1, I1_diag,
                    c=c,
                    n=n,
                    g=float(g),
                    mu=bb.mu,
                    dt=float(args.dt),
                    contact_eps=bb.contact_eps,
                    restitution=bb.restitution,
                    pn_guess=float(pn_bb_guess),
                )
                pn_bb_guess = 0.8 * pn_bb_guess + 0.2 * pn_bb
                # Convert to impulse on block1
                p_bb = -p_on_2
                c_bb = c

        # Apply block-block impulse
        v1_bb = v1_t
        v2_bb = v2_t
        if bb_contact:
            v1_bb = _apply_point_impulse(v1_bb, q1, c_bb, p_bb, mass1, I1_diag, sign=+1.0)
            v2_bb = _apply_point_impulse(v2_bb, q2, c_bb, p_bb, mass2, I2_diag, sign=-1.0)

        # --- 4) integrate blocks ---
        def integrate(q, v):
            qn = np.asarray(q, dtype=np.float64).copy()
            qn[:3] += float(args.dt) * np.asarray(v[:3], dtype=np.float64)
            dq = quat_from_omega_world_np(np.asarray(v[3:], dtype=np.float64), float(args.dt))
            qn[3:7] = quat_mul_wxyz_np(dq, qn[3:7])
            qn[3:7] = _norm_quat(qn[3:7])
            return qn

        q1n = integrate(q1, v1_bb)
        q2n = integrate(q2, v2_bb)

        # --- 5) anti-penetration projection against ground ---
        q1n2, v1n2, min_z1, _ = project_to_ground_and_damp(
            q1n,
            v1_bb,
            float(args.dt),
            local_pts_all,
            pn_ground=float(_pn_ground(solver1)),
            pn_support_thresh=1e-9,
        )
        q2n2, v2n2, min_z2, _ = project_to_ground_and_damp(
            q2n,
            v2_bb,
            float(args.dt),
            local_pts_all,
            pn_ground=float(_pn_ground(solver2)),
            pn_support_thresh=1e-9,
        )

        q1[:] = q1n2
        q2[:] = q2n2
        v1[:] = v1n2
        v2[:] = v2n2

        # For visualization
        ecp1_vis = None if ecp1 is None else np.asarray(ecp1, dtype=np.float64)
        ecp2_vis = None if ecp2 is None else np.asarray(ecp2, dtype=np.float64)
        if ecp1_vis is None and last_good_ecp1 is not None:
            ecp1_vis = last_good_ecp1
        if ecp2_vis is None and last_good_ecp2 is not None:
            ecp2_vis = last_good_ecp2

        F_tool = p_tool_total / max(1e-12, float(args.dt))

        return {
            "t1_ms": float(t1_ms),
            "t2_ms": float(t2_ms),
            "info1": int(info1),
            "info2": int(info2),
            "ecp1": ecp1_vis,
            "ecp2": ecp2_vis,
            "tool_pos": tool_pos.copy(),
            "F_tool": F_tool.copy(),
            "bb": bb_contact,
            "c_bb": c_bb.copy(),
            "p_bb": p_bb.copy(),
            "min_z1": float(min_z1),
            "min_z2": float(min_z2),
            "g1": float(g1),
            "g2": float(g2),
        }

    if not args.view:
        for i in range(int(args.steps)):
            step_once(i)
        return

    with mujoco.viewer.launch_passive(m, d) as viewer:
        t_start = time.perf_counter()
        for i in range(int(args.steps)):
            if not _viewer_running(viewer):
                break

            t_frame0 = time.perf_counter()
            dbg = step_once(i)

            # Write block poses back to MuJoCo (body-origin qpos)
            d.qpos[qadr1 : qadr1 + 7] = com_to_body_origin_qpos(q1, ipos1)
            d.qpos[qadr2 : qadr2 + 7] = com_to_body_origin_qpos(q2, ipos2)

            mujoco.mj_forward(m, d)

            # Overlays
            try:
                scn = viewer.user_scn
                scn.ngeom = 0

                if dbg["ecp1"] is not None:
                    g0 = scn.geoms[scn.ngeom]
                    _set_sphere(g0, dbg["ecp1"], float(args.ecp_radius), rgba_ecp1)
                    scn.ngeom += 1

                if dbg["ecp2"] is not None:
                    g0 = scn.geoms[scn.ngeom]
                    _set_sphere(g0, dbg["ecp2"], float(args.ecp_radius), rgba_ecp2)
                    scn.ngeom += 1

                F = dbg["F_tool"]
                Fn = float(np.linalg.norm(F))
                if Fn > 1e-9:
                    direction = F / (Fn + 1e-12)
                    length = min(float(args.force_max_len), float(args.force_scale) * Fn)
                    g1 = scn.geoms[scn.ngeom]
                    _set_arrow(g1, dbg["tool_pos"], direction, float(args.arrow_radius), length, rgba_force)
                    scn.ngeom += 1

                if dbg["bb"] and float(np.linalg.norm(dbg["p_bb"])) > 1e-12:
                    Fbb = dbg["p_bb"] / max(1e-12, float(args.dt))
                    Fn2 = float(np.linalg.norm(Fbb))
                    if Fn2 > 1e-9:
                        direction2 = Fbb / (Fn2 + 1e-12)
                        length2 = min(float(args.force_max_len), float(args.force_scale) * Fn2)
                        g2 = scn.geoms[scn.ngeom]
                        _set_arrow(g2, dbg["c_bb"], direction2, float(args.arrow_radius), length2, rgba_bb)
                        scn.ngeom += 1
            except Exception:
                pass

            viewer.sync()

            # Print spikes (frame time)
            t_frame = (time.perf_counter() - t_frame0) * 1000.0
            if args.realtime:
                elapsed = (time.perf_counter() - t_frame0)
                remain = float(args.dt) - elapsed
                if remain > 0:
                    time.sleep(remain)

        hz = (i + 1) / max(1e-9, (time.perf_counter() - t_start))
        print(f"[done] steps={i+1} avg_hz={hz:.1f}")


if __name__ == "__main__":
    main()
