#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/push_fr3_waypoints_compsim_live.py

FR3 + compsim: live (compute-while-watch) waypoint pushing demo.

Design
------
- **compsim** owns the physics for the T-block (ground MCP) and the tool-block impulse.
- **MuJoCo** is used for visualization and for kinematic FR3 motion (differential IK).
- MuJoCo contacts are disabled to avoid double-physics.

Overlays (v081-style)
---------------------
- ECP: blue sphere
- Tool contact force: red arrow, from tool center, F = p_tool / dt

Run (example)
-------------
  export JAX_PLATFORM_NAME=cpu
  cd ~/compsim
  PYTHONPATH=. python3 -m scripts.push_fr3_waypoints_compsim_live \
      --xml ./model/scene_fr3_push.xml \
      --body T_siconos \
      --ee_site fr3_ee_site \
      --live_view

Notes
-----
- You MUST pass the correct `--ee_site` name from your MJCF.
- Joint discovery is by prefix; adjust `--arm_joint_prefix` if your joint names differ.
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

import jax.numpy as jnp

import compsim
from compsim import state
from compsim.samples import get_local_points_ref
from compsim.pose import body_origin_qpos_to_com, com_to_body_origin_qpos
from compsim.tool_contact import compute_tool_block_impulse
from compsim.math3d import (
    quat_from_omega_world_np,
    quat_mul_wxyz_np,
    quat_to_R_np_wxyz,
)
from compsim.integration import project_to_ground_and_damp

from scripts.waypoint_planner import WaypointPushPlanner, PlannerConfig


# -----------------------------
# Params
# -----------------------------


@dataclass
class ToolParams:
    radius: float = 0.05
    mu: float = 0.6
    mass: float = 1.0
    restitution: float = 0.0
    contact_eps: float = 1e-6
    enable_margin: float = 1e-4
    k: float = 800.0
    d: float = 80.0
    fmax: float = 500.0
    vcap: float = 3.0


# -----------------------------
# MuJoCo helpers
# -----------------------------


def _abs_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _disable_all_contacts(m: mujoco.MjModel) -> None:
    """Make MuJoCo purely visual (compsim does all contact)."""
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


def _cap_norm(x: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n <= max_norm or n < 1e-12:
        return x
    return x * (max_norm / (n + 1e-12))


def _norm_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _quat_yaw_wxyz(q: np.ndarray) -> float:
    # yaw from quaternion (w,x,y,z)
    w, x, y, z = [float(v) for v in q]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)], dtype=np.float64)


def inertia_world_inv_from_body_diag_np(inertia_body_diag: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    """I_world^{-1} for a diagonal inertia in body frame."""
    R = quat_to_R_np_wxyz(np.asarray(q_wxyz, dtype=np.float64))
    I_body = np.diag(np.asarray(inertia_body_diag, dtype=np.float64))
    I_world = R @ I_body @ R.T
    return np.linalg.inv(I_world)


# -----------------------------
# Obstacle point cloud (2D) for planner.step(..., obs_xy, dt)
# -----------------------------

def _geom_name(m: mujoco.MjModel, geom_id: int) -> str:
    try:
        s = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        return "" if s is None else str(s)
    except Exception:
        return ""


def _sample_geom_perimeter_xy(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    geom_id: int,
    step: float = 0.03,
    n_circle: int = 40,
) -> np.ndarray:
    """Return perimeter samples (Nx2) in world XY for common geom types.

    Notes
    -----
    We only need a *conservative* point cloud for A* + distance projection.
    planner 会用 d_min (e.g. d_nav) 再膨胀一次，因此这里采样的是“几何表面”。
    """
    gid = int(geom_id)
    gtype = int(m.geom_type[gid])
    size = np.asarray(m.geom_size[gid], dtype=np.float64)
    cen = np.asarray(d.geom_xpos[gid], dtype=np.float64)
    # geom_xmat is 9 values row-major
    try:
        R = np.asarray(d.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    except Exception:
        R = np.eye(3)

    # sphere / cylinder / capsule -> circle in XY
    if gtype in (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
        r = float(size[0])
        ang = np.linspace(0.0, 2.0 * np.pi, int(n_circle), endpoint=False)
        pts = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros_like(ang)], axis=1)
        pts_w = (R @ pts.T).T + cen
        return pts_w[:, :2].astype(np.float64, copy=False)

    # box -> sample rectangle edges in local frame then rotate
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        hx, hy = float(size[0]), float(size[1])
        # pick number of samples along edges based on step
        nx = max(2, int(np.ceil((2.0 * hx) / max(1e-6, step))))
        ny = max(2, int(np.ceil((2.0 * hy) / max(1e-6, step))))
        xs = np.linspace(-hx, hx, nx)
        ys = np.linspace(-hy, hy, ny)
        edge = []
        for x in xs:
            edge.append([x, -hy, 0.0])
            edge.append([x, +hy, 0.0])
        for y in ys:
            edge.append([-hx, y, 0.0])
            edge.append([+hx, y, 0.0])
        pts = np.asarray(edge, dtype=np.float64)
        pts_w = (R @ pts.T).T + cen
        return pts_w[:, :2].astype(np.float64, copy=False)

    # ignore plane / other
    return np.zeros((0, 2), dtype=np.float64)


def _collect_obs_xy(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    body_id_tblock: int,
    name_must_contain=("obst", "wall", "pillar", "dyn_obst"),
) -> np.ndarray:
    """Collect obstacle samples as a single (N,2) array.

    We include geoms whose name contains one of keywords and exclude:
      - T-block geoms (body_id_tblock)
      - goal/marker visuals
    """
    pts_list = []
    for gid in range(int(m.ngeom)):
        # Exclude T-block's own geoms
        if int(m.geom_bodyid[gid]) == int(body_id_tblock):
            continue
        name = _geom_name(m, gid).lower()
        if not name:
            continue
        if "goal" in name or "mark" in name:
            continue
        if not any(k in name for k in name_must_contain):
            continue
        pts = _sample_geom_perimeter_xy(m, d, gid)
        if pts.shape[0] > 0:
            pts_list.append(pts)
    if not pts_list:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack(pts_list)


# -----------------------------
# Viewer overlays (user_scn)
# -----------------------------


def _set_geom_identity_mat(g) -> None:
    I = np.eye(3, dtype=np.float32)
    try:
        if hasattr(g, "mat") and g.mat.shape == (3, 3):
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
        if hasattr(g, "mat") and g.mat.shape == (3, 3):
            g.mat[:] = R.astype(np.float32)
        else:
            g.mat[:] = R.astype(np.float32).reshape(-1)
    except Exception:
        pass
    g.rgba[:] = np.asarray(rgba, dtype=np.float32)


# -----------------------------
# Differential IK (position + optional yaw)
# -----------------------------


@dataclass
class DiffIKConfig:
    damping: float = 1e-4
    dq_cap: float = 0.15  # rad per step
    kp_pos: float = 8.0
    kp_yaw: float = 2.0
    track_yaw: bool = False


def _find_arm_joint_ids(m: mujoco.MjModel, prefix: str) -> list[int]:
    """Return joint ids sorted by index for all hinge joints whose name starts with prefix."""
    ids = []
    for jid in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is None:
            continue
        if name.startswith(prefix) and int(m.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE):
            ids.append(jid)
    return sorted(ids)


def _collect_dof_indices_for_joints(m: mujoco.MjModel, joint_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (qpos_adrs, dof_adrs) arrays for hinge joints."""
    qadrs = []
    dadrs = []
    for jid in joint_ids:
        qadrs.append(int(m.jnt_qposadr[jid]))
        dadrs.append(int(m.jnt_dofadr[jid]))
    return np.asarray(qadrs, dtype=np.int32), np.asarray(dadrs, dtype=np.int32)


def diffik_step(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    site_id: int,
    joint_qpos_adrs: np.ndarray,
    joint_dof_adrs: np.ndarray,
    x_des: np.ndarray,
    yaw_des: float | None,
    cfg: DiffIKConfig,
    dt: float,
) -> None:
    """One small IK step to move site towards x_des (and optionally yaw).

    We solve damped least squares on the selected joint dofs only.
    """
    # current
    x_cur = np.asarray(d.site_xpos[site_id], dtype=np.float64)
    e_pos = (np.asarray(x_des, dtype=np.float64) - x_cur)
    v_task = cfg.kp_pos * e_pos

    # Jacobian
    jacp = np.zeros((3, m.nv), dtype=np.float64)
    jacr = np.zeros((3, m.nv), dtype=np.float64)
    mujoco.mj_jacSite(m, d, jacp, jacr, site_id)

    # Reduce to chosen dofs
    Jp = jacp[:, joint_dof_adrs]
    rows = [Jp]
    b = [v_task]

    if cfg.track_yaw and (yaw_des is not None):
        # yaw about world z from site xmat
        R = np.asarray(d.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
        yaw_cur = float(np.arctan2(R[1, 0], R[0, 0]))
        # wrap
        e_yaw = float((yaw_des - yaw_cur + np.pi) % (2 * np.pi) - np.pi)
        w_task = np.array([0.0, 0.0, cfg.kp_yaw * e_yaw], dtype=np.float64)
        Jr = jacr[:, joint_dof_adrs]
        rows.append(Jr)
        b.append(w_task)

    J = np.vstack(rows)
    y = np.concatenate(b)

    # DLS solve: dq = J^T (J J^T + λI)^{-1} y
    JJt = J @ J.T
    JJt.flat[:: JJt.shape[0] + 1] += float(cfg.damping)
    dq = J.T @ np.linalg.solve(JJt, y)

    # cap dq per step
    dq = _cap_norm(dq, float(cfg.dq_cap))

    # integrate into qpos (hinge joints)
    for k, qadr in enumerate(joint_qpos_adrs):
        d.qpos[qadr] += float(dq[k])

    # clip to joint limits
    for k, jid_dof in enumerate(joint_dof_adrs):
        # qposadr of the joint is aligned for hinge
        qadr = int(joint_qpos_adrs[k])
        # find joint id for this dof via lookup: not direct; we clip via qpos range if available
        # easiest: use joint limits from the corresponding joint id (same k ordering)
        pass

    mujoco.mj_forward(m, d)


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, default="model/fr3_xml_pack/fr3_push_modular_compsim.xml", help="MJCF scene containing FR3 + T-block")
    ap.add_argument("--body", type=str, default="T_siconos")
    ap.add_argument("--ee_site", type=str, default="tool_tip", help="FR3 end-effector site name")
    ap.add_argument("--arm_joint_prefix", type=str, default="fr3_joint", help="prefix for 7 arm joints")

    ap.add_argument("--live_view", action="store_true")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--realtime", action="store_true")

    # Ground MCP
    ap.add_argument("--mu_ground", type=float, default=0.7)
    ap.add_argument("--restitution", type=float, default=0.10)
    ap.add_argument("--contact_eps", type=float, default=1e-6)
    ap.add_argument("--ground_max_iter", type=int, default=60)

    # Tool contact / impedance
    ap.add_argument("--tool_radius", type=float, default=0.05)
    ap.add_argument("--tool_mu", type=float, default=0.6)
    ap.add_argument("--tool_mass", type=float, default=1.0)
    ap.add_argument("--tool_k", type=float, default=800.0)
    ap.add_argument("--tool_d", type=float, default=80.0)
    ap.add_argument("--tool_fmax", type=float, default=500.0)
    ap.add_argument("--tool_vcap", type=float, default=3.0)
    ap.add_argument("--tool_enable_margin", type=float, default=1e-4)

    # Overlays
    ap.add_argument("--ecp_radius", type=float, default=0.02)
    ap.add_argument("--arrow_radius", type=float, default=0.006)
    ap.add_argument("--force_scale", type=float, default=0.02)
    ap.add_argument("--force_max_len", type=float, default=0.25)

    # IK
    ap.add_argument("--ik_track_yaw", action="store_true")
    ap.add_argument("--ik_dq_cap", type=float, default=0.15)
    ap.add_argument("--ik_damping", type=float, default=1e-4)

    args = ap.parse_args()
    xml_path = _abs_path(args.xml)

    # ---- compsim init ----
    compsim.init_from_xml(xml_path, body=args.body)
    print(f"[compsim] xml={xml_path} body={args.body} mass={state.total_mass:.6f} inertia_body_diag={state.inertia_body_diag}")

    # Ground solver
    local_pts = get_local_points_ref(verbose=True)
    solver = compsim.TBlockSimulator_Step_NoBounce(
        dt=float(args.dt),
        mass=float(state.total_mass),
        inertia_diag_body=np.asarray(state.inertia_body_diag, dtype=np.float64),
        local_pts_all=local_pts,
        mu_fric=float(args.mu_ground),
        restitution=float(args.restitution),
        contact_eps=float(args.contact_eps),
    )
    try:
        import siconos.numerics as sn
        solver.options.iparam[sn.SICONOS_IPARAM_MAX_ITER] = int(args.ground_max_iter)
    except Exception:
        pass

    tool = ToolParams(
        radius=float(args.tool_radius),
        mu=float(args.tool_mu),
        mass=float(args.tool_mass),
        restitution=0.0,
        contact_eps=float(args.contact_eps),
        enable_margin=float(args.tool_enable_margin),
        k=float(args.tool_k),
        d=float(args.tool_d),
        fmax=float(args.tool_fmax),
        vcap=float(args.tool_vcap),
    )
    compute_tool_block_impulse.tool_mass = float(tool.mass)

    # ---- MuJoCo visual model ----
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    _disable_all_contacts(m)

    # For obstacle sampling: exclude the pushed object body
    tblock_body_id_vis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, args.body)

    # block qpos adr
    qposadr_block = _find_freejoint_qposadr(m, args.body)

    # EE site
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, args.ee_site)
    if site_id < 0:
        raise RuntimeError(f"ee site '{args.ee_site}' not found")

    # arm joints
    joint_ids = _find_arm_joint_ids(m, args.arm_joint_prefix)
    if len(joint_ids) == 0:
        raise RuntimeError(
            f"No hinge joints found with prefix '{args.arm_joint_prefix}'. "
            "Set --arm_joint_prefix to match your FR3 joint names."
        )
    joint_qpos_adrs, joint_dof_adrs = _collect_dof_indices_for_joints(m, joint_ids)
    if joint_qpos_adrs.shape[0] != joint_dof_adrs.shape[0]:
        raise RuntimeError("Internal joint mapping mismatch")

    ik_cfg = DiffIKConfig(
        damping=float(args.ik_damping),
        dq_cap=float(args.ik_dq_cap),
        track_yaw=bool(args.ik_track_yaw),
    )

    mujoco.mj_forward(m, d)

    # ---- init block state from MuJoCo ----
    qpos0 = np.asarray(d.qpos[qposadr_block : qposadr_block + 7], dtype=np.float64).copy()
    q_com = body_origin_qpos_to_com(qpos0, np.asarray(state.T_IPOS_BODY, dtype=np.float64))
    q_com[3:7] = _norm_quat(q_com[3:7])
    v = np.zeros(6, dtype=np.float64)

    # ---- init tool state from EE ----
    tool_pos = np.asarray(d.site_xpos[site_id], dtype=np.float64).copy()
    tool_vel = np.zeros(3, dtype=np.float64)

    # gravity
    f_applied = np.zeros(6, dtype=np.float64)
    f_applied[2] = -float(state.total_mass) * 9.81

    # planner
    pcfg = PlannerConfig(z_tool=float(tool_pos[2]), vxy_cap=min(0.8, float(tool.vcap)))
    planner = WaypointPushPlanner(pcfg)
    # 只需要设置“当前要追的 waypoint 索引”
    planner.reset(goal_idx=1)

    rgba_ecp = np.array([0.2, 0.4, 1.0, 1.0], dtype=np.float32)
    rgba_force = np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32)

    last_tool_imp_norm = 0.0

    def step_once(step_idx: int):
        nonlocal q_com, v, tool_pos, tool_vel, last_tool_imp_norm

        # 0) planner desired tool target
        obs_xy = _build_obs_xy_from_mujoco(m, d, exclude_body_id=tblock_body_id_vis)
        tool_des, yaw_des, _st, _dbg = planner.step(q_com, tool_pos, obs_xy, float(args.dt))

        # 1) impedance to get free tool velocity
        v_des = (tool_des - tool_pos) / max(1e-9, float(args.dt))
        v_des = _cap_norm(v_des, tool.vcap)

        F_cmd = tool.k * (tool_des - tool_pos) + tool.d * (v_des - tool_vel)
        F_cmd = _cap_norm(F_cmd, tool.fmax)

        tool_vel_free = tool_vel + float(args.dt) * (F_cmd / max(1e-9, tool.mass))
        tool_vel_free = _cap_norm(tool_vel_free, tool.vcap)

        # 2) ground MCP
        v_ground, ecp, info = solver.solve_step(
            q_curr_np=q_com,
            v_curr_np=v,
            f_applied_np=f_applied,
            step_idx=int(step_idx),
            return_ecp=True,
            tool_impulse_norm=float(last_tool_imp_norm),
        )

        # 3) tool-block impulse
        p_tool = np.zeros(3, dtype=np.float64)
        a_tool = np.zeros(3, dtype=np.float64)
        g_tool = 1e9

        p_tool, a_tool, _n_tool, g_tool = compute_tool_block_impulse(
            q_block=q_com,
            v_block6=v_ground,
            tool_pos=tool_pos,
            tool_vel=tool_vel_free,
            tool_radius=tool.radius,
            tool_mu=tool.mu,
            dt=float(args.dt),
            contact_eps=tool.contact_eps,
            restitution=tool.restitution,
            enable_margin=tool.enable_margin,
        )

        last_tool_imp_norm = float(np.linalg.norm(p_tool))

        # 4) apply tool impulse to block twist (fast, no 6x6 M_inv)
        if np.linalg.norm(p_tool) > 1e-12:
            m_block = float(state.total_mass)
            r = a_tool - q_com[:3]
            Iw_inv = inertia_world_inv_from_body_diag_np(state.inertia_body_diag, q_com[3:7])
            dv_lin = p_tool / m_block
            dv_ang = Iw_inv @ np.cross(r, p_tool)
            v_next = np.asarray(v_ground, dtype=np.float64).copy()
            v_next[:3] += dv_lin
            v_next[3:] += dv_ang
        else:
            v_next = np.asarray(v_ground, dtype=np.float64)

        # 5) tool reaction impulse (virtual tool mass)
        tool_vel = tool_vel_free - (p_tool / max(1e-9, tool.mass))
        tool_vel = _cap_norm(tool_vel, tool.vcap)
        tool_pos = tool_pos + float(args.dt) * tool_vel

        # 6) integrate block pose
        q_next = np.asarray(q_com, dtype=np.float64).copy()
        q_next[0:3] += float(args.dt) * v_next[0:3]
        dq = quat_from_omega_world_np(v_next[3:6], float(args.dt))
        q_next[3:7] = quat_mul_wxyz_np(dq, q_next[3:7])
        q_next[3:7] = _norm_quat(q_next[3:7])

        # 7) anti-penetration projection + mild damping
        if getattr(solver, "_frozen", False):
            pn_ground = float(getattr(solver, "z_guess_frozen", np.zeros(10))[9])
        else:
            pn_ground = float(getattr(solver, "z_guess", np.zeros(13))[12])

        q_next2, v_next2, min_z, _supported = project_to_ground_and_damp(
            q_next,
            v_next,
            float(args.dt),
            local_pts,
            pn_ground=pn_ground,
            pn_support_thresh=1e-9,
        )
        q_com[:] = q_next2
        v[:] = v_next2

        # 8) drive FR3 EE to tool_pos (differential IK)
        # (this is purely kinematic; compsim tool dynamics already updated tool_pos)
        diffik_step(
            m,
            d,
            site_id,
            joint_qpos_adrs,
            joint_dof_adrs,
            x_des=tool_pos,
            yaw_des=float(yaw_des) if (ik_cfg.track_yaw and yaw_des is not None) else None,
            cfg=ik_cfg,
            dt=float(args.dt),
        )

        # Write block pose back into MuJoCo
        qpos_vis = com_to_body_origin_qpos(q_com, np.asarray(state.T_IPOS_BODY, dtype=np.float64))
        d.qpos[qposadr_block : qposadr_block + 7] = qpos_vis

        # forward for site/world transforms
        mujoco.mj_forward(m, d)

        F_tool = p_tool / max(1e-12, float(args.dt))
        return {
            "info": int(info),
            "ecp": None if ecp is None else np.asarray(ecp, dtype=np.float64),
            "tool_pos": tool_pos.copy(),
            "F_tool": F_tool.copy(),
            "p_tool": p_tool.copy(),
            "min_z": float(min_z),
            "g_tool": float(g_tool),
        }

    if not args.live_view:
        for i in range(int(args.steps)):
            step_once(i)
        return

    with mujoco.viewer.launch_passive(m, d) as viewer:
        t_start = time.perf_counter()
        for i in range(int(args.steps)):
            if not _viewer_running(viewer):
                break
            t0 = time.perf_counter()

            dbg = step_once(i)

            # Overlays
            try:
                scn = viewer.user_scn
                scn.ngeom = 0
                if dbg["ecp"] is not None:
                    g0 = scn.geoms[0]
                    _set_sphere(g0, dbg["ecp"], float(args.ecp_radius), rgba_ecp)
                    scn.ngeom = 1

                F = dbg["F_tool"]
                Fn = float(np.linalg.norm(F))
                if Fn > 1e-9:
                    direction = F / (Fn + 1e-12)
                    length = min(float(args.force_max_len), float(args.force_scale) * Fn)
                    g1 = scn.geoms[scn.ngeom]
                    _set_arrow(g1, dbg["tool_pos"], direction, float(args.arrow_radius), length, rgba_force)
                    scn.ngeom += 1
            except Exception:
                pass

            viewer.sync()

            if args.realtime:
                elapsed = time.perf_counter() - t0
                remain = float(args.dt) - elapsed
                if remain > 0:
                    time.sleep(remain)

        hz = (i + 1) / max(1e-9, (time.perf_counter() - t_start))
        print(f"[done] steps={i+1} avg_hz={hz:.1f}")


if __name__ == "__main__":
    main()
