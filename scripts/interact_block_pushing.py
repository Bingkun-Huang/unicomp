#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from compsim.math3d import inertia_world_inv_from_body_diag_np
import compsim
from compsim import state
from compsim.samples import get_local_points_ref
from compsim.pose import body_origin_qpos_to_com, com_to_body_origin_qpos
from compsim.tool_contact import compute_tool_block_impulse
from compsim.math3d import (
    mass_matrix_inv_6x6,
    build_jacobian_single,
    quat_from_omega_world_np,
    quat_mul_wxyz_np,
)
from compsim.integration import project_to_ground_and_damp


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


# -----------------------------
# Helpers
# -----------------------------

def _abs_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _inject_tool_mocap_if_missing(xml_text: str, tool_name: str, radius: float) -> str:
    """Inject a mocap tool sphere into a *single-file* MJCF string if it's missing."""
    if f'name="{tool_name}"' in xml_text:
        return xml_text
    idx = xml_text.rfind("</worldbody>")
    if idx < 0:
        raise ValueError("MJCF has no </worldbody>; cannot inject tool body")
    tool_xml = f"""
    <body name=\"{tool_name}\" mocap=\"true\" pos=\"0 0 0.25\">
      <!-- Keep group default so it is visible. Contacts are disabled anyway. -->
      <geom name=\"{tool_name}_geom\" type=\"sphere\" size=\"{radius:.6f}\" rgba=\"0 1 0 1\"
            contype=\"0\" conaffinity=\"0\"/>
    </body>
"""
    return xml_text[:idx] + tool_xml + xml_text[idx:]


def _disable_all_contacts(m: mujoco.MjModel) -> None:
    """Make MuJoCo purely visual. (compsim handles all contact)"""
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
        raise ValueError(f"body '{body_name}' not found in visualization model")
    jadr = int(m.body_jntadr[bid])
    jnum = int(m.body_jntnum[bid])
    for k in range(jnum):
        jid = jadr + k
        if int(m.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(m.jnt_qposadr[jid])
    raise ValueError(f"body '{body_name}' has no freejoint")


def _norm_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _find_mocap_id(m: mujoco.MjModel, body_name: str) -> int:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return -1
    return int(m.body_mocapid[bid])


def _viewer_running(viewer) -> bool:
    # mujoco.viewer variants across versions
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


def _init_overlays_v081_like(viewer, ecp_radius: float, arrow_width: float) -> bool:
    """Initialize viewer.user_scn geoms using mjv_initGeom (important for visibility)."""
    if not hasattr(viewer, "user_scn"):
        return False
    try:
        scn = viewer.user_scn
        scn.ngeom = 2
        I = np.eye(3, dtype=np.float64).reshape(-1)
        rgba_force = np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float64)  # red
        rgba_ecp = np.array([0.2, 0.4, 1.0, 1.0], dtype=np.float64)    # blue

        # Arrow geom (will be updated by mjv_connector each step)
        mujoco.mjv_initGeom(
            scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.array([arrow_width, 1e-6, arrow_width], dtype=np.float64),
            np.array([100.0, 100.0, 100.0], dtype=np.float64),
            I,
            rgba_force,
        )

        # ECP sphere
        mujoco.mjv_initGeom(
            scn.geoms[1],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([ecp_radius, 0.0, 0.0], dtype=np.float64),
            np.array([100.0, 100.0, 100.0], dtype=np.float64),
            I,
            rgba_ecp,
        )

        return True
    except Exception:
        return False


def _update_overlays_v081_like(
    viewer,
    ecp: np.ndarray | None,
    ecp_lift: float,
    tool_pos: np.ndarray,
    force_world: np.ndarray,
    force_scale: float,
    force_max_len: float,
    arrow_width: float,
) -> None:
    scn = viewer.user_scn

    # ECP marker
    if ecp is not None:
        p = np.asarray(ecp, dtype=np.float64).copy()
        p[2] += float(ecp_lift)
        scn.geoms[1].pos[:] = p
    else:
        scn.geoms[1].pos[:] = np.array([100.0, 100.0, 100.0], dtype=np.float64)

    # Force arrow from tool center
    F = np.asarray(force_world, dtype=np.float64)
    Fn = float(np.linalg.norm(F))
    if Fn > 1e-9:
        dvec = F / (Fn + 1e-12)
        length = min(float(force_max_len), float(force_scale) * Fn)
        frm = np.asarray(tool_pos, dtype=np.float64)
        to = frm + dvec * length
        mujoco.mjv_connector(scn.geoms[0], mujoco.mjtGeom.mjGEOM_ARROW, float(arrow_width), frm, to)
    else:
        scn.geoms[0].pos[:] = np.array([100.0, 100.0, 100.0], dtype=np.float64)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, default="model/t_block_optimized2.xml")
    ap.add_argument("--body", type=str, default="T_siconos")
    ap.add_argument("--target_mocap", type=str, default="target_mocap")
    ap.add_argument("--tool_mocap", type=str, default="tool_mocap")

    ap.add_argument("--view", action="store_true")
    ap.add_argument("--steps", type=int, default=400000)

    # Simulation accuracy params
    ap.add_argument("--ground_max_iter", type=int, default=180)
    ap.add_argument("--ground_max_iter_fallback", type=int, default=80)
    ap.add_argument("--ground_budget_ms", type=float, default=10.0)
    ap.add_argument("--spike_ms", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.001)

    # Ground / MCP params
    ap.add_argument("--mu_ground", type=float, default=0.9)
    ap.add_argument("--restitution", type=float, default=0.10)
    ap.add_argument("--contact_eps", type=float, default=1e-6)

    # Tool contact params
    ap.add_argument("--use_tool", action="store_true", default=True)
    ap.add_argument("--tool_radius", type=float, default=0.06)
    ap.add_argument("--tool_mu", type=float, default=0.6)
    ap.add_argument("--tool_mass", type=float, default=5.0)
    ap.add_argument("--tool_restitution", type=float, default=0.0)
    ap.add_argument("--tool_enable_margin", type=float, default=1e-4)

    # Tool impedance params
    ap.add_argument("--tool_k", type=float, default=800.0)
    ap.add_argument("--tool_d", type=float, default=80.0)
    ap.add_argument("--tool_fmax", type=float, default=100.0)
    ap.add_argument("--tool_vcap", type=float, default=3.0, help="max tool speed (m/s)")

    # Overlay sizes (only for visualization)
    ap.add_argument("--ecp_radius", type=float, default=0.012)
    ap.add_argument("--ecp_lift", type=float, default=0.003)
    ap.add_argument("--arrow_width", type=float, default=0.01)
    ap.add_argument("--force_scale", type=float, default=0.1, help="arrow length per Newton")
    ap.add_argument("--force_max_len", type=float, default=0.50)

    # Real-time pacing
    ap.add_argument("--realtime", action="store_true", help="sleep to roughly match dt")

    args = ap.parse_args()
    args.xml = _abs_path(args.xml)

    # 1) Initialize compsim state from XML (caches mass/inertia/local offsets)
    compsim.init_from_xml(args.xml, body=args.body)

    print("\n===== [compsim] Block params =====")
    print(f"body name        : {args.body}")
    print(f"total_mass [kg]  : {float(state.total_mass):.6f}")
    print(f"inertia_body_diag [kg*m^2] : {np.asarray(state.inertia_body_diag, dtype=np.float64)}")


    # 2) Ground solver
    local_pts = get_local_points_ref(verbose=True)
    solver = compsim.TBlockSimulator_Step_NoBounce(
        dt=float(args.dt),
        mass=float(state.total_mass),
        inertia_diag_body=np.asarray(state.inertia_body_diag, dtype=np.float64),
        local_pts_all=local_pts,
        mu_fric=float(args.mu_ground),
        restitution=float(args.restitution),
        contact_eps=float(args.contact_eps),
        support_z=0.0,
    )

    # Cap Siconos iterations to avoid long stalls
    try:
        import siconos.numerics as sn
        solver.options.iparam[sn.SICONOS_IPARAM_MAX_ITER] = int(args.ground_max_iter)
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
    compute_tool_block_impulse.tool_mass = float(tool.mass)

    # 3) Visualization model (try to inject tool mocap for single-file MJCF)
    xml_text = open(args.xml, "r", encoding="utf-8").read()
    xml_text_inj = _inject_tool_mocap_if_missing(xml_text, args.tool_mocap, radius=tool.radius)

    try:
        m = mujoco.MjModel.from_xml_string(xml_text_inj)
    except Exception:
        m = mujoco.MjModel.from_xml_path(args.xml)

    d = mujoco.MjData(m)
    _disable_all_contacts(m)

    # addresses / ids
    qposadr = _find_freejoint_qposadr(m, args.body)

    target_mocap_id = _find_mocap_id(m, args.target_mocap)
    if target_mocap_id < 0:
        raise RuntimeError(f"target mocap body '{args.target_mocap}' not found or not mocap=true")

    tool_mocap_id = _find_mocap_id(m, args.tool_mocap)
    if tool_mocap_id < 0:
        raise RuntimeError(
            f"tool mocap body '{args.tool_mocap}' not found. "
            "Either use a single-file XML (so injection works) or add it manually to your XML."
        )

    # Force initial overlap: tool_mocap (green) == target_mocap (red)
    d.mocap_pos[tool_mocap_id][:] = d.mocap_pos[target_mocap_id][:]

    tool_pos = np.asarray(d.mocap_pos[tool_mocap_id], dtype=np.float64).copy()
    tool_vel = np.zeros(3, dtype=np.float64)
    tool_des_prev = np.asarray(d.mocap_pos[target_mocap_id], dtype=np.float64).copy()

    # Initialize internal block state from the visualization qpos
    qpos0 = np.asarray(d.qpos[qposadr : qposadr + 7], dtype=np.float64).copy()
    q_com = body_origin_qpos_to_com(qpos0, np.asarray(state.T_IPOS_BODY, dtype=np.float64))
    q_com[3:7] = _norm_quat(q_com[3:7])
    v = np.zeros(6, dtype=np.float64)

    # gravity wrench on COM
    f_applied = np.zeros(6, dtype=np.float64)
    f_applied[2] = -float(state.total_mass) * 9.81

    last_tool_imp_norm = 0.0
    last_good_ground_v = None
    last_good_ecp = None

    def step_once(step_idx: int):
        nonlocal q_com, v, tool_pos, tool_vel, tool_des_prev
        nonlocal last_tool_imp_norm, last_good_ground_v, last_good_ecp

        # Read target (red mocap)
        tool_des = np.asarray(d.mocap_pos[target_mocap_id], dtype=np.float64).copy()

        # Desired velocity from target motion
        v_des = (tool_des - tool_des_prev) / float(args.dt)
        v_des = _cap_norm(v_des, tool.vcap)
        tool_des_prev = tool_des.copy()

        # Impedance force (world frame)
        F_cmd = tool.k * (tool_des - tool_pos) + tool.d * (v_des - tool_vel)
        F_cmd = _cap_norm(F_cmd, tool.fmax)

        # Free tool velocity (before contact)
        tool_vel_free = tool_vel + float(args.dt) * (F_cmd / max(1e-9, tool.mass))
        tool_vel_free = _cap_norm(tool_vel_free, tool.vcap)

        # 1) ground solve (MCP)
        t0 = time.perf_counter()
        v_ground, ecp, info = solver.solve_step(
            q_curr_np=q_com,
            v_curr_np=v,
            f_applied_np=f_applied,
            step_idx=int(step_idx),
            return_ecp=True,
            tool_impulse_norm=float(last_tool_imp_norm),
        )
        t_ground = (time.perf_counter() - t0) * 1000.0

        # If it fails or runs long, fall back to last successful solution (avoids visible hitch)
        if (int(info) != 0) or (t_ground > float(args.ground_budget_ms)):
            if last_good_ground_v is not None:
                v_ground = last_good_ground_v
                ecp = last_good_ecp
            try:
                import siconos.numerics as sn
                solver.options.iparam[sn.SICONOS_IPARAM_MAX_ITER] = int(args.ground_max_iter_fallback)
            except Exception:
                pass
        else:
            last_good_ground_v = np.asarray(v_ground, dtype=np.float64).copy()
            last_good_ecp = None if ecp is None else np.asarray(ecp, dtype=np.float64).copy()
            try:
                import siconos.numerics as sn
                solver.options.iparam[sn.SICONOS_IPARAM_MAX_ITER] = int(args.ground_max_iter)
            except Exception:
                pass

        # 2) tool-block impulse
        p_tool = np.zeros(3, dtype=np.float64)
        a_tool = np.zeros(3, dtype=np.float64)
        g_tool = 1e9

        if args.use_tool:
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

        # 3) apply tool impulse to block twist
        # com = q_com[0:3]
        # r = a_tool - com
        # M_inv = np.asarray(
        #     mass_matrix_inv_6x6(
        #         jnp.array(float(state.total_mass), dtype=jnp.float64),
        #         jnp.array(np.asarray(state.inertia_body_diag, dtype=np.float64), dtype=jnp.float64),
        #         jnp.array(np.asarray(q_com[3:7], dtype=np.float64), dtype=jnp.float64),
        #     )
        # )
        # J = np.asarray(build_jacobian_single(jnp.array(r, dtype=jnp.float64)))
        # dv = M_inv @ (J.T @ p_tool)
        # v_next = np.asarray(v_ground, dtype=np.float64) + np.asarray(dv, dtype=np.float64)


        if np.linalg.norm(p_tool) > 1e-12:
            m = float(state.total_mass)
            r = a_tool - q_com[:3]
            Iw_inv = inertia_world_inv_from_body_diag_np(state.inertia_body_diag, q_com[3:7])

            dv_lin = p_tool / m
            dv_ang = Iw_inv @ np.cross(r, p_tool)

            v_next = np.asarray(v_ground, dtype=np.float64).copy()
            v_next[:3] += dv_lin
            v_next[3:] += dv_ang
        else:
            v_next = np.asarray(v_ground, dtype=np.float64)

        # 4) tool reaction: v_tool := v_tool_free - p/m
        tool_vel = tool_vel_free - (p_tool / max(1e-9, tool.mass))
        tool_vel = _cap_norm(tool_vel, tool.vcap)
        tool_pos = tool_pos + float(args.dt) * tool_vel
        d.mocap_pos[tool_mocap_id] = tool_pos

        # 5) integrate block pose
        q_next = np.asarray(q_com, dtype=np.float64).copy()
        q_next[0:3] += float(args.dt) * v_next[0:3]
        dq = quat_from_omega_world_np(v_next[3:6], float(args.dt))
        q_next[3:7] = quat_mul_wxyz_np(dq, q_next[3:7])
        q_next[3:7] = _norm_quat(q_next[3:7])

        # 6) anti-penetration projection + mild damping
        if getattr(solver, "_frozen", False):
            pn_ground = float(getattr(solver, "z_guess_frozen", np.zeros(10))[9])
        else:
            pn_ground = float(getattr(solver, "z_guess", np.zeros(13))[12])

        q_next2, v_next2, min_z, supported = project_to_ground_and_damp(
            q_next,
            v_next,
            float(args.dt),
            local_pts,
            pn_ground=pn_ground,
            pn_support_thresh=1e-9,
            support_z=0.0,
        )

        q_com[:] = q_next2
        v[:] = v_next2

        # Visualization
        ecp_vis = None
        if ecp is not None:
            ecp_vis = np.asarray(ecp, dtype=np.float64)
        elif last_good_ecp is not None:
            ecp_vis = np.asarray(last_good_ecp, dtype=np.float64)

        F_tool = p_tool / max(1e-12, float(args.dt))
        # print("Force of tool: ", F_tool)

        return {
            "info": int(info),
            "t_ground_ms": float(t_ground),
            "pn_ground": float(pn_ground),
            "p_tool": p_tool.copy(),
            "F_tool": F_tool.copy(),
            "tool_pos": tool_pos.copy(),
            "ecp": ecp_vis,
            "min_z": float(min_z),
            "g_tool": float(g_tool),
        }

    if not args.view:
        for i in range(int(args.steps)):
            step_once(i)
        return

    # Viewer loop
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Ensure MuJoCo has something rendered once before touching user_scn
        mujoco.mj_forward(m, d)
        viewer.sync()

        overlay_ok = _init_overlays_v081_like(viewer, float(args.ecp_radius), float(args.arrow_width))
        if not overlay_ok:
            print("[warn] viewer.user_scn overlays not available; ECP/arrow will not be drawn")

        t_start = time.perf_counter()
        for i in range(int(args.steps)):
            if not _viewer_running(viewer):
                break

            t_frame0 = time.perf_counter()
            dbg = step_once(i)

            # Write block pose back into MuJoCo for visualization
            qpos_vis = com_to_body_origin_qpos(q_com, np.asarray(state.T_IPOS_BODY, dtype=np.float64))
            d.qpos[qposadr : qposadr + 7] = qpos_vis

            t0 = time.perf_counter()
            mujoco.mj_forward(m, d)
            t_mj_forward = (time.perf_counter() - t0) * 1000.0

            # Overlays: ECP sphere + force arrow
            if overlay_ok:
                try:
                    _update_overlays_v081_like(
                        viewer,
                        dbg["ecp"],
                        float(args.ecp_lift),
                        dbg["tool_pos"],
                        dbg["F_tool"],
                        float(args.force_scale),
                        float(args.force_max_len),
                        float(args.arrow_width),
                    )
                except Exception as e:
                    overlay_ok = False
                    print("[warn] overlay update failed:", e)

            t0 = time.perf_counter()
            viewer.sync()
            t_sync = (time.perf_counter() - t0) * 1000.0

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