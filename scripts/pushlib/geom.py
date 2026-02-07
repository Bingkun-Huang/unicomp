# Auto-split from push_waypoints_compsim_live.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import mujoco

def quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4,)
    return q / (np.linalg.norm(q) + 1e-12)


def enforce_quat_sign_continuity_wxyz(qs_wxyz: np.ndarray) -> np.ndarray:
    qs = np.asarray(qs_wxyz, dtype=np.float64).copy()
    for i in range(1, qs.shape[0]):
        if float(np.dot(qs[i - 1], qs[i])) < 0.0:
            qs[i] = -qs[i]
    return qs


def wrap_to_pi(a: float) -> float:
    a = float(a)
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4,)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = (q / n).tolist()
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def yaw_error_from_quats_wxyz(q_now_wxyz: np.ndarray, q_goal_wxyz: np.ndarray) -> float:
    Rn = _quat_wxyz_to_rotmat(np.asarray(q_now_wxyz, dtype=np.float64).reshape(4,))
    Rg = _quat_wxyz_to_rotmat(np.asarray(q_goal_wxyz, dtype=np.float64).reshape(4,))
    Rrel = Rg @ Rn.T
    yaw_err = float(np.arctan2(Rrel[1, 0], Rrel[0, 0]))
    return wrap_to_pi(yaw_err)


def clamp_norm(x: np.ndarray, max_norm: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = float(np.linalg.norm(x))
    if max_norm <= 0.0 or n <= max_norm:
        return x
    return x * (max_norm / (n + 1e-12))


# ============================================================
# ======================= ROT / GEOM UTIL =====================
# ============================================================

def _rotmat_from_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    return _quat_wxyz_to_rotmat(q_wxyz)


def _box_contains_point_bodyframe(
    p_body: np.ndarray,
    gpos_body: np.ndarray,
    gquat_body_wxyz: np.ndarray,
    gsize_half: np.ndarray,
    eps: float
) -> bool:
    p_body = np.asarray(p_body, dtype=np.float64).reshape(3,)
    gpos_body = np.asarray(gpos_body, dtype=np.float64).reshape(3,)
    gsize_half = np.asarray(gsize_half, dtype=np.float64).reshape(3,)

    Rg = _rotmat_from_quat_wxyz(gquat_body_wxyz)
    plocal = Rg.T @ (p_body - gpos_body)

    return (
        abs(plocal[0]) <= (gsize_half[0] - eps) and
        abs(plocal[1]) <= (gsize_half[1] - eps) and
        abs(plocal[2]) <= (gsize_half[2] - eps)
    )


def _sample_face_points_geomframe(axis: int, sign: float, half: np.ndarray, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    half = np.asarray(half, dtype=np.float64).reshape(3,)
    a = int(axis)
    s = float(sign)

    coord = np.zeros(3, dtype=np.float64)
    coord[a] = s * half[a]

    tang = [0, 1, 2]
    tang.remove(a)
    uax, vax = tang[0], tang[1]

    u_half = float(half[uax])
    v_half = float(half[vax])

    nu = max(3, int(np.ceil((2.0 * u_half) / max(1e-9, spacing))) + 1)
    nv = max(3, int(np.ceil((2.0 * v_half) / max(1e-9, spacing))) + 1)

    us = np.linspace(-u_half, +u_half, nu, dtype=np.float64)
    vs = np.linspace(-v_half, +v_half, nv, dtype=np.float64)

    UU, VV = np.meshgrid(us, vs, indexing="xy")
    pts = np.zeros((UU.size, 3), dtype=np.float64)
    pts[:, a] = coord[a]
    pts[:, uax] = UU.reshape(-1)
    pts[:, vax] = VV.reshape(-1)

    n = np.zeros(3, dtype=np.float64)
    n[a] = s
    return pts, n


# ============================================================
# ===== STRICT "PLOT FACES" SAMPLES (points+normals) ==========
# ============================================================

@dataclass
class SurfaceSamples:
    p_local_com: np.ndarray   # (M,3)
    n_local_com: np.ndarray   # (M,3)


def load_mujoco_body_geoms_desc(xml_path: str, body_name: str = "T_siconos") -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    model = mujoco.MjModel.from_xml_path(xml_path)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Cannot find body '{body_name}' in MuJoCo model.")

    ipos_body = np.array(model.body_ipos[body_id], dtype=np.float64).copy()

    geom_start = int(model.body_geomadr[body_id])
    geom_num = int(model.body_geomnum[body_id])
    geom_ids = np.arange(geom_start, geom_start + geom_num, dtype=np.int32)

    geoms_desc = []
    for gid in geom_ids:
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        size = np.array(model.geom_size[gid], dtype=np.float64).copy()
        gpos_body = np.array(model.geom_pos[gid], dtype=np.float64).copy()
        gquat_body = np.array(model.geom_quat[gid], dtype=np.float64).copy()  # wxyz
        geoms_desc.append((size, gpos_body, gquat_body))

    if len(geoms_desc) == 0:
        raise RuntimeError("No box geoms found under the specified body. Check model definition.")
    return ipos_body.astype(np.float64), geoms_desc


def build_union_side_surface_points_normals_bodyframe(
    geoms_desc: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    q0_body_wxyz: np.ndarray,
    spacing: float,
    side_normal_z_max: float,
    internal_eps: float,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    q0_body_wxyz = np.asarray(q0_body_wxyz, dtype=np.float64).reshape(4,)
    R0 = _rotmat_from_quat_wxyz(q0_body_wxyz)
    z_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    sizes = [np.asarray(s, dtype=np.float64).reshape(3,) for (s, _, _) in geoms_desc]
    gposs = [np.asarray(p, dtype=np.float64).reshape(3,) for (_, p, _) in geoms_desc]
    gquats = [np.asarray(q, dtype=np.float64).reshape(4,) for (_, _, q) in geoms_desc]

    pts_all = []
    n_all = []

    for i, (half, gpos_body, gquat_body) in enumerate(zip(sizes, gposs, gquats)):
        Rg = _rotmat_from_quat_wxyz(gquat_body)

        faces = [
            (0, +1.0), (0, -1.0),
            (1, +1.0), (1, -1.0),
            (2, +1.0), (2, -1.0),
        ]

        for (axis, sign) in faces:
            pts_g, n_g = _sample_face_points_geomframe(axis, sign, half, spacing)

            n_body = Rg @ n_g
            n_world = R0 @ n_body

            if abs(float(np.dot(n_world, z_world))) >= float(side_normal_z_max):
                continue

            pts_b = (Rg @ pts_g.T).T + gpos_body.reshape(1, 3)
            ns_b = np.tile(n_body.reshape(1, 3), (pts_b.shape[0], 1))

            keep = np.ones((pts_b.shape[0],), dtype=bool)
            for j in range(len(geoms_desc)):
                if j == i:
                    continue
                for k in range(pts_b.shape[0]):
                    if not keep[k]:
                        continue
                    if _box_contains_point_bodyframe(
                        pts_b[k], gposs[j], gquats[j], sizes[j], eps=float(internal_eps)
                    ):
                        keep[k] = False

            pts_keep = pts_b[keep]
            ns_keep = ns_b[keep]
            if pts_keep.shape[0] > 0:
                pts_all.append(pts_keep)
                n_all.append(ns_keep)

    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    pts_body = np.concatenate(pts_all, axis=0)
    ns_body = np.concatenate(n_all, axis=0)

    if pts_body.shape[0] > int(max_points):
        stride = int(np.ceil(pts_body.shape[0] / float(max_points)))
        pts_body = pts_body[::stride].copy()
        ns_body = ns_body[::stride].copy()

    ns_n = np.linalg.norm(ns_body, axis=1, keepdims=True)
    ns_body = ns_body / (ns_n + 1e-12)

    return pts_body, ns_body


def build_strict_plot_surface_samples_comframe(
    xml_path: str,
    q0_body_wxyz: np.ndarray,
    body_name: str = "T_siconos",
) -> Tuple[np.ndarray, SurfaceSamples]:
    ipos_body, geoms_desc = load_mujoco_body_geoms_desc(xml_path, body_name=body_name)

    pts_body_origin, ns_body = build_union_side_surface_points_normals_bodyframe(
        geoms_desc=geoms_desc,
        q0_body_wxyz=q0_body_wxyz,
        spacing=float(FACE_SAMPLE_SPACING),
        side_normal_z_max=float(SIDE_NORMAL_Z_MAX),
        internal_eps=float(FACE_INTERNAL_EPS),
        max_points=int(MAX_FACE_POINTS),
    )

    pts_local_com = pts_body_origin - ipos_body.reshape(1, 3)

    samples = SurfaceSamples(
        p_local_com=pts_local_com.astype(np.float64),
        n_local_com=ns_body.astype(np.float64),
    )
    return ipos_body.astype(np.float64), samples


def transform_samples_to_world(q_block: np.ndarray, samples: SurfaceSamples) -> Tuple[np.ndarray, np.ndarray]:
    q_block = np.asarray(q_block, dtype=np.float64).reshape(7,)
    com = q_block[0:3]
    Rb = _rotmat_from_quat_wxyz(q_block[3:7])

    P = samples.p_local_com
    N = samples.n_local_com

    p_w = com.reshape(1, 3) + (Rb @ P.T).T
    n_w = (Rb @ N.T).T
    n_w = n_w / (np.linalg.norm(n_w, axis=1, keepdims=True) + 1e-12)
    return p_w, n_w


# ============================================================
# ===================== CONTACT SELECTION =====================
# ============================================================

def blend_pos_yaw_weights(pos_err_xy: float) -> Tuple[float, float]:
    d = float(max(0.0, pos_err_xy))
    if d >= float(POS_YAW_BLEND_DIST):
        return float(W_POS_FAR), float(W_YAW_FAR)
    a = d / float(POS_YAW_BLEND_DIST + 1e-12)
    w_pos = float(W_POS_NEAR + (W_POS_FAR - W_POS_NEAR) * a)
    w_yaw = float(W_YAW_NEAR + (W_YAW_FAR - W_YAW_NEAR) * a)
    return w_pos, w_yaw


def select_contact_from_surface_samples_translation_only(
    F_xy_des: np.ndarray,
    p_w: np.ndarray,
    n_w: np.ndarray,
    last_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    F_xy_des = np.asarray(F_xy_des, dtype=np.float64).reshape(2,)

    if np.linalg.norm(F_xy_des) > 1e-9:
        d = F_xy_des / (np.linalg.norm(F_xy_des) + 1e-12)
    else:
        d = np.array([1.0, 0.0], dtype=np.float64)

    n_xy = n_w[:, 0:2]
    n_xy_n = np.linalg.norm(n_xy, axis=1)
    valid = n_xy_n > 1e-9

    n_xy_unit = np.zeros_like(n_xy)
    n_xy_unit[valid] = n_xy[valid] / (n_xy_n[valid].reshape(-1, 1) + 1e-12)

    push_dir = -n_xy_unit
    push_align = np.sum(push_dir * d.reshape(1, 2), axis=1)

    score = push_align.copy()
    score[~valid] = -1e18
    score[push_align < float(CONTACT_MIN_ALIGN)] = -1e18

    if last_idx is not None and 0 <= int(last_idx) < score.shape[0]:
        score[int(last_idx)] += float(CONTACT_HYSTERESIS_BONUS)

    best_idx = int(np.argmax(score))
    best_score = float(score[best_idx])
    return p_w[best_idx].copy(), n_w[best_idx].copy(), best_idx, best_score


def select_contact_from_surface_samples_translation_yaw(
    q_block: np.ndarray,
    pos_dir_xy: np.ndarray,
    yaw_err: float,
    w_pos: float,
    w_yaw: float,
    p_w: np.ndarray,
    n_w: np.ndarray,
    last_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    q_block = np.asarray(q_block, dtype=np.float64).reshape(7,)
    com = q_block[0:3].reshape(1, 3)

    pos_dir_xy = np.asarray(pos_dir_xy, dtype=np.float64).reshape(2,)
    if np.linalg.norm(pos_dir_xy) < 1e-9:
        pos_dir_xy = np.array([1.0, 0.0], dtype=np.float64)

    yaw_err = float(wrap_to_pi(yaw_err))
    need_yaw = abs(yaw_err) >= float(YAW_ERR_ACTIVE)
    yaw_sign = 0.0
    yaw_scale = 0.0
    if need_yaw:
        yaw_sign = 1.0 if yaw_err > 0.0 else -1.0
        yaw_scale = min(1.0, abs(yaw_err) / float(YAW_ERR_REF + 1e-12))

    n_xy = n_w[:, 0:2]
    n_xy_n = np.linalg.norm(n_xy, axis=1)
    valid = n_xy_n > 1e-9

    n_xy_unit = np.zeros_like(n_xy)
    n_xy_unit[valid] = n_xy[valid] / (n_xy_n[valid].reshape(-1, 1) + 1e-12)
    push_dir = -n_xy_unit

    pos_align = np.sum(push_dir * pos_dir_xy.reshape(1, 2), axis=1)

    r = (p_w - com)
    tau_z = r[:, 0] * push_dir[:, 1] - r[:, 1] * push_dir[:, 0]
    yaw_score = yaw_sign * tau_z

    w_yaw_eff = float(w_yaw) * float(yaw_scale)
    min_align = float(CONTACT_MIN_ALIGN)
    if w_yaw_eff > 0.5:
        min_align = max(float(MIN_ALIGN_RELAX_YAW), float(CONTACT_MIN_ALIGN) - 0.35)

    score = np.full((p_w.shape[0],), -1e18, dtype=np.float64)
    if np.any(valid):
        base_ok = valid & (pos_align >= min_align)

        if w_yaw_eff > 0.4:
            base_ok = base_ok & (yaw_score >= -1e-12)

        s = (
            float(w_pos) * pos_align
            + float(w_yaw_eff) * (float(TORQUE_SCORE_GAIN) * yaw_score + float(LEVER_SCORE_GAIN) * np.abs(tau_z))
        )
        score[base_ok] = s[base_ok]

    if last_idx is not None and 0 <= int(last_idx) < score.shape[0]:
        score[int(last_idx)] += float(CONTACT_HYSTERESIS_BONUS)

    best_idx = int(np.argmax(score))
    best_score = float(score[best_idx])
    return p_w[best_idx].copy(), n_w[best_idx].copy(), best_idx, best_score


# ============================================================
# ======================= SDF NAV HELPERS =====================
# ============================================================

def enforce_min_sdf_distance(x_des: np.ndarray, q_block: np.ndarray, d_min: float) -> np.ndarray:
    x_des = np.asarray(x_des, dtype=np.float64).reshape(3,)
    if not hasattr(sim, "closest_point_on_tblock_surface_world"):
        return x_des

    a, sdf = sim.closest_point_on_tblock_surface_world(x_des, q_block)
    a = np.asarray(a, dtype=np.float64).reshape(3,)
    d = float(sdf)

    if d >= float(d_min):
        return x_des

    v = x_des - a
    nv = float(np.linalg.norm(v))
    if nv < 1e-12:
        v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        nv = 1.0
    n_out = v / nv
    x_proj = x_des + (float(d_min) - d) * n_out
    return x_proj


def kinematic_step_towards(x_cur: np.ndarray, x_goal: np.ndarray, dt: float, vxy_max: float) -> np.ndarray:
    x_cur = np.asarray(x_cur, dtype=np.float64).reshape(3,)
    x_goal = np.asarray(x_goal, dtype=np.float64).reshape(3,)

    dx = x_goal - x_cur
    dx[2] = 0.0
    dist = float(np.linalg.norm(dx))
    if dist < 1e-12:
        return x_cur.copy()

    step = min(dist, float(vxy_max) * float(dt))
    return x_cur + dx * (step / (dist + 1e-12))


def suppress_tangential_motion_near_contact(
    tool_pos: np.ndarray,
    tool_pos_next: np.ndarray,
    n_out: np.ndarray,
    gap: float,
    # NOTE: keep defaults self-contained to avoid circular imports.
    # In the original monolithic script these came from:
    #   gap_lock = TOUCH_CLEARANCE
    #   gap_free = NAV_CLEARANCE * 1.25
    # If you want to tune them globally, pass explicit values from state_machine.
    gap_lock: float = 5e-4,
    gap_free: float = 1.25e-2,
) -> np.ndarray:
    tool_pos = np.asarray(tool_pos, dtype=np.float64).reshape(3,)
    tool_pos_next = np.asarray(tool_pos_next, dtype=np.float64).reshape(3,)
    n_out = np.asarray(n_out, dtype=np.float64).reshape(3,)

    if gap_free <= gap_lock:
        return tool_pos_next.copy()

    alpha = (float(gap) - float(gap_lock)) / (float(gap_free) - float(gap_lock) + 1e-12)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    n_xy = np.array([n_out[0], n_out[1]], dtype=np.float64)
    nn = float(np.linalg.norm(n_xy))
    if nn < 1e-9:
        return tool_pos_next.copy()
    n_xy /= (nn + 1e-12)

    dp = tool_pos_next[0:2] - tool_pos[0:2]
    dp_n = n_xy * float(np.dot(dp, n_xy))
    dp_t = dp - dp_n

    dp_new = dp_n + alpha * dp_t
    out = tool_pos_next.copy()
    out[0:2] = tool_pos[0:2] + dp_new
    return out


def filter_samples_by_height(p_w: np.ndarray, n_w: np.ndarray, z_tool: float, band: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_w = np.asarray(p_w, dtype=np.float64)
    n_w = np.asarray(n_w, dtype=np.float64)
    dz = np.abs(p_w[:, 2] - float(z_tool))
    mask = dz <= float(band)
    idx = np.where(mask)[0].astype(np.int32)
    if idx.shape[0] < 20:
        idx = np.arange(p_w.shape[0], dtype=np.int32)
    return p_w[idx], n_w[idx], idx


# ============================================================
