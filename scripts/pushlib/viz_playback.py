# Auto-split from push_waypoints_compsim_live.py
from __future__ import annotations

import numpy as np
import mujoco
import mujoco.viewer

import compsim as sim

from .geom import _quat_wxyz_to_rotmat

# Defaults (kept consistent with original script)
GHOST_RGBA = (0.2, 0.9, 0.2, 1.0)
GHOST_ALPHA_REF = 0.30
GHOST_ALPHA_WP = 0.60
VIEW_FORCE_SCALE = 0.03
VIEW_ARROW_WIDTH = 0.01
VIEW_ARROW_LEN_MAX = 0.9

# =================== VISUALIZATION (DO NOT EDIT) =============
# ============================================================

VIEW_SHOW_PATHS = True
VIEW_MAX_PATH_SEGS = 100
VIEW_PATH_RADIUS_TRAJ = 0.0025
VIEW_PATH_RADIUS_REF  = 0.0020

VIEW_SHOW_ECP = True
VIEW_ECP_RADIUS = 0.012
VIEW_SHOW_CONTACT_POINT = True
VIEW_CONTACT_RADIUS = 0.010
VIEW_TOOL_DES_RADIUS = 0.010

VIEW_SHOW_CONTACTABLE_SIDE_FACES = False
FACE_POINT_RADIUS = 0.0032
FACE_POINT_RGBA = np.array([0.15, 0.95, 0.15, 0.95], dtype=np.float32)


def build_union_side_surface_points_bodyframe(
    geoms_desc: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    q0_body_wxyz: np.ndarray,
    spacing: float,
    side_normal_z_max: float,
    internal_eps: float,
    max_points: int,
) -> np.ndarray:
    q0_body_wxyz = np.asarray(q0_body_wxyz, dtype=np.float64).reshape(4,)
    R0 = _rotmat_from_quat_wxyz(q0_body_wxyz)

    z_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    sizes = [np.asarray(s, dtype=np.float64).reshape(3,) for (s, _, _) in geoms_desc]
    gposs = [np.asarray(p, dtype=np.float64).reshape(3,) for (_, p, _) in geoms_desc]
    gquats = [np.asarray(q, dtype=np.float64).reshape(4,) for (_, _, q) in geoms_desc]

    pts_body_all = []

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
            if pts_keep.shape[0] > 0:
                pts_body_all.append(pts_keep)

    if len(pts_body_all) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    pts_body = np.concatenate(pts_body_all, axis=0)

    if pts_body.shape[0] > int(max_points):
        stride = int(np.ceil(pts_body.shape[0] / float(max_points)))
        pts_body = pts_body[::stride].copy()

    return pts_body


# ============================================================
# ===================== PLAYBACK VIEWER (unchanged) ===========
# ============================================================
# （你原本的 visualize_push_waypoints_mujoco 保持不动；此处省略：你贴的版本已包含在你的文件中）
# 下面为了“整文件可直接复制运行”，我保留你贴的完整函数实现（未修改）

def visualize_push_waypoints_mujoco(
    traj_q: np.ndarray,
    ref_p: np.ndarray,
    ref_q: np.ndarray,
    wp_t: np.ndarray,
    wp_p: np.ndarray,
    wp_q: np.ndarray,
    dt: float,
    tool_pos: Optional[np.ndarray] = None,
    tool_des: Optional[np.ndarray] = None,
    tool_force: Optional[np.ndarray] = None,
    tool_contact_pt: Optional[np.ndarray] = None,
    ecp_hist: Optional[np.ndarray] = None,
    xml_path: Optional[str] = None,
    joint_name_free: str = "root_siconos",
    body_name: str = "T_siconos",
    render_fps: float = 60.0,
    playback_speed: float = 1.0,
):
    if xml_path is None:
        xml_path = getattr(sim, "XML_PATH", None)
    if xml_path is None:
        raise RuntimeError("xml_path is None. Pass xml_path or ensure sim.XML_PATH exists.")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name_free)
    if jid < 0:
        raise RuntimeError(f"Cannot find joint '{joint_name_free}' in MuJoCo model.")
    qadr = int(model.jnt_qposadr[jid])

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    ipos_body = np.array(model.body_ipos[body_id], dtype=np.float64) if body_id >= 0 else np.zeros(3, dtype=np.float64)

    geom_start = int(model.body_geomadr[body_id])
    geom_num = int(model.body_geomnum[body_id])
    geom_ids = np.arange(geom_start, geom_start + geom_num, dtype=np.int32)

    geoms_desc = []
    for gid in geom_ids:
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        size = np.array(model.geom_size[gid], dtype=np.float64).copy()
        gpos_body = np.array(model.geom_pos[gid], dtype=np.float64).copy()
        gquat_body = np.array(model.geom_quat[gid], dtype=np.float64).copy()
        geoms_desc.append((size, gpos_body, gquat_body))

    def com_to_body_origin_qpos(q_com: np.ndarray) -> np.ndarray:
        q_com = np.asarray(q_com, dtype=np.float64).reshape(7,)
        p_com = q_com[0:3]
        q_wxyz = q_com[3:7]
        Rm = _quat_wxyz_to_rotmat(q_wxyz)
        p_body = p_com - Rm @ ipos_body
        return np.concatenate([p_body, q_wxyz], axis=0)

    def _mjv_init(scene, geom_type, size_xyz, pos_xyz, mat_3x3, rgba_4):
        if scene.ngeom >= scene.maxgeom:
            return None
        g = scene.geoms[scene.ngeom]
        size64 = np.asarray(size_xyz, dtype=np.float64).reshape(3, 1)
        pos64  = np.asarray(pos_xyz,  dtype=np.float64).reshape(3, 1)
        mat64  = np.asarray(mat_3x3,  dtype=np.float64).reshape(9, 1)
        rgba32 = np.asarray(rgba_4,   dtype=np.float32).reshape(4, 1)
        mujoco.mjv_initGeom(g, int(geom_type), size64, pos64, mat64, rgba32)
        scene.ngeom += 1
        return g

    def add_box(scene, pos_world, q_wxyz, size_xyz, rgba):
        Rm = _quat_wxyz_to_rotmat(q_wxyz)
        _mjv_init(scene, mujoco.mjtGeom.mjGEOM_BOX, size_xyz, pos_world, Rm, rgba)

    def add_sphere(scene, pos_world, radius, rgba):
        _mjv_init(scene, mujoco.mjtGeom.mjGEOM_SPHERE,
                  np.array([radius, 0.0, 0.0], dtype=np.float64),
                  pos_world, np.eye(3, dtype=np.float64), rgba)

    def add_ghost_body(scene, q_com_wxyz7, rgba):
        q_com_wxyz7 = np.asarray(q_com_wxyz7, dtype=np.float64).reshape(7,)
        p_com = q_com_wxyz7[0:3]
        q_body = q_com_wxyz7[3:7]
        R_body = _quat_wxyz_to_rotmat(q_body)
        p_body = p_com - R_body @ ipos_body

        for (size, gpos_body, gquat_body) in geoms_desc:
            p_geom_w = p_body + R_body @ gpos_body
            q_geom_w = sim.quat_mul_wxyz_np(q_body, gquat_body)
            add_box(scene, p_geom_w, q_geom_w, size, rgba)

    def add_force_arrow(scene, p_from, F_world, rgba):
        F_world = np.asarray(F_world, dtype=np.float64).reshape(3,)
        mag = float(np.linalg.norm(F_world))
        g = _mjv_init(
            scene,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.array([VIEW_ARROW_WIDTH, 1e-6, VIEW_ARROW_WIDTH], dtype=np.float64),
            np.array([100.0, 100.0, 100.0], dtype=np.float64),
            np.eye(3, dtype=np.float64),
            rgba,
        )
        if g is None:
            return
        if mag < 1e-10:
            g.pos[:] = np.array([100.0, 100.0, 100.0], dtype=np.float64)
            return

        d = F_world / mag
        L = min(float(VIEW_FORCE_SCALE) * mag, float(VIEW_ARROW_LEN_MAX))
        frm = np.asarray(p_from, dtype=np.float64).reshape(3,)
        to  = frm + d * L
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, float(VIEW_ARROW_WIDTH), frm, to)

    def add_polyline(scene, pts, rgba, radius=0.0025, max_segs=80):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] < 2:
            return
        n = pts.shape[0]
        stride = max(1, n // int(max_segs))
        idx = np.arange(0, n, stride, dtype=int)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        ps = pts[idx]

        for i in range(ps.shape[0] - 1):
            if scene.ngeom >= scene.maxgeom:
                break
            p0 = ps[i]
            p1 = ps[i + 1]
            d  = p1 - p0
            L  = float(np.linalg.norm(d))
            if L < 1e-9:
                continue
            mid  = 0.5 * (p0 + p1)
            z = d / L
            if abs(z[2]) < 0.99:
                a = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            x = np.cross(a, z)
            nx = float(np.linalg.norm(x))
            if nx < 1e-12:
                continue
            x /= nx
            y = np.cross(z, x)
            Rm = np.column_stack([x, y, z])
            _mjv_init(scene, mujoco.mjtGeom.mjGEOM_CAPSULE,
                      np.array([radius, 0.5 * L, 0.0], dtype=np.float64),
                      mid, Rm, rgba)

    contactable_pts_body = np.zeros((0, 3), dtype=np.float64)
    if VIEW_SHOW_CONTACTABLE_SIDE_FACES and len(geoms_desc) > 0 and traj_q.shape[0] > 0:
        q0_com = np.asarray(traj_q[0], dtype=np.float64).reshape(7,)
        q0_body_wxyz = q0_com[3:7]

        contactable_pts_body = build_union_side_surface_points_bodyframe(
            geoms_desc=geoms_desc,
            q0_body_wxyz=q0_body_wxyz,
            spacing=float(FACE_SAMPLE_SPACING),
            side_normal_z_max=float(SIDE_NORMAL_Z_MAX),
            internal_eps=float(FACE_INTERNAL_EPS),
            max_points=int(MAX_FACE_POINTS),
        )

    traj_path = traj_q[:, 0:3]
    ref_path  = ref_p[:, 0:3]
    nframes = int(traj_q.shape[0])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0

        t_start = time.perf_counter()
        t_last = t_start

        while viewer.is_running():
            now = time.perf_counter()
            elapsed = (now - t_start) * float(playback_speed)
            k = int(elapsed / float(dt)) % nframes

            qk_com = traj_q[k].astype(np.float64).reshape(7,)
            qk_vis = com_to_body_origin_qpos(qk_com)
            data.qpos[qadr:qadr + 7] = qk_vis
            mujoco.mj_forward(model, data)

            scene = viewer.user_scn
            scene.ngeom = 0

            rgba_ref = np.array([0.6, 0.6, 0.6, float(GHOST_ALPHA_REF)], dtype=np.float32)
            qref_com = np.concatenate([ref_p[k], ref_q[k]], axis=0)
            add_ghost_body(scene, qref_com, rgba_ref)

            rgba_wp = np.array([0.5, 0.5, 0.5, float(GHOST_ALPHA_WP)], dtype=np.float32)
            for i in range(int(wp_p.shape[0])):
                qwp_com = np.concatenate([wp_p[i], wp_q[i]], axis=0)
                add_ghost_body(scene, qwp_com, rgba_wp)

            if VIEW_SHOW_ECP and (ecp_hist is not None) and k < ecp_hist.shape[0]:
                if np.all(np.isfinite(ecp_hist[k])):
                    rgba_ecp = np.array([0.2, 0.4, 1.0, 0.9], dtype=np.float32)
                    add_sphere(scene, ecp_hist[k], float(VIEW_ECP_RADIUS), rgba_ecp)

            if tool_pos is not None and k < tool_pos.shape[0]:
                rgba_tool = np.array([0.2, 0.9, 0.2, 1.0], dtype=np.float32)
                add_sphere(scene, tool_pos[k], float(TOOL_RADIUS), rgba_tool)

            if tool_des is not None and k < tool_des.shape[0]:
                rgba_des = np.array([1.0, 0.6, 0.1, 1.0], dtype=np.float32)
                add_sphere(scene, tool_des[k], float(VIEW_TOOL_DES_RADIUS), rgba_des)

            if (tool_force is not None) and (tool_contact_pt is not None) and k < tool_force.shape[0]:
                a = tool_contact_pt[k]
                if np.all(np.isfinite(a)):
                    F = tool_force[k]
                    rgba_F = np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float32)
                    add_force_arrow(scene, a, F, rgba_F)

                    if VIEW_SHOW_CONTACT_POINT:
                        rgba_c = np.array([1.0, 0.2, 0.2, 0.95], dtype=np.float32)
                        add_sphere(scene, a, float(VIEW_CONTACT_RADIUS), rgba_c)

            if VIEW_SHOW_PATHS:
                rgba_traj = np.array([0.2, 0.8, 0.2, 0.85], dtype=np.float32)
                rgba_refp = np.array([0.7, 0.7, 0.7, 0.55], dtype=np.float32)
                add_polyline(scene, traj_path, rgba_traj, radius=float(VIEW_PATH_RADIUS_TRAJ), max_segs=int(VIEW_MAX_PATH_SEGS))
                add_polyline(scene, ref_path,  rgba_refp,  radius=float(VIEW_PATH_RADIUS_REF),  max_segs=int(VIEW_MAX_PATH_SEGS))

            viewer.sync()

            if render_fps > 1e-6:
                target = 1.0 / float(render_fps)
                dt_wall = now - t_last
                if dt_wall < target:
                    time.sleep(target - dt_wall)
                t_last = time.perf_counter()


# ============================================================
# ============================ MAIN ===========================
# ============================================================

