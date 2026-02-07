#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np

try:
    from scipy.spatial.transform import Rotation as R  # optional
except Exception:
    R = None

import mujoco
import mujoco.viewer

# -----------------------------
# Compsim backend (replaces legacy v081)
# -----------------------------
# NOTE: keep this block above any JAX imports.
import os as _os
_os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT = _os.path.abspath(_os.path.join(_THIS_DIR, ".."))

import compsim
from compsim import state as _cs_state
from compsim.sim import TBlockSimulator_Step_NoBounce as _TBlockSimulator_Step_NoBounce
from compsim.samples import get_local_points_ref as _get_local_points_ref
from compsim.pose import body_origin_qpos_to_com as _body_origin_qpos_to_com
from compsim.pose import com_to_body_origin_qpos as _com_to_body_origin_qpos
from compsim.integration import project_to_ground_and_damp as _project_to_ground_and_damp
from compsim.tool_contact import compute_tool_block_impulse as _compute_tool_block_impulse
from compsim.tool_contact import closest_point_on_tblock_surface_world as _closest_point_on_tblock_surface_world
from compsim.tool_contact import inertia_world_from_body_diag as _inertia_world_from_body_diag
from compsim.math3d import quat_from_omega_world_np as _quat_from_omega_world_np
from compsim.math3d import quat_mul_wxyz_np as _quat_mul_wxyz_np

_LOCAL_POINTS_REF = None


def _ensure_local_points_ref(verbose: bool = False):
    global _LOCAL_POINTS_REF
    if _LOCAL_POINTS_REF is None:
        _LOCAL_POINTS_REF = _get_local_points_ref(verbose=verbose)
    return _LOCAL_POINTS_REF


class _SimCompat:
    """Compatibility shim: provides the subset of the old v081 API used by this script."""

    __name__ = "compsim"
    # Default XML: repo-root/t_block_optimized.xml (override via --xml)
    XML_PATH = _os.path.abspath(_os.path.join(_REPO_ROOT, "model/t_block_optimized.xml"))

    state = _cs_state

    # Dynamics/contact
    TBlockSimulator_Step_NoBounce = _TBlockSimulator_Step_NoBounce
    project_to_ground_and_damp = staticmethod(_project_to_ground_and_damp)

    # Tool contact
    compute_tool_block_impulse = staticmethod(_compute_tool_block_impulse)
    closest_point_on_tblock_surface_world = staticmethod(_closest_point_on_tblock_surface_world)

    # Rigid-body helpers
    body_origin_qpos_to_com = staticmethod(_body_origin_qpos_to_com)
    com_to_body_origin_qpos = staticmethod(_com_to_body_origin_qpos)
    inertia_world_from_body_diag = staticmethod(_inertia_world_from_body_diag)
    quat_from_omega_world_np = staticmethod(_quat_from_omega_world_np)
    quat_mul_wxyz_np = staticmethod(_quat_mul_wxyz_np)

    @property
    def total_mass(self):
        return float(self.state.total_mass)

    @property
    def inertia_body_diag(self):
        return self.state.inertia_body_diag

    @property
    def local_points_ref(self):
        return _ensure_local_points_ref(verbose=False)


# keep the old name `sim` to minimize diffs below
sim = _SimCompat()

try:
    from recording_helpers import RealtimeDataLogger
except Exception:
    class RealtimeDataLogger:
        def __init__(self, *args, **kwargs): ...
        def log_tool_impulse(self, *args, **kwargs): ...
        def log_block_energy(self, *args, **kwargs): ...


# ============================================================
# ====================== TUNABLE PARAMETERS ===================
# ============================================================
#     {"idx": 0, "p": [0.45, 0.00, 0.32], "q": [0.707, 0.707, 0.0, 0.0]},   # initial
#     {"idx": 1, "p": [0.55, -0.10, 0.32], "q": [0.50, 0.50, 0.5, 0.5]},    # target (planar translation + yaw)
#     {"idx": 2, "p": [0.65, 0.1, 0.32], "q": [0.707, 0.707, 0.0, 0.0]},
WAYPOINTS = [
    {"idx": 0, "p": [0.70, -0.25, 0.02], "q": [0.707, 0.707, 0.0, 0.0]},  # initial
    {"idx": 1, "p": [0.40, 0.2, 0.02], "q": [0.50, 0.50, 0.5, 0.5]},
    {"idx": 2, "p": [0.95, 0.15, 0.02], "q": [0.707, 0.707, 0.0, 0.0]},   # target (planar translation + yaw)

    # another target
]

# WAYPOINTS = [
#      {"idx": 0, "p": [0.70, -0.25, 0.13], "q": [0.707, 0.0, 0.0, 0.707]},   # initial
#      {"idx": 1, "p": [0.75, 0.40, 0.13], "q": [0.0, 0.0, 0.0, 1.0]},    # target (planar translation + yaw)
#      {"idx": 2, "p": [0.40, 0.07, 0.13], "q": [0.382, 0.0 , 0.0, 0.9238]},
#     # another target
# ]


# WAYPOINTS = [
#      {"idx": 0, "p": [0.70, -0.25, 0.02], "q": [0.707, 0.0, 0.0, 0.707]},   # initial
#      {"idx": 1, "p": [0.75, 0.40, 0.02], "q": [0.382, 0.0 , 0.0, 0.9238]},    # target (planar translation + yaw)
#      {"idx": 1, "p": [0.40, 0.07, 0.02], "q": [0.707, 0.0, 0.0, 0.707]},
#     # another target
# ]

# WAYPOINTS = [
#      {"idx": 0, "p": [0.70, -0.25, 0.02], "q": [0.707, 0.0, 0.0, 0.707]},   # initial
#      {"idx": 1, "p": [0.70, 0.25, 0.02], "q": [0.707, 0.0, 0.0, 0.707]},    # target (planar translation + yaw)
#      {"idx": 2, "p": [0.70, 0.75, 0.02], "q": [0.707, 0.0, 0.0, 0.707]},
#      {"idx": 3, "p": [0.20, 0.75, 0.02], "q": [0.707, 0.0, 0.0, 0.707]},   # initial
#      {"idx": 4, "p": [0.20, 0.25, 0.10], "q": [0.707, 0.0, 0.0, 0.707]},    # target (planar translation + yaw)
#      {"idx": 5, "p": [0.20, -0.25, 0.10], "q": [0.707, 0.0, 0.0, 0.707]},
#     # another target
# ]




DT = 0.002
TOTAL_TIME = 100.0
STEPS = int(round(TOTAL_TIME / DT))

# Success thresholds
POS_THRESH_XY = 0.020     # meters
VEL_THRESH_XY = 0.05      # m/s
YAW_THRESH = np.deg2rad(5.0)   # rad
WZ_THRESH  = 0.30              # rad/s

TOOL_RADIUS = 0.01
TOOL_MASS = 2.0
TOOL_MU = 0.6

TOOL_ENABLE_MARGIN = 5e-3
TOOL_RESTITUTION = 0.0
TOOL_CONTACT_EPS = 1e-6

TOOL_VXY_MAX = 0.30        # m/s
TOOL_VXY_MAX_PUSH = 0.40   # m/s

NAV_CLEARANCE = 0.010      # m
TOUCH_CLEARANCE = 0.0005   # m
PUSH_PEN_INIT = 0.003      # m
PUSH_PEN_MIN = 0.002       # m
PUSH_PEN_MAX = 0.015       # m

NAV_D_DES = float(TOOL_RADIUS) + float(NAV_CLEARANCE)
TOUCH_D_DES = float(TOOL_RADIUS) + float(TOUCH_CLEARANCE)

# Force band
FN_MIN = 0.5
FN_MAX = 20.0
PEN_STEP = 0.0005

RETRACT_HOLD_TIME = 0.20
SETTLE_HOLD_TIME = 0.3
APPROACH_D_RATE = 0.02

RETRACT_HOLD_STEPS = int(round(RETRACT_HOLD_TIME / DT))
SETTLE_HOLD_STEPS = int(round(SETTLE_HOLD_TIME / DT))

# --- release-switch hold time ---
RELEASE_HOLD_TIME = 0.12
RELEASE_HOLD_STEPS = int(round(RELEASE_HOLD_TIME / DT))

# Surface sampling
SIDE_NORMAL_Z_MAX = 0.35
FACE_SAMPLE_SPACING = 0.012
FACE_INTERNAL_EPS = 8e-5
MAX_FACE_POINTS = 700

# Contact selection
CONTACT_MIN_ALIGN = 0.15
CONTACT_HYSTERESIS_BONUS = 0.20
FACE_SWITCH_COOLDOWN = 80
FACE_SWITCH_MARGIN = 0.08

# Yaw scoring / blending
POS_YAW_BLEND_DIST = 0.10
W_POS_FAR,  W_YAW_FAR  = 1.0, 0.2
W_POS_NEAR, W_YAW_NEAR = 0.4, 1.0
YAW_ERR_ACTIVE = np.deg2rad(1.0)
YAW_ERR_REF    = np.deg2rad(30.0)
TORQUE_SCORE_GAIN = 20.0
LEVER_SCORE_GAIN  = 5.0
MIN_ALIGN_RELAX_YAW = -0.10

# Face-switch robustness
SWITCH_GAP_TRIGGER = float(NAV_CLEARANCE)

# Tangential suppression near contact (prevents "撞飞" when moving tangentially while near contact)
TANG_SUPPRESS_GAP_LOCK = float(TOUCH_CLEARANCE)
TANG_SUPPRESS_GAP_FREE = float(NAV_CLEARANCE) * 1.25

# NAV TangentBug params (kept but unused now)
NAV_SEGMENT_SAMPLES = 25
NAV_CLEAR_TOL = 2e-4
WALL_RADIAL_K = 8.0
WALL_TAN_SPEED_FRAC = 0.75
WALL_EXIT_HYST = 2e-3
WALL_STAGNATION_TIME = 0.8
WALL_MIN_PROGRESS = 1e-3

# ============================================================
# ============ NAV Recovery (2nd safety net / NEW) ============
# ============================================================
NAV_RECOVER_ENABLE = True
NAV_STALL_EPS = 5e-5
NAV_STALL_TIME = 1.0
NAV_MAX_TIME = 8.0
NAV_RECOVER_COOLDOWN = 1.0
NAV_RECOVER_TOPK = 40
NAV_RECOVER_PRINT = True

# Debug (NAV)
DEBUG_NAV = True
DEBUG_NAV_EVERY = 100
DEBUG_NAV_PRINT_ON_STALL = True
DEBUG_NAV_STALL_EPS = 5e-5

# Height filter for side samples (avoid mixing top/bottom faces with tool height)
Z_SAMPLE_BAND = float(TOOL_RADIUS) + 0.006

# Output
VIEW = True
VIEW_RENDER_FPS = 60.0
VIEW_PLAYBACK_SPEED = 1.0
SAVE_NPZ = True
OUT_NPZ = "push_minimal_framework_yaw_astar_nav.npz"

# --- Visualization constants (kept from your demo) ---
GHOST_ALPHA_REF = 0.25
GHOST_ALPHA_WP  = 0.18

VIEW_FORCE_SCALE = 0.5
VIEW_ARROW_WIDTH = 0.008
VIEW_ARROW_LEN_MAX = 0.9


# -----------------------------
# Live-view (compute while watching)
# -----------------------------
LIVE_VIEW = False
LIVE_VIEW_STRIDE = 5
LIVE_VIEW_REALTIME = False
LIVE_TARGET_MOCAP = "target_mocap"  # drive this mocap marker to the tool position

# IMPORTANT: this is used only for LIVE_VIEW. main() will overwrite it from --body
BODY_NAME = "T_siconos"


def _disable_all_contacts_mj(m: mujoco.MjModel) -> None:
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


_disable_all_contacts = _disable_all_contacts_mj


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


def _scene_clear(scene) -> None:
    """Avoid NameError: just clear user_scn overlays."""
    try:
        scene.ngeom = 0
    except Exception:
        pass


def _scene_add_sphere(scene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= len(scene.geoms):
        return
    g = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    size = np.array([float(radius), 0.0, 0.0], dtype=np.float64)
    p = np.asarray(pos, dtype=np.float64)
    mat = np.eye(3, dtype=np.float64).reshape(-1)
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        size,
        p,
        mat,
        np.asarray(rgba, dtype=np.float32),
    )


def _scene_add_force_arrow(scene, p_from: np.ndarray, force: np.ndarray, rgba: np.ndarray) -> None:
    Fn = float(np.linalg.norm(force))
    if Fn <= 1e-12 or scene.ngeom >= len(scene.geoms):
        return
    # Arrow length in meters
    L = min(float(VIEW_ARROW_LEN_MAX), float(VIEW_FORCE_SCALE) * Fn)
    d = np.asarray(force, dtype=np.float64) / (Fn + 1e-12)
    a = np.asarray(p_from, dtype=np.float64)
    b = a + d * L

    g = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, float(VIEW_ARROW_WIDTH), a, b)


# =========================
# NEW: polyline capsules
# =========================
def _scene_add_polyline_capsules(
    scene,
    pts_xyz: np.ndarray,
    rgba: np.ndarray,
    radius: float = 0.0025,
    max_segs: int = 120,
) -> None:
    """Draw polyline with capsules. pts_xyz: (N,3)."""
    pts = np.asarray(pts_xyz, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return

    n = int(pts.shape[0])
    stride = max(1, n // int(max_segs))
    idx = np.arange(0, n, stride, dtype=int)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    ps = pts[idx]

    maxgeom = int(getattr(scene, "maxgeom", len(scene.geoms)))

    for i in range(ps.shape[0] - 1):
        if scene.ngeom >= maxgeom:
            break
        p0 = ps[i]
        p1 = ps[i + 1]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            continue

        mid = 0.5 * (p0 + p1)
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
        Rm = np.column_stack([x, y, z]).reshape(-1)

        g = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([float(radius), 0.5 * L, 0.0], dtype=np.float64),
            np.asarray(mid, dtype=np.float64),
            np.asarray(Rm, dtype=np.float64),
            np.asarray(rgba, dtype=np.float32),
        )


# ============================================================
# ====================== QUAT UTILITIES ======================
# ============================================================

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
    gap_lock: float = TANG_SUPPRESS_GAP_LOCK,
    gap_free: float = TANG_SUPPRESS_GAP_FREE,
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
# =================== NAV: A* GRID (NO SDF) ===================
# ============================================================

import heapq
from dataclasses import field

# ---- A* params (NAV only) ----
ASTAR_RES = 0.006
ASTAR_MARGIN = 0.18
ASTAR_DIAGONAL = True
ASTAR_MAX_EXPAND = 160000
ASTAR_REPLAN_DIST = 0.03
ASTAR_REPLAN_EVERY = 0.60
ASTAR_GOAL_NUDGE_MAX = 0.08
ASTAR_PROJECT_EPS = 1e-4


@dataclass
class AStarNavState:
    path_xy: Optional[np.ndarray] = None   # (N,2)
    wp_i: int = 0
    goal_xy: np.ndarray = field(default_factory=lambda: np.array([np.nan, np.nan], dtype=np.float64))
    last_plan_step: int = -10**9
    replans: int = 0
    failed_plans: int = 0


def _grid_from_bounds(xmin, xmax, ymin, ymax, res):
    nx = int(np.ceil((xmax - xmin) / res)) + 1
    ny = int(np.ceil((ymax - ymin) / res)) + 1
    return nx, ny


def _xy_to_ij(xy, xmin, ymin, res):
    ix = int(np.round((float(xy[0]) - xmin) / res))
    iy = int(np.round((float(xy[1]) - ymin) / res))
    return ix, iy


def _ij_to_xy(ix, iy, xmin, ymin, res):
    x = xmin + float(ix) * res
    y = ymin + float(iy) * res
    return np.array([x, y], dtype=np.float64)


def _clip_ij(ix, iy, nx, ny):
    ix = int(np.clip(ix, 0, nx - 1))
    iy = int(np.clip(iy, 0, ny - 1))
    return ix, iy


def _build_disk_offsets(r_cells: int):
    offs = []
    rr2 = int(r_cells) * int(r_cells)
    for dx in range(-r_cells, r_cells + 1):
        for dy in range(-r_cells, r_cells + 1):
            if dx*dx + dy*dy <= rr2:
                offs.append((dx, dy))
    return offs


def build_occupancy_from_points(obs_xy: np.ndarray, xmin, ymin, nx, ny, res, inflate_radius: float):
    occ = np.zeros((nx, ny), dtype=np.bool_)
    if obs_xy is None:
        return occ
    obs_xy = np.asarray(obs_xy, dtype=np.float64).reshape(-1, 2)
    if obs_xy.shape[0] == 0:
        return occ

    r_cells = int(np.ceil(float(inflate_radius) / float(res)))
    r_cells = max(1, r_cells)
    disk = _build_disk_offsets(r_cells)

    for p in obs_xy:
        ix0, iy0 = _xy_to_ij(p, xmin, ymin, res)
        for dx, dy in disk:
            ix = ix0 + dx
            iy = iy0 + dy
            if 0 <= ix < nx and 0 <= iy < ny:
                occ[ix, iy] = True
    return occ


def _nearest_free_cell(occ: np.ndarray, ix, iy, max_r=25):
    nx, ny = occ.shape
    ix, iy = _clip_ij(ix, iy, nx, ny)
    if not occ[ix, iy]:
        return ix, iy
    for r in range(1, int(max_r) + 1):
        for dx in range(-r, r + 1):
            for dy in (-r, r):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < nx and 0 <= jy < ny and (not occ[jx, jy]):
                    return jx, jy
        for dy in range(-r + 1, r):
            for dx in (-r, r):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < nx and 0 <= jy < ny and (not occ[jx, jy]):
                    return jx, jy
    return None


def astar_plan(start_xy: np.ndarray, goal_xy: np.ndarray, obs_xy: np.ndarray, inflate_radius: float,
               res: float, margin: float, max_expand: int = ASTAR_MAX_EXPAND, diagonal: bool = True):
    start_xy = np.asarray(start_xy, dtype=np.float64).reshape(2,)
    goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(2,)
    obs_xy = np.asarray(obs_xy, dtype=np.float64).reshape(-1, 2)

    xmin = float(min(start_xy[0], goal_xy[0], float(np.min(obs_xy[:, 0])) if obs_xy.size else start_xy[0])) - float(margin) - float(inflate_radius)
    xmax = float(max(start_xy[0], goal_xy[0], float(np.max(obs_xy[:, 0])) if obs_xy.size else start_xy[0])) + float(margin) + float(inflate_radius)
    ymin = float(min(start_xy[1], goal_xy[1], float(np.min(obs_xy[:, 1])) if obs_xy.size else start_xy[1])) - float(margin) - float(inflate_radius)
    ymax = float(max(start_xy[1], goal_xy[1], float(np.max(obs_xy[:, 1])) if obs_xy.size else start_xy[1])) + float(margin) + float(inflate_radius)

    nx, ny = _grid_from_bounds(xmin, xmax, ymin, ymax, float(res))
    if nx * ny > 450000:
        scale = np.sqrt((nx * ny) / 450000.0)
        res2 = float(res) * float(scale)
        nx, ny = _grid_from_bounds(xmin, xmax, ymin, ymax, res2)
        res = res2

    occ = build_occupancy_from_points(obs_xy, xmin, ymin, nx, ny, float(res), float(inflate_radius))

    s_ix, s_iy = _xy_to_ij(start_xy, xmin, ymin, float(res))
    g_ix, g_iy = _xy_to_ij(goal_xy, xmin, ymin, float(res))
    s_ix, s_iy = _clip_ij(s_ix, s_iy, nx, ny)
    g_ix, g_iy = _clip_ij(g_ix, g_iy, nx, ny)

    s2 = _nearest_free_cell(occ, s_ix, s_iy, max_r=30)
    g2 = _nearest_free_cell(occ, g_ix, g_iy, max_r=30)
    if s2 is None or g2 is None:
        return None, {'ok': False, 'reason': 'start_or_goal_blocked', 'nx': nx, 'ny': ny, 'res': res}
    s_ix, s_iy = s2
    g_ix, g_iy = g2

    if diagonal:
        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def h(ix, iy):
        dx = (ix - g_ix)
        dy = (iy - g_iy)
        return np.hypot(dx, dy)

    inf = 1e18
    gscore = np.full((nx, ny), inf, dtype=np.float64)
    parent = np.full((nx, ny, 2), -1, dtype=np.int32)

    pq = []
    gscore[s_ix, s_iy] = 0.0
    heapq.heappush(pq, (h(s_ix, s_iy), 0.0, s_ix, s_iy))

    expanded = 0
    found = False

    while pq and expanded < int(max_expand):
        f, gcur, ix, iy = heapq.heappop(pq)
        if (ix == g_ix) and (iy == g_iy):
            found = True
            break
        if gcur > gscore[ix, iy] + 1e-12:
            continue
        expanded += 1

        for dx, dy in nbrs:
            jx, jy = ix + dx, iy + dy
            if not (0 <= jx < nx and 0 <= jy < ny):
                continue
            if occ[jx, jy]:
                continue
            step = np.hypot(dx, dy)
            ng = gcur + step
            if ng + 1e-12 < gscore[jx, jy]:
                gscore[jx, jy] = ng
                parent[jx, jy, 0] = ix
                parent[jx, jy, 1] = iy
                nf = ng + h(jx, jy)
                heapq.heappush(pq, (nf, ng, jx, jy))

    if not found:
        return None, {'ok': False, 'reason': 'no_path', 'expanded': expanded, 'nx': nx, 'ny': ny, 'res': res}

    path_ij = [(g_ix, g_iy)]
    ix, iy = g_ix, g_iy
    while not (ix == s_ix and iy == s_iy):
        pix, piy = int(parent[ix, iy, 0]), int(parent[ix, iy, 1])
        if pix < 0:
            break
        path_ij.append((pix, piy))
        ix, iy = pix, piy
    path_ij.reverse()

    path_xy = np.stack([_ij_to_xy(ix, iy, xmin, ymin, float(res)) for (ix, iy) in path_ij], axis=0)
    info = {'ok': True, 'expanded': expanded, 'nx': nx, 'ny': ny, 'res': res}
    return path_xy, info


def project_out_of_obstacles(xyz: np.ndarray, obs_xy: np.ndarray, inflate_radius: float, z_tool: float):
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3,)
    obs_xy = np.asarray(obs_xy, dtype=np.float64).reshape(-1, 2)
    if obs_xy.shape[0] == 0:
        xyz[2] = float(z_tool)
        return xyz

    dxy = obs_xy - xyz[0:2].reshape(1, 2)
    d2 = np.sum(dxy*dxy, axis=1)
    j = int(np.argmin(d2))
    d = float(np.sqrt(d2[j]))
    if d >= float(inflate_radius) + float(ASTAR_PROJECT_EPS):
        xyz[2] = float(z_tool)
        return xyz

    v = xyz[0:2] - obs_xy[j]
    nv = float(np.linalg.norm(v))
    if nv < 1e-12:
        v = np.array([1.0, 0.0], dtype=np.float64)
        nv = 1.0
    v = v / (nv + 1e-12)
    xyz[0:2] = obs_xy[j] + v * (float(inflate_radius) + float(ASTAR_PROJECT_EPS))
    xyz[2] = float(z_tool)
    return xyz


def nudge_goal_if_in_collision(goal_xyz: np.ndarray, n_out: np.ndarray, obs_xy: np.ndarray, inflate_radius: float, z_tool: float):
    goal_xyz = np.asarray(goal_xyz, dtype=np.float64).reshape(3,)
    goal_xyz[2] = float(z_tool)
    n_out = np.asarray(n_out, dtype=np.float64).reshape(3,)
    nxy = n_out[0:2]
    nn = float(np.linalg.norm(nxy))
    if nn < 1e-9:
        nxy = np.array([1.0, 0.0], dtype=np.float64)
        nn = 1.0
    nxy = nxy / (nn + 1e-12)

    obs_xy = np.asarray(obs_xy, dtype=np.float64).reshape(-1, 2)
    if obs_xy.shape[0] == 0:
        return goal_xyz

    dxy = obs_xy - goal_xyz[0:2].reshape(1, 2)
    d2 = np.sum(dxy*dxy, axis=1)
    if float(np.min(d2)) >= float(inflate_radius)**2:
        return goal_xyz

    step = float(min(0.004, float(ASTAR_RES)))
    max_steps = int(np.ceil(float(ASTAR_GOAL_NUDGE_MAX) / step))
    x = goal_xyz.copy()
    for _ in range(max_steps):
        x[0:2] += nxy * step
        dxy = obs_xy - x[0:2].reshape(1, 2)
        d2 = np.sum(dxy*dxy, axis=1)
        if float(np.min(d2)) >= float(inflate_radius)**2:
            return x
    return x


def nav_step_astar(
    tool_pos: np.ndarray,
    goal_pos: np.ndarray,
    obs_xy: np.ndarray,
    inflate_radius: float,
    dt: float,
    vxy_cap: float,
    nav: AStarNavState,
    z_tool: float,
    step_idx: int,
) -> Tuple[np.ndarray, np.ndarray, AStarNavState]:
    tool_pos = np.asarray(tool_pos, dtype=np.float64).reshape(3,)
    goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3,)
    obs_xy = np.asarray(obs_xy, dtype=np.float64).reshape(-1, 2)

    tool_xy = tool_pos[0:2].copy()
    goal_xy = goal_pos[0:2].copy()

    replan_every_steps = int(round(float(ASTAR_REPLAN_EVERY) / float(dt)))
    goal_moved = float(np.linalg.norm(goal_xy - nav.goal_xy))
    need_plan = (nav.path_xy is None) or (goal_moved > float(ASTAR_REPLAN_DIST)) or ((step_idx - nav.last_plan_step) >= replan_every_steps)

    if need_plan:
        path, info = astar_plan(tool_xy, goal_xy, obs_xy, float(inflate_radius), float(ASTAR_RES), float(ASTAR_MARGIN), diagonal=bool(ASTAR_DIAGONAL))
        nav.last_plan_step = int(step_idx)
        if path is None or (not info.get('ok', False)):
            nav.failed_plans += 1
            nav.path_xy = None
            nav.wp_i = 0
            tool_des = goal_pos.copy()
            tool_des[2] = float(z_tool)
            tool_next = kinematic_step_towards(tool_pos, tool_des, float(dt), float(vxy_cap))
            tool_next[2] = float(z_tool)
            tool_next = project_out_of_obstacles(tool_next, obs_xy, float(inflate_radius), float(z_tool))
            return tool_next, tool_des, nav

        nav.path_xy = path
        nav.wp_i = 0
        nav.goal_xy = goal_xy.copy()
        nav.replans += 1

    path = nav.path_xy
    if path is None or path.shape[0] == 0:
        tool_des = goal_pos.copy()
        tool_des[2] = float(z_tool)
        tool_next = kinematic_step_towards(tool_pos, tool_des, float(dt), float(vxy_cap))
        tool_next[2] = float(z_tool)
        tool_next = project_out_of_obstacles(tool_next, obs_xy, float(inflate_radius), float(z_tool))
        return tool_next, tool_des, nav

    while nav.wp_i < path.shape[0] - 1:
        wp = path[int(nav.wp_i)]
        if float(np.linalg.norm(wp - tool_xy)) <= float(max(ASTAR_RES, 0.5 * vxy_cap * dt)):
            nav.wp_i += 1
        else:
            break

    wp = path[int(nav.wp_i)]
    tool_des = tool_pos.copy()
    tool_des[0:2] = wp
    tool_des[2] = float(z_tool) 

    tool_next = kinematic_step_towards(tool_pos, tool_des, float(dt), float(vxy_cap))
    tool_next[2] = float(z_tool)
    tool_next = project_out_of_obstacles(tool_next, obs_xy, float(inflate_radius), float(z_tool))
    return tool_next, tool_des, nav


def pick_reachable_contact_idx_for_nav(
    tool_pos: np.ndarray,
    q_block: np.ndarray,
    z_tool: float,
    d_nav: float,
    F_xy_des: np.ndarray,
    p_w_use: np.ndarray,
    n_w_use: np.ndarray,
    idx_map: np.ndarray,
    last_idx_use: Optional[int] = None,
    top_k: int = NAV_RECOVER_TOPK,
) -> Tuple[Optional[int], float, Optional[np.ndarray]]:
    tool_pos = np.asarray(tool_pos, dtype=np.float64).reshape(3,)
    F_xy_des = np.asarray(F_xy_des, dtype=np.float64).reshape(2,)
    p_w_use = np.asarray(p_w_use, dtype=np.float64)
    n_w_use = np.asarray(n_w_use, dtype=np.float64)
    idx_map = np.asarray(idx_map, dtype=np.int32)

    if p_w_use.shape[0] == 0:
        return None, -1e18, None

    if np.linalg.norm(F_xy_des) > 1e-9:
        d = F_xy_des / (np.linalg.norm(F_xy_des) + 1e-12)
    else:
        d = np.array([1.0, 0.0], dtype=np.float64)

    n_xy = n_w_use[:, 0:2]
    n_xy_n = np.linalg.norm(n_xy, axis=1)
    valid = n_xy_n > 1e-9

    n_xy_unit = np.zeros_like(n_xy)
    n_xy_unit[valid] = n_xy[valid] / (n_xy_n[valid].reshape(-1, 1) + 1e-12)
    push_dir = -n_xy_unit
    push_align = np.sum(push_dir * d.reshape(1, 2), axis=1)

    score = push_align.copy().astype(np.float64)
    score[~valid] = -1e18
    score[push_align < float(CONTACT_MIN_ALIGN)] = -1e18
    if last_idx_use is not None and 0 <= int(last_idx_use) < score.shape[0]:
        score[int(last_idx_use)] += float(CONTACT_HYSTERESIS_BONUS)

    order = np.argsort(-score)
    if int(top_k) > 0:
        order = order[: min(int(top_k), order.shape[0])]

    obs_xy = p_w_use[:, 0:2]

    for j in order:
        s = float(score[int(j)])
        if s <= -1e17:
            break

        p_c = p_w_use[int(j)]
        n_c = n_w_use[int(j)]
        n_out = n_c / (np.linalg.norm(n_c) + 1e-12)

        x_nav = (p_c + n_out * float(d_nav)).astype(np.float64)
        x_nav[2] = float(z_tool)
        x_goal = nudge_goal_if_in_collision(x_nav, n_out, obs_xy, float(d_nav), float(z_tool))

        path, info = astar_plan(tool_pos[0:2], x_goal[0:2], obs_xy, float(d_nav), float(ASTAR_RES), float(ASTAR_MARGIN), diagonal=bool(ASTAR_DIAGONAL))
        if path is not None and info.get('ok', False):
            return int(idx_map[int(j)]), s, x_goal.copy()

    return None, -1e18, None


# ============================================================
# =================== FORCE / PENETRATION LOOP =================
# ============================================================

def update_penetration_mm_band(pen: float, Fn: float) -> float:
    pen = float(pen)
    Fn = float(Fn)

    if Fn < float(FN_MIN):
        pen += float(PEN_STEP)
    elif Fn > float(FN_MAX):
        pen -= float(PEN_STEP)

    pen = float(np.clip(pen, float(PUSH_PEN_MIN), float(PUSH_PEN_MAX)))
    return pen


# ============================================================
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

def run_push_minimal():
    PN_SUPPORT_THRESH = 1e-9  # support threshold for projection

    print("[Minimal] Using simulator module:", sim.__name__)
    print(f"[Minimal] total_mass = {float(sim.total_mass):.6f}")

    if len(WAYPOINTS) < 2:
        raise ValueError("Need at least 2 waypoints.")
    wp_t = np.array([float(w["idx"]) for w in WAYPOINTS], dtype=np.float64)
    wp_p = np.array([w["p"] for w in WAYPOINTS], dtype=np.float64).reshape(-1, 3)
    wp_q = np.array([quat_normalize_wxyz(w["q"]) for w in WAYPOINTS], dtype=np.float64).reshape(-1, 4)
    wp_q = enforce_quat_sign_continuity_wxyz(wp_q)

    # ground solver
    simobj = sim.TBlockSimulator_Step_NoBounce(
        dt=float(DT),
        mass=float(sim.total_mass),
        inertia_diag_body=sim.inertia_body_diag,
        local_pts_all=sim.local_points_ref,
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
        proj_tol=1e-6,
        ground_enable_margin=2e-3,
        support_z=0.0,
    )
    simobj.ecp_xy_reg = 1e-2
    simobj.jac_reg = 1e-8

    # tool mass (your impulse model uses it)
    if hasattr(sim, "compute_tool_block_impulse"):
        try:
            sim.compute_tool_block_impulse.tool_mass = float(TOOL_MASS)
        except Exception:
            pass

    # init block state
    q = np.zeros(7, dtype=np.float64)
    q[0:3] = wp_p[0]
    q[3:7] = wp_q[0].copy()
    v = np.zeros(6, dtype=np.float64)

    # strict surface samples
    xml_path = getattr(sim, "XML_PATH", None)
    if xml_path is None:
        raise RuntimeError("sim.XML_PATH is None. Please set it in your sim module.")

    q0_body_wxyz = q[3:7].copy()
    _, surface_samples = build_strict_plot_surface_samples_comframe(
        xml_path=xml_path,
        q0_body_wxyz=q0_body_wxyz,
        body_name="T_siconos",
    )
    if surface_samples.p_local_com.shape[0] == 0:
        raise RuntimeError("Surface samples are empty. Check FACE_SAMPLE_SPACING / SIDE_NORMAL_Z_MAX / model geoms.")
    print(f"[Surface] samples={surface_samples.p_local_com.shape[0]}  NAV_D_DES={NAV_D_DES:.4f}  TOUCH_D_DES={TOUCH_D_DES:.4f}  Z_SAMPLE_BAND={Z_SAMPLE_BAND:.3f}")

    # init tool
    z_tool = float(wp_p[0, 2])
    tool_pos = np.array([q[0] - 0.50, q[1], z_tool], dtype=np.float64)
    tool_vel = np.zeros(3, dtype=np.float64)

    # Optional live view: update MuJoCo viewer while computing (no MuJoCo physics).
    viewer = None
    viewer_ctx = None
    m_vis = None
    d_vis = None
    qposadr_vis = None
    target_mocap_id = -1
    last_view_t = time.perf_counter()

    # store ipos_body from the visualization MJCF (needed to convert COM-qpos -> body-origin qpos)
    ipos_body_vis = None

    # -------- NEW: ghost geoms + block trajectory (live_view) --------
    geoms_desc_vis: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    _scene_add_ghost_tblock = None

    live_block_path: List[np.ndarray] = []
    LIVE_BLOCK_PATH_MAX = 2500
    # ---------------------------------------------------------------

    if LIVE_VIEW:
        try:
            if getattr(sim, "XML_PATH", None) is None:
                raise RuntimeError("sim.XML_PATH is None (live_view needs a MJCF to visualize).")

            m_vis = mujoco.MjModel.from_xml_path(getattr(sim, "XML_PATH"))
            d_vis = mujoco.MjData(m_vis)
            _disable_all_contacts(m_vis)

            # body + freejoint address
            qposadr_vis = _find_freejoint_qposadr(m_vis, BODY_NAME)
            body_id_vis = mujoco.mj_name2id(m_vis, mujoco.mjtObj.mjOBJ_BODY, BODY_NAME)
            if body_id_vis < 0:
                raise RuntimeError(f"body '{BODY_NAME}' not found in vis model")
            ipos_body_vis = np.array(m_vis.body_ipos[body_id_vis], dtype=np.float64)

            # -------- NEW: collect BOX geoms under this body for ghost drawing --------
            geom_start = int(m_vis.body_geomadr[body_id_vis])
            geom_num = int(m_vis.body_geomnum[body_id_vis])
            for gid in range(geom_start, geom_start + geom_num):
                if int(m_vis.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                    continue
                size = np.array(m_vis.geom_size[gid], dtype=np.float64).copy()
                gpos_body = np.array(m_vis.geom_pos[gid], dtype=np.float64).copy()
                gquat_body = np.array(m_vis.geom_quat[gid], dtype=np.float64).copy()  # wxyz
                geoms_desc_vis.append((size, gpos_body, gquat_body))

            def _scene_add_ghost_tblock(scene, q_com7: np.ndarray, rgba: np.ndarray) -> None:
                """Draw translucent T-block at COM pose q_com7=(p(3), quat_wxyz(4))."""
                if ipos_body_vis is None or len(geoms_desc_vis) == 0:
                    return
                q_com7 = np.asarray(q_com7, dtype=np.float64).reshape(7,)
                p_com = q_com7[0:3]
                q_body = q_com7[3:7]
                Rb = _quat_wxyz_to_rotmat(q_body)

                # COM -> body-origin
                p_body = p_com - Rb @ ipos_body_vis

                maxgeom = int(getattr(scene, "maxgeom", len(scene.geoms)))
                for (size, gpos_body, gquat_body) in geoms_desc_vis:
                    if scene.ngeom >= maxgeom:
                        break
                    p_geom = p_body + Rb @ gpos_body
                    q_geom = sim.quat_mul_wxyz_np(q_body, gquat_body)
                    mat = _quat_wxyz_to_rotmat(q_geom).reshape(-1)

                    g = scene.geoms[scene.ngeom]
                    scene.ngeom += 1
                    mujoco.mjv_initGeom(
                        g,
                        mujoco.mjtGeom.mjGEOM_BOX,
                        np.asarray(size, dtype=np.float64),
                        np.asarray(p_geom, dtype=np.float64),
                        np.asarray(mat, dtype=np.float64),
                        np.asarray(rgba, dtype=np.float32),
                    )
            # -------------------------------------------------------------------------

            # mocap marker (optional)
            target_mocap_id = _find_mocap_id(m_vis, LIVE_TARGET_MOCAP)

            viewer_ctx = mujoco.viewer.launch_passive(m_vis, d_vis)
            viewer = viewer_ctx.__enter__()

            # initialize poses
            try:
                qpos_vis0 = _com_to_body_origin_qpos(q, ipos_body_vis)
            except Exception:
                # fallback if compsim.pose signature differs
                Rm = _quat_wxyz_to_rotmat(q[3:7])
                p_body0 = q[0:3] - Rm @ ipos_body_vis
                qpos_vis0 = np.concatenate([p_body0, q[3:7]], axis=0)

            d_vis.qpos[qposadr_vis: qposadr_vis + 7] = qpos_vis0
            if target_mocap_id >= 0:
                d_vis.mocap_pos[target_mocap_id] = tool_pos
            mujoco.mj_forward(m_vis, d_vis)

            try:
                viewer.sync()
            except Exception:
                pass

        except Exception as e:
            print(f"[live_view] disabled due to error: {e}")
            viewer = None
            if viewer_ctx is not None:
                try:
                    viewer_ctx.__exit__(None, None, None)
                except Exception:
                    pass
            viewer_ctx = None

    phase = "RETRACT"
    retract_counter = 0
    settle_counter = 0
    release_counter = 0

    last_contact_idx: Optional[int] = None
    last_contact_score: float = -1e18
    last_switch_step: int = -10**9

    pending_contact_idx: Optional[int] = None
    lock_contact_idx: Optional[int] = None

    pen = float(PUSH_PEN_INIT)
    nav_state = AStarNavState()

    # NAV recovery watchdog state
    nav_stall_steps = 0
    nav_phase_steps = 0
    nav_last_recover_step = -10**9

    NAV_STALL_STEPS = int(round(float(NAV_STALL_TIME) / float(DT)))
    NAV_MAX_STEPS   = int(round(float(NAV_MAX_TIME) / float(DT)))
    NAV_RECOVER_COOLDOWN_STEPS = int(round(float(NAV_RECOVER_COOLDOWN) / float(DT)))

    # logs
    traj_q, traj_v = [], []
    ref_p_hist, ref_q_hist = [], []
    tool_pos_hist, tool_vel_hist, tool_des_hist = [], [], []
    tool_contact_pt_hist, tool_impulse_hist, tool_force_hist = [], [], []
    info_hist, minz_hist, ecp_hist = [], [], []

    last_tool_impulse_norm = 0.0

    goal_idx = 1
    t_grid = np.arange(STEPS + 1, dtype=np.float64) * float(DT)

    # initial log
    traj_q.append(q.copy())
    traj_v.append(v.copy())
    ref_p_hist.append(wp_p[goal_idx].copy())
    ref_q_hist.append(wp_q[goal_idx].copy())
    tool_pos_hist.append(tool_pos.copy())
    tool_vel_hist.append(tool_vel.copy())
    tool_des_hist.append(tool_pos.copy())
    tool_contact_pt_hist.append(np.full((3,), np.nan, dtype=np.float64))
    tool_impulse_hist.append(np.zeros((3,), np.float64))
    tool_force_hist.append(np.zeros((3,), np.float64))
    info_hist.append(0)
    minz_hist.append(0.0)
    ecp_hist.append(np.full((3,), np.nan, dtype=np.float64))

    print(f"[Minimal] Start: steps={STEPS}, DT={DT}, goal idx={goal_idx}")

    logger = RealtimeDataLogger(
        out_dir="logs",
        energy_csv="block_energy.csv",
        impulse_csv="tool_impulse.csv",
        flush_every=200,
    )

    for k in range(STEPS):
        p_goal = wp_p[goal_idx].copy()
        q_goal = wp_q[goal_idx].copy()
        z_tool = float(p_goal[2])

        

        # gap at START
        if hasattr(sim, "closest_point_on_tblock_surface_world"):
            _, sdf_prev = sim.closest_point_on_tblock_surface_world(tool_pos, q)
            sdf_prev = float(sdf_prev)
        else:
            sdf_prev = 1e9
        gap_prev = float(sdf_prev) - float(TOOL_RADIUS)
        tool_pos_prev = tool_pos.copy()

        # errors
        e_p_xy = (p_goal[0:2] - q[0:2])
        pos_err_xy = float(np.linalg.norm(e_p_xy))
        yaw_err = yaw_error_from_quats_wxyz(q[3:7], q_goal)

        if pos_err_xy > 1e-9:
            pos_dir = e_p_xy / (pos_err_xy + 1e-12)
        else:
            pos_dir = np.array([0.0, -1.0], dtype=np.float64)

        # surface samples -> world
        p_w_s, n_w_s = transform_samples_to_world(q, surface_samples)
        p_w_use, n_w_use, idx_map = filter_samples_by_height(p_w_s, n_w_s, z_tool, float(Z_SAMPLE_BAND))

        # map last idx to filtered idx
        last_idx_use = None
        if last_contact_idx is not None:
            hits = np.where(idx_map == int(last_contact_idx))[0]
            if hits.size > 0:
                last_idx_use = int(hits[0])

        # choose contact (or lock after switch)
        if (lock_contact_idx is not None) and (phase != "PUSH_TRANSLATE"):
            chosen_idx_full = int(np.clip(int(lock_contact_idx), 0, p_w_s.shape[0] - 1))
            p_c = p_w_s[chosen_idx_full].copy()
            n_c = n_w_s[chosen_idx_full].copy()
            chosen_score = float(last_contact_score)
        else:
            if phase in ("PUSH_TRANSLATE",):
                w_pos, w_yaw = blend_pos_yaw_weights(pos_err_xy)
                p_c_best, n_c_best, idx_best_use, score_best = select_contact_from_surface_samples_translation_yaw(
                    q_block=q,
                    pos_dir_xy=pos_dir,
                    yaw_err=yaw_err,
                    w_pos=w_pos,
                    w_yaw=w_yaw,
                    p_w=p_w_use,
                    n_w=n_w_use,
                    last_idx=last_idx_use,
                )
            else:
                p_c_best, n_c_best, idx_best_use, score_best = select_contact_from_surface_samples_translation_only(
                    F_xy_des=e_p_xy,
                    p_w=p_w_use,
                    n_w=n_w_use,
                    last_idx=last_idx_use,
                )

            chosen_idx_full = int(idx_map[int(idx_best_use)])
            chosen_score = float(score_best)
            p_c = p_c_best
            n_c = n_c_best

            # cooldown / margin
            if last_contact_idx is not None and chosen_idx_full != int(last_contact_idx):
                if (k - last_switch_step) < int(FACE_SWITCH_COOLDOWN):
                    if not (chosen_score >= float(last_contact_score) + float(FACE_SWITCH_MARGIN)):
                        chosen_idx_full = int(last_contact_idx)
                        p_c = p_w_s[chosen_idx_full].copy()
                        n_c = n_w_s[chosen_idx_full].copy()
                        chosen_score = float(last_contact_score)

            # near-contact switching -> RELEASE_SWITCH
            if (last_contact_idx is not None) and (chosen_idx_full != int(last_contact_idx)):
                near_contact = (gap_prev < float(SWITCH_GAP_TRIGGER))
                in_sensitive_phase = phase in ("APPROACH", "SETTLE", "PUSH_TRANSLATE")
                if near_contact and in_sensitive_phase:
                    pending_contact_idx = int(chosen_idx_full)
                    chosen_idx_full = int(last_contact_idx)
                    p_c = p_w_s[chosen_idx_full].copy()
                    n_c = n_w_s[chosen_idx_full].copy()
                    chosen_score = float(last_contact_score)
                    phase = "RELEASE_SWITCH"
                    release_counter = 0
                    lock_contact_idx = None

        # update switch bookkeeping
        if last_contact_idx is None or chosen_idx_full != int(last_contact_idx):
            last_switch_step = k
        last_contact_idx = int(chosen_idx_full)
        last_contact_score = float(chosen_score)

        n_out = n_c / (np.linalg.norm(n_c) + 1e-12)

        # distance targets
        d_nav = float(NAV_D_DES)
        d_touch = float(TOUCH_D_DES)

        pen = float(np.clip(pen, 1e-6, float(TOOL_RADIUS) * 0.9))
        d_push = float(TOOL_RADIUS) - float(pen)
        d_push = float(np.clip(
            d_push,
            float(TOOL_RADIUS) - float(PUSH_PEN_MAX),
            float(TOOL_RADIUS) - float(PUSH_PEN_MIN),
        ))

        x_nav = (p_c + n_out * d_nav).astype(np.float64)
        x_nav[2] = z_tool
        x_nav_goal = x_nav.copy()
        x_nav_goal[2] = z_tool

        x_touch = (p_c + n_out * d_touch).astype(np.float64)
        x_touch[2] = z_tool
        x_push = (p_c + n_out * d_push).astype(np.float64)
        x_push[2] = z_tool

        tool_des = tool_pos.copy()
        vxy_cap = float(TOOL_VXY_MAX)

        # state machine: tool motion
        if phase in ("RETRACT", "NAV", "RELEASE_SWITCH"):
            if phase == "RETRACT":
                nav_state = AStarNavState()
            obs_xy_nav = p_w_use[:, 0:2]
            x_nav_goal = nudge_goal_if_in_collision(x_nav.copy(), n_out, obs_xy_nav, float(d_nav), float(z_tool))
            tool_pos_next, tool_des, nav_state = nav_step_astar(
                tool_pos=tool_pos,
                goal_pos=x_nav_goal,
                obs_xy=obs_xy_nav,
                inflate_radius=float(d_nav),
                dt=float(DT),
                vxy_cap=float(vxy_cap),
                nav=nav_state,
                z_tool=float(z_tool),
                step_idx=int(k),
            )
        elif phase == "APPROACH":
            if not hasattr(run_push_minimal, "_dmin_cur"):
                run_push_minimal._dmin_cur = float(d_nav)
            run_push_minimal._dmin_cur = max(
                float(d_touch),
                float(run_push_minimal._dmin_cur) - float(APPROACH_D_RATE) * float(DT),
            )
            d_min_cur = float(run_push_minimal._dmin_cur)

            tool_des = enforce_min_sdf_distance(x_touch.copy(), q, d_min=float(d_min_cur))
            tool_des[2] = z_tool
            vxy_cap = min(vxy_cap, float(TOOL_VXY_MAX_PUSH))

            tool_pos_next = kinematic_step_towards(tool_pos, tool_des, float(DT), float(vxy_cap))
            tool_pos_next[2] = z_tool

        elif phase == "SETTLE":
            tool_des = enforce_min_sdf_distance(x_touch.copy(), q, d_min=float(d_touch))
            tool_des[2] = z_tool
            vxy_cap = min(vxy_cap, float(TOOL_VXY_MAX_PUSH))

            tool_pos_next = kinematic_step_towards(tool_pos, tool_des, float(DT), float(vxy_cap))
            tool_pos_next[2] = z_tool

        else:  # PUSH_TRANSLATE
            tool_des = enforce_min_sdf_distance(x_push.copy(), q, d_min=float(d_push))
            tool_des[2] = z_tool
            vxy_cap = min(vxy_cap, float(TOOL_VXY_MAX_PUSH))

            tool_pos_next = kinematic_step_towards(tool_pos, tool_des, float(DT), float(vxy_cap))
            tool_pos_next[2] = z_tool

        # Tangential suppression near contact
        if phase in ("APPROACH", "SETTLE", "PUSH_TRANSLATE", "RELEASE_SWITCH"):
            tool_pos_next = suppress_tangential_motion_near_contact(
                tool_pos=tool_pos,
                tool_pos_next=tool_pos_next,
                n_out=n_out,
                gap=gap_prev,
            )

        # enforce distance after step
        if phase in ("RETRACT", "NAV", "RELEASE_SWITCH"):
            tool_pos_next = project_out_of_obstacles(tool_pos_next, p_w_use[:, 0:2], float(d_nav), float(z_tool))
        elif phase in ("APPROACH", "SETTLE"):
            tool_pos_next = enforce_min_sdf_distance(tool_pos_next, q, d_min=float(d_touch))
        else:
            tool_pos_next = enforce_min_sdf_distance(tool_pos_next, q, d_min=float(d_push))
        tool_pos_next[2] = z_tool

        tool_vel_free = (tool_pos_next - tool_pos) / float(DT)
        tool_vel_free[2] = 0.0

        # ground solve
        f_ext = np.zeros(6, dtype=np.float64)
        f_ext[2] += -9.81 * float(sim.total_mass)

        v_ground, ecp, info = simobj.solve_step(
            q, v, f_ext, step_idx=k, return_ecp=True, tool_impulse_norm=float(last_tool_impulse_norm)
        )

        if getattr(simobj, "_frozen", False) and hasattr(simobj, "z_guess_frozen"):
            pn_ground = float(simobj.z_guess_frozen[9])
        else:
            pn_ground = float(simobj.z_guess[12]) if hasattr(simobj, "z_guess") else 0.0

        # tool-block impulse
        v_next = np.asarray(v_ground, dtype=np.float64).copy()
        tool_vel_next = tool_vel_free.copy()

        p_lin, a_c, n_used, _gap_unused = sim.compute_tool_block_impulse(
            q_block=q,
            v_block6=v_ground,
            tool_pos=tool_pos,
            tool_vel=tool_vel_free,
            tool_radius=float(TOOL_RADIUS),
            tool_mu=float(TOOL_MU),
            dt=float(DT),
            contact_eps=float(TOOL_CONTACT_EPS),
            restitution=float(TOOL_RESTITUTION),
            enable_margin=float(TOOL_ENABLE_MARGIN),
        )

        pnorm = float(np.linalg.norm(p_lin))
        last_tool_impulse_norm = pnorm

        # log tool impulse
        t_sim = float(k) * float(DT)
        logger.log_tool_impulse(
            step=int(k),
            t=t_sim,
            p_lin=np.asarray(p_lin, dtype=np.float64),
            contact_pt=None if a_c is None else np.asarray(a_c, dtype=np.float64),
            normal=np.asarray(n_c, dtype=np.float64),
            gap=float(_gap_unused),
            tool_pos=np.asarray(tool_pos, dtype=np.float64),
            tool_vel=np.asarray(tool_vel_free, dtype=np.float64),
        )

        pn_est = 0.0
        Fn_est = 0.0
        if pnorm > 0.0:
            n_used = np.asarray(n_used, dtype=np.float64).reshape(3,)
            pn_est = float(np.dot(np.asarray(p_lin, dtype=np.float64), n_used))
            Fn_est = pn_est / float(DT)

        F_contact = (np.asarray(p_lin, dtype=np.float64) / float(DT)) if float(DT) > 0 else np.zeros(3, dtype=np.float64)
        contact_pt = np.full((3,), np.nan, dtype=np.float64)

        if pnorm > 0.0:
            com_now = q[:3]
            r_arm = np.asarray(a_c, dtype=np.float64) - com_now
            Iw = sim.inertia_world_from_body_diag(sim.inertia_body_diag, q[3:7])
            Iw_inv = np.linalg.inv(Iw + 1e-12 * np.eye(3))

            v_next[0:3] = v_ground[0:3] + np.asarray(p_lin, dtype=np.float64) / float(sim.total_mass)
            v_next[3:6] = v_ground[3:6] + (Iw_inv @ np.cross(r_arm, np.asarray(p_lin, dtype=np.float64)))

            tool_vel_next = tool_vel_free - np.asarray(p_lin, dtype=np.float64) / float(TOOL_MASS)
            tool_vel_next[2] = 0.0

            contact_pt = np.asarray(a_c, dtype=np.float64).reshape(3,)

        # integrate block
        pos_next = q[:3] + v_next[:3] * float(DT)
        dq = sim.quat_from_omega_world_np(v_next[3:6], float(DT))
        quat_next = sim.quat_mul_wxyz_np(dq, q[3:7])
        quat_next /= (np.linalg.norm(quat_next) + 1e-12)

        q_next = np.hstack([pos_next, quat_next])
        v_next2 = v_next.copy()

        tool_pos = tool_pos_next.copy()
        tool_vel = tool_vel_next.copy()

        q_next, v_next2, min_z, _ = sim.project_to_ground_and_damp(
            q_next,
            v_next2,
            float(DT),
            sim.local_points_ref,
            pn_ground=float(pn_ground),
            pn_support_thresh=float(PN_SUPPORT_THRESH),
            support_z=0.0,
        )

        q, v = q_next, v_next2

        # ============================================================
        # Live-view update (compute+watch) + ghosts + block trajectory
        # ============================================================
        if viewer is not None and (k % LIVE_VIEW_STRIDE == 0):
            if not _viewer_running(viewer):
                try:
                    if viewer_ctx is not None:
                        viewer_ctx.__exit__(None, None, None)
                except Exception:
                    pass
                viewer = None
                viewer_ctx = None
            else:
                try:
                    # update qpos + tool marker
                    if ipos_body_vis is None:
                        raise RuntimeError("ipos_body_vis is None (unexpected).")
                    try:
                        qpos_vis = _com_to_body_origin_qpos(q, ipos_body_vis)
                    except Exception:
                        Rm = _quat_wxyz_to_rotmat(q[3:7])
                        p_body = q[0:3] - Rm @ ipos_body_vis
                        qpos_vis = np.concatenate([p_body, q[3:7]], axis=0)

                    d_vis.qpos[qposadr_vis: qposadr_vis + 7] = qpos_vis
                    if target_mocap_id >= 0:
                        d_vis.mocap_pos[target_mocap_id] = tool_pos
                    mujoco.mj_forward(m_vis, d_vis)

                    # clear overlays
                    _scene_clear(viewer.user_scn)

                    # (1) block trajectory (COM path)
                    live_block_path.append(q[0:3].copy())
                    if len(live_block_path) > int(LIVE_BLOCK_PATH_MAX):
                        live_block_path = live_block_path[-int(LIVE_BLOCK_PATH_MAX):]

                    if len(live_block_path) >= 2:
                        rgba_traj = np.array([0.2, 0.8, 0.2, 0.85], dtype=np.float32)
                        _scene_add_polyline_capsules(
                            viewer.user_scn,
                            np.asarray(live_block_path, dtype=np.float64),
                            rgba_traj,
                            radius=float(VIEW_PATH_RADIUS_TRAJ),
                            max_segs=int(VIEW_MAX_PATH_SEGS),
                        )

                    # (2) waypoint ghosts + current goal ghost
                    if _scene_add_ghost_tblock is not None:
                        rgba_wp  = np.array([0.55, 0.55, 0.55, float(GHOST_ALPHA_WP)],  dtype=np.float32)
                        rgba_ref = np.array([0.70, 0.70, 0.70, float(GHOST_ALPHA_REF)], dtype=np.float32)

                        for i in range(int(wp_p.shape[0])):
                            qwp7 = np.hstack([wp_p[i], wp_q[i]])
                            _scene_add_ghost_tblock(viewer.user_scn, qwp7, rgba_wp)

                        qref7 = np.hstack([p_goal, q_goal])
                        _scene_add_ghost_tblock(viewer.user_scn, qref7, rgba_ref)

                    # (3) your original overlays: ECP + tool + force arrow
                    if ecp is not None and np.all(np.isfinite(ecp)):
                        _scene_add_sphere(
                            viewer.user_scn,
                            np.asarray(ecp, dtype=np.float64),
                            float(VIEW_ECP_RADIUS),
                            np.array([0.2, 0.4, 1.0, 1.0], dtype=np.float32),
                        )

                    _scene_add_sphere(
                        viewer.user_scn,
                        np.asarray(tool_pos, dtype=np.float64),
                        float(TOOL_RADIUS),
                        np.array([0.2, 0.9, 0.2, 1.0], dtype=np.float32),
                    )

                    _scene_add_force_arrow(
                        viewer.user_scn,
                        np.asarray(tool_pos, dtype=np.float64),
                        np.asarray(p_lin, dtype=np.float64) / float(DT),
                        np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32),
                    )

                    viewer.sync()

                    if LIVE_VIEW_REALTIME:
                        now = time.perf_counter()
                        target = last_view_t + float(DT) * float(LIVE_VIEW_STRIDE)
                        if now < target:
                            time.sleep(target - now)
                        last_view_t = time.perf_counter()
                except Exception:
                    pass

        # log block energies
        t_sim = float(k) * float(DT)
        logger.log_block_energy(
            step=int(k),
            t=t_sim,
            q_block=np.asarray(q, dtype=np.float64),
            v_block6=np.asarray(v, dtype=np.float64),
            mass=float(sim.total_mass),
            inertia_body_diag=np.asarray(sim.inertia_body_diag, dtype=np.float64),
            frozen_flag=int(getattr(simobj, "_frozen", False)),
            g=9.81,
            z0=0.0,
        )

        # gap now
        if hasattr(sim, "closest_point_on_tblock_surface_world"):
            _, sdf_now = sim.closest_point_on_tblock_surface_world(tool_pos, q)
            sdf_now = float(sdf_now)
        else:
            sdf_now = 1e9
        gap_now = float(sdf_now) - float(TOOL_RADIUS)

        GAP_NAV_TOL   = 2e-4
        GAP_TOUCH_TOL = 2e-4

        # phase transitions
        if phase == "RETRACT":
            if gap_now >= float(NAV_CLEARANCE) - float(GAP_NAV_TOL):
                retract_counter += 1
            else:
                retract_counter = 0
            if retract_counter >= int(RETRACT_HOLD_STEPS):
                phase = "NAV"
                retract_counter = 0
                nav_state = AStarNavState()
                nav_stall_steps = 0
                nav_phase_steps = 0

        elif phase == "RELEASE_SWITCH":
            if gap_now >= float(NAV_CLEARANCE) - float(GAP_NAV_TOL):
                release_counter += 1
            else:
                release_counter = 0

            if release_counter >= int(RELEASE_HOLD_STEPS):
                if pending_contact_idx is not None:
                    lock_contact_idx = int(pending_contact_idx)
                    pending_contact_idx = None
                phase = "NAV"
                release_counter = 0
                nav_state = AStarNavState()
                nav_stall_steps = 0
                nav_phase_steps = 0

        elif phase == "NAV":
            if float(np.linalg.norm(tool_pos[0:2] - x_nav_goal[0:2])) < 0.03:
                phase = "APPROACH"
                run_push_minimal._dmin_cur = float(d_nav)
                nav_state = AStarNavState()
                nav_stall_steps = 0
                nav_phase_steps = 0

        elif phase == "APPROACH":
            if gap_now <= float(TOUCH_CLEARANCE) + float(GAP_TOUCH_TOL):
                phase = "SETTLE"
                settle_counter = 0

        elif phase == "SETTLE":
            if abs(gap_now - float(TOUCH_CLEARANCE)) <= float(GAP_TOUCH_TOL):
                settle_counter += 1
            else:
                settle_counter = 0

            if settle_counter >= int(SETTLE_HOLD_STEPS):
                phase = "PUSH_TRANSLATE"
                lock_contact_idx = None

        else:  # PUSH_TRANSLATE
            pen = update_penetration_mm_band(pen, Fn_est)

        # success check
        pos_err_xy = float(np.linalg.norm(p_goal[0:2] - q[0:2]))
        vel_xy = float(np.linalg.norm(v[0:2]))
        yaw_err = yaw_error_from_quats_wxyz(q[3:7], q_goal)
        wz = float(v[5])

        done = (
            (pos_err_xy <= float(POS_THRESH_XY)) and
            (vel_xy <= float(VEL_THRESH_XY)) and
            (abs(yaw_err) <= float(YAW_THRESH)) and
            (abs(wz) <= float(WZ_THRESH))
        )

        # logging
        traj_q.append(q.copy())
        traj_v.append(v.copy())
        ref_p_hist.append(p_goal.copy())
        ref_q_hist.append(q_goal.copy())

        tool_pos_hist.append(tool_pos.copy())
        tool_vel_hist.append(tool_vel.copy())
        tool_des_hist.append(tool_des.copy())

        tool_contact_pt_hist.append(contact_pt.copy())
        tool_impulse_hist.append(np.asarray(p_lin, dtype=np.float64).reshape(3,))
        tool_force_hist.append(np.asarray(F_contact, dtype=np.float64).reshape(3,))

        info_hist.append(int(info))
        minz_hist.append(float(min_z))
        ecp_hist.append(np.asarray(ecp, dtype=np.float64).reshape(3,))

        # Extra NAV debug (A*)
        if DEBUG_NAV and phase == "NAV":
            tool_step = float(np.linalg.norm(tool_pos - tool_pos_prev))
            do_dbg = ((k % int(DEBUG_NAV_EVERY)) == 0) or (DEBUG_NAV_PRINT_ON_STALL and tool_step < float(DEBUG_NAV_STALL_EPS))
            if do_dbg:
                d_raw  = float(np.linalg.norm(tool_pos[0:2] - x_nav[0:2]))
                d_goal = float(np.linalg.norm(tool_pos[0:2] - x_nav_goal[0:2]))
                plen = 0 if (getattr(nav_state, 'path_xy', None) is None) else int(nav_state.path_xy.shape[0])
        # NAV recovery watchdog
        if NAV_RECOVER_ENABLE and phase == "NAV":
            tool_step = float(np.linalg.norm(tool_pos - tool_pos_prev))
            nav_phase_steps += 1
            if tool_step < float(NAV_STALL_EPS):
                nav_stall_steps += 1
            else:
                nav_stall_steps = 0

            need_recover = (nav_stall_steps >= int(NAV_STALL_STEPS)) or (nav_phase_steps >= int(NAV_MAX_STEPS))
            cooldown_ok = (k - int(nav_last_recover_step)) >= int(NAV_RECOVER_COOLDOWN_STEPS)

            d_goal_now = float(np.linalg.norm(tool_pos[0:2] - x_nav_goal[0:2]))

            if need_recover and cooldown_ok and (d_goal_now > 0.035):
                reason = "STALL" if (nav_stall_steps >= int(NAV_STALL_STEPS)) else "TIMEOUT"

                idx_new, score_new, x_goal_new = pick_reachable_contact_idx_for_nav(
                    tool_pos=tool_pos,
                    q_block=q,
                    z_tool=float(z_tool),
                    d_nav=float(d_nav),
                    F_xy_des=e_p_xy,
                    p_w_use=p_w_use,
                    n_w_use=n_w_use,
                    idx_map=idx_map,
                    last_idx_use=None,
                    top_k=int(NAV_RECOVER_TOPK),
                )

                if idx_new is not None:
                    lock_contact_idx = int(idx_new)
                    pending_contact_idx = None
                    last_contact_idx = int(idx_new)
                    last_contact_score = float(score_new)
                    last_switch_step = int(k)
                    nav_state = AStarNavState()

                    nav_stall_steps = 0
                    nav_phase_steps = 0
                    nav_last_recover_step = int(k)

                    if NAV_RECOVER_PRINT:
                        if x_goal_new is None:
                            x_goal_new = x_nav_goal
                else:
                    lock_contact_idx = None
                    pending_contact_idx = None
                    last_contact_idx = None
                    last_contact_score = -1e18
                    nav_state = AStarNavState()

                    nav_stall_steps = 0
                    nav_phase_steps = 0
                    nav_last_recover_step = int(k)


        if done:
            if goal_idx < (len(WAYPOINTS) - 1):
                goal_idx += 1

                phase = "RETRACT"
                retract_counter = 0
                settle_counter = 0
                release_counter = 0

                pending_contact_idx = None
                lock_contact_idx = None
                nav_state = AStarNavState()
                nav_stall_steps = 0
                nav_phase_steps = 0
                nav_last_recover_step = -10**9

                pen = float(PUSH_PEN_INIT)
                last_contact_idx = None
                last_contact_score = -1e18
                last_switch_step = -10**9

                if hasattr(run_push_minimal, "_dmin_cur"):
                    delattr(run_push_minimal, "_dmin_cur")

                print(f"[Minimal] Switching to next goal_idx={goal_idx}")
            else:
                print("[Minimal] All waypoints reached.")
                break

    # convert logs
    traj_q = np.asarray(traj_q, dtype=np.float64)
    traj_v = np.asarray(traj_v, dtype=np.float64)
    ref_p  = np.asarray(ref_p_hist, dtype=np.float64)
    ref_q  = np.asarray(ref_q_hist, dtype=np.float64)

    tool_pos_hist = np.asarray(tool_pos_hist, dtype=np.float64)
    tool_vel_hist = np.asarray(tool_vel_hist, dtype=np.float64)
    tool_des_hist = np.asarray(tool_des_hist, dtype=np.float64)

    tool_contact_pt_hist = np.asarray(tool_contact_pt_hist, dtype=np.float64)
    tool_impulse_hist    = np.asarray(tool_impulse_hist, dtype=np.float64)
    tool_force_hist      = np.asarray(tool_force_hist, dtype=np.float64)

    info_hist = np.asarray(info_hist, dtype=np.int32)
    minz_hist = np.asarray(minz_hist, dtype=np.float64)
    ecp_hist  = np.asarray(ecp_hist, dtype=np.float64)

    t_grid_used = t_grid[:traj_q.shape[0]]

    if SAVE_NPZ:
        np.savez(
            OUT_NPZ,
            traj_q=traj_q,
            traj_v=traj_v,
            tool_pos=tool_pos_hist,
            tool_vel=tool_vel_hist,
            tool_des=tool_des_hist,
            tool_contact_pt=tool_contact_pt_hist,
            tool_impulse=tool_impulse_hist,
            tool_force=tool_force_hist,
            ecp=ecp_hist,
            info=info_hist,
            minz=minz_hist,
            t_grid=t_grid_used,
            p_ref=ref_p,
            q_ref=ref_q,
            wp_t=wp_t,
            wp_p=wp_p,
            wp_q=wp_q,
            dt=np.array([DT], dtype=np.float64),
            tool_radius=np.array([TOOL_RADIUS], dtype=np.float64),
            tool_mass=np.array([TOOL_MASS], dtype=np.float64),
            mode=np.array(["astar_nav_no_sdf_translate_plus_yaw"], dtype=object),
        )
        print("[Minimal] Saved:", OUT_NPZ)

    # close live viewer if used
    if viewer_ctx is not None:
        try:
            viewer_ctx.__exit__(None, None, None)
        except Exception:
            pass

    if VIEW:
        visualize_push_waypoints_mujoco(
            traj_q=traj_q,
            ref_p=ref_p,
            ref_q=ref_q,
            wp_t=wp_t,
            wp_p=wp_p,
            wp_q=wp_q,
            dt=float(DT),
            tool_pos=tool_pos_hist,
            tool_des=tool_des_hist,
            tool_force=tool_force_hist,
            tool_contact_pt=tool_contact_pt_hist,
            ecp_hist=ecp_hist,
            xml_path=getattr(sim, "XML_PATH", None),
            render_fps=float(VIEW_RENDER_FPS),
            playback_speed=float(VIEW_PLAYBACK_SPEED),
        )


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Waypoint pushing demo (compsim backend)")
    ap.add_argument("--xml", type=str, default="model/t_block_optimized.xml", help="MJCF used for both compsim params + MuJoCo visualization")
    ap.add_argument("--body", type=str, default="T_siconos", help="free body name (must match MJCF)")
    ap.add_argument("--view", action="store_true", help="force-enable MuJoCo viewer")
    ap.add_argument("--live_view", action="store_true", help="Compute while watching (live viewer)")
    ap.add_argument("--view_stride", type=int, default=5, help="In live_view, update viewer every N simulation steps")
    ap.add_argument("--view_realtime", action="store_true", help="In live_view, sleep to match dt (slower but real-time)")
    ap.add_argument("--no_view", action="store_true", help="force-disable MuJoCo viewer")
    args = ap.parse_args()

    xml_path = _os.path.abspath(_os.path.expanduser(args.xml))
    sim.XML_PATH = xml_path

    # IMPORTANT: update live-view body name from args
    global BODY_NAME
    BODY_NAME = str(args.body)

    # Make sure compsim is initialized before we touch state.total_mass/inertia.
    compsim.init_from_xml(xml_path, body=args.body)
    print(f"[compsim] xml={xml_path} body={args.body} mass={sim.total_mass:.6f} inertia_body_diag={np.asarray(sim.inertia_body_diag, dtype=np.float64)}")

    # Precompute local points once (used by ground MCP + ground projection)
    _ensure_local_points_ref(verbose=True)

    global VIEW, LIVE_VIEW, LIVE_VIEW_STRIDE, LIVE_VIEW_REALTIME
    if args.live_view:
        VIEW = 1
    if args.view:
        VIEW = 1
    if args.no_view:
        VIEW = 0

    LIVE_VIEW = bool(args.live_view)
    LIVE_VIEW_STRIDE = max(1, int(args.view_stride))
    LIVE_VIEW_REALTIME = bool(args.view_realtime)

    run_push_minimal()


if __name__ == "__main__":
    main()
