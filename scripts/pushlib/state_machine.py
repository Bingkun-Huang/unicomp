#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
push_minimal_framework_yaw_tangentbug.py

方案1（系统解法-升级版）：用 A* 网格导航（基于侧面采样点障碍）彻底解决“拐角卡死”
================================================================================
你遇到的 deadlock 本质是：NAV 阶段只做“朝目标直线走 + SDF 投影”，在拐角处
closest-point/normal 不连续 + 直线投影没有“沿墙走”的行为，导致在 corner 周围来回抖动/锁死。

本文件把 NAV 替换为 A* 网格路径规划：
  - 用侧面 surface samples (点云) 当作障碍物，按 d_nav 膨胀后在 2D 网格上跑 A*。
  - 只要 tool center 与任意 sample 点的距离 >= d_nav，就视为无碰撞。（点云近似，非严格几何）

同时（你提出的“切线方向移动会把 T 撞飞”）：
  - 加了 **near-contact tangential suppression**：当 gap 很小，抑制切向分量，避免球体实体切向运动时
    横向撞击把 T “撞飞”。只在接近/接触时生效，远离时不影响绕障。

结构保持不变：
  RETRACT -> NAV -> APPROACH -> SETTLE -> PUSH_TRANSLATE
并加了更稳的面切换：
  RELEASE_SWITCH：先脱离（到 NAV clearance），再去新面（避免贴着面切换导致拖拽/撞飞）。

依赖：
  - compsim（你的后端）
  - mujoco
  - (可选) recording_helpers.RealtimeDataLogger（若不存在会自动 fallback 为 no-op logger）

运行：
  python3 push_minimal_framework_yaw_tangentbug.py --xml model/t_block_optimized2.xml --body T_siconos --view
  python3 push_minimal_framework_yaw_tangentbug.py --xml model/t_block_optimized2.xml --body T_siconos --live_view --view_stride 5
"""

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
# NOTE:
# This module lives under:  <repo>/scripts/pushlib/
# We want the repo root for default paths (e.g., <repo>/model/...).
_REPO_ROOT = _os.path.abspath(_os.path.join(_THIS_DIR, "..", ".."))

import compsim
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

    def __init__(self):
        # Lazy MuJoCo fallback cache (used when compsim/state does not yet
        # expose mass/inertia, or when APIs change across versions).
        self._mj_model_cache = None
        self._mj_body_id_cache = None
        self._mj_cache_key = None

    @property
    def state(self):
        """Return the *current* simulator state handle.

        We fetch `compsim.state` dynamically instead of caching it at
        import-time, because some compsim versions replace `compsim.state`
        during initialization (e.g., init_from_xml)."""

        return getattr(compsim, "state", compsim)

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

    def _mj_model_and_body_id(self):
        """Load MuJoCo model lazily (fallback) and cache (model, body_id)."""

        key = (self.XML_PATH, self.BODY_NAME)
        if (
            self._mj_model_cache is None
            or self._mj_body_id_cache is None
            or self._mj_cache_key != key
        ):
            model = mujoco.MjModel.from_xml_path(self.XML_PATH)
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.BODY_NAME)
            if bid < 0:
                raise ValueError(f"Body '{self.BODY_NAME}' not found in XML: {self.XML_PATH}")
            self._mj_model_cache = model
            self._mj_body_id_cache = int(bid)
            self._mj_cache_key = key
        return self._mj_model_cache, self._mj_body_id_cache

    @property
    def total_mass(self):
        st = self.state

        for attr in ("total_mass", "mass"):
            if hasattr(st, attr):
                return float(getattr(st, attr))
        for attr in ("total_mass", "mass"):
            if hasattr(compsim, attr):
                return float(getattr(compsim, attr))

        model, bid = self._mj_model_and_body_id()
        return float(model.body_mass[bid])

    @property
    def inertia_body_diag(self):
        st = self.state

        for attr in ("inertia_body_diag", "inertia_diag_body", "inertia_diag", "inertia"):
            if hasattr(st, attr):
                return np.asarray(getattr(st, attr), dtype=np.float64)
        for attr in ("inertia_body_diag", "inertia_diag_body", "inertia_diag", "inertia"):
            if hasattr(compsim, attr):
                return np.asarray(getattr(compsim, attr), dtype=np.float64)

        model, bid = self._mj_model_and_body_id()
        return np.asarray(model.body_inertia[bid], dtype=np.float64)

    @property
    def local_points_ref(self):
        return _ensure_local_points_ref(verbose=False)


# keep the old name `sim` to minimize diffs below
sim = _SimCompat()

# ------------------------------------------------------------------
# Public defaults exported for CLI wrappers / run_core
# ------------------------------------------------------------------
# IMPORTANT: callers are expected to override these via arguments
# (e.g., `push_waypoints_compsim_live.py --xml ... --body ...`).
XML_PATH = getattr(sim, "XML_PATH", "model/t_block_optimized.xml")

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

DEFAULT_WAYPOINTS = [
    {"idx": 0, "p": [0.60, 0.00, 0.02], "q": [0.707, 0.707, 0.0, 0.0]},   # initial
    {"idx": 1, "p": [0.70, 0.50, 0.02], "q": [0.50, 0.50, 0.5, 0.5]},    # target (planar translation + yaw)
    {"idx": 2, "p": [0.40, -0.25, 0.02], "q": [0.707, 0.707, 0.0, 0.0]},   # another target
]

DT = 0.002
TOTAL_TIME = 60.0
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
FN_MIN = 10.0
FN_MAX = 20.0
PEN_STEP = 0.0005

RETRACT_HOLD_TIME = 0.20
SETTLE_HOLD_TIME = 0.25
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

VIEW_FORCE_SCALE = 0.03
VIEW_ARROW_WIDTH = 0.01
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



# ---- split imports ----
from .geom import *  # noqa: F401,F403
from .nav_astar import *  # noqa: F401,F403
from .viz_playback import *  # noqa: F401,F403

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


def run_push_minimal(
    WAYPOINTS=None,
    XML_PATH=XML_PATH,
    BODY_NAME=BODY_NAME,
    STEPS=STEPS,
    LIVE_VIEW=LIVE_VIEW,
    LIVE_VIEW_STRIDE=LIVE_VIEW_STRIDE,
    SAVE_NPZ=False,
    OUT_NPZ=OUT_NPZ,
    VIEW=False,
):
    # Allow the caller (e.g. scripts/push_waypoints_compsim_live.py) to provide custom waypoints
    if WAYPOINTS is None:
        WAYPOINTS = DEFAULT_WAYPOINTS

    # Allow caller to override the visualization MJCF
    try:
        sim.XML_PATH = XML_PATH
    except Exception:
        pass

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

            m_vis = mujoco.MjModel.from_xml_path(XML_PATH)
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
                        np.asarray(size, dtype=np.float64).reshape(3, 1),
                        np.asarray(p_geom, dtype=np.float64).reshape(3, 1),
                        np.asarray(mat, dtype=np.float64).reshape(9, 1),
                        np.asarray(rgba, dtype=np.float32).reshape(4, 1),
                    )
            # -------------------------------------------------------------------------

            # mocap marker (optional)
            target_mocap_id = _find_mocap_id(m_vis, LIVE_TARGET_MOCAP)

            viewer_ctx = mujoco.viewer.launch_passive(m_vis, d_vis)
            viewer = viewer_ctx.__enter__()

            # ensure translucent geoms (ghosts) render
            try:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
            except Exception:
                pass

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
                gap_lock=float(TANG_SUPPRESS_GAP_LOCK),
                gap_free=float(TANG_SUPPRESS_GAP_FREE),
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

        if (k % 200) == 0:
            print(
                f"[k={k:6d} t={t_grid[k]:.3f}s phase={phase:>14s}] "
                f"pos_xy_err={pos_err_xy:.3e} vel_xy={vel_xy:.3e} "
                f"yaw_err={np.rad2deg(yaw_err):+.2f}deg wz={wz:+.3f} "
                f"sdf={sdf_now:.4e} gap={gap_now*1e3:+.3f}mm "
                f"pen={pen*1e3:.3f}mm Fn={Fn_est:.2f}N "
                f"|p|={pnorm:.3e} min_z={min_z:.2e} frozen={int(getattr(simobj,'_frozen',False))} info={info} "
                f"lock={int(lock_contact_idx is not None)} nav=A*:{getattr(nav_state,'wp_i',0)}/{(0 if getattr(nav_state,'path_xy',None) is None else int(nav_state.path_xy.shape[0]))} repl={getattr(nav_state,'replans',0)}"
            )

        # Extra NAV debug (A*)
        if DEBUG_NAV and phase == "NAV":
            tool_step = float(np.linalg.norm(tool_pos - tool_pos_prev))
            do_dbg = ((k % int(DEBUG_NAV_EVERY)) == 0) or (DEBUG_NAV_PRINT_ON_STALL and tool_step < float(DEBUG_NAV_STALL_EPS))
            if do_dbg:
                d_raw  = float(np.linalg.norm(tool_pos[0:2] - x_nav[0:2]))
                d_goal = float(np.linalg.norm(tool_pos[0:2] - x_nav_goal[0:2]))
                plen = 0 if (getattr(nav_state, 'path_xy', None) is None) else int(nav_state.path_xy.shape[0])
                print(f"  [NAVDBG] tool_step={tool_step:.3e} d_raw={d_raw:.4f} d_goal={d_goal:.4f} path_len={plen} wp={getattr(nav_state,'wp_i',0)} repl={getattr(nav_state,'replans',0)} fail={getattr(nav_state,'failed_plans',0)}")
                print(f"          x_nav_raw=({x_nav[0]:+.3f},{x_nav[1]:+.3f}) x_nav_goal=({x_nav_goal[0]:+.3f},{x_nav_goal[1]:+.3f}) tool=({tool_pos[0]:+.3f},{tool_pos[1]:+.3f})")

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
                    contact_min_align=float(CONTACT_MIN_ALIGN),
                    contact_hysteresis_bonus=float(CONTACT_HYSTERESIS_BONUS),
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
                        print(
                            f"[NAV-RECOVER:{reason}] unlock->lock idx={int(idx_new)} score={float(score_new):+.3f} "
                            f"goal=({float(x_goal_new[0]):+.3f},{float(x_goal_new[1]):+.3f}) "
                            f"mode_reset=1"
                        )
                else:
                    lock_contact_idx = None
                    pending_contact_idx = None
                    last_contact_idx = None
                    last_contact_score = -1e18
                    nav_state = AStarNavState()

                    nav_stall_steps = 0
                    nav_phase_steps = 0
                    nav_last_recover_step = int(k)

                    if NAV_RECOVER_PRINT:
                        print(f"[NAV-RECOVER:{reason}] no reachable candidates in TOPK={int(NAV_RECOVER_TOPK)} -> unlock all")

        if done:
            print(
                f"[Done] reached waypoint {goal_idx} at t={t_grid[k]:.3f}s "
                f"pos_xy_err={pos_err_xy:.3e}, vel_xy={vel_xy:.3e}, "
                f"yaw_err={np.rad2deg(yaw_err):.2f}deg, wz={wz:.3e}."
            )

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

    out = dict(
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
        dt=float(DT),
        tool_radius=float(TOOL_RADIUS),
        tool_mass=float(TOOL_MASS),
        mode="astar_nav_no_sdf_translate_plus_yaw",
        xml_path=getattr(sim, "XML_PATH", None),
    )

    # close live viewer if used
    if viewer_ctx is not None:
        try:
            viewer_ctx.__exit__(None, None, None)
        except Exception:
            pass

    return out
