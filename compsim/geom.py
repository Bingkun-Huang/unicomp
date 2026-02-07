from __future__ import annotations

import numpy as np
import mujoco

from . import state
from .math3d import rotate_world_to_local_np, rotate_vector_by_quaternion_np, quat_conj_np


def _box_sdf_and_closest_point(p_g: np.ndarray, half_ext: np.ndarray) -> tuple[float, np.ndarray]:
    p = np.asarray(p_g, dtype=np.float64)
    b = np.asarray(half_ext, dtype=np.float64)
    p_closest = np.clip(p, -b, b)
    q = np.abs(p) - b
    outside = float(np.linalg.norm(np.maximum(q, 0.0)))
    inside = float(min(max(q[0], q[1], q[2]), 0.0))
    sdf = outside + inside
    return sdf, p_closest


def project_point_to_tblock_surface_world(p_world: np.ndarray, q_curr: np.ndarray) -> tuple[np.ndarray, float]:
    """Project a world point to the nearest point on the block surface.

    Returns (p_proj_world, sdf), where sdf is the signed distance (positive outside).
    """
    state.ensure_initialized()
    pos_com = np.asarray(q_curr[:3], dtype=np.float64)
    q_body = np.asarray(q_curr[3:7], dtype=np.float64)

    p_body = rotate_world_to_local_np(np.asarray(p_world, dtype=np.float64) - pos_com, q_body)

    best_sdf = 1e9
    best_p_body = None

    for gt, gp, gq, gs in zip(state.T_GEOM_TYPE, state.T_GEOM_POS, state.T_GEOM_QUAT, state.T_GEOM_SIZE):
        if int(gt) != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        p_g = rotate_vector_by_quaternion_np(p_body - gp, quat_conj_np(gq))
        sdf, p_g_closest = _box_sdf_and_closest_point(p_g, gs[:3])
        if float(sdf) < best_sdf:
            best_sdf = float(sdf)
            p_body_closest = rotate_vector_by_quaternion_np(p_g_closest, gq) + gp
            best_p_body = p_body_closest

    if best_p_body is None:
        return np.asarray(p_world, dtype=np.float64).copy(), 1e9

    p_proj_world = rotate_vector_by_quaternion_np(best_p_body, q_body) + pos_com
    return p_proj_world, best_sdf


def add_world_wrench_projected(
    f_ext_acc: np.ndarray,
    p_world: np.ndarray,
    F_world: np.ndarray,
    tau_world: np.ndarray,
    q_curr: np.ndarray,
) -> np.ndarray:
    """Accumulate external wrench expressed at a world point, projected to surface point."""
    p_proj, _ = project_point_to_tblock_surface_world(p_world, q_curr)
    pos_com = np.asarray(q_curr[:3], dtype=np.float64)
    r = p_proj - pos_com
    f_ext_acc[0:3] += np.asarray(F_world, dtype=np.float64)
    f_ext_acc[3:6] += (np.asarray(tau_world, dtype=np.float64) + np.cross(r, np.asarray(F_world, dtype=np.float64)))
    return p_proj
