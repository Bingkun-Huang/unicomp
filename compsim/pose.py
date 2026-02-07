from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .math3d import quat_to_R_wxyz, quat_to_R_np_wxyz


def com_to_body_origin_qpos(q_com_wxyz: np.ndarray, ipos_body: np.ndarray) -> np.ndarray:
    """Convert internal COM state to MuJoCo freejoint qpos (body origin).

    q_com_wxyz: [com_world(3), quat_wxyz(4)]
    ipos_body: model.body_ipos[body_id] (COM in body frame)
    """
    q = np.asarray(q_com_wxyz, dtype=np.float64).copy()
    com = q[:3]
    quat = q[3:7]
    R = np.asarray(quat_to_R_wxyz(jnp.array(quat, dtype=jnp.float64)))  # body->world
    body_origin = com - R @ np.asarray(ipos_body, dtype=np.float64)
    return np.hstack([body_origin, quat])


def body_origin_qpos_to_com(qpos_wxyz: np.ndarray, ipos_body: np.ndarray) -> np.ndarray:
    """Convert MuJoCo freejoint qpos (body origin) to internal COM state."""
    qpos = np.asarray(qpos_wxyz, dtype=np.float64).copy()
    body_origin = qpos[:3]
    quat = qpos[3:7]
    R = np.asarray(quat_to_R_wxyz(jnp.array(quat, dtype=jnp.float64)))  # body->world
    com = body_origin + R @ np.asarray(ipos_body, dtype=np.float64)
    return np.hstack([com, quat])


def place_body_on_ground_qpos_from_quat(
    quat_wxyz,
    ipos_body,
    local_pts_com,
    origin_xy=(0.0, 0.0),
    clearance=1e-6,
):
    """Return qpos0 (body origin) such that the body is placed on plane z=0."""
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    quat = quat / (np.linalg.norm(quat) + 1e-12)

    R = quat_to_R_np_wxyz(quat)
    pts = np.asarray(np.array(local_pts_com), dtype=np.float64)
    zmin_rot = float(np.min((pts @ R.T)[:, 2]))

    com = np.array([origin_xy[0], origin_xy[1], clearance - zmin_rot], dtype=np.float64)
    origin = com - R @ np.asarray(ipos_body, dtype=np.float64)

    qpos0 = np.zeros(7, dtype=np.float64)
    qpos0[0:3] = origin
    qpos0[3:7] = quat
    return qpos0, com
