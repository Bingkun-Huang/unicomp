from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp


# ===========================
# Quaternion helpers (NumPy)
# ===========================

def quat_to_R_np_wxyz(q):
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return np.array([
        [ww+xx-yy-zz, 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   ww-xx+yy-zz, 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   ww-xx-yy+zz],
    ], dtype=np.float64)

def inertia_world_inv_from_body_diag_np(inertia_body_diag, quat_wxyz):
    I_body = np.diag(np.asarray(inertia_body_diag, dtype=np.float64))
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    R = quat_to_R_np_wxyz(q)
    Iw = R @ I_body @ R.T
    return np.linalg.inv(Iw + 1e-12 * np.eye(3))


def quat_conj_np(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=np.float64)
    return np.array([q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3]], dtype=np.float64)


def quat_mul_wxyz_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def rotate_vector_by_quaternion_np(v: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q=[w,x,y,z]."""
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64)
    qv = np.array([x, y, z], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + w * v)


def rotate_world_to_local_np(v_world: np.ndarray, q_body_wxyz: np.ndarray) -> np.ndarray:
    return rotate_vector_by_quaternion_np(v_world, quat_conj_np(q_body_wxyz))


def quat_from_omega_world_np(omega_world: np.ndarray, dt: float) -> np.ndarray:
    omega = np.asarray(omega_world, dtype=np.float64)
    ang = float(np.linalg.norm(omega) * dt)
    if ang < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = omega / (np.linalg.norm(omega) + 1e-12)
    half = 0.5 * ang
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def quat_to_R_np_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float64,
    )


def skew_np(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]], dtype=np.float64)


# ===========================
# JAX helpers
# ===========================

@jax.jit
def quaternion_rotate(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    w, x, y, z = q
    return v + 2.0 * jnp.cross(q[1:], jnp.cross(q[1:], v) + w * v)


@jax.jit
def get_world_points(q_pos: jnp.ndarray, local_pts: jnp.ndarray) -> jnp.ndarray:
    pos = q_pos[0:3]
    quat = q_pos[3:7]
    return jax.vmap(lambda p: quaternion_rotate(quat, p))(local_pts) + pos


@jax.jit
def build_jacobian_single(r_world: jnp.ndarray) -> jnp.ndarray:
    r = r_world
    skew = jnp.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]], dtype=jnp.float64)
    return jnp.concatenate([jnp.eye(3, dtype=jnp.float64), -skew], axis=1)


@jax.jit
def quat_to_R_wxyz(q: jnp.ndarray) -> jnp.ndarray:
    w, x, y, z = q
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return jnp.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=jnp.float64,
    )


@jax.jit
def mass_matrix_inv_6x6(mass: jnp.ndarray, inertia_body_diag: jnp.ndarray, quat_wxyz: jnp.ndarray) -> jnp.ndarray:
    R = quat_to_R_wxyz(quat_wxyz)
    I_body = jnp.diag(inertia_body_diag)
    I_world = R @ I_body @ R.T
    I_world_inv = jnp.linalg.inv(I_world)
    Minv = jnp.zeros((6, 6), dtype=jnp.float64)
    Minv = Minv.at[0:3, 0:3].set((1.0 / mass) * jnp.eye(3, dtype=jnp.float64))
    Minv = Minv.at[3:6, 3:6].set(I_world_inv)
    return Minv


@jax.jit
def project_ellipsoid3(pt, po, pr, limit, e_t, e_o, e_r):
    limit = jnp.maximum(limit, 0.0)
    ut = pt / (e_t + 1e-18)
    uo = po / (e_o + 1e-18)
    ur = pr / (e_r + 1e-18)

    nrm = jnp.sqrt(ut * ut + uo * uo + ur * ur + 1e-18)
    scale = jnp.where(nrm > limit, limit / nrm, 1.0)

    utp = ut * scale
    uop = uo * scale
    urp = ur * scale

    return utp * e_t, uop * e_o, urp * e_r

# -----------------------------------------------------------------------------
# Backwards-compatible alias (from original v081)
# -----------------------------------------------------------------------------
_skew = skew_np
