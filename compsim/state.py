from __future__ import annotations

import os
import numpy as np
import mujoco

# NOTE: This module keeps the original 'global cache' style used by v081,
# but the cache is now explicitly initialized via `init_from_xml`.

XML_PATH: str | None = None
model: mujoco.MjModel | None = None

body_name: str = "T_siconos"
body_id: int = -1

total_mass: float = 0.0
inertia_body_diag: np.ndarray | None = None

# Geoms belonging to T_siconos (in COM frame)
T_GEOM_IDS: np.ndarray | None = None
T_GEOM_TYPE: np.ndarray | None = None
T_GEOM_POS: np.ndarray | None = None
T_GEOM_QUAT: np.ndarray | None = None
T_GEOM_SIZE: np.ndarray | None = None
T_IPOS_BODY: np.ndarray | None = None


def init_from_xml(xml_path: str, body: str = "T_siconos") -> None:
    """(Re)initialize cached MuJoCo model and T-block geometry.

    The cache is used by geometry and contact routines.
    """
    global XML_PATH, model, body_name, body_id
    global total_mass, inertia_body_diag
    global T_GEOM_IDS, T_GEOM_TYPE, T_GEOM_POS, T_GEOM_QUAT, T_GEOM_SIZE, T_IPOS_BODY

    XML_PATH = xml_path
    body_name = body

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"[compsim.state] body '{body_name}' not found in {XML_PATH}")

    total_mass = float(model.body_mass[body_id])
    inertia_body_diag = model.body_inertia[body_id].astype(np.float64)

    geom_start = int(model.body_geomadr[body_id])
    geom_num = int(model.body_geomnum[body_id])

    T_GEOM_IDS = np.arange(geom_start, geom_start + geom_num, dtype=np.int32)
    T_GEOM_TYPE = np.array(model.geom_type[T_GEOM_IDS], dtype=np.int32)
    T_GEOM_POS = np.array(model.geom_pos[T_GEOM_IDS], dtype=np.float64)
    T_GEOM_QUAT = np.array(model.geom_quat[T_GEOM_IDS], dtype=np.float64)  # wxyz
    T_GEOM_SIZE = np.array(model.geom_size[T_GEOM_IDS], dtype=np.float64)

    # COM in body frame
    T_IPOS_BODY = np.array(model.body_ipos[body_id], dtype=np.float64)

    # Shift geom positions to COM frame (matches original v081)
    T_GEOM_POS = (T_GEOM_POS - T_IPOS_BODY).astype(np.float64)


def ensure_initialized(default_xml: str | None = None) -> None:
    """Initialize caches if needed (defaulting to v081's xml)."""
    global model
    if model is not None:
        return

    if default_xml is None:
        # prefer local file next to the caller; fallback to current working dir
        default_xml = os.environ.get("COMPSIM_DEFAULT_XML", "81_t_block_optimized.xml")
    init_from_xml(default_xml, body="T_siconos")
