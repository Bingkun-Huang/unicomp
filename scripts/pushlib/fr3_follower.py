from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import mujoco


DEFAULT_FR3_JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))


@dataclass(frozen=True)
class Fr3FollowerConfig:
    ee_site: str = "tool_tip"
    joint_names: Sequence[str] = DEFAULT_FR3_JOINT_NAMES
    kp: float = 50.0
    damping: float = 1e-3
    dq_max: float = 1.0
    ee_vmax: float = 0.8
    q_home: Sequence[float] | None = (0.0, -0.8, 0.0, -2.35, 0.0, 1.57, 0.78)


class Fr3DlsIkFollower:
    """Position-only damped least-squares IK follower for FR3 visualization.

    The follower moves the visual robot's qpos so its end-effector site tracks
    the planner tool used by the compsim pushing pipeline. It intentionally
    does not own pushing physics.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: Fr3FollowerConfig):
        self.model = model
        self.data = data
        self.config = config

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, str(config.ee_site))
        if self.site_id < 0:
            raise ValueError(f"FR3 end-effector site '{config.ee_site}' not found")

        self.joint_names = [str(name) for name in config.joint_names]
        self.joint_ids: list[int] = []
        self.qpos_adrs: list[int] = []
        self.dof_adrs: list[int] = []
        self.joint_ranges: list[tuple[float, float]] = []

        for name in self.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"FR3 joint '{name}' not found")
            self.joint_ids.append(int(jid))
            self.qpos_adrs.append(int(model.jnt_qposadr[jid]))
            self.dof_adrs.append(int(model.jnt_dofadr[jid]))
            self.joint_ranges.append(
                (
                    float(model.jnt_range[jid, 0]),
                    float(model.jnt_range[jid, 1]),
                )
            )

        self.qpos_adrs_np = np.asarray(self.qpos_adrs, dtype=np.int32)
        self.dof_adrs_np = np.asarray(self.dof_adrs, dtype=np.int32)
        self._jacp = np.zeros((3, model.nv), dtype=np.float64)
        self._jacr = np.zeros((3, model.nv), dtype=np.float64)

        if config.q_home is not None:
            self.set_qpos(np.asarray(config.q_home, dtype=np.float64))
        mujoco.mj_forward(self.model, self.data)

    def set_qpos(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64).reshape(len(self.qpos_adrs),)
        for i, adr in enumerate(self.qpos_adrs):
            lo, hi = self.joint_ranges[i]
            self.data.qpos[adr] = float(np.clip(q[i], lo, hi))

    def site_pos(self) -> np.ndarray:
        return np.asarray(self.data.site_xpos[self.site_id], dtype=np.float64).copy()

    def step(self, target_pos: np.ndarray, dt: float) -> float:
        """Advance the IK follower one step and return EE tracking error."""
        dt = float(dt)
        target = np.asarray(target_pos, dtype=np.float64).reshape(3,)
        current = self.site_pos()
        error = target - current

        v_des = float(self.config.kp) * error
        v_norm = float(np.linalg.norm(v_des))
        if v_norm > float(self.config.ee_vmax) and v_norm > 1e-12:
            v_des *= float(self.config.ee_vmax) / v_norm

        self._jacp[...] = 0.0
        self._jacr[...] = 0.0
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        J = self._jacp[:, self.dof_adrs_np]

        lam2 = float(self.config.damping) ** 2
        A = J @ J.T + lam2 * np.eye(3, dtype=np.float64)
        dq = J.T @ np.linalg.solve(A, v_des)
        dq = np.clip(dq, -float(self.config.dq_max), float(self.config.dq_max))

        for i, adr in enumerate(self.qpos_adrs):
            lo, hi = self.joint_ranges[i]
            q_next = float(self.data.qpos[adr]) + float(dq[i]) * dt
            self.data.qpos[adr] = float(np.clip(q_next, lo, hi))

        mujoco.mj_forward(self.model, self.data)
        return float(np.linalg.norm(error))
