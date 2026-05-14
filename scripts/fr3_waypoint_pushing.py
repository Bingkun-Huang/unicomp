#!/usr/bin/env python3
"""Run FR3 waypoint pushing on the compsim contact pipeline.

This is the user-facing script for the current FR3 pushing mainline.

Architecture:
    - waypoint/contact state machine: scripts.pushlib.state_machine
    - block dynamics + ground support + tool impulse: compsim
    - FR3 visual/kinematic follower: scripts.pushlib.fr3_follower
    - MuJoCo: live visualization and geometry lookup

Recommended command:
    python3 scripts/fr3_waypoint_pushing.py \
      --xml model/fr3_xml_pack/fr3_push_modular_compsim.xml \
      --body T_siconos \
      --live_view \
      --fr3_enable_ik \
      --fr3_physical_tool
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pushlib.io_and_logging import save_npz  # noqa: E402
from scripts.pushlib.state_machine import (  # noqa: E402
    BODY_NAME as DEFAULT_BODY_NAME,
    DEFAULT_WAYPOINTS,
    STEPS as DEFAULT_STEPS,
    XML_PATH as DEFAULT_XML_PATH,
    run_push_minimal,
)
from scripts.pushlib.viz_playback import visualize_push_waypoints_mujoco  # noqa: E402


DEFAULT_FR3_JOINT_NAMES = "fr3_joint1,fr3_joint2,fr3_joint3,fr3_joint4,fr3_joint5,fr3_joint6,fr3_joint7"


def parse_joint_names(raw: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in str(raw).split(",") if name.strip())
    if len(names) != 7:
        raise argparse.ArgumentTypeError(
            f"expected 7 comma-separated FR3 joint names, got {len(names)}"
        )
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FR3 waypoint pushing: kinematic FR3/tool_tip driving compsim contact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--xml",
        type=str,
        default=DEFAULT_XML_PATH,
        help="MuJoCo XML used for visualization and geometry queries.",
    )
    parser.add_argument(
        "--body",
        type=str,
        default=DEFAULT_BODY_NAME,
        help="Pushed body name in the MJCF.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Number of simulation steps.",
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="compsim",
        choices=("compsim",),
        help="Simulator backend.",
    )

    parser.add_argument("--live_view", action="store_true", help="Show MuJoCo passive viewer during the run.")
    parser.add_argument("--view", action="store_true", help="Show playback viewer after the run.")
    parser.add_argument("--save_npz", action="store_true", help="Save trajectory output to NPZ.")
    parser.add_argument("--out_npz", type=str, default=None, help="Output NPZ path.")
    parser.add_argument("--render_fps", type=float, default=None, help="Playback render FPS.")
    parser.add_argument("--playback_speed", type=float, default=None, help="Playback speed multiplier.")

    parser.add_argument(
        "--fr3_enable_ik",
        action="store_true",
        help="Enable FR3 end-effector IK follower in live view.",
    )
    parser.add_argument("--fr3_ee_site", type=str, default="tool_tip", help="FR3 end-effector site name.")
    parser.add_argument(
        "--fr3_joint_names",
        type=parse_joint_names,
        default=parse_joint_names(DEFAULT_FR3_JOINT_NAMES),
        help="Comma-separated FR3 joint names.",
    )
    parser.add_argument("--fr3_ik_kp", type=float, default=50.0, help="FR3 IK position gain.")
    parser.add_argument("--fr3_ik_damping", type=float, default=1e-3, help="FR3 damped least-squares IK damping.")
    parser.add_argument("--fr3_ik_dq_max", type=float, default=1.0, help="FR3 max IK joint velocity command.")
    parser.add_argument("--fr3_ee_vmax", type=float, default=0.8, help="FR3 max end-effector tracking speed.")
    parser.add_argument(
        "--fr3_physical_tool",
        action="store_true",
        help="Use actual FR3 tool_tip pose/velocity as the compsim pushing tool.",
    )

    parser.add_argument(
        "--support_z",
        type=float,
        default=None,
        help="Manual support plane height. If omitted, inferred from --support_geom.",
    )
    parser.add_argument(
        "--support_geom",
        type=str,
        default="table_top",
        help="Geom used to infer support plane height.",
    )

    return parser


def initialize_compsim(xml_path: str, body_name: str) -> None:
    """Initialize compsim and clear geometry caches for the selected XML/body."""
    try:
        import compsim
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import compsim. Activate the unicomp environment and run from the repo root."
        ) from exc

    if hasattr(compsim, "init_from_xml"):
        compsim.init_from_xml(xml_path, body=body_name)
    elif hasattr(compsim, "init"):
        compsim.init(xml_path, body=body_name)
    else:
        raise RuntimeError("compsim has neither init_from_xml(...) nor init(...).")

    # These caches depend on XML/body. Clear them so repeated runs in one Python
    # process cannot silently reuse stale geometry.
    try:
        import scripts.pushlib.state_machine as state_machine

        state_machine._LOCAL_POINTS_REF = None
    except Exception:
        pass

    try:
        import compsim.samples as samples

        samples._local_points_ref = None
    except Exception:
        pass


def maybe_save_output(args: argparse.Namespace, out: dict) -> None:
    if not bool(args.save_npz):
        return
    out_path = args.out_npz or "push_waypoints_compsim_live.npz"
    save_npz(out_path, out)
    print(f"[io] saved NPZ -> {out_path}")


def maybe_playback(args: argparse.Namespace, out: dict) -> None:
    if not bool(args.view):
        return

    render_fps = args.render_fps if args.render_fps is not None else float(out.get("render_fps", 60.0))
    playback_speed = (
        args.playback_speed
        if args.playback_speed is not None
        else float(out.get("playback_speed", 1.0))
    )

    visualize_push_waypoints_mujoco(
        traj_q=out["traj_q"],
        ref_p=out["ref_p"],
        ref_q=out["ref_q"],
        wp_t=out["wp_t"],
        wp_p=out["wp_p"],
        wp_q=out["wp_q"],
        dt=float(out.get("dt", 0.002)),
        tool_pos=out.get("tool_pos", None),
        tool_des=out.get("tool_des", None),
        tool_force=out.get("tool_force", None),
        tool_contact_pt=out.get("tool_contact_pt", None),
        ecp_hist=out.get("ecp", None),
        xml_path=args.xml or out.get("xml_path", None),
        body_name=args.body or out.get("body_name", "T_siconos"),
        render_fps=render_fps,
        playback_speed=playback_speed,
    )


def run(args: argparse.Namespace) -> dict:
    if str(args.sim) != "compsim":
        raise ValueError(f"unsupported simulator backend: {args.sim}")

    initialize_compsim(args.xml, args.body)

    out = run_push_minimal(
        WAYPOINTS=DEFAULT_WAYPOINTS,
        XML_PATH=args.xml,
        BODY_NAME=args.body,
        STEPS=int(args.steps),
        LIVE_VIEW=bool(args.live_view),
        FR3_ENABLE_IK=bool(args.fr3_enable_ik),
        FR3_EE_SITE=str(args.fr3_ee_site),
        FR3_JOINT_NAMES=tuple(args.fr3_joint_names),
        FR3_IK_KP=float(args.fr3_ik_kp),
        FR3_IK_DAMPING=float(args.fr3_ik_damping),
        FR3_IK_DQ_MAX=float(args.fr3_ik_dq_max),
        FR3_EE_VMAX=float(args.fr3_ee_vmax),
        FR3_PHYSICAL_TOOL=bool(args.fr3_physical_tool),
        SUPPORT_Z=args.support_z,
        SUPPORT_GEOM_NAME=str(args.support_geom),
    )

    maybe_save_output(args, out)
    maybe_playback(args, out)
    return out


def main(argv: Sequence[str] | None = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
