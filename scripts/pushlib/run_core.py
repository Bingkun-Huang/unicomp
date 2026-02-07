"""pushlib.run_core

Entry-point glue:
- Parse CLI args
- Call state_machine.run_push_minimal(...)
- Optional: save NPZ, playback view

The heavy lifting is in:
- pushlib.state_machine
- pushlib.io_and_logging
"""

from __future__ import annotations

import argparse
from typing import Any, List, Optional

from .state_machine import run_push_minimal, DEFAULT_WAYPOINTS, XML_PATH as DEFAULT_XML_PATH, BODY_NAME as DEFAULT_BODY_NAME, STEPS as DEFAULT_STEPS
from .io_and_logging import save_npz
from .viz_playback import visualize_push_waypoints_mujoco


def main(
    argv: Optional[List[str]] = None,
    *,
    default_waypoints: Optional[Any] = None,
    default_xml: Optional[str] = None,
    default_body: Optional[str] = None,
):
    """Run the waypoint pushing demo.

    Args:
        argv: CLI argv (without program name). If None, uses sys.argv[1:].
        default_waypoints: Optional override for WAYPOINTS (list of dicts with keys: idx, p, q).
        default_xml: Optional override for MuJoCo MJCF xml_path.
        default_body: Optional override for the body name in the MJCF.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        type=str,
        default=(default_xml if default_xml is not None else DEFAULT_XML_PATH),
        help="MuJoCo XML (MJCF) used for visualization + geometry queries.",
    )
    parser.add_argument(
        "--body",
        type=str,
        default=(default_body if default_body is not None else DEFAULT_BODY_NAME),
        help="Body name of the pushed object in the MuJoCo model.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Number of simulation steps.")
    parser.add_argument(
        "--sim",
        type=str,
        default="compsim",
        choices=["compsim"],
        help="Simulator backend (currently only 'compsim').",
    )
    parser.add_argument("--live_view", action="store_true", help="Realtime MuJoCo passive viewer.")
    parser.add_argument("--view", action="store_true", help="Playback viewer after run.")
    parser.add_argument("--save_npz", action="store_true", help="Save run to NPZ.")
    parser.add_argument("--out_npz", type=str, default=None, help="Output NPZ path.")
    parser.add_argument("--render_fps", type=float, default=None, help="Playback render FPS.")
    parser.add_argument("--playback_speed", type=float, default=None, help="Playback speed multiplier.")

    args = parser.parse_args(argv)

    sim_name = str(getattr(args, "sim", "compsim"))

    WAYPOINTS = default_waypoints if default_waypoints is not None else DEFAULT_WAYPOINTS

    # ------------------------------------------------------------------
    # IMPORTANT: initialize simulator-side cached geometry/state
    # ------------------------------------------------------------------
    # In the original monolithic script, XML/body initialization happened
    # in the top-level entrypoint before the state-machine ran. After the
    # refactor, we must do it here so that compsim exposes mass/inertia and
    # precomputed contact geometry.
    if sim_name == "compsim":
        try:
            import compsim as _compsim
            if hasattr(_compsim, "init_from_xml"):
                _compsim.init_from_xml(args.xml, body=args.body)
            elif hasattr(_compsim, "init"):
                # fallback for older API
                _compsim.init(args.xml, body=args.body)
        except ImportError:
            raise RuntimeError(
                "Simulator 'compsim' selected but cannot import compsim. "
                "Check PYTHONPATH / package installation."
            )


    # Call core
    out = run_push_minimal(
        WAYPOINTS=WAYPOINTS,
        XML_PATH=(args.xml if args.xml is not None else DEFAULT_XML_PATH),
        BODY_NAME=(args.body if args.body is not None else DEFAULT_BODY_NAME),
        STEPS=(args.steps if args.steps is not None else DEFAULT_STEPS),
        LIVE_VIEW=args.live_view,
    )

    # Optional save
    if args.save_npz:
        out_path = args.out_npz or "push_waypoints_compsim_live.npz"
        save_npz(out_path, out)
        print(f"[io] saved NPZ -> {out_path}")

    # Optional playback
    if args.view:
        xml_path = args.xml or out.get("xml_path", None)
        body_name = args.body or out.get("body_name", "T_siconos")
        render_fps = args.render_fps if args.render_fps is not None else float(out.get("render_fps", 60.0))
        playback_speed = args.playback_speed if args.playback_speed is not None else float(out.get("playback_speed", 1.0))

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
            xml_path=xml_path,
            body_name=body_name,
            render_fps=render_fps,
            playback_speed=playback_speed,
        )

    return out


if __name__ == "__main__":
    main()
