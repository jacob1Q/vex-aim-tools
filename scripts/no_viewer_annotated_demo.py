#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2

from aim_fsm import Robot
from aim_fsm import evbase, program
from aim_fsm.program import StateMachineProgram


def _ensure_repo_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _pose_to_dict(pose) -> dict | None:
    if pose is None:
        return None
    for attr in ("x", "y", "theta"):
        if not hasattr(pose, attr):
            return None
    try:
        return {
            "x": float(pose.x),
            "y": float(pose.y),
            "theta": float(pose.theta),
        }
    except Exception:
        return None


def _normalize_meta(meta: dict, image_shape: tuple[int, int, int]) -> dict:
    clean = dict(meta)
    clean["pose"] = _pose_to_dict(meta.get("pose"))
    clean["image_shape"] = tuple(int(v) for v in image_shape)
    return clean


def main() -> int:
    _ensure_repo_on_path()

    parser = argparse.ArgumentParser(
        description="Headless annotated frame callback demo (no viewer).",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("ROBOT", "192.168.4.1"),
        help="Robot IP address (default: env ROBOT or 192.168.4.1).",
    )
    parser.add_argument(
        "--out-dir",
        default="frames",
        help="Directory to save annotated frames.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Save every Nth frame (default: 10).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames (0 means run forever).",
    )
    parser.add_argument(
        "--save-meta",
        action="store_true",
        help="Write per-frame metadata to frames_meta.jsonl.",
    )
    parser.add_argument(
        "--no-aruco",
        action="store_true",
        help="Disable ArUco detection and annotation.",
    )
    parser.add_argument(
        "--no-sdk",
        action="store_true",
        help="Disable SDK object annotation.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "frames_meta.jsonl")
    meta_lock = threading.Lock()
    meta_fh = open(meta_path, "a", encoding="utf-8") if args.save_meta else None

    stop_event = threading.Event()

    def on_frame(image_rgb, meta):
        frame = meta.get("frame_count")
        if frame is None:
            return
        if args.stride > 1 and frame % args.stride != 0:
            return
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        filename = os.path.join(args.out_dir, f"frame_{frame:06d}.png")
        cv2.imwrite(filename, bgr)
        if meta_fh is not None:
            record = _normalize_meta(meta, image_rgb.shape)
            line = json.dumps(record, default=str)
            with meta_lock:
                meta_fh.write(line + "\n")
                meta_fh.flush()
        if args.max_frames > 0 and frame >= args.max_frames:
            stop_event.set()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def loopthread():
        loop.run_forever()

    th = threading.Thread(target=loopthread, daemon=True)
    th.start()

    robot = Robot(loop=loop, host=args.host, launch_speech_listener=False)
    evbase.robot_for_loading = robot
    program.robot_for_loading = robot

    fsm = StateMachineProgram(
        launch_cam_viewer=False,
        launch_worldmap_viewer=False,
        launch_particle_viewer=False,
        launch_path_viewer=False,
        speech=False,
        annotated_image_callback=on_frame,
        aruco=not args.no_aruco,
        annotate_sdk=not args.no_sdk,
    )

    loop.call_soon_threadsafe(fsm.start)

    print("Headless annotated callback demo running.")
    print(f"  Saving annotated frames to: {args.out_dir}")
    if args.save_meta:
        print(f"  Saving metadata to: {meta_path}")
    if args.max_frames > 0:
        print(f"  Will stop after frame: {args.max_frames}")
    print("Press Ctrl+C to stop.")

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    if meta_fh is not None:
        meta_fh.close()
    try:
        loop.call_soon_threadsafe(fsm.stop)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
