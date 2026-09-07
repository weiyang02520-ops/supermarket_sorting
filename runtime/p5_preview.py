#!/usr/bin/env python3
"""P5 preview: kele grasp (official sequence) then optional table place.

Does not rewrite client_task_1.py. Speeds go through RosRobotController.
YOLO stays on CPU by default so Server GS keeps the 4060.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from runtime.head_camera_kinematics import AISLE_SCAN_HEAD, AISLE_SCAN_SPINE
from runtime.motion_planning import IKSolveError, MMK2KdlBackend
from runtime.orchestrator import MissionOrchestrator
from runtime.p4_nav import navigate
from runtime.p5_grasp import (
    GraspObservation,
    GraspPhase,
    GraspSession,
    cam_to_world,
    footprint_to_world,
    patch_depth_m,
    slot_head_hint,
)
from runtime.p5_params import (
    DEFAULT_KIND,
    DEFAULT_WEIGHTS,
    GRASP_TIMEOUT_S,
    LOG_DIR,
    MAX_ANG,
    MAX_LIN,
    NAV_TIMEOUT_S,
    PLACE_TIMEOUT_S,
    RATE_HZ,
)
from runtime.ros_contract import (
    DEPTH_TOPIC,
    JOINT_STATES_TOPIC,
    ODOM_TOPIC,
    RGB_CAMERA_INFO_TOPIC,
    RGB_TOPIC,
    TASK_TOPIC,
)
from runtime.ros_robot_control import RosRobotController
from runtime.ros_sensor_utils import SensorCache, quaternion_matrix
from runtime.scene_zones import DELIVERY_TARGET_XYZ, in_delivery_base, in_picking_zone
from runtime.waypoint_nav import pose_from_odom


def _joint_cb(sensors: SensorCache):
    def wrapped(message):
        try:
            sensors.update_joint_state(message)
        except Exception:
            pass
    return wrapped


def _image_cb(update):
    def wrapped(message):
        try:
            update(message)
        except Exception:
            pass
    return wrapped


def _joint_map(sensors: SensorCache) -> dict:
    return dict(zip(sensors.joint_names, [float(value) for value in sensors.joint_positions]))


def _camera_world(fk, odom, slide: float, head) -> np.ndarray:
    position = odom.pose.pose.position
    quat = odom.pose.pose.orientation
    fk.set_base_pose([position.x, position.y, position.z], [quat.w, quat.x, quat.y, quat.z])
    fk.set_slide_joint(float(slide))
    fk.set_head_joints([float(head[0]), float(head[1])])
    cam_pos, cam_wxyz = fk.get_head_camera_pose()
    matrix = np.eye(4)
    matrix[:3, 3] = cam_pos
    matrix[:3, :3] = quaternion_matrix(cam_wxyz[1], cam_wxyz[2], cam_wxyz[3], cam_wxyz[0])
    return matrix


def _products_and_world(detector, kind, rgb, depth, intrinsics, t_cam_world):
    from perception.aruco_slots import ProductDetection

    bgr = np.ascontiguousarray(rgb[..., ::-1])
    products = []
    worlds = []
    if detector is None or intrinsics is None or t_cam_world is None:
        return products, worlds
    for item in detector.detect(bgr):
        if item["class"] != kind:
            continue
        u, v = int(item["x"]), int(item["y"])
        depth_m = patch_depth_m(depth, u, v)
        products.append(ProductDetection(
            kind=item["class"],
            u=float(u),
            v=float(v),
            conf=float(item["conf"]),
            w=float(item["w"]),
            h=float(item["h"]),
        ))
        if depth_m <= 0.0:
            continue
        try:
            p_cam = intrinsics.project_pixel(u, v, depth_m)
        except Exception:
            continue
        worlds.append(cam_to_world(p_cam, t_cam_world))
    return products, worlds


def _slot_hint(rgb, products, kind: str):
    from perception.aruco_slots import (
        associate_products_to_slots,
        dedupe_markers_by_pixel,
        detect_aruco_markers,
        slot_from_marker_id,
    )

    bgr = np.ascontiguousarray(rgb[..., ::-1])
    markers = dedupe_markers_by_pixel(detect_aruco_markers(bgr))
    for item in associate_products_to_slots(products, markers):
        if item.get("kind") == kind and item.get("marker_id") is not None:
            return slot_from_marker_id(int(item["marker_id"])).name
    return None


def _publish(controller: RosRobotController, cmd) -> None:
    controller.publish_velocity(cmd.linear, cmd.angular)
    controller.command_spine(cmd.slide)
    controller.command_head((cmd.head_yaw, cmd.head_pitch))
    controller.command_arm("l", cmd.left_arm, cmd.left_grip)
    controller.command_arm("r", cmd.right_arm, cmd.right_grip)


def _ee_pose(backend: MMK2KdlBackend, slide, right_arm, base_xy, yaw):
    try:
        transform = backend.forward("r", float(slide), right_arm)
    except (ValueError, IKSolveError):
        return None, None
    return footprint_to_world(transform[:3, 3], base_xy, yaw), transform[:3, :3]


def _make_obs(t, sensors, worlds, backend: MMK2KdlBackend) -> GraspObservation:
    values = _joint_map(sensors)
    x, y, yaw = pose_from_odom(sensors.odom)
    right = np.array([values[f"right_arm_joint{i}"] for i in range(1, 7)])
    left = np.array([values[f"left_arm_joint{i}"] for i in range(1, 7)])
    slide = values["slide_joint"]
    ee_world, ee_rot = _ee_pose(backend, slide, right, (x, y), yaw)
    return GraspObservation(
        t=t,
        base_xy=(x, y),
        base_yaw=yaw,
        slide=slide,
        head_yaw=values["head_yaw_joint"],
        head_pitch=values["head_pitch_joint"],
        left_arm=left,
        left_grip=values["left_arm_eef_gripper_joint"],
        right_arm=right,
        right_grip=values["right_arm_eef_gripper_joint"],
        detections_world=worlds,
        ee_world=ee_world,
        ee_rot_footprint=ee_rot,
    )


def _ik_fn(backend: MMK2KdlBackend):
    def solve(footprint_pos, rot, slide, q_ref):
        try:
            return backend.solve(footprint_pos, rot, "r", slide, q_ref)
        except (IKSolveError, ValueError):
            return None
    return solve


def _wait_ready(rclpy_mod, node, sensors: SensorCache, timeout_sec: float) -> None:
    deadline = time.monotonic() + float(timeout_sec)
    while time.monotonic() < deadline:
        rclpy_mod.spin_once(node, timeout_sec=0.05)
        joints_ok = len(sensors.joint_names) >= 17
        if sensors.odom is not None and joints_ok and sensors.rgb is not None:
            return
    raise TimeoutError("timed out waiting for odom/joints/rgb")


def _open_node(rclpy_mod, name: str):
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from std_msgs.msg import Float64MultiArray, String
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    node = rclpy_mod.create_node(name)
    sensors = SensorCache()
    controller = RosRobotController(
        node, Twist, Float64MultiArray, sensors, max_linear=MAX_LIN, max_angular=MAX_ANG,
    )
    latch = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    node.create_subscription(Odometry, ODOM_TOPIC, sensors.update_odom, 10)
    node.create_subscription(JointState, JOINT_STATES_TOPIC, _joint_cb(sensors), 10)
    node.create_subscription(Image, RGB_TOPIC, _image_cb(sensors.update_rgb), 10)
    node.create_subscription(Image, DEPTH_TOPIC, _image_cb(sensors.update_depth), 10)
    node.create_subscription(CameraInfo, RGB_CAMERA_INFO_TOPIC, sensors.update_camera_info, 10)
    node.create_subscription(String, TASK_TOPIC, sensors.update_task, latch)
    return node, sensors, controller


def _nav_subset(report: dict) -> dict:
    return {
        "arrived": bool(report.get("arrived")),
        "in_zone": bool(report.get("in_zone")),
        "odom_xy": report.get("odom_xy"),
        "yaw": report.get("yaw"),
        "elapsed_s": report.get("elapsed_s"),
        "failure_reason": report.get("failure_reason") or "",
        "status": report.get("status"),
    }


def _run_session(
    rclpy_mod,
    session: GraspSession,
    sensors: SensorCache,
    controller: RosRobotController,
    node,
    fk,
    backend: MMK2KdlBackend,
    detector,
    kind: str,
    timeout_sec: float,
) -> dict:
    period = 1.0 / RATE_HZ
    start = time.monotonic()
    last_log = 0.0
    cmd = None
    hint_tried = False
    while time.monotonic() - start < timeout_sec:
        rclpy_mod.spin_once(node, timeout_sec=period)
        if sensors.odom is None or len(sensors.joint_names) < 17:
            continue
        values = _joint_map(sensors)
        t_cam = None
        worlds = []
        products = []
        if sensors.rgb is not None and sensors.depth is not None and sensors.intrinsics is not None:
            t_cam = _camera_world(
                fk, sensors.odom, values["slide_joint"],
                (values["head_yaw_joint"], values["head_pitch_joint"]),
            )
            products, worlds = _products_and_world(
                detector, kind, sensors.rgb, sensors.depth, sensors.intrinsics, t_cam,
            )
            if not hint_tried:
                hint = _slot_hint(sensors.rgb, products, kind)
                session.note_slot_hint(hint)
                if hint:
                    print(f"p5: slot hint {hint} (look only, not grasp xyz)", flush=True)
                hint_tried = True
        obs = _make_obs(time.monotonic() - start, sensors, worlds, backend)
        cmd = session.tick(obs)
        _publish(controller, cmd)
        if time.monotonic() - last_log >= 1.0:
            x, y, yaw = obs.base_xy[0], obs.base_xy[1], obs.base_yaw
            print(
                f"p5: phase={cmd.phase.value} xy=({x:.2f},{y:.2f}) yaw={yaw:.2f} "
                f"lin={cmd.linear:.2f} grip={cmd.right_grip:.2f} dets={len(worlds)} "
                f"{cmd.log}",
                flush=True,
            )
            last_log = time.monotonic()
        if cmd.done or cmd.failed:
            controller.stop_base()
            break
    else:
        controller.stop_all()
        if session.phase not in (GraspPhase.HELD, GraspPhase.DONE, GraspPhase.FAILED):
            session.fail("wall_timeout")
    extras = session.report_extras()
    x = y = yaw = 0.0
    if sensors.odom is not None:
        x, y, yaw = pose_from_odom(sensors.odom)
    grasped = session.phase in (GraspPhase.HELD, GraspPhase.DONE)
    extras.update({
        "elapsed_s": round(time.monotonic() - start, 2),
        "odom_xy": [round(x, 3), round(y, 3)],
        "yaw": round(yaw, 3),
        "in_picking_zone": bool(in_picking_zone((x, y))),
        "in_delivery_base": bool(in_delivery_base((x, y))),
        "grasped": grasped,
        "placed": session.phase == GraspPhase.DONE,
        "failed": session.phase == GraspPhase.FAILED,
        "look_hint": list(slot_head_hint(session.slot_hint)),
    })
    return extras


def run_p5(
    rclpy_mod,
    weights: Path,
    device: str,
    kind: str,
    skip_nav: bool,
    grasp_only: bool,
    nav_timeout: float,
    grasp_timeout: float,
    place_timeout: float,
) -> dict:
    from kinematics.mmk2_fk import MMK2FK
    from perception.yolo_backend import YoloBackend

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = LOG_DIR / f"p5_{stamp}.json"
    report = {
        "grasped": False,
        "placed": False,
        "kind": kind,
        "used_json_xyz": False,
        "nav_shelf": None,
        "nav_delivery": None,
        "log_json": str(json_path),
        "note": "placed is the client sequence, not Server referee",
    }
    orch = None
    backend = MMK2KdlBackend()
    fk = MMK2FK()
    detector = None
    if Path(weights).is_file():
        print(f"[p5] loading YOLO on {device}", flush=True)
        detector = YoloBackend(weights, confidence=0.65, device=device)
    else:
        report["failure_reason"] = f"weights_missing:{weights}"
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (LOG_DIR / "last.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report

    if not skip_nav:
        shelf = navigate(rclpy_mod, goal="shelf", timeout_sec=nav_timeout)
        report["nav_shelf"] = _nav_subset(shelf)
        if not shelf.get("arrived"):
            report["failure_reason"] = shelf.get("failure_reason") or "shelf_nav_failed"
            json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            (LOG_DIR / "last.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return report

    node, sensors, controller = _open_node(rclpy_mod, "supermarket_p5_grasp")
    try:
        _wait_ready(rclpy_mod, node, sensors, timeout_sec=12.0)
        controller.command_spine(AISLE_SCAN_SPINE)
        controller.command_head(AISLE_SCAN_HEAD)
        settle = time.monotonic() + 0.8
        while time.monotonic() < settle:
            controller.command_spine(AISLE_SCAN_SPINE)
            controller.command_head(AISLE_SCAN_HEAD)
            rclpy_mod.spin_once(node, timeout_sec=0.05)
        if sensors.task_raw:
            orch = MissionOrchestrator()
            orch.load_task(sensors.task_raw)
            target = orch.start_pick()
            kind = target.kind
            report["kind"] = kind
            report["target_id"] = target.id
        if kind != DEFAULT_KIND:
            report["failure_reason"] = f"p5 preview only supports {DEFAULT_KIND}.pt (got {kind})"
            if orch is not None:
                orch.complete_pick(False, report["failure_reason"])
            return report
        session = GraspSession(ik_fn=_ik_fn(backend), kind=kind, now=0.0)
        grasp = _run_session(
            rclpy_mod, session, sensors, controller, node, fk, backend,
            detector, kind, grasp_timeout,
        )
        report.update({k: grasp[k] for k in grasp})
        if orch is not None:
            orch.complete_pick(bool(grasp.get("grasped") and not grasp.get("failed")), grasp.get("reason"))
        if grasp_only or not grasp.get("grasped") or grasp.get("failed"):
            if not grasp.get("grasped"):
                report["failure_reason"] = grasp.get("reason") or "grasp_failed"
            return report
    finally:
        controller.stop_base()
        node.destroy_node()

    delivery = navigate(rclpy_mod, goal="delivery", timeout_sec=nav_timeout)
    report["nav_delivery"] = _nav_subset(delivery)
    if not delivery.get("arrived"):
        report["failure_reason"] = delivery.get("failure_reason") or "delivery_nav_failed"
        if orch is not None:
            orch.complete_deliver(False, report["failure_reason"])
        return report

    node, sensors, controller = _open_node(rclpy_mod, "supermarket_p5_place")
    try:
        _wait_ready(rclpy_mod, node, sensors, timeout_sec=12.0)
        session.begin_place(now=0.0, table_xyz=DELIVERY_TARGET_XYZ)
        placed = _run_session(
            rclpy_mod, session, sensors, controller, node, fk, backend,
            detector, kind, place_timeout,
        )
        report["placed"] = bool(placed.get("placed"))
        report["phase"] = placed.get("phase")
        report["place_reason"] = placed.get("reason")
        report["place_odom_xy"] = placed.get("odom_xy")
        report["place_target_in_box"] = placed.get("place_target_in_box")
        if orch is not None:
            orch.complete_deliver(bool(placed.get("placed")), placed.get("reason"))
            report["orchestrator"] = orch.state.value
        if not placed.get("placed"):
            report["failure_reason"] = placed.get("reason") or "place_failed"
    finally:
        controller.stop_all()
        node.destroy_node()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P5 kele grasp/place preview")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--kind", default=DEFAULT_KIND)
    parser.add_argument("--skip-nav", action="store_true", help="already at the shelf approach")
    parser.add_argument("--grasp-only", action="store_true", help="stop after retreat, keep the grasp")
    parser.add_argument("--timeout-nav", type=float, default=NAV_TIMEOUT_S)
    parser.add_argument("--timeout-grasp", type=float, default=GRASP_TIMEOUT_S)
    parser.add_argument("--timeout-place", type=float, default=PLACE_TIMEOUT_S)
    args = parser.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[p5] CUDA_VISIBLE_DEVICES= (hidden; Server GS keeps the 4060)", flush=True)

    import rclpy

    rclpy.init()
    report = None
    try:
        report = run_p5(
            rclpy,
            weights=args.weights,
            device=args.device,
            kind=args.kind,
            skip_nav=args.skip_nav,
            grasp_only=args.grasp_only,
            nav_timeout=args.timeout_nav,
            grasp_timeout=args.timeout_grasp,
            place_timeout=args.timeout_place,
        )
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        text = json.dumps(report, indent=2)
        json_path = Path(report.get("log_json") or (LOG_DIR / "last.json"))
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(text + "\n", encoding="utf-8")
        (LOG_DIR / "last.json").write_text(text + "\n", encoding="utf-8")
        print(text)
        if args.grasp_only:
            return 0 if report.get("grasped") else 1
        return 0 if report.get("grasped") and report.get("placed") else 1
    except TimeoutError as exc:
        payload = {"grasped": False, "placed": False, "error": str(exc), "used_json_xyz": False}
        print(json.dumps(payload, indent=2))
        return 1
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
