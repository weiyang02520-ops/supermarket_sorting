#!/usr/bin/env python3
"""P5 preview: kele grasp then optional table place.

Does not rewrite client_task_1.py. Speeds go through RosRobotController.
YOLO stays on CPU by default so Server GS keeps the 4060.
Control ticks do not wait for inference.
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
    LatestFrameWorker,
    ObjectHoldState as GraspHold,
    PerceptionNeed,
    bbox_depth_m,
    cam_to_world,
    footprint_to_world,
    slot_head_hint,
    vision_to_object_center,
)
from runtime.p5_params import (
    CARRY_ANG,
    CARRY_LIN,
    DEFAULT_KIND,
    DEFAULT_WEIGHTS,
    GRASP_TIMEOUT_S,
    KELE_RECIPE,
    LOG_DIR,
    MAX_ANG,
    MAX_LIN,
    NAV_TIMEOUT_S,
    PLACE_TIMEOUT_S,
    RATE_HZ,
    RGB_DEPTH_SKEW_MAX,
    recipe_for,
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


def _products_and_world(detector, kind, rgb, depth, intrinsics, t_cam_world, recipe=KELE_RECIPE):
    from perception.aruco_slots import ProductDetection
    from runtime.p5_grasp import DetectionSample

    bgr = np.ascontiguousarray(rgb[..., ::-1])
    products = []
    worlds = []
    samples = []
    if detector is None or intrinsics is None or t_cam_world is None:
        return products, worlds, samples
    cam_xy = (float(t_cam_world[0, 3]), float(t_cam_world[1, 3]))
    rgb_stamp = 0.0
    for item in detector.detect(bgr):
        if item["class"] != kind:
            continue
        u, v = int(item["x"]), int(item["y"])
        depth_m, pu, pv = bbox_depth_m(depth, u, v, int(item.get("w", 0)), int(item.get("h", 0)))
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
            p_cam = intrinsics.project_pixel(pu, pv, depth_m)
        except Exception:
            continue
        surface = cam_to_world(p_cam, t_cam_world)
        worlds.append(surface)
        samples.append(DetectionSample(
            kind=item["class"],
            stamp=rgb_stamp,
            frame_id=0,
            p_surface=surface,
            p_center=vision_to_object_center(surface, (0.0, 0.0), 0.0, recipe, camera_xy=cam_xy),
            depth=depth_m,
            u=float(pu),
            v=float(pv),
            w=float(item["w"]),
            h=float(item["h"]),
            conf=float(item["conf"]),
        ))
    return products, worlds, samples


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
    if cmd.hard_stop:
        controller.stop_base()
    else:
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


def _make_obs(t, sensors, worlds, backend: MMK2KdlBackend, detections=(), camera_xy=None) -> GraspObservation:
    values = _joint_map(sensors)
    x, y, yaw = pose_from_odom(sensors.odom)
    right = np.array([values[f"right_arm_joint{i}"] for i in range(1, 7)])
    left = np.array([values[f"left_arm_joint{i}"] for i in range(1, 7)])
    slide = values["slide_joint"]
    ee_world, ee_rot = _ee_pose(backend, slide, right, (x, y), yaw)
    lin = ang = 0.0
    if sensors.last_pose_sample is not None:
        lin = sensors.last_pose_sample.linear_speed
        ang = sensors.last_pose_sample.angular_speed
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
        detections=list(detections),
        ee_world=ee_world,
        ee_rot_footprint=ee_rot,
        rgb_frame_id=int(sensors.rgb_id),
        rgb_stamp=sensors.rgb_stamp,
        camera_xy=camera_xy,
        linear_speed=lin,
        angular_speed=ang,
        t_sim=sensors.clock_sim(),
        t_wall=time.monotonic(),
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
        "carry_hold": bool(report.get("carry_hold")),
        "max_lin": report.get("max_lin"),
    }


def _hold_from_session(session: GraspSession) -> dict:
    return {
        "slide": float(session.action[0]),
        "head_yaw": float(session.action[1]),
        "head_pitch": float(session.action[2]),
        "left_arm": session.action[3:9].copy(),
        "left_grip": float(session.action[9]),
        "right_arm": session.action[10:16].copy(),
        "right_grip": float(session.action[16]),
    }


def _session_clock(sensors: SensorCache, wall0: float, sim0) -> float:
    sim = sensors.clock_sim()
    if sim is not None and sim0 is not None:
        return float(sim) - float(sim0)
    return time.monotonic() - wall0


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
    wall0 = time.monotonic()
    sim0 = sensors.clock_sim()
    last_log = 0.0
    cmd = None
    last_tick_wall = wall0
    tick_gaps = []
    last_rgb_id = None
    recipe = session.recipe
    infer_times = []
    frames_seen = 0
    yolo_hits = 0
    depth_hits = 0
    last_yolo_n = 0
    last_depth_n = 0
    last_result_id = None
    last_rgb = None

    def _infer(payload):
        rgb, depth, intrinsics, t_cam = payload
        return _products_and_world(detector, kind, rgb, depth, intrinsics, t_cam, recipe)

    worker = LatestFrameWorker(_infer)
    try:
        while time.monotonic() - wall0 < timeout_sec:
            rclpy_mod.spin_once(node, timeout_sec=period)
            if sensors.odom is None or len(sensors.joint_names) < 17:
                continue
            now_wall = time.monotonic()
            tick_gaps.append(now_wall - last_tick_wall)
            last_tick_wall = now_wall
            values = _joint_map(sensors)
            t_cam = None
            worlds = []
            detections = []
            camera_xy = None
            need = session.perception_need()
            if (
                need != PerceptionNeed.NONE
                and detector is not None
                and sensors.rgbd_ready(max_skew=RGB_DEPTH_SKEW_MAX)
            ):
                t_cam = _camera_world(
                    fk, sensors.odom, values["slide_joint"],
                    (values["head_yaw_joint"], values["head_pitch_joint"]),
                )
                camera_xy = (float(t_cam[0, 3]), float(t_cam[1, 3]))
                if last_rgb_id != sensors.rgb_id:
                    last_rgb = sensors.rgb.copy()
                    worker.submit(
                        int(sensors.rgb_id),
                        (last_rgb, sensors.depth.copy(), sensors.intrinsics, t_cam.copy()),
                    )
                    last_rgb_id = sensors.rgb_id
                got = worker.latest()
                if got is not None:
                    frame_id, payload = got
                    products, worlds, detections = payload
                    last_yolo_n = len(products)
                    last_depth_n = len(detections)
                    if frame_id != last_result_id:
                        last_result_id = frame_id
                        if products:
                            yolo_hits += 1
                        if detections:
                            depth_hits += 1
                    for sample in detections:
                        sample.frame_id = int(frame_id)
                        sample.stamp = float(sensors.rgb_stamp or 0.0)
                    infer_times = list(worker.infer_times)
                    frames_seen = worker.frames
                    if products:
                        hint = _slot_hint(sensors.rgb, products, kind)
                        if hint and hint != session.slot_hint:
                            session.note_slot_hint(hint)
                            print(f"p5: slot hint {hint} (look only, not grasp xyz)", flush=True)
            obs = _make_obs(
                _session_clock(sensors, wall0, sim0), sensors, worlds, backend,
                detections=detections, camera_xy=camera_xy,
            )
            cmd = session.tick(obs)
            _publish(controller, cmd)
            if time.monotonic() - last_log >= 1.0:
                x, y, yaw = obs.base_xy[0], obs.base_xy[1], obs.base_yaw
                print(
                    f"p5: phase={cmd.phase.value} xy=({x:.2f},{y:.2f}) yaw={yaw:.2f} "
                    f"lin={cmd.linear:.2f} grip={cmd.right_grip:.2f} "
                    f"yolo={last_yolo_n} depth={last_depth_n} "
                    f"head=({obs.head_yaw:.2f},{obs.head_pitch:.2f}) "
                    f"need={cmd.perception} hold={cmd.hold_state} {cmd.log}",
                    flush=True,
                )
                last_log = time.monotonic()
            if cmd.done or cmd.failed or cmd.phase == GraspPhase.PAUSED:
                controller.stop_base()
                break
        else:
            controller.stop_all()
            if session.phase not in (GraspPhase.HELD, GraspPhase.DONE, GraspPhase.FAILED, GraspPhase.PAUSED):
                session.fail("wall_timeout")
    finally:
        worker.close()
    if session.phase == GraspPhase.FAILED and session.reason == "lock_timeout" and last_rgb is not None:
        try:
            import cv2
            debug_path = LOG_DIR / "lock_timeout_rgb.jpg"
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_path), np.ascontiguousarray(last_rgb[..., ::-1]))
            print(f"p5: saved {debug_path} (head look at lock_timeout)", flush=True)
        except Exception:
            debug_path = None
    else:
        debug_path = None
    extras = session.report_extras()
    x = y = yaw = 0.0
    if sensors.odom is not None:
        x, y, yaw = pose_from_odom(sensors.odom)
    grasped = session.phase == GraspPhase.HELD and session.hold_state == GraspHold.HELD
    sim_now = sensors.clock_sim()
    extras.update({
        "elapsed_s": round(time.monotonic() - wall0, 2),
        "elapsed_wall_s": round(time.monotonic() - wall0, 2),
        "elapsed_sim_s": None if sim0 is None or sim_now is None else round(float(sim_now) - float(sim0), 2),
        "odom_xy": [round(x, 3), round(y, 3)],
        "yaw": round(yaw, 3),
        "in_picking_zone": bool(in_picking_zone((x, y))),
        "in_delivery_base": bool(in_delivery_base((x, y))),
        "grasped": grasped,
        "placed": session.phase == GraspPhase.DONE and bool(session.place_verified_local),
        "failed": session.phase == GraspPhase.FAILED,
        "paused": session.phase == GraspPhase.PAUSED,
        "look_hint": extras.get("head_cmd") or list(slot_head_hint(session.slot_hint)),
        "infer_s_mean": None if not infer_times else round(float(np.mean(infer_times)), 3),
        "infer_frames": int(frames_seen),
        "yolo_hits": int(yolo_hits),
        "depth_hits": int(depth_hits),
        "last_yolo_n": int(last_yolo_n),
        "last_depth_n": int(last_depth_n),
        "lock_timeout_rgb": None if debug_path is None else str(debug_path),
        "control_dt_max": None if not tick_gaps else round(max(tick_gaps), 3),
        "control_dt_mean": None if not tick_gaps else round(float(np.mean(tick_gaps)), 3),
        "snapshot": sensors.snapshot_meta(),
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
        "place_sequence_done": False,
        "place_verified_local": False,
        "referee_confirmed": None,
        "note": "placed/place_verified_local are client evidence, not Server referee",
    }
    orch = None
    planner = None
    backend = MMK2KdlBackend()
    fk = MMK2FK()
    detector = None
    recipe = recipe_for(kind)
    if recipe is None or not recipe.enabled:
        report["failure_reason"] = f"template_unavailable:{kind}"
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (LOG_DIR / "last.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
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
        planner = shelf.pop("_planner", None)
        report["nav_shelf"] = _nav_subset(shelf)
        if not shelf.get("arrived"):
            report["failure_reason"] = shelf.get("failure_reason") or "shelf_nav_failed"
            json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            (LOG_DIR / "last.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return report

    node, sensors, controller = _open_node(rclpy_mod, "supermarket_p5_grasp")
    hold_joints = None
    try:
        _wait_ready(rclpy_mod, node, sensors, timeout_sec=12.0)
        controller.command_spine(AISLE_SCAN_SPINE)
        controller.command_head(AISLE_SCAN_HEAD)
        settle = time.monotonic() + 0.4
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
            recipe = recipe_for(kind)
            if recipe is None or not recipe.enabled:
                report["failure_reason"] = f"template_unavailable:{kind}"
                orch.complete_pick(False, report["failure_reason"])
                return report
        session = GraspSession(ik_fn=_ik_fn(backend), kind=kind, now=0.0, recipe=recipe)
        grasp = _run_session(
            rclpy_mod, session, sensors, controller, node, fk, backend,
            detector, kind, grasp_timeout,
        )
        report.update({k: grasp[k] for k in grasp})
        hold_joints = _hold_from_session(session)
        if orch is not None:
            if grasp.get("grasped"):
                orch.complete_pick(True, grasp.get("reason"))
            elif grasp.get("paused") or session.hold_state == GraspHold.UNKNOWN:
                orch.mark_hold_unknown(grasp.get("reason") or "c2_unknown")
            else:
                orch.complete_pick(False, grasp.get("reason"))
        if grasp_only or not grasp.get("grasped") or grasp.get("failed"):
            if not grasp.get("grasped"):
                report["failure_reason"] = grasp.get("reason") or "grasp_failed"
            return report
    finally:
        controller.stop_base()
        node.destroy_node()

    delivery = navigate(
        rclpy_mod,
        goal="delivery",
        timeout_sec=nav_timeout,
        max_lin=CARRY_LIN,
        max_ang=CARRY_ANG,
        hold_joints=hold_joints,
        planner=planner,
    )
    planner = delivery.pop("_planner", planner)
    report["nav_delivery"] = _nav_subset(delivery)
    if not delivery.get("arrived"):
        report["failure_reason"] = delivery.get("failure_reason") or "delivery_nav_failed"
        if orch is not None:
            orch.mark_delivery_blocked(report["failure_reason"])
        return report

    node, sensors, controller = _open_node(rclpy_mod, "supermarket_p5_place")
    try:
        _wait_ready(rclpy_mod, node, sensors, timeout_sec=12.0)
        occupied = report.get("occupied_slots") or ()
        session.begin_place(now=0.0, table_xyz=DELIVERY_TARGET_XYZ, occupied=occupied)
        placed = _run_session(
            rclpy_mod, session, sensors, controller, node, fk, backend,
            detector, kind, place_timeout,
        )
        report["placed"] = bool(placed.get("placed"))
        report["phase"] = placed.get("phase")
        report["place_reason"] = placed.get("reason")
        report["place_odom_xy"] = placed.get("odom_xy")
        report["place_target_in_box"] = placed.get("place_target_in_box")
        report["place_sequence_done"] = bool(placed.get("place_sequence_done"))
        report["place_verified_local"] = bool(placed.get("place_verified_local"))
        report["referee_confirmed"] = placed.get("referee_confirmed")
        report["c3"] = bool(placed.get("c3"))
        if orch is not None:
            if placed.get("placed"):
                orch.complete_deliver(True, placed.get("reason"))
            else:
                orch.note_place_unverified(placed.get("reason") or "place_unverified")
            report["orchestrator"] = orch.state.value
            report["hold_state"] = orch.hold_state.value
        if not placed.get("placed"):
            report["failure_reason"] = placed.get("reason") or "place_unverified"
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
        text = json.dumps({k: v for k, v in report.items() if not str(k).startswith("_")}, indent=2, default=str)
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
