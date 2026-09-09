"""P5 grasp/place state machine. No ROS import.

Cola first: YOLO+depth lock, calibrated template, C1/C2/C3 evidence.
Internal close/lift/retreat steps stay for control logs; extra dwells do not.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Deque, List, Optional, Sequence, Tuple

import numpy as np

from common.control import step_func
from runtime.head_camera_kinematics import (
    AISLE_SCAN_POSES,
    HEAD_PITCH_LIMITS,
    HEAD_YAW_LIMITS,
    SLIDE_LIMITS,
    base_to_head_camera,
)
from runtime.p5_params import (
    C2_LOOK_S,
    CREEP_SLOW_ALONG,
    CREEP_SLOW_SPEED,
    CREEP_YAW_KP,
    DEFAULT_KIND,
    DEPLOY_CART_TOL,
    DEPLOY_JOINT_TOL,
    DEPLOY_ROT_TOL,
    DEPLOY_SLIDE_TOL,
    DT_JUMP_S,
    GRASP_HEAD_PITCH,
    GRASP_ROT,
    GRASP_TIMEOUT_S,
    HEAD_PITCH,
    HEAD_YAW,
    INIT_ARM_L,
    INIT_ARM_R,
    JOINT_SLEW,
    LIFT_TIMEOUT_S,
    KELE_RECIPE,
    LOCK_ATTEMPTS,
    LOCK_TIMEOUT_S,
    MAX_ANG,
    MAX_ANG_ACC,
    MAX_LIN,
    MAX_LIN_ACC,
    PLACE_CART_TOL,
    PLACE_OCCUPIED_RADIUS,
    PLACE_RETRACT_FWD,
    PLACE_SLOT_XY,
    PLACE_TIMEOUT_S,
    RATE_HZ,
    SLOT_COLUMN_YAW,
    SLOT_LAYER_PITCH,
    SLOT_LAYER_Z,
    GraspRecipe,
    gripper_fits,
    recipe_for,
)
from runtime.scene_zones import DELIVERY_TARGET_XYZ, in_delivery_box
from runtime.waypoint_nav import wrap_to_pi

IkFn = Callable[[np.ndarray, np.ndarray, float, np.ndarray], Optional[np.ndarray]]


class GraspPhase(str, Enum):
    INIT = "init"
    AIM = "aim"
    DEPLOY = "deploy"
    CREEP = "creep"
    CLOSE = "close"
    LIFT = "lift"
    RETREAT = "retreat"
    HELD = "held"
    PLACE_DEPLOY = "place_deploy"
    PLACE_LOWER = "place_lower"
    PLACE_OPEN = "place_open"
    PLACE_RETRACT = "place_retract"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"


class ObjectHoldState(str, Enum):
    EMPTY = "empty"
    HELD = "held"
    UNKNOWN = "unknown"


class PerceptionNeed(str, Enum):
    NONE = "none"
    C1 = "c1"
    C2 = "c2"
    C3 = "c3"


@dataclass
class DetectionSample:
    kind: str
    stamp: float
    frame_id: int
    p_surface: np.ndarray
    p_center: np.ndarray
    depth: float = 0.0
    u: float = 0.0
    v: float = 0.0
    w: float = 0.0
    h: float = 0.0
    conf: float = 0.0
    slot_hint: Optional[str] = None


@dataclass
class LockedTarget:
    object_world: np.ndarray
    deploy_world: np.ndarray
    approach_world: np.ndarray
    base_xy: Tuple[float, float]
    base_yaw: float
    stamp: float
    frame_ids: Tuple[int, ...]
    sample_count: int
    quality: float
    instance_id: str
    recipe_kind: str
    recipe_version: str


@dataclass
class GraspObservation:
    t: float
    base_xy: Tuple[float, float]
    base_yaw: float
    slide: float
    head_yaw: float
    head_pitch: float
    left_arm: np.ndarray
    left_grip: float
    right_arm: np.ndarray
    right_grip: float
    detections_world: Sequence[np.ndarray] = ()
    detections: Sequence[DetectionSample] = ()
    ee_world: Optional[np.ndarray] = None
    ee_rot_footprint: Optional[np.ndarray] = None
    rgb_frame_id: Optional[int] = None
    rgb_stamp: Optional[float] = None
    camera_xy: Optional[Tuple[float, float]] = None
    linear_speed: float = 0.0
    angular_speed: float = 0.0
    t_sim: Optional[float] = None
    t_wall: Optional[float] = None


@dataclass
class GraspCommand:
    linear: float
    angular: float
    slide: float
    head_yaw: float
    head_pitch: float
    left_arm: np.ndarray
    left_grip: float
    right_arm: np.ndarray
    right_grip: float
    phase: GraspPhase
    done: bool = False
    failed: bool = False
    reason: str = ""
    log: str = ""
    hard_stop: bool = False
    hold_state: str = ObjectHoldState.EMPTY.value
    perception: str = PerceptionNeed.NONE.value


def clip_base(linear: float, angular: float,
              max_lin: float = MAX_LIN, max_ang: float = MAX_ANG) -> Tuple[float, float]:
    return (
        float(np.clip(linear, -max_lin, max_lin)),
        float(np.clip(angular, -max_ang, max_ang)),
    )


def world_to_footprint(p_world, base_xy, base_yaw: float) -> np.ndarray:
    point = np.asarray(p_world, dtype=float)
    delta = point - np.array([base_xy[0], base_xy[1], 0.0])
    cosine, sine = math.cos(-base_yaw), math.sin(-base_yaw)
    return np.array([
        cosine * delta[0] - sine * delta[1],
        sine * delta[0] + cosine * delta[1],
        delta[2],
    ])


def footprint_to_world(fp, base_xy, base_yaw: float) -> np.ndarray:
    point = np.asarray(fp, dtype=float)
    cosine, sine = math.cos(base_yaw), math.sin(base_yaw)
    return np.array([
        base_xy[0] + cosine * point[0] - sine * point[1],
        base_xy[1] + sine * point[0] + cosine * point[1],
        point[2],
    ])


def approach_axis_world(base_yaw: float) -> np.ndarray:
    return np.array([math.cos(base_yaw), math.sin(base_yaw), 0.0], dtype=float)


def reachable_in_footprint(p_world, base_xy, base_yaw: float, recipe: GraspRecipe = KELE_RECIPE) -> bool:
    footprint = world_to_footprint(p_world, base_xy, base_yaw)
    if footprint[0] < recipe.fwd_min or footprint[0] > recipe.fwd_max:
        return False
    if abs(footprint[1]) > recipe.lateral_max:
        return False
    height = float(np.asarray(p_world, dtype=float)[2])
    return recipe.z_min <= height <= recipe.z_max


def vision_to_object_center(
    p_world,
    base_xy,
    base_yaw: float,
    recipe: GraspRecipe = KELE_RECIPE,
    camera_xy=None,
) -> np.ndarray:
    point = np.asarray(p_world, dtype=float).copy()
    if camera_xy is not None:
        view = point[:2] - np.asarray(camera_xy[:2], dtype=float)
        norm = float(np.linalg.norm(view))
        if norm > 1e-6:
            point[:2] = point[:2] + recipe.radius * (view / norm)
    else:
        footprint = world_to_footprint(point, base_xy, base_yaw)
        footprint[0] += recipe.radius
        point = footprint_to_world(footprint, base_xy, base_yaw)
    point[2] += recipe.z_bias
    return point


def pregrasp_world(object_world, base_yaw: float, recipe: GraspRecipe = KELE_RECIPE) -> np.ndarray:
    axis = approach_axis_world(base_yaw)
    lateral = np.array([-math.sin(base_yaw), math.cos(base_yaw), 0.0], dtype=float)
    return (
        np.asarray(object_world, dtype=float)
        - recipe.pregrasp_along * axis
        + recipe.pregrasp_lateral * lateral
        + np.array([0.0, 0.0, recipe.pregrasp_z])
    )


def pick_centered_detection(
    detections_world: Sequence[np.ndarray],
    base_xy,
    base_yaw: float,
    recipe: GraspRecipe = KELE_RECIPE,
) -> Optional[np.ndarray]:
    best = None
    best_lat = recipe.lateral_max
    for point in detections_world:
        if not reachable_in_footprint(point, base_xy, base_yaw, recipe):
            continue
        footprint = world_to_footprint(point, base_xy, base_yaw)
        lateral = abs(float(footprint[1]))
        if lateral < best_lat:
            best_lat = lateral
            best = np.asarray(point, dtype=float)
    return best


def associate_detection(
    detections: Sequence[DetectionSample],
    previous: np.ndarray,
    recipe: GraspRecipe,
) -> Optional[DetectionSample]:
    best = None
    best_dist = recipe.associate_tol
    prev = np.asarray(previous, dtype=float)
    for item in detections:
        dist = min(
            float(np.linalg.norm(np.asarray(item.p_center, dtype=float) - prev)),
            float(np.linalg.norm(np.asarray(item.p_surface, dtype=float) - prev)),
        )
        if dist <= best_dist:
            best_dist = dist
            best = item
    return best


def bbox_depth_m(
    depth_m: np.ndarray,
    u: int,
    v: int,
    w: int = 0,
    h: int = 0,
    patch: int = 9,
) -> Tuple[float, int, int]:
    """Median depth in a small block inside the bbox. Nearby blocks if the center is bad."""
    height, width = depth_m.shape[:2]
    half = max(1, int(patch) // 2)
    x0 = max(0, int(u) - (int(w) // 2 if w > 0 else half))
    x1 = min(width, int(u) + (int(w) // 2 if w > 0 else half) + 1)
    y0 = max(0, int(v) - (int(h) // 2 if h > 0 else half))
    y1 = min(height, int(v) + (int(h) // 2 if h > 0 else half) + 1)

    def _median(cx: int, cy: int) -> float:
        yy0, yy1 = max(y0, cy - half), min(y1, cy + half + 1)
        xx0, xx1 = max(x0, cx - half), min(x1, cx + half + 1)
        if yy1 <= yy0 or xx1 <= xx0:
            return 0.0
        block = np.asarray(depth_m[yy0:yy1, xx0:xx1], dtype=float)
        valid = block[np.isfinite(block) & (block > 0.05) & (block < 4.0)]
        if valid.size < 1:
            return 0.0
        median = float(np.median(valid))
        spread = float(np.median(np.abs(valid - median)))
        kept = valid[np.abs(valid - median) <= max(0.08, 3.0 * spread)]
        if kept.size == 0:
            return 0.0
        return float(np.median(kept))

    cu = int(np.clip(u, 0, width - 1))
    cv = int(np.clip(v, 0, height - 1))
    depth = _median(cu, cv)
    if depth > 0.0:
        return depth, cu, cv
    offsets = ((0, -half), (0, half), (-half, 0), (half, 0), (-half, -half), (half, half))
    for du, dv in offsets:
        nu, nv = int(np.clip(cu + du, x0, max(x0, x1 - 1))), int(np.clip(cv + dv, y0, max(y0, y1 - 1)))
        depth = _median(nu, nv)
        if depth > 0.0:
            return depth, nu, nv
    return 0.0, cu, cv


def patch_depth_m(depth_m: np.ndarray, u: int, v: int, radius: int = 4) -> float:
    depth, _, _ = bbox_depth_m(depth_m, u, v, patch=max(1, 2 * int(radius) + 1))
    return depth


def cam_to_world(p_cam, t_cam_world) -> np.ndarray:
    point = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=float)
    return (np.asarray(t_cam_world, dtype=float) @ point)[:3]


def place_world_targets(
    table_xyz=DELIVERY_TARGET_XYZ,
    recipe: GraspRecipe = KELE_RECIPE,
    occupied: Sequence[Sequence[float]] = (),
) -> Tuple[np.ndarray, np.ndarray]:
    table = np.asarray(table_xyz, dtype=float)
    chosen = (float(table[0]), float(table[1]))
    for slot in PLACE_SLOT_XY:
        taken = False
        for item in occupied:
            if math.hypot(slot[0] - float(item[0]), slot[1] - float(item[1])) < PLACE_OCCUPIED_RADIUS:
                taken = True
                break
        if not taken:
            chosen = slot
            break
    z_g = recipe.table_support_z + recipe.h_bottom_from_g + recipe.release_clearance
    deploy = np.array([chosen[0], chosen[1], z_g + recipe.place_above])
    lower = np.array([chosen[0], chosen[1], z_g])
    return deploy, lower


def slot_layer_from_name(slot_name: Optional[str]) -> Optional[int]:
    if not slot_name:
        return None
    match = re.search(r"L([123])", str(slot_name))
    return int(match.group(1)) if match else None


def layer_from_height(z: float) -> int:
    lo, hi = SLOT_LAYER_Z
    if z < lo:
        return 1
    if z < hi:
        return 2
    return 3


def slot_hint_agrees_with_height(slot_name: Optional[str], z: float) -> bool:
    layer = slot_layer_from_name(slot_name)
    if layer is None:
        return True
    return layer == layer_from_height(float(z))


def slot_head_hint(slot_name: Optional[str]) -> Tuple[float, float]:
    """Head look for a bound slot. Returns yaw/pitch only — never a grasp xyz."""
    if not slot_name:
        return HEAD_YAW, HEAD_PITCH
    layer_match = re.search(r"L([123])", slot_name)
    column_match = re.search(r"C([123])", slot_name)
    pitch = SLOT_LAYER_PITCH.get(int(layer_match.group(1)), HEAD_PITCH) if layer_match else HEAD_PITCH
    yaw = SLOT_COLUMN_YAW.get(int(column_match.group(1)), HEAD_YAW) if column_match else HEAD_YAW
    return float(yaw), float(pitch)


def locate_head_hint(
    slot_name: Optional[str],
    attempt: int = 1,
    height: Optional[float] = None,
) -> Tuple[float, float]:
    """C1 look from the aisle. Do not use close-creep pitch -0.6 here."""
    hint = slot_name
    if hint and height is not None and not slot_hint_agrees_with_height(hint, height):
        hint = None
    if hint:
        return slot_head_hint(hint)
    if height is not None:
        pitch = float(SLOT_LAYER_PITCH.get(layer_from_height(float(height)), HEAD_PITCH))
        if int(attempt) <= 1:
            return HEAD_YAW, pitch
        others = [pose for pose in AISLE_SCAN_POSES if abs(float(pose[1]) - pitch) > 0.02]
        if not others:
            others = list(AISLE_SCAN_POSES)
        pose = others[(int(attempt) - 2) % len(others)]
        return float(pose[0]), pitch
    if int(attempt) <= 1:
        return HEAD_YAW, HEAD_PITCH
    others = [pose for pose in AISLE_SCAN_POSES if abs(float(pose[1]) - HEAD_PITCH) > 0.02]
    if not others:
        others = list(AISLE_SCAN_POSES)
    pose = others[(int(attempt) - 2) % len(others)]
    return float(pose[0]), float(pose[1])


def look_at_world(target, base_xy, base_yaw: float, slide: float) -> Tuple[float, float]:
    """Head yaw/pitch that points the optical axis at a world point."""
    target_fp = world_to_footprint(target, base_xy, base_yaw)
    slide = float(np.clip(slide, SLIDE_LIMITS[0], SLIDE_LIMITS[1]))

    def score(yaw: float, pitch: float) -> float:
        try:
            matrix = base_to_head_camera(slide, yaw, pitch)
        except ValueError:
            return -1.0
        vec = target_fp - matrix[:3, 3]
        norm = float(np.linalg.norm(vec))
        if norm < 1e-4:
            return -1.0
        return float(np.dot(matrix[:3, 2], vec / norm))

    try:
        cam = base_to_head_camera(slide, HEAD_YAW, HEAD_PITCH)[:3, 3]
    except ValueError:
        return HEAD_YAW, HEAD_PITCH
    delta = target_fp - cam
    yaw0 = float(np.clip(math.atan2(delta[1], delta[0]), HEAD_YAW_LIMITS[0], HEAD_YAW_LIMITS[1]))
    best_yaw, best_pitch, best = yaw0, HEAD_PITCH, -1.0
    for pitch in np.linspace(HEAD_PITCH_LIMITS[0], HEAD_PITCH_LIMITS[1], 17):
        value = score(yaw0, float(pitch))
        if value > best:
            best_pitch, best = float(pitch), value
    for yaw in np.linspace(best_yaw - 0.12, best_yaw + 0.12, 7):
        yaw = float(np.clip(yaw, HEAD_YAW_LIMITS[0], HEAD_YAW_LIMITS[1]))
        for pitch in np.linspace(best_pitch - 0.10, best_pitch + 0.10, 7):
            pitch = float(np.clip(pitch, HEAD_PITCH_LIMITS[0], HEAD_PITCH_LIMITS[1]))
            value = score(yaw, pitch)
            if value > best:
                best_yaw, best_pitch, best = yaw, pitch, value
    if best < 0.85:
        return HEAD_YAW, HEAD_PITCH
    return best_yaw, best_pitch


def creep_remaining_along(ee, object_world, approach, recipe: GraspRecipe = KELE_RECIPE) -> float:
    """Positive until the EE has wrapped past the object along approach."""
    axis = np.asarray(approach, dtype=float)
    goal = np.asarray(object_world, dtype=float)
    now = np.asarray(ee, dtype=float)
    return float(np.dot(goal - now, axis)) - float(recipe.creep_stop_along)


def rotation_error(measured_rot, target_rot=GRASP_ROT) -> float:
    delta = np.asarray(measured_rot, dtype=float).T @ np.asarray(target_rot, dtype=float)
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cosine))


def _pack(slide, head_yaw, head_pitch, left_arm, left_grip, right_arm, right_grip) -> np.ndarray:
    return np.concatenate([
        [float(slide), float(head_yaw), float(head_pitch)],
        np.asarray(left_arm, dtype=float),
        [float(left_grip)],
        np.asarray(right_arm, dtype=float),
        [float(right_grip)],
    ])


def lock_from_frames(
    frames: Sequence[DetectionSample],
    base_xy,
    base_yaw: float,
    recipe: GraspRecipe = KELE_RECIPE,
    now: float = 0.0,
) -> Optional[LockedTarget]:
    if not recipe.enabled or not gripper_fits(recipe):
        return None
    unique: List[DetectionSample] = []
    seen = set()
    for item in frames:
        if int(item.frame_id) in seen:
            continue
        seen.add(int(item.frame_id))
        unique.append(item)
    if len(unique) < 2:
        return None
    last = unique[-2:]
    if len(unique) >= 3:
        last = unique[-3:]
        dist_ab = float(np.linalg.norm(last[0].p_center - last[1].p_center))
        dist_bc = float(np.linalg.norm(last[1].p_center - last[2].p_center))
        dist_ac = float(np.linalg.norm(last[0].p_center - last[2].p_center))
        if dist_ab > recipe.two_frame_pos_tol and dist_bc > recipe.two_frame_pos_tol and dist_ac > recipe.two_frame_pos_tol:
            return None
        if dist_bc <= recipe.two_frame_pos_tol:
            last = last[-2:]
        elif dist_ac <= recipe.two_frame_pos_tol:
            last = [last[0], last[2]]
        else:
            last = last[:2]
    a, b = last[0], last[1]
    if a.kind != b.kind:
        return None
    delta = float(np.linalg.norm(np.asarray(a.p_center) - np.asarray(b.p_center)))
    if delta > recipe.two_frame_pos_tol:
        return None
    center = 0.5 * (np.asarray(a.p_center, dtype=float) + np.asarray(b.p_center, dtype=float))
    if not reachable_in_footprint(center, base_xy, base_yaw, recipe):
        return None
    axis = approach_axis_world(base_yaw)
    return LockedTarget(
        object_world=center,
        deploy_world=pregrasp_world(center, base_yaw, recipe),
        approach_world=axis,
        base_xy=(float(base_xy[0]), float(base_xy[1])),
        base_yaw=float(base_yaw),
        stamp=float(now),
        frame_ids=(int(a.frame_id), int(b.frame_id)),
        sample_count=2,
        quality=float(max(0.0, 1.0 - delta / max(recipe.two_frame_pos_tol, 1e-6))),
        instance_id=f"{recipe.kind}:{int(a.frame_id)}:{int(b.frame_id)}",
        recipe_kind=recipe.kind,
        recipe_version=recipe.version,
    )


class LatestFrameWorker:
    """Keep at most one pending frame. Control must not join this thread."""

    def __init__(self, infer_fn: Callable):
        self._infer = infer_fn
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending = None
        self._result = None
        self._result_id = None
        self._processed_id = None
        self._stop = False
        self.infer_times: List[float] = []
        self.frames = 0
        self._thread = threading.Thread(target=self._loop, name="p5-yolo", daemon=True)
        self._thread.start()

    def submit(self, frame_id: int, payload) -> bool:
        with self._lock:
            if self._stop:
                return False
            if self._processed_id == int(frame_id) and self._pending is None:
                return False
            if self._pending is not None and self._pending[0] == int(frame_id):
                return False
            self._pending = (int(frame_id), payload)
            self._cond.notify()
            return True

    def latest(self):
        with self._lock:
            if self._result is None:
                return None
            return self._result_id, self._result

    def close(self) -> None:
        with self._lock:
            self._stop = True
            self._cond.notify()

    def _loop(self) -> None:
        while True:
            with self._lock:
                while not self._stop and self._pending is None:
                    self._cond.wait(timeout=0.05)
                if self._stop:
                    return
                frame_id, payload = self._pending
                self._pending = None
                if frame_id == self._processed_id:
                    continue
            started = time.monotonic()
            try:
                result = self._infer(payload)
            except Exception:
                result = None
            elapsed = time.monotonic() - started
            with self._lock:
                self._processed_id = frame_id
                self.infer_times.append(elapsed)
                self.frames += 1
                if result is not None:
                    self._result = result
                    self._result_id = frame_id


class GraspSession:
    """Official cola pick, then optional table place. No JSON product xyz."""

    def __init__(
        self,
        ik_fn: IkFn,
        kind: str = DEFAULT_KIND,
        now: float = 0.0,
        dt: float = 1.0 / RATE_HZ,
        recipe: Optional[GraspRecipe] = None,
    ):
        self.ik_fn = ik_fn
        self.kind = str(kind)
        self.recipe = recipe if recipe is not None else recipe_for(self.kind) or KELE_RECIPE
        self.nominal_dt = float(dt)
        self.dt = float(dt)
        self.phase = GraspPhase.INIT
        self.reason = ""
        self.hold_state = ObjectHoldState.EMPTY
        self.slot_hint: Optional[str] = None
        self.lock_attempt = 1
        self.close_attempts = 0
        self.used_json_xyz = False
        self.frames: Deque[DetectionSample] = deque(maxlen=8)
        self._seen_frame_ids: Deque[int] = deque(maxlen=16)
        self._instance: Optional[np.ndarray] = None
        self.locked: Optional[LockedTarget] = None
        self.place_deploy_world: Optional[np.ndarray] = None
        self.place_lower_world: Optional[np.ndarray] = None
        self.place_occupied: List[np.ndarray] = []
        self.place_sequence_done = False
        self.place_verified_local = False
        self.referee_confirmed = None
        self.c1 = False
        self.c2 = False
        self.c3 = False
        self.arm_target_set = False
        self.phase_t0 = float(now)
        self.session_t0 = float(now)
        self.last_log = ""
        self._last_obs_t: Optional[float] = None
        self._ready_t: Optional[float] = None
        self._grip_reached_t: Optional[float] = None
        self._c2_look_t0: Optional[float] = None
        self._c3_stable_t0: Optional[float] = None
        self._c3_last = None
        self._pause_ticks = 0
        self.phase_timings: List[dict] = [{"phase": GraspPhase.INIT.value, "t0": float(now)}]
        self.tc = _pack(
            self.recipe.slide, HEAD_YAW, HEAD_PITCH,
            INIT_ARM_L, self.recipe.grip_open, INIT_ARM_R, self.recipe.grip_open,
        )
        self.action = self.tc.copy()
        self.tc_prev = self.tc.copy()
        self.move_ratio = np.ones(17)
        self.des_lin = 0.0
        self.des_ang = 0.0
        self.cur_lin = 0.0
        self.cur_ang = 0.0

    def perception_need(self) -> PerceptionNeed:
        if self.phase == GraspPhase.AIM:
            return PerceptionNeed.C1
        if self.phase in (GraspPhase.LIFT, GraspPhase.RETREAT, GraspPhase.PAUSED):
            return PerceptionNeed.C2
        if self.phase in (GraspPhase.PLACE_OPEN, GraspPhase.PLACE_RETRACT):
            return PerceptionNeed.C3
        return PerceptionNeed.NONE

    def note_slot_hint(self, slot_name: Optional[str]) -> None:
        nxt = slot_name or None
        if nxt and self.frames:
            height = float(self.frames[-1].p_center[2])
            if not slot_hint_agrees_with_height(nxt, height):
                self.last_log = f"ignore slot_hint {nxt} vs z={height:.2f}"
                return
        if nxt and nxt != self.slot_hint:
            self.slot_hint = nxt
            self.frames.clear()
            self._seen_frame_ids.clear()
            self._instance = None
            self._ready_t = None
            return
        if self.slot_hint is None:
            self.slot_hint = nxt

    def begin_place(self, now: float, table_xyz=DELIVERY_TARGET_XYZ, occupied=()) -> None:
        if self.phase not in (GraspPhase.HELD, GraspPhase.RETREAT) or self.hold_state != ObjectHoldState.HELD:
            self.phase = GraspPhase.FAILED
            self.reason = "place_without_grasp"
            return
        self.place_occupied = [np.asarray(item, dtype=float) for item in occupied]
        self.place_deploy_world, self.place_lower_world = place_world_targets(
            table_xyz, self.recipe, self.place_occupied,
        )
        self.arm_target_set = False
        self.place_sequence_done = False
        self.place_verified_local = False
        self.c3 = False
        self._c3_stable_t0 = None
        self._enter(GraspPhase.PLACE_DEPLOY, now)
        self.session_t0 = float(now)
        self.reason = ""

    def tick(self, obs: GraspObservation) -> GraspCommand:
        if self.phase == GraspPhase.FAILED:
            return self._command(obs, done=False, failed=True, hard_stop=True)
        if self.phase == GraspPhase.DONE:
            return self._command(obs, done=True)
        if self.phase == GraspPhase.HELD and self.hold_state == ObjectHoldState.HELD:
            return self._command(obs, done=True)
        if self.phase == GraspPhase.PAUSED:
            self._hard_stop_base()
            self._update_c2(obs)
            self._slew()
            return self._command(obs)

        jumped = self._update_dt(obs)
        timeout = PLACE_TIMEOUT_S if self.phase.value.startswith("place") else GRASP_TIMEOUT_S
        if obs.t - self.session_t0 > timeout and self.phase not in {
            GraspPhase.HELD, GraspPhase.DONE, GraspPhase.FAILED, GraspPhase.PAUSED,
        }:
            self._fail("grasp_timeout" if timeout == GRASP_TIMEOUT_S else "place_timeout")
            return self._command(obs, failed=True, hard_stop=True)

        if jumped:
            self._hard_stop_base()
            self.last_log = f"dt_pause dt={self.dt:.3f}"
            self._pause_ticks += 1
            return self._command(obs, hard_stop=True)

        if self.phase == GraspPhase.INIT:
            self._set_twist(0.0, 0.0)
            self._set_joints(slide=self.recipe.slide, head_yaw=HEAD_YAW, head_pitch=HEAD_PITCH, right_grip=self.recipe.grip_open)
            if self._look_ready(obs) and self._joints_ready(obs, need_cart=False):
                self._enter(GraspPhase.AIM, obs.t)
        elif self.phase == GraspPhase.AIM:
            self._aim_and_lock(obs)
        elif self.phase == GraspPhase.DEPLOY:
            self._deploy(obs)
        elif self.phase == GraspPhase.CREEP:
            self._creep(obs)
        elif self.phase == GraspPhase.CLOSE:
            self._close(obs)
        elif self.phase == GraspPhase.LIFT:
            self._lift(obs)
        elif self.phase == GraspPhase.RETREAT:
            self._retreat(obs)
        elif self.phase == GraspPhase.PLACE_DEPLOY:
            self._place_move(obs, self.place_deploy_world, GraspPhase.PLACE_LOWER)
        elif self.phase == GraspPhase.PLACE_LOWER:
            self._place_move(obs, self.place_lower_world, GraspPhase.PLACE_OPEN)
        elif self.phase == GraspPhase.PLACE_OPEN:
            self._place_open(obs)
        elif self.phase == GraspPhase.PLACE_RETRACT:
            self._place_retract(obs)

        self._slew()
        self._ramp()
        return self._command(obs)

    def _update_dt(self, obs: GraspObservation) -> bool:
        if self._last_obs_t is None:
            self.dt = self.nominal_dt
            self._last_obs_t = float(obs.t)
            return False
        raw = float(obs.t) - self._last_obs_t
        self._last_obs_t = float(obs.t)
        if raw < 0.0:
            self.dt = self.nominal_dt
            return False
        self.dt = raw
        return raw > max(DT_JUMP_S, self.recipe.dt_jump_s)

    def _ingest(
        self,
        obs: GraspObservation,
        associate_at=None,
        allow_reselect: bool = False,
        tol: Optional[float] = None,
    ) -> List[DetectionSample]:
        samples: List[DetectionSample] = list(obs.detections)
        if not samples and obs.detections_world:
            frame_id = int(obs.rgb_frame_id) if obs.rgb_frame_id is not None else int(round(obs.t * 1000.0))
            for point in obs.detections_world:
                surface = np.asarray(point, dtype=float)
                center = vision_to_object_center(
                    surface, obs.base_xy, obs.base_yaw, self.recipe, obs.camera_xy,
                )
                samples.append(DetectionSample(
                    kind=self.kind,
                    stamp=float(obs.rgb_stamp if obs.rgb_stamp is not None else obs.t),
                    frame_id=frame_id,
                    p_surface=surface,
                    p_center=center,
                ))
        if not samples:
            return []
        frame_id = int(samples[0].frame_id)
        if frame_id in self._seen_frame_ids:
            return []
        recipe = self.recipe if tol is None else replace(self.recipe, associate_tol=float(tol))
        chosen = None
        if associate_at is not None:
            chosen = associate_detection(samples, associate_at, recipe)
        if chosen is None and self._instance is not None and not allow_reselect:
            chosen = associate_detection(samples, self._instance, recipe)
        if chosen is None and (allow_reselect or self._instance is None):
            worlds = [item.p_center for item in samples]
            picked = pick_centered_detection(worlds, obs.base_xy, obs.base_yaw, self.recipe)
            if picked is not None:
                for item in samples:
                    if np.allclose(item.p_center, picked):
                        chosen = item
                        break
        if chosen is None:
            return []
        self._seen_frame_ids.append(frame_id)
        if self._instance is None:
            self._instance = np.asarray(chosen.p_center, dtype=float).copy()
        self.frames.append(chosen)
        return [chosen]

    def _look_ready(self, obs: GraspObservation) -> bool:
        head_err = max(
            abs(obs.head_yaw - float(self.tc[1])),
            abs(obs.head_pitch - float(self.tc[2])),
        )
        slide_err = abs(obs.slide - float(self.tc[0]))
        return head_err < 0.05 and slide_err < DEPLOY_SLIDE_TOL and abs(obs.linear_speed) <= 0.02

    def _aim_and_lock(self, obs: GraspObservation) -> None:
        self._set_twist(0.0, 0.0)
        height = float(self.frames[-1].p_center[2]) if self.frames else None
        if (
            self.slot_hint is not None
            and height is not None
            and not slot_hint_agrees_with_height(self.slot_hint, height)
        ):
            self.last_log = f"drop slot_hint {self.slot_hint} (z={height:.2f})"
            self.slot_hint = None
        yaw, pitch = locate_head_hint(self.slot_hint, self.lock_attempt, height)
        self._set_joints(slide=self.recipe.slide, head_yaw=yaw, head_pitch=pitch, right_grip=self.recipe.grip_open)
        if not self.recipe.enabled:
            self._fail("template_unavailable")
            return
        if not gripper_fits(self.recipe):
            self._fail("gripper_too_narrow")
            return
        if not self._look_ready(obs):
            self.last_log = (
                f"aim wait head yaw={obs.head_yaw:.2f}/{yaw:.2f} "
                f"pitch={obs.head_pitch:.2f}/{pitch:.2f}"
            )
            if obs.t - self.phase_t0 <= LOCK_TIMEOUT_S:
                return
        else:
            self._ingest(obs, allow_reselect=self._instance is None)
            locked = lock_from_frames(list(self.frames), obs.base_xy, obs.base_yaw, self.recipe, now=obs.t)
            if locked is not None:
                if not self._arm_to(locked.deploy_world, obs, grip=self.recipe.grip_open):
                    self.frames.clear()
                    self._seen_frame_ids.clear()
                    if self.lock_attempt < LOCK_ATTEMPTS:
                        self.lock_attempt += 1
                        self.phase_t0 = obs.t
                        self.last_log = "ik_unreachable on deploy; retry lock"
                        return
                    self._fail("ik_unreachable")
                    return
                self.locked = locked
                self.c1 = True
                self._enter(GraspPhase.DEPLOY, obs.t)
                self.last_log = (
                    f"locked kind={self.kind} frames={locked.frame_ids} "
                    f"obj={np.round(locked.object_world, 3).tolist()}"
                )
                return
            self.last_log = (
                f"c1 wait frames={len(self.frames)} yolo={len(obs.detections) or len(obs.detections_world)} "
                f"look=({yaw:.2f},{pitch:.2f}) slot_hint={self.slot_hint}"
            )
            if obs.t - self.phase_t0 <= LOCK_TIMEOUT_S:
                return
        if self.lock_attempt < LOCK_ATTEMPTS:
            self.lock_attempt += 1
            self.frames.clear()
            self._seen_frame_ids.clear()
            self._instance = None
            self.phase_t0 = obs.t
            self._ready_t = None
            nxt = locate_head_hint(self.slot_hint, self.lock_attempt, height)
            self.last_log = (
                f"lock retry {self.lock_attempt}/{LOCK_ATTEMPTS} "
                f"look={nxt} slot_hint={self.slot_hint} (no json xyz)"
            )
            return
        self._fail("lock_timeout")

    def _deploy(self, obs: GraspObservation) -> None:
        self._set_twist(0.0, 0.0)
        self._look_at_locked(obs)
        if self.locked is None:
            self._fail("lock_timeout")
            return
        if not self.arm_target_set:
            if not self._arm_to(self.locked.deploy_world, obs, grip=self.recipe.grip_open):
                self._fail("ik_unreachable")
                return
        if self._deploy_done(obs, self.locked.deploy_world):
            self._enter(GraspPhase.CREEP, obs.t)

    def _look_at_locked(self, obs: GraspObservation, target=None) -> None:
        point = target
        if point is None and self.locked is not None:
            point = self.locked.object_world
        if point is None:
            return
        yaw, pitch = look_at_world(point, obs.base_xy, obs.base_yaw, obs.slide)
        self._set_joints(head_yaw=yaw, head_pitch=pitch)

    def _creep(self, obs: GraspObservation) -> None:
        self._look_at_locked(obs)
        axis = self.locked.approach_world if self.locked is not None else approach_axis_world(obs.base_yaw)
        goal = self.locked.object_world if self.locked is not None else np.array(
            [obs.base_xy[0], obs.base_xy[1], 0.0]
        )
        ee = obs.ee_world if obs.ee_world is not None else np.array([obs.base_xy[0], obs.base_xy[1], 0.0])
        remaining = creep_remaining_along(ee, goal, axis, self.recipe)
        if remaining > 0.0:
            yaw_err = wrap_to_pi(math.atan2(axis[1], axis[0]) - obs.base_yaw)
            speed = self.recipe.creep_speed
            ang = CREEP_YAW_KP * yaw_err
            if remaining <= CREEP_SLOW_ALONG:
                speed = min(speed, CREEP_SLOW_SPEED)
                ang = 0.0
            self._set_twist(speed, ang)
            self.last_log = f"creep remaining={remaining:.3f}"
        else:
            self._hard_stop_base()
            if abs(obs.linear_speed) <= 0.02:
                self._enter(GraspPhase.CLOSE, obs.t)

    def _close(self, obs: GraspObservation) -> None:
        self._hard_stop_base()
        self._look_at_locked(obs)
        self._set_joints(right_grip=self.recipe.grip_close)
        if self.close_attempts == 0:
            self.close_attempts = 1
        if abs(self.action[16] - self.recipe.grip_close) > 0.03:
            self._grip_reached_t = None
            return
        if self._grip_reached_t is None:
            self._grip_reached_t = obs.t
        if obs.t - self._grip_reached_t >= self.recipe.close_hold_s and abs(obs.linear_speed) <= 0.02:
            self._enter(GraspPhase.LIFT, obs.t)

    def _lift(self, obs: GraspObservation) -> None:
        self._set_twist(0.0, 0.0)
        self._look_at_hold(obs)
        self._set_joints(slide=self.recipe.slide - self.recipe.lift_clearance, right_grip=self.recipe.grip_close)
        self._update_c2(obs)
        target = self.recipe.slide - self.recipe.lift_clearance
        if abs(obs.slide - target) < 0.02:
            self._enter(GraspPhase.RETREAT, obs.t)
            return
        if obs.t - self.phase_t0 >= LIFT_TIMEOUT_S:
            self.last_log = "lift stall; extract"
            self._enter(GraspPhase.RETREAT, obs.t)

    def _look_at_hold(self, obs: GraspObservation) -> None:
        """C2 looks at the gripper/object, not the empty L2 shelf."""
        target = obs.ee_world
        if target is None and self.locked is not None:
            target = self.locked.object_world
        if target is None:
            self._set_joints(head_yaw=HEAD_YAW, head_pitch=GRASP_HEAD_PITCH)
            return
        self._look_at_locked(obs, target)

    def _slide_at_carry(self, obs: GraspObservation) -> bool:
        return abs(float(obs.slide) - float(self.recipe.carry_slide)) < 0.05

    def _retreat(self, obs: GraspObservation) -> None:
        self._look_at_hold(obs)
        axis = self.locked.approach_world if self.locked is not None else approach_axis_world(obs.base_yaw)
        lock_xy = np.array(self.locked.base_xy if self.locked is not None else obs.base_xy, dtype=float)
        now_xy = np.array(obs.base_xy, dtype=float)
        backed = float(np.dot(lock_xy - now_xy, axis[:2]))
        yaw_err = wrap_to_pi(math.atan2(axis[1], axis[0]) - obs.base_yaw)
        need = max(self.recipe.pregrasp_along, self.recipe.carry_out_min + 0.05)
        if backed < need - 0.03:
            self._set_twist(-self.recipe.retreat_speed, 1.0 * yaw_err)
            self._c2_look_t0 = None
        else:
            self._hard_stop_base()
            # Clear of the shelf: raise the torso (slide down in value = up).
            # Do not do this while the hand is still in a layer bay.
            self._set_joints(slide=self.recipe.carry_slide, right_grip=self.recipe.grip_close)
        self._update_c2(obs)
        clear = backed >= need - 0.03
        if self.hold_state == ObjectHoldState.HELD and self.c2 and clear:
            if not self._slide_at_carry(obs):
                self.last_log = f"c2 held; raising slide {obs.slide:.3f}->{self.recipe.carry_slide:.3f}"
                return
            self.reason = "held"
            self.last_log = "c2 held; grasp retained"
            self._enter(GraspPhase.HELD, obs.t)
        elif self.hold_state == ObjectHoldState.EMPTY and backed >= self.recipe.carry_out_min:
            self._fail("empty_grasp")
        elif backed >= need and abs(obs.linear_speed) <= 0.02:
            if self._c2_look_t0 is None:
                self._c2_look_t0 = float(obs.t)
                self.last_log = "c2 look at gripper"
                return
            if obs.t - self._c2_look_t0 < C2_LOOK_S:
                self.last_log = f"c2 wait look {obs.t - self._c2_look_t0:.2f}/{C2_LOOK_S:.1f}"
                return
            self.hold_state = ObjectHoldState.UNKNOWN
            self.reason = "c2_unknown"
            self.last_log = "c2 unknown; keep grip, do not haul"
            self._enter(GraspPhase.PAUSED, obs.t)

    def _update_c2(self, obs: GraspObservation) -> None:
        if self.locked is None:
            self.hold_state = ObjectHoldState.UNKNOWN
            return
        pred = obs.ee_world
        ingested = []
        if pred is not None:
            ingested = self._ingest(
                obs, associate_at=pred, allow_reselect=False, tol=self.recipe.follow_tol,
            )
        if not ingested:
            ingested = self._ingest(
                obs, associate_at=self.locked.object_world, allow_reselect=False, tol=self.recipe.associate_tol,
            )
        samples = list(obs.detections)
        if pred is not None and samples and not ingested:
            nearest = min(
                min(
                    float(np.linalg.norm(item.p_center - pred)),
                    float(np.linalg.norm(item.p_surface - pred)),
                )
                for item in samples
            )
            self.last_log = f"c2 miss nearest={nearest:.3f} n={len(samples)}"
        if pred is None or not ingested:
            if self.hold_state != ObjectHoldState.HELD:
                self.hold_state = ObjectHoldState.UNKNOWN
            return
        seen = ingested[0]
        follow = float(np.linalg.norm(seen.p_center - pred))
        follow = min(follow, float(np.linalg.norm(seen.p_surface - pred)))
        away = float(np.linalg.norm(seen.p_center[:2] - self.locked.object_world[:2]))
        still_on_shelf = away < 0.06
        if follow <= self.recipe.follow_tol and away >= 0.04:
            self.hold_state = ObjectHoldState.HELD
            if away >= self.recipe.carry_out_min:
                self.c2 = True
            self.last_log = f"c2 follow={follow:.3f} away={away:.3f}"
            return
        if still_on_shelf and follow > self.recipe.follow_tol:
            self.hold_state = ObjectHoldState.EMPTY
            self.c2 = False
            self.last_log = f"c2 empty away={away:.3f} follow={follow:.3f}"
            return
        self.hold_state = ObjectHoldState.UNKNOWN
        self.last_log = f"c2 unknown follow={follow:.3f} away={away:.3f}"

    def _place_move(self, obs: GraspObservation, world_pos, next_phase: GraspPhase) -> None:
        self._set_twist(0.0, 0.0)
        if world_pos is None:
            self._fail("place_without_grasp")
            return
        grip = self.recipe.grip_close if self.phase != GraspPhase.PLACE_OPEN else self.recipe.grip_open
        if not self.arm_target_set:
            if not self._arm_to(world_pos, obs, grip=grip):
                self._fail("place_ik_unreachable")
                return
        if self._deploy_done(obs, world_pos, cart_tol=PLACE_CART_TOL):
            self.arm_target_set = False
            self._enter(next_phase, obs.t)

    def _place_open(self, obs: GraspObservation) -> None:
        self._set_twist(0.0, 0.0)
        self._set_joints(right_grip=self.recipe.grip_open)
        self._update_c3(obs)
        if abs(self.action[16] - self.recipe.grip_open) <= 0.05:
            self.arm_target_set = False
            self._enter(GraspPhase.PLACE_RETRACT, obs.t)

    def _place_retract(self, obs: GraspObservation) -> None:
        self._set_twist(0.0, 0.0)
        if not self.arm_target_set:
            if self.place_lower_world is None:
                self._fail("place_without_grasp")
                return
            footprint = world_to_footprint(self.place_lower_world, obs.base_xy, obs.base_yaw)
            footprint[0] = max(0.20, float(footprint[0]) - PLACE_RETRACT_FWD)
            retract_world = footprint_to_world(footprint, obs.base_xy, obs.base_yaw)
            if not self._arm_to(retract_world, obs, grip=self.recipe.grip_open):
                self._set_joints(right_arm=INIT_ARM_R, right_grip=self.recipe.grip_open)
                self.arm_target_set = True
        self._update_c3(obs)
        if self.c3:
            self.place_sequence_done = True
            self.place_verified_local = True
            self.hold_state = ObjectHoldState.EMPTY
            self.reason = "placed"
            self.last_log = "c3 placed (not Server referee)"
            self._enter(GraspPhase.DONE, obs.t)

    def _update_c3(self, obs: GraspObservation) -> None:
        hint = self.place_lower_world if self.place_lower_world is not None else (
            self._c3_last[0] if self._c3_last is not None else None
        )
        ingested = self._ingest(obs, associate_at=hint, allow_reselect=hint is None, tol=0.18)
        if not ingested:
            self._c3_stable_t0 = None
            self._c3_last = None
            return
        seen = np.asarray(ingested[0].p_center, dtype=float)
        separated = True
        if obs.ee_world is not None:
            separated = float(np.linalg.norm(seen[:2] - np.asarray(obs.ee_world)[:2])) > 0.08
        on_table = bool(in_delivery_box(seen))
        speed = 0.0
        if self._c3_last is not None:
            prev, t_prev = self._c3_last
            dt = max(1e-3, float(obs.t) - t_prev)
            speed = float(np.linalg.norm(seen[:2] - prev[:2]) / dt)
        self._c3_last = (seen, float(obs.t))
        if on_table and separated and speed <= self.recipe.place_stable_speed:
            if self._c3_stable_t0 is None:
                self._c3_stable_t0 = float(obs.t)
            if float(obs.t) - self._c3_stable_t0 >= self.recipe.place_stable_s:
                self.c3 = True
                self.last_log = f"c3 stable at {np.round(seen, 3).tolist()}"
        else:
            self._c3_stable_t0 = None

    def _arm_to(self, world_pos, obs: GraspObservation, grip: float) -> bool:
        footprint = world_to_footprint(world_pos, obs.base_xy, obs.base_yaw)
        joints = self.ik_fn(footprint, self.recipe.grasp_rot, float(self.tc[0]), np.asarray(obs.right_arm, dtype=float))
        if joints is None:
            return False
        self._set_joints(right_arm=joints, right_grip=grip)
        self.arm_target_set = True
        return True

    def _joints_ready(self, obs: GraspObservation, need_cart: bool = True, world_pos=None, cart_tol: float = DEPLOY_CART_TOL) -> bool:
        slide_err = abs(obs.slide - float(self.tc[0]))
        joint_err = float(np.max(np.abs(obs.right_arm - self.tc[10:16])))
        cart_err = 0.0
        rot_err = 0.0
        if need_cart:
            cart_err = float("inf")
            rot_err = float("inf")
            if world_pos is not None and obs.ee_world is not None:
                cart_err = float(np.linalg.norm(np.asarray(obs.ee_world) - np.asarray(world_pos)))
            if obs.ee_rot_footprint is not None:
                rot_err = rotation_error(obs.ee_rot_footprint, self.recipe.grasp_rot)
        joints_ok = joint_err < DEPLOY_JOINT_TOL or not need_cart
        cart_ok = (not need_cart) or (cart_err < cart_tol and rot_err < DEPLOY_ROT_TOL)
        ready = slide_err < DEPLOY_SLIDE_TOL and joints_ok and cart_ok
        if not ready:
            self._ready_t = None
            return False
        if self._ready_t is None:
            self._ready_t = obs.t
        return obs.t - self._ready_t >= self.recipe.settle_evidence_s

    def _deploy_done(self, obs: GraspObservation, world_pos, cart_tol: float = DEPLOY_CART_TOL) -> bool:
        ready = self._joints_ready(obs, need_cart=True, world_pos=world_pos, cart_tol=cart_tol)
        if not ready:
            slide_err = abs(obs.slide - float(self.tc[0]))
            joint_err = float(np.max(np.abs(obs.right_arm - self.tc[10:16]))) if self.arm_target_set else float("inf")
            cart_err = float("inf")
            rot_err = float("inf")
            if world_pos is not None and obs.ee_world is not None:
                cart_err = float(np.linalg.norm(np.asarray(obs.ee_world) - np.asarray(world_pos)))
            if obs.ee_rot_footprint is not None:
                rot_err = rotation_error(obs.ee_rot_footprint, self.recipe.grasp_rot)
            self.last_log = (
                f"deploy wait slide={slide_err:.3f} joint={joint_err:.3f} "
                f"cart={cart_err:.3f} rot={rot_err:.3f}"
            )
        return ready

    def _set_joints(
        self,
        slide=None,
        head_yaw=None,
        head_pitch=None,
        left_arm=None,
        left_grip=None,
        right_arm=None,
        right_grip=None,
    ) -> None:
        if slide is not None:
            self.tc[0] = float(slide)
        if head_yaw is not None:
            self.tc[1] = float(head_yaw)
        if head_pitch is not None:
            self.tc[2] = float(head_pitch)
        if left_arm is not None:
            self.tc[3:9] = np.asarray(left_arm, dtype=float)
        if left_grip is not None:
            self.tc[9] = float(left_grip)
        if right_arm is not None:
            self.tc[10:16] = np.asarray(right_arm, dtype=float)
        if right_grip is not None:
            self.tc[16] = float(right_grip)

    def _set_twist(self, linear: float, angular: float) -> None:
        self.des_lin, self.des_ang = clip_base(linear, angular)

    def _hard_stop_base(self) -> None:
        self.des_lin = 0.0
        self.des_ang = 0.0
        self.cur_lin = 0.0
        self.cur_ang = 0.0

    def _slew(self) -> None:
        if not np.allclose(self.tc, self.tc_prev):
            delta = np.abs(self.action - self.tc)
            self.move_ratio = delta / (np.max(delta) + 1e-6)
            self.move_ratio[0] *= 0.3
            self.tc_prev = self.tc.copy()
        step = JOINT_SLEW * min(self.dt, self.nominal_dt * 2.0)
        for index in range(17):
            self.action[index] = step_func(self.action[index], self.tc[index], self.move_ratio[index] * step)

    def _ramp(self) -> None:
        self.cur_lin += float(np.clip(self.des_lin - self.cur_lin, -MAX_LIN_ACC * self.dt, MAX_LIN_ACC * self.dt))
        self.cur_ang += float(np.clip(self.des_ang - self.cur_ang, -MAX_ANG_ACC * self.dt, MAX_ANG_ACC * self.dt))

    def _enter(self, phase: GraspPhase, now: float) -> None:
        self._close_phase(now)
        self.phase = phase
        self.phase_t0 = float(now)
        self._ready_t = None
        self._grip_reached_t = None
        self.phase_timings.append({"phase": phase.value, "t0": float(now)})
        if phase == GraspPhase.DEPLOY:
            self.arm_target_set = True

    def _close_phase(self, now: float) -> None:
        if self.phase_timings and "t1" not in self.phase_timings[-1]:
            self.phase_timings[-1]["t1"] = float(now)
            self.phase_timings[-1]["elapsed_s"] = round(float(now) - self.phase_timings[-1]["t0"], 3)

    def _note_phase(self, now: float) -> None:
        self._close_phase(now)
        self.phase_timings.append({"phase": self.phase.value, "t0": float(now)})

    def fail(self, reason: str) -> None:
        self.phase = GraspPhase.FAILED
        self.reason = reason
        self._hard_stop_base()
        self.last_log = reason

    def _fail(self, reason: str) -> None:
        self.fail(reason)

    def _command(self, _obs: GraspObservation, done: bool = False, failed: bool = False, hard_stop: bool = False) -> GraspCommand:
        if hard_stop or self.phase == GraspPhase.FAILED:
            linear, angular = 0.0, 0.0
            self._hard_stop_base()
        else:
            linear, angular = clip_base(self.cur_lin, self.cur_ang)
        grasped = self.phase == GraspPhase.HELD and self.hold_state == ObjectHoldState.HELD
        return GraspCommand(
            linear=linear,
            angular=angular,
            slide=float(self.action[0]),
            head_yaw=float(self.action[1]),
            head_pitch=float(self.action[2]),
            left_arm=self.action[3:9].copy(),
            left_grip=float(self.action[9]),
            right_arm=self.action[10:16].copy(),
            right_grip=float(self.action[16]),
            phase=self.phase,
            done=done or grasped or self.phase == GraspPhase.DONE,
            failed=failed or self.phase == GraspPhase.FAILED,
            reason=self.reason,
            log=self.last_log,
            hard_stop=hard_stop or self.phase == GraspPhase.FAILED,
            hold_state=self.hold_state.value,
            perception=self.perception_need().value,
        )

    def report_extras(self) -> dict:
        obj = None if self.locked is None else np.round(self.locked.object_world, 4).tolist()
        deploy = None if self.locked is None else np.round(self.locked.deploy_world, 4).tolist()
        place_ok = False
        if self.place_lower_world is not None:
            place_ok = bool(in_delivery_box(self.place_lower_world))
        return {
            "kind": self.kind,
            "phase": self.phase.value,
            "slot_hint": self.slot_hint,
            "lock_attempt": self.lock_attempt,
            "used_json_xyz": bool(self.used_json_xyz),
            "object_world": obj,
            "deploy_world": deploy,
            "approach_world": None if self.locked is None else np.round(self.locked.approach_world, 4).tolist(),
            "recipe_version": self.recipe.version,
            "c1": bool(self.c1),
            "c2": bool(self.c2),
            "c3": bool(self.c3),
            "hold_state": self.hold_state.value,
            "place_sequence_done": bool(self.place_sequence_done),
            "place_verified_local": bool(self.place_verified_local),
            "referee_confirmed": self.referee_confirmed,
            "place_target_in_box": place_ok,
            "phase_timings": list(self.phase_timings),
            "close_attempts": int(self.close_attempts),
            "reason": self.reason,
            "head_cmd": [round(float(self.tc[1]), 3), round(float(self.tc[2]), 3)],
            "slide_cmd": round(float(self.tc[0]), 3),
        }
