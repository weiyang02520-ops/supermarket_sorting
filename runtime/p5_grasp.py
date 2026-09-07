"""P5 grasp/place state machine. No ROS import.

Ports the official cola sequence from client_task_1.py (deploy / creep /
close / lift / retreat) plus a table-place finish. Vision xyz comes from
RGB-D + head FK; lock retry may use an ArUco slot as a look hint only.
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Deque, Optional, Sequence, Tuple

import numpy as np

from common.control import step_func
from runtime.p5_params import (
    CLOSE_HOLD_S,
    CREEP_SPEED,
    CREEP_STOP_DY,
    CREEP_YAW_KP,
    DEFAULT_KIND,
    DEPLOY_CART_TOL,
    DEPLOY_DWELL_S,
    DEPLOY_JOINT_TOL,
    DEPLOY_OFFSET,
    DEPLOY_ROT_TOL,
    DEPLOY_SLIDE_TOL,
    DETECT_DWELL_S,
    DETECT_MIN_SAMPLES,
    GRASP_ROT,
    GRASP_TIMEOUT_S,
    GRASP_YAW,
    GRIP_CLOSE,
    GRIP_OPEN,
    HEAD_PITCH,
    HEAD_YAW,
    INIT_ARM_L,
    INIT_ARM_R,
    INIT_DWELL_S,
    JOINT_SLEW,
    LIFT_AMOUNT,
    LOCK_ATTEMPTS,
    LOCK_TIMEOUT_S,
    MAX_ANG,
    MAX_ANG_ACC,
    MAX_LIN,
    MAX_LIN_ACC,
    PLACE_CART_TOL,
    PLACE_DEPLOY_Z,
    PLACE_LOWER_Z,
    PLACE_OPEN_HOLD_S,
    PLACE_RETRACT_FWD,
    PLACE_TIMEOUT_S,
    POS_TOL,
    RATE_HZ,
    REACH_FWD_MAX,
    REACH_FWD_MIN,
    REACH_LATERAL_MAX,
    REACH_Z_MAX,
    REACH_Z_MIN,
    RETREAT_SPEED,
    RETREAT_Y,
    SLIDE_GRASP,
    SLOT_COLUMN_YAW,
    SLOT_LAYER_PITCH,
    VISION_SURFACE_TO_CENTER_FWD,
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
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class LockedTarget:
    object_world: np.ndarray
    deploy_world: np.ndarray
    creep_stop_y: float
    sample_count: int


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
    ee_world: Optional[np.ndarray] = None
    ee_rot_footprint: Optional[np.ndarray] = None


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


def reachable_in_footprint(p_world, base_xy, base_yaw: float) -> bool:
    footprint = world_to_footprint(p_world, base_xy, base_yaw)
    if footprint[0] < REACH_FWD_MIN or footprint[0] > REACH_FWD_MAX:
        return False
    if abs(footprint[1]) > REACH_LATERAL_MAX:
        return False
    height = float(np.asarray(p_world, dtype=float)[2])
    return REACH_Z_MIN <= height <= REACH_Z_MAX


def vision_to_object_center(p_world, base_xy, base_yaw: float) -> np.ndarray:
    footprint = world_to_footprint(p_world, base_xy, base_yaw)
    footprint[0] += VISION_SURFACE_TO_CENTER_FWD
    return footprint_to_world(footprint, base_xy, base_yaw)


def lock_from_samples(
    samples: Sequence[np.ndarray],
    base_xy,
    base_yaw: float,
    min_samples: int = DETECT_MIN_SAMPLES,
) -> Optional[LockedTarget]:
    if len(samples) < min_samples:
        return None
    candidate = np.median(np.asarray(samples, dtype=float), axis=0)
    if not reachable_in_footprint(candidate, base_xy, base_yaw):
        return None
    object_world = vision_to_object_center(candidate, base_xy, base_yaw)
    return LockedTarget(
        object_world=object_world,
        deploy_world=object_world + DEPLOY_OFFSET,
        creep_stop_y=float(object_world[1] + CREEP_STOP_DY),
        sample_count=len(samples),
    )


def pick_centered_detection(
    detections_world: Sequence[np.ndarray],
    base_xy,
    base_yaw: float,
) -> Optional[np.ndarray]:
    best = None
    best_lat = REACH_LATERAL_MAX
    for point in detections_world:
        if not reachable_in_footprint(point, base_xy, base_yaw):
            continue
        footprint = world_to_footprint(point, base_xy, base_yaw)
        lateral = abs(float(footprint[1]))
        if lateral < best_lat:
            best_lat = lateral
            best = np.asarray(point, dtype=float)
    return best


def patch_depth_m(depth_m: np.ndarray, u: int, v: int, radius: int = 4) -> float:
    height, width = depth_m.shape[:2]
    y0, y1 = max(0, int(v) - radius), min(height, int(v) + radius + 1)
    x0, x1 = max(0, int(u) - radius), min(width, int(u) + radius + 1)
    patch = np.asarray(depth_m[y0:y1, x0:x1], dtype=float)
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def cam_to_world(p_cam, t_cam_world) -> np.ndarray:
    point = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=float)
    return (np.asarray(t_cam_world, dtype=float) @ point)[:3]


def place_world_targets(table_xyz=DELIVERY_TARGET_XYZ) -> Tuple[np.ndarray, np.ndarray]:
    table = np.asarray(table_xyz, dtype=float)
    deploy = np.array([table[0], table[1], PLACE_DEPLOY_Z])
    lower = np.array([table[0], table[1], PLACE_LOWER_Z])
    return deploy, lower


def slot_head_hint(slot_name: Optional[str]) -> Tuple[float, float]:
    """Head look for a bound slot. Returns yaw/pitch only — never a grasp xyz."""
    if not slot_name:
        return HEAD_YAW, HEAD_PITCH
    layer_match = re.search(r"L([123])", slot_name)
    column_match = re.search(r"C([123])", slot_name)
    pitch = SLOT_LAYER_PITCH.get(int(layer_match.group(1)), HEAD_PITCH) if layer_match else HEAD_PITCH
    yaw = SLOT_COLUMN_YAW.get(int(column_match.group(1)), HEAD_YAW) if column_match else HEAD_YAW
    return float(yaw), float(pitch)


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


class GraspSession:
    """Official cola pick, then optional table place. No JSON product xyz."""

    def __init__(
        self,
        ik_fn: IkFn,
        kind: str = DEFAULT_KIND,
        now: float = 0.0,
        dt: float = 1.0 / RATE_HZ,
    ):
        self.ik_fn = ik_fn
        self.kind = str(kind)
        self.dt = float(dt)
        self.phase = GraspPhase.INIT
        self.reason = ""
        self.slot_hint: Optional[str] = None
        self.lock_attempt = 1
        self.used_json_xyz = False
        self.samples: Deque[np.ndarray] = deque(maxlen=30)
        self.locked: Optional[LockedTarget] = None
        self.place_deploy_world: Optional[np.ndarray] = None
        self.place_lower_world: Optional[np.ndarray] = None
        self.arm_target_set = False
        self.phase_t0 = float(now)
        self.session_t0 = float(now)
        self.last_log = ""
        self.tc = _pack(
            SLIDE_GRASP, HEAD_YAW, HEAD_PITCH,
            INIT_ARM_L, GRIP_OPEN, INIT_ARM_R, GRIP_OPEN,
        )
        self.action = self.tc.copy()
        self.tc_prev = self.tc.copy()
        self.move_ratio = np.ones(17)
        self.des_lin = 0.0
        self.des_ang = 0.0
        self.cur_lin = 0.0
        self.cur_ang = 0.0

    def note_slot_hint(self, slot_name: Optional[str]) -> None:
        self.slot_hint = slot_name or None

    def begin_place(self, now: float, table_xyz=DELIVERY_TARGET_XYZ) -> None:
        if self.phase not in (GraspPhase.HELD, GraspPhase.RETREAT):
            self.phase = GraspPhase.FAILED
            self.reason = "place_without_grasp"
            return
        self.place_deploy_world, self.place_lower_world = place_world_targets(table_xyz)
        self.arm_target_set = False
        self.phase = GraspPhase.PLACE_DEPLOY
        self.phase_t0 = float(now)
        self.session_t0 = float(now)
        self.reason = ""

    def tick(self, obs: GraspObservation) -> GraspCommand:
        if self.phase == GraspPhase.FAILED:
            return self._command(obs, done=False, failed=True)
        if self.phase == GraspPhase.DONE:
            return self._command(obs, done=True)
        if self.phase == GraspPhase.HELD:
            return self._command(obs, done=True)

        timeout = PLACE_TIMEOUT_S if self.phase.value.startswith("place") else GRASP_TIMEOUT_S
        if obs.t - self.session_t0 > timeout and self.phase not in {
            GraspPhase.HELD, GraspPhase.DONE, GraspPhase.FAILED,
        }:
            self._fail("grasp_timeout" if timeout == GRASP_TIMEOUT_S else "place_timeout")
            self._slew()
            self._ramp()
            return self._command(obs, failed=True)

        if self.phase == GraspPhase.INIT:
            self._set_twist(0.0, 0.0)
            self._set_joints(slide=SLIDE_GRASP, head_pitch=HEAD_PITCH, right_grip=GRIP_OPEN)
            if obs.t - self.phase_t0 >= INIT_DWELL_S:
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

    def _aim_and_lock(self, obs: GraspObservation) -> None:
        self._set_twist(0.0, 0.0)
        yaw, pitch = slot_head_hint(self.slot_hint) if self.lock_attempt > 1 else (HEAD_YAW, HEAD_PITCH)
        self._set_joints(slide=SLIDE_GRASP, head_yaw=yaw, head_pitch=pitch, right_grip=GRIP_OPEN)
        sample = pick_centered_detection(obs.detections_world, obs.base_xy, obs.base_yaw)
        if sample is not None:
            self.samples.append(sample)
        if obs.t - self.phase_t0 < DETECT_DWELL_S:
            return
        locked = lock_from_samples(list(self.samples), obs.base_xy, obs.base_yaw)
        if locked is None:
            if obs.t - self.phase_t0 > LOCK_TIMEOUT_S:
                if self.lock_attempt < LOCK_ATTEMPTS:
                    self.lock_attempt += 1
                    self.samples.clear()
                    self.phase_t0 = obs.t
                    self.last_log = (
                        f"lock retry {self.lock_attempt}/{LOCK_ATTEMPTS} "
                        f"slot_hint={self.slot_hint} (no json xyz)"
                    )
                else:
                    self._fail("lock_timeout")
            return
        if not self._arm_to(locked.deploy_world, obs, grip=GRIP_OPEN):
            self.samples.clear()
            if self.lock_attempt < LOCK_ATTEMPTS:
                self.lock_attempt += 1
                self.phase_t0 = obs.t
                self.last_log = "ik_unreachable on deploy; retry lock"
                return
            self._fail("ik_unreachable")
            return
        self.locked = locked
        self._enter(GraspPhase.DEPLOY, obs.t)
        self.last_log = (
            f"locked kind={self.kind} samples={locked.sample_count} "
            f"obj={np.round(locked.object_world, 3).tolist()}"
        )

    def _deploy(self, obs: GraspObservation) -> None:
        self._set_twist(0.0, 0.0)
        if self.locked is not None:
            self._arm_to(self.locked.deploy_world, obs, grip=GRIP_OPEN)
        if self._deploy_done(obs, self.locked.deploy_world if self.locked else None):
            self._enter(GraspPhase.CREEP, obs.t)

    def _creep(self, obs: GraspObservation) -> None:
        stop_y = self.locked.creep_stop_y if self.locked is not None else obs.base_xy[1]
        ee = obs.ee_world if obs.ee_world is not None else np.array(
            [obs.base_xy[0], obs.base_xy[1], 0.0]
        )
        if ee[1] < stop_y:
            yaw_err = wrap_to_pi(GRASP_YAW - obs.base_yaw)
            self._set_twist(CREEP_SPEED, CREEP_YAW_KP * yaw_err)
        else:
            self._set_twist(0.0, 0.0)
            self._enter(GraspPhase.CLOSE, obs.t)

    def _close(self, obs: GraspObservation) -> None:
        self._set_twist(0.0, 0.0)
        self._set_joints(right_grip=GRIP_CLOSE)
        if obs.t - self.phase_t0 > CLOSE_HOLD_S:
            self._enter(GraspPhase.LIFT, obs.t)

    def _lift(self, obs: GraspObservation) -> None:
        self._set_twist(0.0, 0.0)
        self._set_joints(slide=SLIDE_GRASP - LIFT_AMOUNT, right_grip=GRIP_CLOSE)
        if abs(obs.slide - (SLIDE_GRASP - LIFT_AMOUNT)) < 0.02:
            self._enter(GraspPhase.RETREAT, obs.t)

    def _retreat(self, obs: GraspObservation) -> None:
        yaw_err = wrap_to_pi(GRASP_YAW - obs.base_yaw)
        if obs.base_xy[1] > RETREAT_Y + POS_TOL:
            self._set_twist(-RETREAT_SPEED, 1.0 * yaw_err)
        else:
            self._set_twist(0.0, 0.0)
            self.phase = GraspPhase.HELD
            self.reason = "held"
            self.last_log = "retreat done; grasp retained"

    def _place_move(self, obs: GraspObservation, world_pos, next_phase: GraspPhase) -> None:
        self._set_twist(0.0, 0.0)
        if world_pos is None:
            self._fail("place_without_grasp")
            return
        grip = GRIP_CLOSE if self.phase != GraspPhase.PLACE_OPEN else GRIP_OPEN
        if not self.arm_target_set:
            if not self._arm_to(world_pos, obs, grip=grip):
                self._fail("place_ik_unreachable")
                return
        if self._deploy_done(obs, world_pos, cart_tol=PLACE_CART_TOL):
            self.arm_target_set = False
            self._enter(next_phase, obs.t)

    def _place_open(self, obs: GraspObservation) -> None:
        self._set_twist(0.0, 0.0)
        self._set_joints(right_grip=GRIP_OPEN)
        if obs.t - self.phase_t0 > PLACE_OPEN_HOLD_S:
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
            if not self._arm_to(retract_world, obs, grip=GRIP_OPEN):
                self._set_joints(right_arm=INIT_ARM_R, right_grip=GRIP_OPEN)
                self.arm_target_set = True
        if obs.t - self.phase_t0 > 0.80:
            self.phase = GraspPhase.DONE
            self.reason = "placed"
            self.last_log = "place sequence complete (not Server referee)"

    def _arm_to(self, world_pos, obs: GraspObservation, grip: float) -> bool:
        footprint = world_to_footprint(world_pos, obs.base_xy, obs.base_yaw)
        joints = self.ik_fn(footprint, GRASP_ROT, float(self.tc[0]), np.asarray(obs.right_arm, dtype=float))
        if joints is None:
            return False
        self._set_joints(right_arm=joints, right_grip=grip)
        self.arm_target_set = True
        return True

    def _deploy_done(self, obs: GraspObservation, world_pos, cart_tol: float = DEPLOY_CART_TOL) -> bool:
        if obs.t - self.phase_t0 < DEPLOY_DWELL_S:
            return False
        slide_err = abs(obs.slide - float(self.tc[0]))
        joint_err = (
            float(np.max(np.abs(obs.right_arm - self.tc[10:16])))
            if self.arm_target_set else float("inf")
        )
        cart_err = float("inf")
        rot_err = float("inf")
        if world_pos is not None and obs.ee_world is not None:
            cart_err = float(np.linalg.norm(np.asarray(obs.ee_world) - np.asarray(world_pos)))
        if obs.ee_rot_footprint is not None:
            rot_err = rotation_error(obs.ee_rot_footprint)
        ready = slide_err < DEPLOY_SLIDE_TOL and (
            joint_err < DEPLOY_JOINT_TOL
            or (cart_err < cart_tol and rot_err < DEPLOY_ROT_TOL)
        )
        if not ready:
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

    def _slew(self) -> None:
        if not np.allclose(self.tc, self.tc_prev):
            delta = np.abs(self.action - self.tc)
            self.move_ratio = delta / (np.max(delta) + 1e-6)
            self.move_ratio[0] *= 0.3
            self.tc_prev = self.tc.copy()
        step = JOINT_SLEW * self.dt
        for index in range(17):
            self.action[index] = step_func(self.action[index], self.tc[index], self.move_ratio[index] * step)

    def _ramp(self) -> None:
        self.cur_lin += float(np.clip(self.des_lin - self.cur_lin, -MAX_LIN_ACC * self.dt, MAX_LIN_ACC * self.dt))
        self.cur_ang += float(np.clip(self.des_ang - self.cur_ang, -MAX_ANG_ACC * self.dt, MAX_ANG_ACC * self.dt))

    def _enter(self, phase: GraspPhase, now: float) -> None:
        self.phase = phase
        self.phase_t0 = float(now)
        if phase == GraspPhase.DEPLOY:
            self.arm_target_set = True

    def fail(self, reason: str) -> None:
        self.phase = GraspPhase.FAILED
        self.reason = reason
        self._set_twist(0.0, 0.0)
        self.last_log = reason

    def _fail(self, reason: str) -> None:
        self.fail(reason)

    def _command(self, _obs: GraspObservation, done: bool = False, failed: bool = False) -> GraspCommand:
        linear, angular = clip_base(self.cur_lin, self.cur_ang)
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
            done=done or self.phase in (GraspPhase.HELD, GraspPhase.DONE),
            failed=failed or self.phase == GraspPhase.FAILED,
            reason=self.reason,
            log=self.last_log,
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
            "creep_stop_y": None if self.locked is None else round(self.locked.creep_stop_y, 4),
            "place_target_in_box": place_ok,
            "reason": self.reason,
        }
