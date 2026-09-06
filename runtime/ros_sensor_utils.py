"""ROS-message decoding, synchronized sensor storage, and TF geometry.

This module intentionally does not import ROS.  It can therefore be tested on
the host and used with either rclpy messages or small message-shaped fakes.
"""

from collections import deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from runtime.nav_types import PoseSample, ScanSample
from runtime.waypoint_nav import yaw_from_quaternion


class SensorDataError(RuntimeError):
    pass


def message_stamp(message) -> float:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return time.monotonic()
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9


def message_frame(message) -> str:
    return str(getattr(getattr(message, "header", None), "frame_id", "") or "").lstrip("/")


def _native_array(data, dtype, is_bigendian: bool):
    wire = np.dtype(dtype).newbyteorder(">" if is_bigendian else "<")
    return np.frombuffer(bytes(data), dtype=wire).astype(dtype, copy=False)


def decode_image(message) -> np.ndarray:
    """Decode common sensor_msgs/Image encodings while respecting row stride."""
    encoding = str(message.encoding).lower()
    formats = {
        "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3),
        "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4),
        "mono8": (np.uint8, 1), "8uc1": (np.uint8, 1),
        "16uc1": (np.uint16, 1), "mono16": (np.uint16, 1),
        "32fc1": (np.float32, 1),
    }
    if encoding not in formats:
        raise SensorDataError(f"unsupported image encoding: {message.encoding!r}")
    dtype, channels = formats[encoding]
    height, width, step = int(message.height), int(message.width), int(message.step)
    itemsize = np.dtype(dtype).itemsize
    required = width * channels * itemsize
    if height <= 0 or width <= 0 or step < required:
        raise SensorDataError(f"invalid image dimensions/step: {width}x{height}, step={step}")
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if raw.size < height * step:
        raise SensorDataError(f"truncated image payload: {raw.size} < {height * step}")
    rows = raw[:height * step].reshape(height, step)[:, :required].copy()
    values = _native_array(rows.tobytes(), dtype, bool(getattr(message, "is_bigendian", False)))
    shape = (height, width, channels) if channels > 1 else (height, width)
    image = values.reshape(shape)
    if encoding in ("bgr8", "bgra8"):
        image = image[..., [2, 1, 0] + ([3] if channels == 4 else [])]
    if channels == 4:
        image = image[..., :3]
    return image


def decode_depth(message) -> np.ndarray:
    depth = decode_image(message).astype(np.float32, copy=True)
    encoding = str(message.encoding).lower()
    if encoding in ("16uc1", "mono16"):
        depth = depth * 0.001
    if encoding not in ("16uc1", "mono16", "32fc1"):
        raise SensorDataError(f"unsupported depth encoding: {message.encoding!r}")
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    frame_id: str = ""

    @classmethod
    def from_message(cls, message):
        k = list(message.k)
        if len(k) != 9 or k[0] <= 0 or k[4] <= 0:
            raise SensorDataError("CameraInfo.k must contain valid fx/fy values")
        return cls(int(message.width), int(message.height), float(k[0]), float(k[4]), float(k[2]), float(k[5]), message_frame(message))

    def project_pixel(self, u: float, v: float, depth_m: float) -> np.ndarray:
        z = float(depth_m)
        if not math.isfinite(z) or z <= 0:
            raise SensorDataError(f"invalid depth: {depth_m!r}")
        return np.array([(float(u) - self.cx) * z / self.fx, (float(v) - self.cy) * z / self.fy, z])


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    q = np.asarray([x, y, z, w], dtype=float)
    norm = float(q @ q)
    if norm < 1e-16:
        raise SensorDataError("zero-length quaternion")
    q *= math.sqrt(2.0 / norm)
    outer = np.outer(q, q)
    return np.array([
        [1 - outer[1, 1] - outer[2, 2], outer[0, 1] - outer[2, 3], outer[0, 2] + outer[1, 3]],
        [outer[0, 1] + outer[2, 3], 1 - outer[0, 0] - outer[2, 2], outer[1, 2] - outer[0, 3]],
        [outer[0, 2] - outer[1, 3], outer[1, 2] + outer[0, 3], 1 - outer[0, 0] - outer[1, 1]],
    ])


def transform_matrix(translation, rotation) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_matrix(rotation.x, rotation.y, rotation.z, rotation.w)
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


class TransformStore:
    """Latest-value TF graph supporting direct and multi-hop lookup."""

    def __init__(self):
        self._edges: Dict[Tuple[str, str], np.ndarray] = {}
        self._lock = threading.RLock()

    def update(self, tf_message) -> None:
        with self._lock:
            for stamped in tf_message.transforms:
                parent = message_frame(stamped)
                child = str(stamped.child_frame_id or "").lstrip("/")
                if not parent or not child:
                    continue
                matrix = transform_matrix(stamped.transform.translation, stamped.transform.rotation)
                self._edges[(child, parent)] = matrix
                self._edges[(parent, child)] = np.linalg.inv(matrix)

    def set_transform(self, parent: str, child: str, matrix) -> None:
        parent, child = parent.lstrip("/"), child.lstrip("/")
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("transform must be 4x4")
        with self._lock:
            self._edges[(child, parent)] = matrix.copy()
            self._edges[(parent, child)] = np.linalg.inv(matrix)

    def lookup(self, target: str, source: str) -> np.ndarray:
        """Return a matrix that maps a point in source into target."""
        target, source = target.lstrip("/"), source.lstrip("/")
        if target == source:
            return np.eye(4)
        with self._lock:
            edges = dict(self._edges)
        queue = deque([(source, np.eye(4))])
        visited = {source}
        while queue:
            frame, source_to_frame = queue.popleft()
            for (edge_source, edge_target), edge_matrix in edges.items():
                if edge_source != frame or edge_target in visited:
                    continue
                source_to_target = edge_matrix @ source_to_frame
                if edge_target == target:
                    return source_to_target
                visited.add(edge_target)
                queue.append((edge_target, source_to_target))
        raise SensorDataError(f"no TF path from {source!r} to {target!r}")


@dataclass(frozen=True)
class SensorSnapshot:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    rgb_stamp: float
    depth_stamp: float
    camera_frame: str


def _odom_speeds(message) -> Tuple[float, float]:
    twist = getattr(message, "twist", None)
    inner = getattr(twist, "twist", twist)
    linear = getattr(getattr(inner, "linear", None), "x", 0.0)
    angular = getattr(getattr(inner, "angular", None), "z", 0.0)
    try:
        return float(linear or 0.0), float(angular or 0.0)
    except (TypeError, ValueError):
        return 0.0, 0.0


class SensorCache:
    """Thread-safe latest sensor values with finite synchronized waits."""

    def __init__(self):
        self._condition = threading.Condition()
        self.rgb = self.depth = self.intrinsics = None
        self.rgb_stamp = self.depth_stamp = None
        self.rgb_received_at = self.depth_received_at = None
        self.rgb_frame = self.depth_frame = ""
        self.camera_frame = ""
        self.joint_names = ()
        self.joint_positions = np.empty(0)
        self.odom = None
        self.odom_received_at = None
        self.odom_history: deque = deque(maxlen=80)
        self.task_raw = None
        self.scan = None
        self.scan_received_at = None
        self.scan_id = 0
        self.last_scan_stamp = None
        self.last_scan_sample: Optional[ScanSample] = None
        self.last_pose_sample: Optional[PoseSample] = None

    def update_rgb(self, message) -> None:
        value = decode_image(message)
        with self._condition:
            self.rgb, self.rgb_stamp = value, message_stamp(message)
            self.rgb_received_at = time.monotonic()
            self.rgb_frame = message_frame(message) or self.rgb_frame
            self.camera_frame = self.rgb_frame or self.camera_frame
            self._condition.notify_all()

    def update_depth(self, message) -> None:
        value = decode_depth(message)
        with self._condition:
            self.depth, self.depth_stamp = value, message_stamp(message)
            self.depth_received_at = time.monotonic()
            self.depth_frame = message_frame(message) or self.depth_frame
            self.camera_frame = self.depth_frame or self.camera_frame
            self._condition.notify_all()

    def update_camera_info(self, message) -> None:
        value = CameraIntrinsics.from_message(message)
        with self._condition:
            self.intrinsics = value
            self.camera_frame = value.frame_id or self.camera_frame
            self._condition.notify_all()

    def update_joint_state(self, message) -> None:
        names = tuple(str(name) for name in message.name)
        positions = np.asarray(message.position, dtype=float)
        if len(names) != len(positions) or len(names) == 0 or not np.all(np.isfinite(positions)):
            raise SensorDataError("joint state names/positions are missing or non-finite")
        with self._condition:
            self.joint_names = names
            self.joint_positions = positions.copy()
            self._condition.notify_all()

    def update_odom(self, message) -> None:
        received = time.monotonic()
        stamp = message_stamp(message)
        lin, ang = _odom_speeds(message)
        pose = PoseSample(
            stamp_sim=stamp,
            received_wall=received,
            frame_id=message_frame(message) or "odom",
            x=float(message.pose.pose.position.x),
            y=float(message.pose.pose.position.y),
            yaw=yaw_from_quaternion(
                message.pose.pose.orientation.x,
                message.pose.pose.orientation.y,
                message.pose.pose.orientation.z,
                message.pose.pose.orientation.w,
            ),
            linear_speed=lin,
            angular_speed=ang,
            child_frame_id=str(getattr(message, "child_frame_id", "") or "base_link").lstrip("/"),
        )
        with self._condition:
            self.odom = message
            self.odom_received_at = received
            self.last_pose_sample = pose
            self.odom_history.append(pose)
            cutoff = received - 2.0
            while self.odom_history and self.odom_history[0].received_wall < cutoff:
                self.odom_history.popleft()
            self._condition.notify_all()

    def update_task(self, message) -> None:
        with self._condition:
            self.task_raw = getattr(message, "data", message)
            self._condition.notify_all()

    def update_scan(self, message) -> None:
        received = time.monotonic()
        stamp = message_stamp(message)
        with self._condition:
            if self.last_scan_stamp is not None and abs(stamp - self.last_scan_stamp) < 1e-9:
                self.scan = message
                self.scan_received_at = received
                self._condition.notify_all()
                return
            self.scan_id += 1
            self.last_scan_stamp = stamp
            self.scan = message
            self.scan_received_at = received
            ranges = tuple(float(v) for v in message.ranges)
            self.last_scan_sample = ScanSample(
                scan_id=self.scan_id,
                stamp_sim=stamp,
                received_wall=received,
                frame_id=message_frame(message) or "laser",
                ranges=ranges,
                angle_min=float(getattr(message, "angle_min", -math.pi)),
                angle_increment=float(getattr(message, "angle_increment", 0.0)),
                range_min=float(getattr(message, "range_min", 0.05)),
                range_max=float(getattr(message, "range_max", 8.0)),
                time_increment=float(getattr(message, "time_increment", 0.0)),
            )
            self._condition.notify_all()

    def pose_at(self, stamp_sim: float) -> Optional[PoseSample]:
        """Interpolate odom at a scan stamp. None if the gap is too large."""
        with self._condition:
            history = list(self.odom_history)
        if not history:
            return None
        best = min(history, key=lambda p: abs(p.stamp_sim - float(stamp_sim)))
        if abs(best.stamp_sim - float(stamp_sim)) > 0.10:
            return None
        before = [p for p in history if p.stamp_sim <= stamp_sim]
        after = [p for p in history if p.stamp_sim >= stamp_sim]
        if not before or not after:
            return best
        a, b = before[-1], after[0]
        if abs(b.stamp_sim - a.stamp_sim) < 1e-9:
            return b
        t = (float(stamp_sim) - a.stamp_sim) / (b.stamp_sim - a.stamp_sim)
        yaw = a.yaw + t * ((b.yaw - a.yaw + math.pi) % (2.0 * math.pi) - math.pi)
        return PoseSample(
            stamp_sim=float(stamp_sim),
            received_wall=a.received_wall + t * (b.received_wall - a.received_wall),
            frame_id=b.frame_id,
            x=a.x + t * (b.x - a.x),
            y=a.y + t * (b.y - a.y),
            yaw=yaw,
            linear_speed=a.linear_speed + t * (b.linear_speed - a.linear_speed),
            angular_speed=a.angular_speed + t * (b.angular_speed - a.angular_speed),
            child_frame_id=b.child_frame_id,
        )

    def wait_snapshot(self, timeout: float = 3.0, max_skew: float = 0.15) -> SensorSnapshot:
        deadline = time.monotonic() + float(timeout)
        with self._condition:
            while True:
                ready = self.rgb is not None and self.depth is not None and self.intrinsics is not None
                skew = abs(self.rgb_stamp - self.depth_stamp) if ready else float("inf")
                age = max(
                    time.monotonic() - self.rgb_received_at,
                    time.monotonic() - self.depth_received_at,
                ) if ready else float("inf")
                frames_match = not self.rgb_frame or not self.depth_frame or self.rgb_frame == self.depth_frame
                if ready and self.rgb.shape[:2] == self.depth.shape and skew <= max_skew and age <= 1.0 and frames_match:
                    frame = self.rgb_frame or self.depth_frame or self.camera_frame or self.intrinsics.frame_id
                    return SensorSnapshot(self.rgb.copy(), self.depth.copy(), self.intrinsics, self.rgb_stamp, self.depth_stamp, frame)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if not ready:
                        detail = "missing RGB/depth/CameraInfo"
                    elif not frames_match:
                        detail = f"RGB/depth frame mismatch: {self.rgb_frame!r} != {self.depth_frame!r}"
                    elif age > 1.0:
                        detail = f"sensor data is stale: age={age:.3f}s"
                    else:
                        detail = f"RGB-depth skew={skew:.3f}s"
                    raise TimeoutError(f"sensor snapshot timeout: {detail}")
                self._condition.wait(min(remaining, 0.1))

    def joint_vector(self, names: Iterable[str]) -> np.ndarray:
        with self._condition:
            values = dict(zip(self.joint_names, self.joint_positions))
        missing = [name for name in names if name not in values]
        if missing:
            raise SensorDataError(f"joint feedback missing: {missing}")
        return np.asarray([values[name] for name in names], dtype=float)
