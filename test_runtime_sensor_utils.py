import unittest
from types import SimpleNamespace

import numpy as np

from runtime.ros_sensor_utils import SensorCache, SensorDataError, TransformStore, decode_depth, decode_image


def image_message(array, encoding, stamp=1.0, frame="camera"):
    array = np.asarray(array)
    if encoding == "16UC1":
        payload = array.astype("<u2").tobytes()
        step = array.shape[1] * 2
    elif encoding == "32FC1":
        payload = array.astype("<f4").tobytes()
        step = array.shape[1] * 4
    else:
        payload = array.astype(np.uint8).tobytes()
        step = array.shape[1] * array.shape[2] if array.ndim == 3 else array.shape[1]
    sec = int(stamp)
    header = SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=int((stamp - sec) * 1e9)), frame_id=frame)
    return SimpleNamespace(height=array.shape[0], width=array.shape[1], step=step, data=payload, encoding=encoding, is_bigendian=False, header=header)


class SensorUtilityTests(unittest.TestCase):
    def test_bgr_is_normalized_to_rgb_and_depth_is_meters(self):
        rgb = decode_image(image_message(np.array([[[3, 2, 1]]]), "bgr8"))
        np.testing.assert_array_equal(rgb, [[[1, 2, 3]]])
        depth = decode_depth(image_message(np.array([[1000]], dtype=np.uint16), "16UC1"))
        self.assertAlmostEqual(float(depth[0, 0]), 1.0)

    def test_transform_store_finds_multihop_path(self):
        store = TransformStore()
        first = np.eye(4)
        first[0, 3] = 1.0
        second = np.eye(4)
        second[1, 3] = 2.0
        store.set_transform("base_link", "camera", first)
        store.set_transform("odom", "base_link", second)
        np.testing.assert_allclose(store.lookup("odom", "camera")[:3, 3], [1.0, 2.0, 0.0])

    def test_joint_state_rejects_invalid_feedback_without_overwriting_cache(self):
        cache = SensorCache()
        valid = SimpleNamespace(name=["joint"], position=[0.25])
        cache.update_joint_state(valid)
        with self.assertRaises(SensorDataError):
            cache.update_joint_state(SimpleNamespace(name=["joint"], position=[np.nan]))
        np.testing.assert_allclose(cache.joint_vector(["joint"]), [0.25])

    def test_joint_history_and_rgbd_ids(self):
        cache = SensorCache()
        joints = SimpleNamespace(
            name=["slide_joint"],
            position=[0.11],
            header=SimpleNamespace(stamp=SimpleNamespace(sec=2, nanosec=0), frame_id=""),
        )
        cache.update_joint_state(joints)
        self.assertAlmostEqual(cache.joint_stamp, 2.0)
        self.assertEqual(len(cache.joint_history), 1)
        rgb = image_message(np.zeros((2, 2, 3), dtype=np.uint8), "rgb8", stamp=3.0)
        depth = image_message(np.ones((2, 2), dtype=np.float32), "32FC1", stamp=3.02)
        cache.update_rgb(rgb)
        cache.update_depth(depth)
        self.assertEqual(cache.rgb_id, 1)
        cache.update_rgb(rgb)
        self.assertEqual(cache.rgb_id, 1)
        self.assertAlmostEqual(cache.rgbd_skew(), 0.02, places=5)
        self.assertFalse(cache.rgbd_ready(max_skew=0.05, max_age=5.0))

    def test_scan_id_does_not_advance_on_the_same_stamp(self):
        cache = SensorCache()
        header = SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0), frame_id="laser")
        scan = SimpleNamespace(
            header=header,
            ranges=(1.0, 2.0, 3.0),
            angle_min=-3.14,
            angle_increment=0.1,
            range_min=0.05,
            range_max=8.0,
            time_increment=0.0,
        )
        cache.update_scan(scan)
        first = cache.scan_id
        cache.update_scan(scan)
        self.assertEqual(cache.scan_id, first)
        header2 = SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=50000000), frame_id="laser")
        scan2 = SimpleNamespace(
            header=header2,
            ranges=(1.0, 2.0, 3.0),
            angle_min=-3.14,
            angle_increment=0.1,
            range_min=0.05,
            range_max=8.0,
            time_increment=0.0,
        )
        cache.update_scan(scan2)
        self.assertEqual(cache.scan_id, first + 1)


if __name__ == "__main__":
    unittest.main()
