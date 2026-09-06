import math
import unittest

from runtime.grid_planner import STOP_DIST as GRID_STOP
from runtime.p4_params import (
    CENTER_WALL_ZONE,
    DELIVERY_APPROACH_XY,
    DELIVERY_PLACE_ZONE,
    DELIVERY_WORK_ZONE,
    EAST_CORRIDOR_ZONE,
    NAV_BOUNDS,
    PICKING_AISLE_ZONE,
    SHELF_APPROACH_XY,
    SHELF_BODY_ZONE,
    SPAWN_XY,
    SPAWN_YAW,
    SPAWN_ZONE,
    STOP_DIST,
    WEST_CORRIDOR_ZONE,
    as_dict,
    in_east_corridor,
    in_shelf_body,
    in_spawn_zone,
    in_west_corridor,
    point_in_zone,
)
from runtime.p4_params import SPEED_SCALE


class P4ParamsTests(unittest.TestCase):
    def test_spawn_sits_in_the_named_start_box(self):
        self.assertTrue(in_spawn_zone(SPAWN_XY))
        self.assertTrue(point_in_zone(SPAWN_XY, NAV_BOUNDS))
        self.assertAlmostEqual(SPAWN_YAW, math.pi / 2.0)

    def test_shelf_and_delivery_boxes_have_area(self):
        for zone in (
            SHELF_BODY_ZONE,
            PICKING_AISLE_ZONE,
            SPAWN_ZONE,
            EAST_CORRIDOR_ZONE,
            WEST_CORRIDOR_ZONE,
            CENTER_WALL_ZONE,
            DELIVERY_WORK_ZONE,
            DELIVERY_PLACE_ZONE,
        ):
            self.assertLess(zone["x"][0], zone["x"][1])
            self.assertLess(zone["y"][0], zone["y"][1])

    def test_shelf_body_is_north_of_the_aisle(self):
        self.assertTrue(in_shelf_body(0.85, 2.60))
        self.assertFalse(in_shelf_body(*SHELF_APPROACH_XY))

    def test_corridors_do_not_cover_the_center_wall(self):
        self.assertFalse(in_east_corridor(0.80, 0.00))
        self.assertFalse(in_west_corridor(0.80, 0.00))
        self.assertTrue(in_east_corridor(1.92, -1.00))
        self.assertTrue(in_west_corridor(-1.20, 0.00))

    def test_delivery_place_is_inside_the_work_zone(self):
        self.assertTrue(point_in_zone(DELIVERY_APPROACH_XY, DELIVERY_WORK_ZONE) or DELIVERY_APPROACH_XY[1] > -3.0)
        mid = (
            0.5 * (DELIVERY_PLACE_ZONE["x"][0] + DELIVERY_PLACE_ZONE["x"][1]),
            0.5 * (DELIVERY_PLACE_ZONE["y"][0] + DELIVERY_PLACE_ZONE["y"][1]),
        )
        self.assertTrue(point_in_zone(mid, DELIVERY_WORK_ZONE))

    def test_planner_and_runner_read_the_same_file(self):
        self.assertEqual(GRID_STOP, STOP_DIST)
        self.assertGreater(SPEED_SCALE, 0.0)
        snap = as_dict()
        self.assertEqual(snap["spawn_xy"], list(SPAWN_XY))
        self.assertEqual(snap["stop_dist"], STOP_DIST)


if __name__ == "__main__":
    unittest.main()
