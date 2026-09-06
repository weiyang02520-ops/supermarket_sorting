import math
import unittest

from runtime.nav_geometry import (
    GEOMETRY_VERSION,
    R_PLAN_REF_M,
    catalog_item,
    forbidden_intrusion_m,
    geometry_catalog,
    in_center_forbidden,
    in_physical_solid,
    in_south_gap,
    preferred_corridor_line,
)
from runtime.nav_types import infer_goal_kind
from runtime.scene_zones import (
    DELIVERY_APPROACH_XY,
    NORTH_RACK_Y,
    SHELF_APPROACH_XY,
    SPAWN_XY,
)


class NavGeometryTests(unittest.TestCase):
    def test_catalog_splits_solids_bans_and_soft_cost(self):
        cat = geometry_catalog()
        self.assertEqual(cat["geometry_version"], GEOMETRY_VERSION)
        kinds = {item["name"]: item["kind"] for item in cat["physical_solids"]}
        self.assertEqual(kinds["center_wall"], "physical_solid")
        bans = {item["name"]: item["inflate"] for item in cat["center_forbidden"]}
        self.assertFalse(bans["north_racks"])
        self.assertFalse(bans["south_gap"])
        self.assertIn("actual chassis collision radius", cat["unverified"])
        self.assertIn("T_base_laser", cat["unverified"])
        self.assertEqual(cat["chassis"]["status"], "UNVERIFIED")
        self.assertEqual(cat["laser"]["status"], "TF_REQUIRED")
        self.assertFalse(cat["scan_no_echo"]["inf_is_free"])

    def test_shelf_approach_is_not_inside_the_rack_ban(self):
        self.assertLess(SHELF_APPROACH_XY[1], NORTH_RACK_Y)
        self.assertAlmostEqual(NORTH_RACK_Y - SHELF_APPROACH_XY[1], 0.005, places=3)
        self.assertFalse(in_center_forbidden(*SHELF_APPROACH_XY))
        self.assertTrue(in_center_forbidden(0.85, 2.50))
        self.assertFalse(in_physical_solid(*SHELF_APPROACH_XY))
        rack = catalog_item("north_racks")
        self.assertEqual(rack["kind"], "center_forbidden")
        self.assertIn("5 mm", rack["note"])

    def test_south_gap_is_bidirectional_and_leaves_corridors(self):
        self.assertTrue(in_south_gap(0.80, -3.30))
        self.assertFalse(in_south_gap(-1.20, -2.90))
        self.assertFalse(in_south_gap(1.92, -1.00))
        self.assertGreater(forbidden_intrusion_m(0.85, 2.50), 0.0)
        self.assertEqual(forbidden_intrusion_m(*SPAWN_XY), 0.0)

    def test_goal_kind_does_not_use_sign_of_x(self):
        self.assertEqual(infer_goal_kind(SHELF_APPROACH_XY), "shelf_approach")
        self.assertEqual(infer_goal_kind(DELIVERY_APPROACH_XY), "delivery_approach")
        self.assertEqual(infer_goal_kind((-1.20, 2.40)), "inspection_approach")
        west_line = preferred_corridor_line("delivery_approach")
        self.assertTrue(any(px < 0.0 for px, _py in west_line))

    def test_plan_radius_is_marked_unverified(self):
        self.assertAlmostEqual(R_PLAN_REF_M, 0.30)
        self.assertEqual(geometry_catalog()["chassis"]["r_plan_m"], R_PLAN_REF_M)


if __name__ == "__main__":
    unittest.main()
