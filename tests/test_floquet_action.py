import copy
import json
import unittest

import numpy as np

from pdegen.floquet_action import SampledPeriodMap, capture_orbit, verify_action


class SampledPeriodMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads(json.dumps(capture_orbit()))

    def test_sampled_action_against_dense_exact_orbit(self):
        result = verify_action(self.record)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["action_checks"]), 9)
        self.assertLess(result["orbit_checks"]["sampled_pde_residual_max"], 1e-7)
        self.assertLess(result["phase_action_error"], 1e-8)
        self.assertLess(max(row["error_per_input_norm"] for row in result["action_checks"]), 1e-8)

    def test_wrong_phase_open_orbit_and_interior_change_are_rejected(self):
        changed = copy.deepcopy(self.record)
        changed["samples"] = changed["samples"][1:] + changed["samples"][:1]
        changed["integrated_endpoint"] = changed["samples"][0]
        with self.assertRaisesRegex(ValueError, "phase or"):
            SampledPeriodMap(changed)

        changed = copy.deepcopy(self.record)
        changed["integrated_endpoint"][0][0] += 1e-3
        with self.assertRaisesRegex(ValueError, "closure"):
            SampledPeriodMap(changed)

        changed = copy.deepcopy(self.record)
        changed["samples"][5][0][1] += 1e-3
        with self.assertRaisesRegex(ValueError, "PDE residual"):
            SampledPeriodMap(changed)

    def test_vector_and_grid_contract(self):
        operator = SampledPeriodMap(self.record)
        with self.assertRaisesRegex(ValueError, "envelope vector"):
            operator.apply(0, 2, np.zeros(9))
        with self.assertRaisesRegex(ValueError, "Bloch"):
            operator.apply(1.0, 2, np.zeros(10))
