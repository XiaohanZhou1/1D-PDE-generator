import unittest

import numpy as np

from pdegen.followup_pilot import D2_POINTS
from pdegen.compare_a1_d2 import d2_batch
from pdegen.delay_feedback import shifted_delay_kink
from pdegen.interface_feedback import boundary_transfer, interface_wave, validate_interface_wave


class FollowupPilotTests(unittest.TestCase):
    def test_d2_parameters_are_fresh(self):
        previous = {row["specification_hash"] for row in d2_batch(0)}
        new = {shifted_delay_kink(*point, 0)["specification_hash"] for point in D2_POINTS}
        self.assertEqual(len(new), 4)
        self.assertTrue(previous.isdisjoint(new))

    def test_transfer_decoupled_limit(self):
        beta, lam, speed = -8, 0.2, 1.5
        mu = (-speed + np.sqrt(speed**2 + 4 * (1 + lam))) / 2
        nu = (-speed + np.sqrt(speed**2 + 4 * (2.5 + lam))) / 2
        expected = -1.5 / (mu + 0.5 + speed) - beta / (nu + 1 + speed) - 1
        self.assertAlmostEqual(boundary_transfer(lam, 0, beta, 30, 1e-9), expected, places=7)

    def test_e2_contract(self):
        item = interface_wave(2, -4, 1)
        self.assertEqual(validate_interface_wave(item), item)
        with self.assertRaises(ValueError):
            validate_interface_wave({**item, "target_wave": {"profile": "unknown"}})
