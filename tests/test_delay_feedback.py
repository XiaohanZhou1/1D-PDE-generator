from fractions import Fraction
import unittest

import numpy as np

from pdegen.compare_a1_d2 import d2_batch
from pdegen.delay_feedback import (laplace_average, shifted_delay_kink, solve_shifted_delay,
                                  validate_shifted_delay, verify_root)


class ShiftedDelayTests(unittest.TestCase):
    def test_frozen_grid_and_validation(self):
        cases = d2_batch(0)
        self.assertEqual(len({item["specification_hash"] for item in cases}), 6)
        self.assertEqual(sum(item["calibration"] for item in cases), 2)
        self.assertEqual(validate_shifted_delay(cases[0]), cases[0])
        with self.assertRaises(ValueError):
            validate_shifted_delay({**cases[0], "target_wave": {"speed": 1}})

    def test_shooting_matches_symmetric_characteristic(self):
        root = verify_root(0.32 + 2.05j, 5, 1, 0)[-1]
        lam = complex(*root["root"])
        depth = np.sqrt(2 * (1 + lam))
        characteristic = 5 * laplace_average(lam, 1) + depth * (depth + 1) / 2
        self.assertLess(abs(characteristic), 1e-8)
        self.assertGreater(lam.real, 0.3)
        self.assertGreater(lam.imag, 2.0)

    def test_asymmetric_stable_label_has_a_closed_verified_contour(self):
        item = shifted_delay_kink(2, "3/2", "1/4", 0)
        self.assertFalse(item["calibration"])

        def analyse(*args, **kwargs):
            return dict(label="STABLE", wave=dict(computed=True, left=[-1, 0], right=[1, 0],
                                                 speed=0, profile_error=0))

        result = solve_shifted_delay(item, analyse)
        self.assertEqual((result["label"], result["evidence_grade"], result["open_rhp_count"]),
                         ("STABLE", "RIGOROUSLY_CERTIFIED", 0))
        cert = result["certificate"]
        self.assertGreater(Fraction(cert["outside_resolvent_margin"]), 0)
        self.assertGreater(Fraction(cert["contour_min_zero_margin"]), 0)
        segments = cert["segments"]
        self.assertEqual(len(segments), cert["segment_count"])
        for segment, following in zip(segments, [*segments[1:], segments[0]], strict=True):
            self.assertEqual(segment["end"], following["start"])
            self.assertEqual(segment["polygon_end"], following["polygon_start"])
            self.assertTrue(any(lo > 0 or hi < 0 for lo, hi in segment["image_box"]))
