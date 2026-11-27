from fractions import Fraction
import unittest

from pdegen.dynamic_interface import (continuous_trace, dynamic_interface, fd_trace,
                                      small_gain_certificate, solve_dynamic_interface,
                                      stability_certificate, validate_dynamic_interface)


class DynamicInterfaceTests(unittest.TestCase):
    def test_whole_domain_bounds_and_contract(self):
        self.assertIsNotNone(small_gain_certificate(Fraction(1, 2)))
        self.assertIsNone(small_gain_certificate(Fraction(4)))
        self.assertEqual(stability_certificate(Fraction(4))["method"], "positive bulk resolvent and trace moment")
        self.assertIsNone(stability_certificate(Fraction(5)))
        item = dynamic_interface(4, 1)
        self.assertEqual(validate_dynamic_interface(item), item)
        with self.assertRaises(ValueError):
            validate_dynamic_interface({**item, "target_wave": {"profile": "other"}})

    def test_independent_trace_and_oscillatory_root(self):
        lam = 0.03 + 2.13j
        self.assertLess(abs(fd_trace(lam, 30, 4500) - continuous_trace(lam, 24, 1e-11)), 1e-4)
        result = solve_dynamic_interface(dynamic_interface(8, 1))
        self.assertEqual(result["label"], "UNSTABLE")
        self.assertGreater(result["witness"]["positive_margin"], 0.03)
        self.assertEqual(solve_dynamic_interface(dynamic_interface(5, 1))["label"], "UNCERTAIN")
