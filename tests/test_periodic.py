import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from pdegen.audit import audit_batch
from pdegen.periodic import periodic_batch, periodic_roll, solve_periodic_roll, validate_periodic_roll


class PeriodicTests(unittest.TestCase):
    def test_canonical_roll(self):
        item = periodic_roll("22/20", "2/10", 1)
        self.assertEqual(validate_periodic_roll(item), item)
        self.assertEqual(item["specification_hash"], periodic_roll("11/10", "1/5", 2)["specification_hash"])
        self.assertEqual(item["validity_checks"]["profile"], "UNVERIFIED")
        with self.assertRaises(ValueError):
            validate_periodic_roll({**item, "target_wave": {**item["target_wave"], "speed": 1}})
        with self.assertRaises(ValueError):
            periodic_roll("3/2", "1/5", 1)

    def test_profile_and_bloch_workflow(self):
        items = periodic_batch(1)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            results = []

            def solve(item):
                result = solve_periodic_roll(item)
                results.append(result)
                return result

            manifest = audit_batch(items, destination, validate_periodic_roll, solve)
            self.assertEqual([row["label"] for row in manifest], ["UNCERTAIN", "UNSTABLE", "UNCERTAIN"])
            self.assertTrue(all(row["open_rhp_count"] is None for row in manifest))
            self.assertTrue(all((destination / row["result"]).is_file() for row in manifest))

        for item, result in zip(items, results, strict=True):
            profile, bloch = result["profile"], result["bloch"]
            self.assertTrue(profile["checks_passed"])
            self.assertLess(profile["fine"]["residual_max"], 1e-10)
            self.assertLess(profile["fine"]["phase_residual_max"], 1e-9)
            self.assertEqual(len(profile["fine"]["steps"]), 16)
            self.assertNotEqual(profile["fine"]["cosine_coefficients"][1], 0.0)
            q = float(Fraction(item["parameters"]["q"]))
            self.assertEqual((bloch["fine_fibers"][0], bloch["fine_fibers"][-1]), (-q / 2, q / 2))
            self.assertTrue(bloch["simple_phase_observed"])
            self.assertLess(bloch["coperiodic_top_three"][1], -1e-3)
            self.assertFalse(bloch["continuous_zone_excluded"])
        unstable = results[1]["bloch"]
        self.assertGreater(unstable["peak_xi"], 0)
        self.assertGreater(unstable["peak_growth"], 1e-2)
        self.assertTrue(unstable["physical_check"]["confirmed"])
        self.assertGreater(unstable["phase_band_growth_over_xi_squared"][-1], 0)
        self.assertLess(results[2]["bloch"]["phase_band_growth_over_xi_squared"][-1], 0)
