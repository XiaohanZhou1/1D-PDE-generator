import math
import unittest

from pdegen.g4_pilot import eigenvalue_probe, review_existing


class ProofReadinessTests(unittest.TestCase):
    def test_frozen_evidence_keeps_all_cases_on_hold(self):
        report = review_existing()
        self.assertEqual(report["cases"], 22)
        self.assertEqual(report["distinct_specifications"], 22)
        self.assertEqual(report["findings"]["ANALYTIC_CALIBRATION"], 10)
        self.assertEqual(report["findings"]["NUMERICAL_EVIDENCE_ONLY"], 6)
        self.assertEqual(report["proof_training_admitted"], 0)

    def test_symmetric_probe_matches_exact_spatial_levels(self):
        result = eigenvalue_probe(0.0, 20, 4000)
        expected_levels = (3.5, 1.0, -0.5)
        for observed, expected in zip(result["top_three_at_zero"], expected_levels, strict=True):
            self.assertAlmostEqual(observed, expected, places=4)
        for n, lam in enumerate(result["positive_roots"]):
            depth = -math.expm1(-lam)/(2*lam) - math.expm1(-2*lam)/(4*lam)
            shape = (math.sqrt(1 + 48*depth) - 1)/2
            self.assertAlmostEqual(lam, (shape - n)**2/2 - 1, places=4)
