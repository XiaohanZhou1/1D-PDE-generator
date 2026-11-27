from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pdegen.anisotropic_bloch import fourier_matrix, solve_roll
from pdegen.b3_certificate import fourier_coefficients
from pdegen.batch_workflow import BCF_PLAN, candidates, run
from pdegen.blind_prompt import render_problem
from pdegen.f3_certificate import SCALE, monodromy_data, product
from pdegen.hamiltonian_polarization import solve_polarization
from pdegen.switched_floquet import fourier_map, solve_switch


class BCFExtensionTests(unittest.TestCase):
    def test_floquet_prompt_uses_unit_disk(self):
        item = candidates(0, [BCF_PLAN[-1]])[0]
        prompt = render_problem(item)
        self.assertIn("单位圆外", prompt)
        self.assertNotIn("右半平面", prompt)
        self.assertNotIn("证书", prompt)

    def test_bcf_preset_retains_blindness(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            summary = run(out, candidates(0, BCF_PLAN), {})
            self.assertEqual(summary["attempted"], 10)
            self.assertEqual(summary["outcomes"], {"PASS": 10})
            self.assertEqual(summary["labels"], {"STABLE": 4, "UNSTABLE": 6})
            self.assertEqual(summary["reliable_with_blind_threshold"], 0)
            rows = json.loads((out / "internal" / "manifest.json").read_bytes())["rows"]
            self.assertEqual(Counter(row["family"] for row in rows), {"B3": 3, "C2": 3, "F3": 4})
            for row in rows:
                self.assertEqual(row["blind_status"], "PENDING_SINGLE_CASE_REVIEW")
                if row["label"] == "UNCERTAIN":
                    self.assertIsNone(row["prompt"])
                    continue
                prompt = (out / row["prompt"]).read_text(encoding="utf-8")
                self.assertNotIn(row["label"], prompt)
                self.assertNotIn(row["specification_hash"], prompt)
                grade = ("RIGOROUSLY_CERTIFIED" if row["label"] == "STABLE" or
                         (row["family"] == "C2" and row["parameters"] == {"alpha": "5/2", "beta": "1"})
                         else "CONVERGED_NUMERICAL")
                self.assertEqual(row["evidence_grade"], grade)

    def test_f3_certificate_checks_continuous_zone_and_period_map(self):
        gain = Fraction(31, 40)
        item, = candidates(0, [{"family": "F3", "parameters": {"gain": str(gain)}}])
        self.assertFalse(item["calibration"])
        result = solve_switch(item)
        self.assertEqual((result["label"], result["evidence_grade"]), ("STABLE", "RIGOROUSLY_CERTIFIED"))
        cert = result["certificate"]
        end = Fraction(0)
        for entry in cert["intervals"]:
            left, right = map(Fraction, entry["interval"])
            self.assertEqual(left, end)
            self.assertGreater(right, left)
            self.assertLess(Fraction(entry["power_norm_upper"]), 1)
            end = right
        self.assertEqual(end, Fraction(1, 2))
        self.assertTrue(any(entry["power"] > 1 for entry in cert["intervals"]))

        xi = Fraction(7, 64)
        real_imag, error, _, _ = monodromy_data(gain, xi, cert["cutoff"])
        real, imag = real_imag
        approx = (real+1j*imag)/SCALE
        independent = fourier_map(float(gain), float(xi), cert["cutoff"])
        self.assertLess(np.linalg.norm(approx-independent, ord=2), float(error))

    def test_f3_fixed_point_product_preserves_signed_floor(self):
        a = np.array([[SCALE-1, SCALE//3], [-SCALE-1, -3]], dtype=np.int64)
        b = np.array([[2, -SCALE], [SCALE+3, -SCALE//2]], dtype=np.int64)
        np.testing.assert_array_equal(product(a, b), (a.astype(object) @ b.astype(object))//SCALE)

    def test_b3_certificate_covers_neutral_sector_and_the_rest_of_the_zone(self):
        plan = [{"family": "B3", "parameters": {"damping": "23/4", "skew": "15/4"}}]
        item, = candidates(0, plan)
        self.assertFalse(item["calibration"])
        result = solve_roll(item)
        self.assertEqual((result["label"], result["evidence_grade"]), ("STABLE", "RIGOROUSLY_CERTIFIED"))
        cert = result["certificate"]
        intervals = [cert["near_zero"], *cert["intervals"]]
        edge = Fraction(0)
        for lo, hi in intervals:
            self.assertEqual(Fraction(lo), edge)
            self.assertGreater(Fraction(hi), edge)
            edge = Fraction(hi)
        self.assertEqual(edge, Fraction(1, 2))
        self.assertGreater(Fraction(cert["tail_bound"]), 0)

        cutoff = cert["cutoff"]
        b, c = Fraction(23, 4), Fraction(15, 4)
        coeffs, scale, _ = fourier_coefficients(b, c, cutoff)
        xi = .3125
        size = 2*(2*cutoff+1)
        exact = sum(coeff[:size, :size]*xi**i for i, coeff in enumerate(coeffs))/scale
        np.testing.assert_allclose(exact, fourier_matrix(float(b), float(c), xi, cutoff), atol=1e-12)

    def test_c2_reaches_the_semidefinite_boundary_and_verifies_a_slow_mode(self):
        boundary, slow = candidates(0, [
            {"family": "C2", "parameters": {"alpha": "3/4", "beta": "1/4"}},
            {"family": "C2", "parameters": {"alpha": "5/2", "beta": "1/2"}},
        ])
        self.assertFalse(boundary["calibration"])
        stable = solve_polarization(boundary)
        self.assertEqual((stable["label"], stable["evidence_grade"]), ("STABLE", "RIGOROUSLY_CERTIFIED"))
        self.assertEqual(stable["certificate"]["depths"][-1], "1")

        unstable = solve_polarization(slow)
        self.assertEqual((unstable["label"], unstable["evidence_grade"]), ("UNSTABLE", "CONVERGED_NUMERICAL"))
        witness = unstable["witness"]
        self.assertGreater(witness["physical"][-1]["length"], 85)
        self.assertGreater(witness["continuous"][-1]["root"][0], 10*witness["discrepancy"])
