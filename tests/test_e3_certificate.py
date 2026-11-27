from fractions import Fraction
import json
import unittest

from pdegen.contract import ROOT
from pdegen.e3_certificate import SCALE, Interval, certify_gain, polygon_winding


class E3CertificateTests(unittest.TestCase):
    def test_outward_rounding(self):
        value = Fraction(2, 3)
        interval = Interval.point(value)
        reciprocal = interval.reciprocal()
        self.assertLessEqual(Fraction(reciprocal.lo, SCALE), 1 / value)
        self.assertGreaterEqual(Fraction(reciprocal.hi, SCALE), 1 / value)
        square_root = interval.sqrt()
        self.assertLessEqual(Fraction(square_root.lo, SCALE)**2, value)
        self.assertGreaterEqual(Fraction(square_root.hi, SCALE)**2, value)
        self.assertEqual(polygon_winding([(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]), 1)

    def test_certificates_cover_closed_contours_and_replay(self):
        output = ROOT / "evidence" / "e3-verified"
        manifest = json.loads((output / "manifest.json").read_bytes())
        for row, expected in zip(manifest["rows"], (0, 0, 2, 2), strict=True):
            with self.subTest(gain=row["gain"]):
                saved = json.loads((output / row["certificate"]).read_bytes())
                instance = json.loads((output / row["instance"]).read_bytes())
                self.assertEqual(saved["specification_hash"], instance["specification_hash"])
                self.assertEqual(row["open_rhp_count"], expected)
                replay = json.loads(json.dumps(certify_gain(row["gain"])))
                self.assertEqual(replay, {name: saved[name] for name in replay})
                self.assertGreater(Fraction(saved["outside_margin_lower_bound"]), 0)
                segments = saved["segments"]
                for segment, following in zip(segments, segments[1:] + segments[:1], strict=True):
                    self.assertEqual(segment["end"], following["start"])
                    self.assertEqual(segment["polygon_end"], following["polygon_start"])
                    self.assertGreater(max(max(0, lo, -hi) for lo, hi in segment["image_box"]), 0)
