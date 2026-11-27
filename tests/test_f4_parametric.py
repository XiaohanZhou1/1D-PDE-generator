from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pdegen.batch_workflow import candidates, read_json, run


class ParametricFloquetTests(unittest.TestCase):
    def test_close_threshold_pair_runs_through_batch_and_exports_blind_prompts(self):
        plan = [dict(family="F4", parameters=dict(stiffness="31/32", modulation=h))
                for h in ("1/2", "17/32")]
        stable, unstable = items = candidates(0, plan)
        self.assertFalse(stable["calibration"])
        self.assertFalse(unstable["calibration"])

        with TemporaryDirectory() as directory:
            out = Path(directory)/"run"
            summary = run(out, items, {})
            self.assertEqual(summary["labels"], {"STABLE": 1, "UNSTABLE": 1})
            rows = read_json(out/"internal/manifest.json")["rows"]
            self.assertEqual([row["label"] for row in rows], ["STABLE", "UNSTABLE"])
            for row in rows:
                self.assertEqual(row["evidence_grade"], "RIGOROUSLY_CERTIFIED")
                certificate = read_json(out/row["evidence"])["certificate"]
                lo, hi = map(Fraction, certificate["undamped_trace_interval"])
                self.assertGreater(Fraction(certificate["trace_threshold_margin_lower"]), 0)
                self.assertLess(hi, -2)
                self.assertLess(hi-lo, Fraction(1, 10**8))
                prompt = (out/row["prompt"]).read_text(encoding="utf-8")
                self.assertIn("单位圆外", prompt)
                for private in ("STABLE", "F4", "Hill", "证书", row["specification_hash"]):
                    self.assertNotIn(private, prompt)
