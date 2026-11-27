import unittest

from pdegen.bloch import roll
from pdegen.label_pilot import readiness, summarise


class LabelPilotTests(unittest.TestCase):
    def test_numerical_stability_requires_full_domain_evidence(self):
        item = roll("1/4", 0)
        audit = {"status": "EVALUATED", "proof_screen": {}}
        stable = {"label": "STABLE", "evidence_grade": "CONVERGED_NUMERICAL"}
        self.assertEqual(readiness(item, stable, audit),
                         ("NEEDS_FULL_DOMAIN_CERTIFICATE", "CALIBRATION"))
        stable["evidence_grade"] = "EXACT_LINEAR_SPECTRUM"
        self.assertEqual(readiness(item, stable, audit), ("PASS", "CALIBRATION"))

    def test_yield_includes_held_and_unresolved_cases(self):
        rows = [
            dict(label_gate="PASS", difficulty_gate="PRELIMINARY_PASS",
                 sampling_group="A1.explicit-family", solver_label="UNSTABLE"),
            dict(label_gate="PASS", difficulty_gate="CALIBRATION", solver_label="STABLE"),
            dict(label_gate="HOLD"),
            dict(label_gate="UNCERTAIN"),
        ]
        report = summarise(rows)
        self.assertEqual(report["label_pass_rate"], 0.5)
        self.assertEqual(report["preliminary_joint_pass_rate"], 0.25)
        self.assertEqual(report["joint_mechanism_groups"], ["A1.explicit-family"])
