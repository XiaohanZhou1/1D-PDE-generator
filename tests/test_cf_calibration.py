import json
import tempfile
import unittest
from pathlib import Path

from pdegen.audit import audit_batch
from pdegen.cf_calibration import (CALIBRATION_FAMILIES, calibration_batch, certify_calibration,
                                  native_reference, validate_calibration)
from pdegen.contract import ROOT


class CrossFamilyCalibrationTests(unittest.TestCase):
    def test_reviewed_coverage_and_hold(self):
        items = [item for family in CALIBRATION_FAMILIES for item in calibration_batch(family, 1)]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            rows = audit_batch(items, out, validate_calibration, certify_calibration)
            self.assertEqual([row["label"] for row in rows],
                             ["STABLE", "STABLE", "UNSTABLE", "STABLE", "UNSTABLE", "STABLE"])
            self.assertEqual([row["open_rhp_count"] for row in rows], [0, 0, None, 0, 1, None])
            for row in rows:
                audit = json.loads((out / row["audit"]).read_text())
                result = json.loads((out / row["result"]).read_text())
                self.assertEqual(audit["dataset_admission"], "HOLD")
                self.assertEqual(result["evidence_grade"], "RIGOROUSLY_CERTIFIED")
                self.assertTrue((ROOT / result["certificate"]["proof"].split("#")[0]).is_file())
                self.assertEqual(audit["specification_hash"], result["specification_hash"])
            self.assertEqual(result["numerical_check"]["status"], "PASS")
            self.assertLess(result["numerical_check"]["zero_fiber_error"], 1e-9)

    def test_changed_operator_cannot_inherit_certificate(self):
        e1 = calibration_batch("E1", 1)[0]
        tampered = {**e1, "eigenproblem": {**e1["eigenproblem"], "domain": "Neumann"}}
        with self.assertRaises(ValueError):
            certify_calibration(tampered)
        f1 = calibration_batch("F1", 1)[0]
        self.assertEqual(native_reference(f1, None)["status"], "UNSUPPORTED")
