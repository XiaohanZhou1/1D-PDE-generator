import json
from pathlib import Path
import tempfile
import unittest

from pdegen.audit import a5_evidence, audit_batch
from pdegen.contract import construct, validate
from pdegen.families import a5


class AuditTests(unittest.TestCase):
    def test_decisions_preserve_rejections_and_unknown_results(self):
        original = construct(a5(1),42)
        duplicate = construct(a5(1),43)
        nearby = construct(a5("9/8"),42)
        held = construct(a5("5/3"),42)
        tampered = {**original,"function_space":"altered"}

        def solve(item):
            raise RuntimeError("deliberate numerical failure")

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            rows = audit_batch([original,duplicate,nearby,held,tampered],path,validate,solve)
            audit = [json.loads((path/row["audit"]).read_text()) for row in rows]
            self.assertEqual([r["status"] for r in audit],
                             ["UNCERTAIN","DUPLICATE","UNCERTAIN","HOLD","REJECTED"])
            self.assertEqual(audit[2]["near_duplicate_of"],original["instance_id"])
            self.assertEqual(audit[0]["sampling_group"],audit[2]["sampling_group"])
            self.assertEqual(audit[0]["dataset_admission"],"HOLD")
            self.assertIsNone(rows[3]["result"])
            self.assertTrue((path/audit[4]["rejected_candidate"]).is_file())

    def test_native_stability_without_evidence_is_uncertain(self):
        evidence = a5_evidence({"label":"STABLE","reason":"no evidence attached"})
        self.assertEqual(evidence["label"],"UNCERTAIN")
        self.assertIsNone(evidence["evidence_grade"])
