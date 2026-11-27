import copy
import unittest

from pdegen.contract import digest
from pdegen.frozen_production import freeze, group_metrics, review_selection, validate_plan


class FrozenProductionTests(unittest.TestCase):
    def test_frozen_sampling_reproduces_and_rejects_tampering(self):
        plan = freeze(set())
        rebuilt = freeze(set(plan["excluded_known_hashes"]))
        self.assertEqual(digest(rebuilt), digest(plan))
        items = validate_plan(plan)
        self.assertEqual(len({row["specification_hash"] for row in items}), 48)
        altered = copy.deepcopy(plan)
        altered["cases"][0]["parameters"]["a"] = 4
        with self.assertRaises(ValueError):
            validate_plan(altered)

    def test_blind_rate_excludes_unreviewed_and_group_attempts(self):
        base = dict(family="E3", preliminary_difficulty="REVIEW_PENDING", label_gate="PASS",
                    exact_duplicate_of=None, sampling_group="E3")
        rows = [dict(**base, label=label, specification_hash=str(index))
                for index, label in enumerate(("STABLE", "UNSTABLE", "UNSTABLE"))]
        sample = review_selection(rows)
        self.assertEqual({row["label"] for row in sample}, {"STABLE", "UNSTABLE"})
        key = sample[0]["specification_hash"]
        protocol = dict(no_hints=True, no_numerics=True, outcome="incomplete_proof")
        pending = group_metrics(rows, sample, {key: [dict(**protocol, session="group")]})
        self.assertIsNone(pending["blind_pass_rate"])
        self.assertEqual(pending["blind_reviewed"], 0)
        reviewed = group_metrics(rows, sample, {key: [dict(**protocol, session="single")]})
        self.assertEqual(reviewed["blind_pass_rate"], 1)
        self.assertEqual(reviewed["blind_coverage"], .5)
        self.assertEqual(reviewed["repetition_reliable"]["repeats"], 2)
