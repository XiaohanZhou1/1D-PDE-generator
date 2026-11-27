import json
from pathlib import Path
import tempfile
import unittest

from pdegen.batch_workflow import candidates, load_reviews, review_index, review_status, run
from pdegen.blind_prompt import render_problem


class UnifiedBatchTests(unittest.TestCase):
    def test_pilot_preserves_single_and_group_review_scope(self):
        cases = candidates(20261126)
        self.assertEqual(len(cases), 14)
        self.assertEqual(load_reviews(), {})
        keys = [item["specification_hash"] for item in cases[:3]]
        reviews = review_index([
            dict(specification_hash=key, no_hints=True, no_numerics=True, session=session, outcome=outcome)
            for key, session, outcome in zip(keys, ("single", "group", "single"),
                                             ("incomplete_proof", "complete_proof", "complete_proof"), strict=True)
        ])
        self.assertEqual([review_status(reviews[key]) for key in keys],
                         ["NOT_SOLVED_IN_BLIND_REVIEW", "PENDING_SINGLE_CASE_REVIEW", "PROOF_COMPLETED"])

    def test_prompt_preserves_parameters_and_flux_component(self):
        a1, d2 = candidates(1, [
            {"family": "A1", "parameters": {"q": "125/128", "a": "129/64", "beta": "-1/4"}},
            {"family": "D2", "parameters": {"sigma": 3, "tau": 1, "epsilon": "3/4"}},
        ])
        text = render_problem(a1)
        self.assertIn(r"\begin{pmatrix}0\\6u^3\\0\\0\end{pmatrix}", text)
        self.assertIn(r"\beta=-\frac{1}{4}", text)
        text = render_problem(d2)
        self.assertIn(r"-12(1-u^2)", text)
        self.assertIn(r"\int_{1}^{\frac{5}{4}}", text)

    def test_e3_only_run_deduplicates_and_separates_outputs(self):
        plan = [{"family": "E3", "parameters": {"gain": gain}} for gain in (5, 5, 8)]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            summary = run(out, candidates(1, plan), load_reviews())
            self.assertEqual(summary["reliable_labels"], 2)
            self.assertEqual(summary["outcomes"], {"PASS": 2, "DUPLICATE": 1})
            prompts = sorted((out / "blind").iterdir())
            self.assertEqual([path.name for path in prompts], ["problem-0001.md", "problem-0002.md"])
            manifest = json.loads((out / "internal" / "manifest.json").read_bytes())
            self.assertEqual(manifest["rows"][0]["label"], "STABLE")
            self.assertEqual(manifest["rows"][1]["exact_duplicate_of"], manifest["rows"][0]["instance_id"])
            self.assertIsNone(manifest["rows"][1]["prompt"])
            self.assertEqual(manifest["rows"][2]["label"], "UNSTABLE")
            for path, expected in zip(prompts, ("g=5", "g=8"), strict=True):
                contents = path.read_text(encoding="utf-8")
                self.assertIn(expected, contents)
                for secret in ("STABLE", "UNSTABLE", "证书", "E3", "RIGOROUSLY_CERTIFIED"):
                    self.assertNotIn(secret, contents)
