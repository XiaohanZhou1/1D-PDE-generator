import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pdegen.contract import ROOT
from pdegen.floquet import PERIOD, fourier_generator, monodromy
from solver.twstab.temporal_floquet import RESULT_VERSION, apply_actions


ORBIT = ROOT / "evidence" / "f2-action" / "orbit.json"
REQUEST = ROOT / "evidence" / "f2-native" / "request.json"


class TwstabFloquetTests(unittest.TestCase):
    def test_solver_cli_on_nine_reference_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            command = [sys.executable, "-m", "twstab", "floquet", "--orbit", str(ORBIT),
                       "--request", str(REQUEST), "--out", str(output)]
            run = subprocess.run(command, cwd=ROOT / "solver", capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(output.read_bytes())
            self.assertEqual(result["schema_version"], RESULT_VERSION)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["orbit_sha256"], hashlib.sha256(ORBIT.read_bytes()).hexdigest())
            self.assertEqual(result["request_sha256"], hashlib.sha256(REQUEST.read_bytes()).hexdigest())
            repeated = subprocess.run(command, cwd=ROOT / "solver", capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(json.loads(output.read_bytes()), result)

        requested = json.loads(REQUEST.read_bytes())["actions"]
        self.assertEqual(len(result["actions"]), 9)
        references = {}
        for action, returned in zip(requested, result["actions"], strict=True):
            for field in ("id", "xi", "cutoff", "tolerance"):
                self.assertEqual(returned[field], action[field])
            xi, cutoff = action["xi"], action["cutoff"]
            if (xi, cutoff) not in references:
                generator, dimension = fourier_generator(xi, cutoff)
                references[xi, cutoff] = monodromy(generator, dimension, PERIOD, 1e-12)
            vector = np.array([complex(*pair) for pair in action["vector"]])
            output_vector = np.array([complex(*pair) for pair in returned["vector"]])
            expected = references[xi, cutoff] @ vector
            self.assertLess(np.linalg.norm(output_vector - expected) / np.linalg.norm(vector), 1e-8)
            if action["id"].endswith("/time_phase"):
                self.assertLess(np.linalg.norm(output_vector - vector) / np.linalg.norm(vector), 1e-8)

    def test_wrong_wire_inputs_are_rejected(self):
        orbit = json.loads(ORBIT.read_bytes())
        request = json.loads(REQUEST.read_bytes())
        action = request["actions"][0]
        with self.assertRaisesRegex(ValueError, "nonempty and unique"):
            apply_actions(orbit, {**request, "actions": [action, action]})
        wrong = {**action, "vector": [[1.0], *action["vector"][1:]]}
        with self.assertRaisesRegex(ValueError, "real, imaginary"):
            apply_actions(orbit, {**request, "actions": [wrong]})
        with self.assertRaisesRegex(ValueError, "specification"):
            apply_actions({**orbit, "specification_hash": "other"}, request)
