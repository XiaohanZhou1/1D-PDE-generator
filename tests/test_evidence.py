import cmath
import gzip
import json
import math
import unittest
from pdegen.contract import ROOT, digest, schema_check, validate
from pdegen.generate import batch
from pdegen.families import a5_fixture
from pdegen.contract import construct


class PilotEvidence(unittest.TestCase):
    def test_shipped_pilot_and_raw_evans_windings(self):
        examples=ROOT/"examples"
        manifest=json.loads((examples/"manifest.json").read_text())
        files=[examples/"instances"/(k+".json") for k in manifest["instances"]]
        files.append(examples/manifest["fixture"])
        regenerated=batch(manifest["seed"])+[construct(a5_fixture(),manifest["seed"])]
        expected_counts=[1,2,2,None,0]
        for path,expected,count in zip(files,regenerated,expected_counts,strict=True):
            item=validate(json.loads(path.read_text()))
            self.assertEqual(item,expected)
            evidence=ROOT/"evidence"/item["instance_id"]
            result=json.loads((evidence/"result.json").read_text())
            schema_check("result",result)
            self.assertEqual(result["instance_id"],item["instance_id"])
            self.assertEqual(result["specification_hash"],item["specification_hash"])
            self.assertEqual(count,result["unstable_count"])
            self.assertTrue(all((evidence/name).is_file() for name in result["artifact_refs"]))
            if count is None:
                self.assertEqual(result["solver_status"],"NOT_RUN")
                self.assertIsNone(result["label"])
                continue
            self.assertEqual(result["solver_status"],"COMPLETED")
            self.assertEqual(result["evidence_grade"],"CONVERGED_NUMERICAL")
            self.assertEqual(result["label"],"UNSTABLE" if count else "STABLE_MOD_TRANSLATION")
            native,=list((evidence/"raw").glob("*/case.json"))
            raw=json.loads(native.read_text())
            self.assertEqual(digest(raw["settings"]),result["settings_hash"])
            self.assertEqual(raw["numerical_open_rhp_count"],count)
            diag=json.loads((native.parent/"diagnostics.json").read_text())
            with gzip.open(native.parent/"contours.json.gz","rt") as f:
                contours=json.load(f)
            for grid in diag["evans"]:
                key="L{}_h{}".format(grid["L"],grid["h"])
                checks=[("full",grid["outer"])]
                checks += [(f"root{i}",r) for i,r in enumerate(grid["local"])]
                if item["family_id"]=="A5":
                    checks += [("integrated_full",grid["integrated_outer"]),
                               ("integrated_origin",grid["integrated_origin"])]
                for suffix,record in checks:
                    with self.subTest(case=item["instance_id"],grid=key,contour=suffix):
                        samples=contours[key+"_"+suffix]
                        phases=[complex(row[3],row[4]) for row in samples]
                        increments=[cmath.phase(b/a) for a,b in zip(phases,phases[1:]+phases[:1])]
                        self.assertAlmostEqual(sum(increments)/math.tau,record["count"],places=8)
                        self.assertLessEqual(max(map(abs,increments)),raw["settings"]["phase_limit"])
