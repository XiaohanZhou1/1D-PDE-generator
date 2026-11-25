import cmath
import gzip
import json
import math
import unittest

from pdegen.algebra import poly, symbol
from pdegen.contract import ROOT, construct, digest, schema_check, validate
from pdegen.families import a5
from pdegen.generate import a5_batch


class A5GenerationTests(unittest.TestCase):
    def test_exact_branch_and_sampling(self):
        records = a5_batch(20261124)
        self.assertEqual(records, a5_batch(20261124))
        self.assertEqual([r["parameters"]["mu"] for r in records], ["1","3","6","5/2","29/8"])
        self.assertNotEqual(records[-2:], a5_batch(20261125)[-2:])
        fixture = json.loads((ROOT/"examples/fixtures/a5-mu3.json").read_text())
        self.assertEqual(records[1], fixture)
        for record in records:
            self.assertEqual(validate(record), record)
            self.assertEqual(record["validity_checks"]["solver_dispatch"], "PASS")
            self.assertEqual(record["novelty_assessment"]["training_admission"], "HOLD")

    def test_invalid_parameters_and_inexact_native_conversion(self):
        for mu,rho in ((0,0),(7,0),(3,"1/10"),(2.5,0)):
            with self.subTest(mu=mu,rho=rho), self.assertRaises(ValueError):
                a5(mu,rho)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            a5_batch(0,["5/2","10/4"])
        held = construct(a5("5/3"), 0)
        self.assertEqual(held["validity_checks"]["profile"], "PASS")
        self.assertEqual(held["validity_checks"]["solver_dispatch"], "FAIL")

    def test_wrong_flux_and_viscosity_are_rejected(self):
        wrong = a5(3)
        wrong["D"] = [2,1]
        wrong["G"][0] += symbol("u")**2-1
        with self.assertRaisesRegex(ValueError,"identity viscosity"):
            construct(wrong,0)
        wrong = a5(3)
        wrong["G"][0] += 1
        with self.assertRaisesRegex(ValueError,"zero far-field flux"):
            construct(wrong,0)
        wrong = a5(3)
        u,v = map(symbol,("u","v"))
        wrong["G"][1] += v-u*u*poly("1/2")-u**3*poly("5/24")
        with self.assertRaisesRegex(ValueError,"hyperbolicity formula"):
            construct(wrong,0)

    def test_published_batch_and_evans_evidence(self):
        examples = ROOT/"examples/g2"
        manifest = json.loads((examples/"manifest.json").read_text())
        for ident, expected in zip(manifest["instances"],a5_batch(manifest["seed"]),strict=True):
            with self.subTest(instance=ident):
                item = validate(json.loads((examples/"instances"/(ident+".json")).read_text()))
                self.assertEqual(item,expected)
                folder = ROOT/"evidence/g2"/ident
                result = json.loads((folder/"result.json").read_text())
                schema_check("result",result)
                self.assertEqual(result["specification_hash"],item["specification_hash"])
                self.assertEqual(result["instance_id"],ident)
                self.assertEqual(result["solver_status"],"COMPLETED")
                self.assertEqual(result["label"],"STABLE_MOD_TRANSLATION")
                self.assertEqual(result["evidence_grade"],"CONVERGED_NUMERICAL")
                self.assertEqual(result["unstable_count"],0)
                self.assertTrue(all((folder/path).is_file() for path in result["artifact_refs"]))
                cfg = json.loads((folder/"settings.json").read_text())
                self.assertEqual(result["settings_hash"],digest(cfg))
                native, = folder.glob("raw/*/diagnostics.json")
                diag = json.loads(native.read_text())
                self.assertEqual([r["count"] for r in diag["checks"]["shooting"]],[1,1])
                self.assertGreater(diag["checks"]["outgoing_obstruction"][0],0)
                self.assertLess(diag["checks"]["outgoing_obstruction"][1],0)
                with gzip.open(native.parent/"contours.json.gz","rt") as stream:
                    contours = json.load(stream)
                for grid in diag["evans"]:
                    tag = "L{}_h{}".format(grid["L"],grid["h"])
                    for suffix in ("full","root0","integrated_full","integrated_origin"):
                        phases = [complex(row[3],row[4]) for row in contours[tag+"_"+suffix]]
                        increments = [cmath.phase(b/a) for a,b in zip(phases,phases[1:]+phases[:1])]
                        self.assertAlmostEqual(sum(increments)/math.tau,1,places=8)
                        self.assertLessEqual(max(map(abs,increments)),cfg["phase_limit"])
