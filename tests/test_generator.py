import copy
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from jsonschema import Draft202012Validator, ValidationError
from pdegen.algebra import evaluate, symbol
from pdegen.contract import ROOT, construct, schema_check, unique, validate
from pdegen.families import a1, a5_fixture, dispatch_limitation
from pdegen.adapter import execute
from pdegen.generate import batch


class ConstructionTests(unittest.TestCase):
    def test_exact_identities_on_rectangle_boundary(self):
        for q in (0,1):
            for a in (2,6):
                for beta in (0,"-1/4","1/4"):
                    with self.subTest(q=q,a=a,beta=beta):
                        item=construct(a1(q,a,beta),1)
                        self.assertEqual(item["validity_checks"]["translation"],"PASS")

    def test_invalid_profile_and_missing_differential_flux(self):
        for change in ("wave","G","F","D"):
            m=a1(1,2)
            if change=="wave":
                m["wave"][1]=-m["wave"][1]
            elif change=="G":
                m["G"][1]=0
            elif change=="F":
                m["F"][2]+=symbol("u")*symbol("y")
            else:
                m["D"][0]=0
            with self.subTest(change=change),self.assertRaises(ValueError):
                construct(m,1)

    def test_operator_product_rule_and_normal_feedback(self):
        base=construct(a1(1,2),1)
        extra=construct(a1(1,2,"1/4"),1)
        L=base["linearised_operator"]
        self.assertEqual(L["B"][1][0],{"s*s":"18"})
        self.assertEqual(L["C"][1][0],{"s*s*t":"-36"})
        self.assertEqual(extra["linearised_operator"]["C"][1][2],{"s*s":"1/4"})
        self.assertEqual(extra["linearised_operator"]["C"][1][0]["s*s*s"],"-1/2")
        self.assertEqual(extra["far_field_data"],base["far_field_data"])
        self.assertEqual(extra["target_wave"],base["target_wave"])
        self.assertNotEqual(extra["specification_hash"],base["specification_hash"])

    def test_changed_far_field_with_exact_wave_is_rejected(self):
        model=a1(1,2)
        u,v=map(symbol,["u","v"])
        model["F"][2]+=8*(v-u*u)
        with self.assertRaisesRegex(ValueError,"far-field"):
            construct(model,1)

    def test_full_frechet_derivative_on_arbitrary_perturbation_jets(self):
        import numpy as np
        rng=np.random.default_rng(164)
        for beta in ("0","-1/4","1/4"):
            m=a1("3/4",3,beta)
            item=construct(m,1)
            L=item["linearised_operator"]
            for z in (-1.2,0.,.7):
                env={"s":1/np.cosh(z),"t":np.tanh(z)}
                U,U1,U2=([evaluate(p,env) for p in item["target_wave"][k]]
                         for k in ("profile","dz","dzz"))
                p,dp,ddp=rng.standard_normal((3,4))
                h=1e-25
                X,dx,dxx=(np.asarray(x)+1j*h*d for x,d in zip((U,U1,U2),(p,dp,ddp)))
                state=dict(zip(m["names"],X))
                reaction=np.array([f.at(state) for f in m["F"]],dtype=complex)
                rhs=np.array(m["D"],float)*dxx+(1-np.array(m["A"],float))*dx+reaction
                rhs[1]+=18*X[0]**2*dx[0]
                B,C=(np.array([[evaluate(f,env) for f in row] for row in L[k]]) for k in ("B","C"))
                expected=np.array(m["D"],float)*ddp+B@dp+C@p
                np.testing.assert_allclose(rhs.imag/h,expected,atol=3e-13,rtol=3e-13)

    def test_rational_inputs_and_scouting_domain(self):
        for q,a,b in ((-1,2,0),(2,2,0),(1,7,0),(1,4,1),(.5,4,0),(True,4,0)):
            with self.assertRaises(ValueError):
                a1(q,a,b)
        self.assertIsNotNone(dispatch_limitation(a1("99/100",2)))
        self.assertIsNotNone(dispatch_limitation(a1("3/4",3)))
        self.assertIsNone(dispatch_limitation(a1("127/128","257/128")))

    def test_reproducibility_and_deduplication(self):
        self.assertEqual(batch(15),batch(15))
        self.assertNotEqual(batch(15)[2]["parameters"],batch(21)[2]["parameters"])
        x,y=construct(a1("2/4",4),1),construct(a1("1/2","8/2"),77)
        self.assertEqual(x["instance_id"],y["instance_id"])
        with self.assertRaisesRegex(ValueError,"duplicate"):
            unique([x,y])

    def test_invalid_contract(self):
        item=construct(a1(1,4),2)
        for field,value in (("weights","exponential"),("wave_id","translated"),("specification_hash","0"*64)):
            with self.assertRaises(ValueError):
                validate(dict(item,**{field:value}))
        with self.assertRaises(ValidationError):
            validate(dict(item,label="UNSTABLE"))
        bad=copy.deepcopy(item)
        bad["equations"]["F"][0]={"__import__('os')":"1"}
        with self.assertRaises(ValidationError):
            validate(bad)
        bad=copy.deepcopy(item)
        bad["linearised_operator"]["C"][1][0]={}
        with self.assertRaises(ValueError):
            validate(bad)

    def test_a5_contract_and_rh(self):
        m=a5_fixture()
        self.assertEqual(m["ends"],[[-1,Fraction(7,24)],[1,Fraction(17,24)]])
        for end in m["ends"]:
            for g in m["G"]:
                self.assertEqual(g.at(dict(zip(m["names"],end))),0)
        item=validate(construct(m,1))
        self.assertEqual(item["stability_convention"],"unweighted_shock_edge")
        bad=copy.deepcopy(item)
        bad["parameters"]["rho"]="1/2"
        with self.assertRaises(ValueError):
            validate(bad)

    def test_schema_definitions(self):
        for s in ("instance","result"):
            Draft202012Validator.check_schema(json.loads((ROOT/"schemas"/(s+".json")).read_text()))

    def test_quarantine_and_failed_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            held=execute(construct(a1(1,2,"1/4"),1),path/"missing",path/"held")
            self.assertEqual(held["solver_status"],"NOT_RUN")
            self.assertIsNone(held["label"])
            with self.assertRaises(ValidationError):
                schema_check("result",dict(held,label="UNSTABLE"))
            failed=execute(construct(a1(1,2),1),path/"missing",path/"failed")
            self.assertEqual(failed["solver_status"],"FAILED")
            self.assertIsNone(failed["unstable_count"])
            self.assertTrue((path/"failed/result.json").is_file())
