"""Regression tests for exact conservative input and the continued A5 origin."""
import unittest

import sympy as sp

from twstab import solve
from twstab.label import analyse


def a5_entry():
    x, t, z = sp.symbols("x t z", real=True)
    u, v = (sp.Function(name)(x,t) for name in ("u","v"))
    def H(q):
        return q**2/2 + 5*q**3/24
    hp = u+5*u**2/8
    g, h, k = 1-u*u,v-H(u),3*(1+u/2)
    f, j = g+k*h,hp*g+(hp*k+u)*h
    return dict(x=x,t=t,equations=[sp.Eq(u.diff(t),u.diff(x,2)-f.diff(x)),
                                   sp.Eq(v.diff(t),v.diff(x,2)-j.diff(x))],
                wave=dict(kind="front",left=dict(u=-1,v=H(sp.S.NegativeOne)),
                          right=dict(u=1,v=H(sp.S.One)),speed=0,L=20,
                          exact_profile=dict(u=sp.tanh(z),v=H(sp.tanh(z)))),
                options=dict(conservation_method="axis",conservation_origin_radius=0.08,region=[120,477]))


class ConservativeFront(unittest.TestCase):
    def test_public_batch_entry(self):
        result, = solve([a5_entry()])
        wave, = result["waves"]
        self.assertEqual(wave["label"],"STABLE")
        self.assertIn("modulo translation",wave["qualifier"])

    def test_incorrect_profile_is_rejected_even_with_correct_endstates(self):
        entry = a5_entry()
        u = entry["wave"]["exact_profile"]["u"]
        entry["wave"]["exact_profile"]["v"] += (1-u**2)/8
        result = analyse(entry)
        self.assertEqual(result["label"],"UNCERTAIN")
        self.assertIn("travelling-wave equations",result["reason"])

    def test_radius_cannot_reach_branch_point(self):
        result = analyse(a5_entry(),conservation_origin_radius=0.25)
        self.assertEqual(result["label"],"UNCERTAIN")
        self.assertIn("branch points",result["reason"])

    def test_other_viscosity_is_rejected(self):
        entry = a5_entry()
        entry["equations"] = [sp.Eq(eq.lhs,2*eq.rhs) for eq in entry["equations"]]
        result = analyse(entry)
        self.assertEqual(result["label"],"UNCERTAIN")
        self.assertIn("identity viscosity",result["reason"])

    def test_nonconservative_exact_input_is_not_silently_ignored(self):
        entry = a5_entry()
        u = sp.Function("u")(entry["x"],entry["t"])
        first,second = entry["equations"]
        entry["equations"] = [sp.Eq(first.lhs,first.rhs-u),second]
        result = analyse(entry)
        self.assertEqual(result["label"],"UNCERTAIN")
        self.assertIn("local conservative front",result["reason"])
