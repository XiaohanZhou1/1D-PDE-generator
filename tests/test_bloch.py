from fractions import Fraction
import unittest

import numpy as np
import sympy as sp

from pdegen.bloch import bands, bloch_batch, exact_labels, physical_fd, roll, validate_roll


class BlochTests(unittest.TestCase):
    def test_exact_profile_linearisation_and_phase_mode(self):
        a, q, theta, u, v = sp.symbols("a q theta u v",real=True)
        X = sp.Matrix([u,v])
        U = sp.sqrt(a)*sp.Matrix([sp.cos(theta),sp.sin(theta)])
        reaction = (1-X.dot(X))*X
        on_wave = dict(zip(X,U,strict=True))
        jacobian = reaction.jacobian(X).subs(on_wave).applyfunc(sp.trigsimp)
        expected = (1-a)*sp.eye(2)-2*U*U.T
        self.assertEqual((jacobian-expected).applyfunc(sp.trigsimp),sp.zeros(2,2))
        residual = q*q*U.diff(theta,2)+reaction.subs(on_wave)
        self.assertEqual(residual.subs(a,1-q*q).applyfunc(sp.trigsimp),sp.zeros(2,1))
        phase = U.diff(theta)
        self.assertEqual((q*q*phase.diff(theta,2)+jacobian*phase).subs(a,1-q*q)
                         .applyfunc(sp.trigsimp),sp.zeros(2,1))

    def test_paired_controls_and_threshold(self):
        cases = bloch_batch(20261124)
        self.assertEqual([exact_labels(Fraction(r["parameters"]["q_squared"])) for r in cases],
                         [("STABLE","STABLE"),("UNSTABLE","STABLE"),("MARGINAL","STABLE")])
        self.assertEqual(exact_labels(Fraction(2,5)),("UNSTABLE","MARGINAL"))
        self.assertEqual(exact_labels(Fraction(1,2)),("UNSTABLE","UNSTABLE"))
        for item in cases:
            self.assertEqual(validate_roll(item),item)
            self.assertFalse(item["instance_id"]==roll("7/8",0)["instance_id"])
        with self.assertRaises(ValueError):
            roll("1",0)
        with self.assertRaises(ValueError):
            validate_roll({**cases[0],"stability_question":"Change task semantics."})

    def test_physical_operator_detects_nonzero_fiber_instability(self):
        q2 = 9/25
        eigen_zero = physical_fd(q2,0.0,64,np)
        self.assertLess(abs(eigen_zero[0]),5e-4)
        self.assertLess(eigen_zero[1],0)
        coarse = physical_fd(q2,.2,32,np)[0]
        fine = physical_fd(q2,.2,96,np)[0]
        expected = bands(q2,.2,np)[0]
        self.assertGreater(fine,0)
        self.assertLess(abs(fine-expected),.2*abs(coarse-expected))
