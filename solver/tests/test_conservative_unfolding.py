import unittest

import sympy as sp

from twstab import solve
from twstab.label import analyse


def shock(mode="unfold", target=0.0):
    x,t = sp.symbols("x t",real=True)
    rho = sp.Symbol("rho")
    u,v = [sp.Function(name)(x,t) for name in ("u","v")]
    g = 1-u*u
    f = (g,u*g+u*(v-u*u/2))
    if mode=="unfold":
        f = (f[0],f[1]+rho*g)
    elif mode=="changes_jump":
        f = (f[0],f[1]+rho*u)
    else:
        f = tuple((1+rho)*e for e in f)
    return dict(x=x,t=t,equations=[sp.Eq(q.diff(t),q.diff(x,2)-flux.diff(x))
                                  for q,flux in zip((u,v),f,strict=True)],parameters={rho:target},
                wave=dict(kind="front",left={u:-1,v:sp.S.Half},right={u:1,v:sp.S.Half},speed=0,
                          unfolding=dict(parameter="rho",target=target,guess=.02)))


class ConservativeUnfolding(unittest.TestCase):
    def test_public_batch_runs_unfolding(self):
        entry = shock()
        record = analyse(entry)
        self.assertTrue(record["wave"]["computed"])
        self.assertEqual(record["wave"]["started_from"],"parameter_unfolding")
        self.assertLess(abs(record["wave"]["unfolding"]["fitted"]),1e-7)
        self.assertLess(record["wave"]["target_equation_residual"],1e-7)
        result, = solve([entry])
        self.assertEqual(len(result["waves"]),1)

    def test_wrong_target_is_rejected_before_spectrum(self):
        record = analyse(shock(target=.05))
        self.assertEqual(record["label"],"UNCERTAIN")
        self.assertIn("differs from target",record["reason"])
        self.assertNotIn("closed_half_plane",record)

    def test_parameter_cannot_change_jump(self):
        record = analyse(shock(mode="changes_jump"))
        self.assertEqual(record["label"],"UNCERTAIN")
        self.assertIn("changes the jump conditions",record["reason"])

    def test_pure_rescaling_parameter_is_rejected(self):
        record = analyse(shock(mode="rescale"))
        self.assertEqual(record["label"],"UNCERTAIN")
        self.assertIn("only rescales",record["reason"])
