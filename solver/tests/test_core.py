"""Unit tests.  Run from Version_2:  python tests/test_core.py -v
Fast set (after every change, about 70 s):  python tests/test_core.py --fast

Every test compares with a result that is known in closed form, from Version_1,
or from the literature (FitzHugh-Nagumo: fast pulse stable, slow pulse unstable;
Fisher-Stefan: the semi-wave is stable).
"""
import json
import runpy
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import sympy as sp
from scipy import sparse
from scipy.optimize import brentq
from scipy.sparse.linalg import eigs, eigsh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAST = "--fast" in sys.argv
if FAST:
    sys.argv.remove("--fast")
slow = unittest.skipIf(FAST, "slow test: run without --fast")

from twstab.asymptotic import (SlowModeError, axis_tolerance, essential_bound, high_wavenumber_limit,   # noqa: E402
                               quadratic_essential_bound, split, split_touching)
from twstab import batch                                         # noqa: E402
from twstab.batch import lines, read_list, solve                 # noqa: E402
from twstab.classify import classify                             # noqa: E402
from twstab.conservation import ConservativeSystem, antiderivative   # noqa: E402
from twstab.continuation import trace                            # noqa: E402
from twstab.contour import circle, isolate, locate, winding      # noqa: E402
from twstab import delay, kernel, weighted                       # noqa: E402
from twstab.scout import (_difference_matrices, coefficient_bounds, fd_companion, fd_operator,   # noqa: E402
                          fd_probe, search_region)
from twstab.evans import Asymptotics, Evans                      # noqa: E402
from twstab.freeboundary import (BoundaryAsymptotics, boundary_bounds, boundary_estimate,   # noqa: E402
                                 build_engine, fd_operator_boundary, solve_semi_wave)
from twstab.label import analyse                                 # noqa: E402
from twstab import problem as problem_module                     # noqa: E402
from twstab.problem import Problem, ProblemError, load_problem   # noqa: E402
from twstab.symbolic import LocalSystem, StructureError          # noqa: E402
from twstab.symmetry import RotatingFrame, rotation_pairs        # noqa: E402
from twstab.taxis import TaxisSystem                             # noqa: E402
from twstab.survey import find_equilibria, survey                # noqa: E402
from twstab.wave import Wave, WaveError, reference_from_guess, solve_wave, solve_wave_at_speed   # noqa: E402

PROBLEMS = ROOT / "problems"
EQUATIONS = ROOT / "equations"                                  # files without `wave`: the solver searches
# Labels of examples/example_list.py, in the order of the output list (waves, then families).
EXAMPLE_LABELS = {"eqn1": ["STABLE", "UNSTABLE", "UNSTABLE"], "eqn2": ["UNSTABLE", "UNSTABLE"], "eqn3": ["STABLE"],
                  "eqn4": ["STABLE", "UNSTABLE"], "eqn5": ["STABLE", "UNSTABLE"],
                  "eqn6": ["STABLE", "UNSTABLE", "UNSTABLE"], "eqn7": ["STABLE", "UNSTABLE", "UNSTABLE"],
                  "eqn8": ["STABLE", "UNSTABLE"], "eqn9": ["STABLE"], "eqn10": ["STABLE"],
                  "fitzhugh_nagumo": ["UNSTABLE", "STABLE", "STABLE", "UNSTABLE", "STABLE", "UNSTABLE"]}


def exact_a1_wave(system, L=18.0):
    """Exact pulse (s, s', s^2, s^3) of family A1 as a Wave object, speed 0."""
    def profile(z):
        s, th = 1 / np.cosh(z), np.tanh(z)
        rows = [s, -s * th,                                   # u, u'
                -s * th, s * (2 * th ** 2 - 1),               # y = s', y' = s''
                s ** 2, -2 * s ** 2 * th,                     # v, v'
                s ** 3, -3 * s ** 3 * th]                     # w, w'
        return np.array(rows)
    zero = np.zeros(system.N)
    return Wave(c=0.0, L=L, N=system.N, Y_minus=zero, Y_plus=zero, interpolant=profile,
                mesh=np.linspace(-L, L, 3), info=dict(profile_error=0.0))


class Asymptotic(unittest.TestCase):
    def test_projector(self):
        rng = np.random.default_rng(3)
        V = rng.normal(size=(6, 6))
        M = V @ np.diag([2.0, 1.0, 0.5, -0.3, -1.0, -4.0]) @ np.linalg.inv(V)
        k, P, ev_u, ev_s = split(M)
        self.assertEqual(k, 3)
        np.testing.assert_allclose(P @ P, P, atol=1e-10)
        np.testing.assert_allclose(P @ M, M @ P, atol=1e-9)
        np.testing.assert_allclose(P @ V[:, :3], V[:, :3], atol=1e-9)
        np.testing.assert_allclose(P @ V[:, 3:], 0 * V[:, 3:], atol=1e-9)

    def test_essential_spectrum_of_fisher(self):
        system = LocalSystem(load_problem(PROBLEMS / "fisher_kpp.py"))
        s, sides = essential_bound(system.end_state([1.0], 2.5)[2], system.end_state([0.0], 2.5)[2])
        self.assertAlmostEqual(s, 1.0, places=9)
        self.assertAlmostEqual(sides["left"]["s"], -1.0, places=9)

    def test_essential_spectrum_of_a_partly_parabolic_system(self):
        # FitzHugh-Nagumo: the branch of w tends to Re = -eps*gamma as kappa -> infinity (not attained).
        system = LocalSystem(load_problem(PROBLEMS / "fhn_fast_pulse.py"))
        self.assertEqual(system.first_order_components, [1])
        a_k = system.end_state([0.0, 0.0], 0.4)[2]
        s, sides = essential_bound(a_k, a_k, first_order=system.first_order_components)
        self.assertAlmostEqual(s, -0.01, places=12)
        self.assertAlmostEqual(sides["left"]["limit_at_infinity"], -0.01, places=12)
        self.assertFalse(sides["left"]["growing_at_large_kappa"])


class Symbolic(unittest.TestCase):
    def test_nagumo_linearisation(self):
        system = LocalSystem(load_problem(PROBLEMS / "nagumo_front.py"))
        self.assertEqual((system.n, system.N, system.mu), (1, 2, [2]))
        U = np.array([0.2, 0.7])
        Y = np.vstack((U, [0.1, -0.3]))
        a = system.coefficients(Y, 0.4)                       # (3, 2, 1, 1)
        fprime = -3 * U ** 2 + 2 * 1.25 * U - 0.25
        np.testing.assert_allclose(a[0, :, 0, 0], fprime, atol=1e-14)
        np.testing.assert_allclose(a[1, :, 0, 0], 0.4, atol=1e-14)
        np.testing.assert_allclose(a[2, :, 0, 0], 1.0, atol=1e-14)

    def test_a1_exact_pulse_solves_the_profile_equation(self):
        system = LocalSystem(load_problem(PROBLEMS / "han_A1.py"))
        wave = exact_a1_wave(system)
        z = np.linspace(-6, 6, 241)
        Y = wave.Y(z)
        h = 1e-5
        numerical = (wave.Y(z + h) - wave.Y(z - h)) / (2 * h)
        np.testing.assert_allclose(system.rhs(Y, 0.0), numerical, atol=2e-9)


class Classifier(unittest.TestCase):
    def _problem(self, equations, unknowns, x, t, **options):
        return Problem(path=Path("memory"), name="memory", x=x, t=t, equations=equations,
                       unknowns=unknowns, parameters={}, wave={}, options=options)

    def test_classes(self):
        x, t, tau = sp.symbols("x t tau")
        f = sp.Function("u")
        u = f(x, t)
        s = sp.Symbol("s")
        cases = {
            "L": sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u)),
            "D": sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - f(x, t - tau))),
            "N": sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - sp.Integral(sp.exp(-s ** 2) * f(x - s, t), (s, -sp.oo, sp.oo)))),
            "OUT": sp.Eq(u.diff(t), u.diff(x, 2) + sp.sin(x) * u),
        }
        for expected, eq in cases.items():
            with self.subTest(expected=expected):
                out = classify(self._problem([eq], [u], x, t))
                self.assertEqual(out.sub_solver, expected)
                self.assertEqual(out.supported, expected != "OUT")
        # Second order in time: class L.  Order three, or u_tt behind a free boundary: not supported.
        second = classify(self._problem([sp.Eq(u.diff(t, 2), u.diff(x, 2) - u)], [u], x, t))
        self.assertEqual((second.sub_solver, second.supported), ("L", True))
        third = classify(self._problem([sp.Eq(u.diff(t, 3), u.diff(x, 2) - u)], [u], x, t))
        self.assertEqual((third.sub_solver, third.supported), ("L", False))
        moving = self._problem([sp.Eq(u.diff(t, 2), u.diff(x, 2) - u)], [u], x, t)
        moving.boundary = dict(conditions=[], speed=sp.Integer(0), values=[None])
        self.assertEqual((classify(moving).sub_solver, classify(moving).supported), ("F", False))
        # One delay and several delays are supported; a derivative of a delayed term is not.
        delayed = sp.Eq(u.diff(t), u.diff(x, 2) - u + f(x, t - tau) * f(x, t - 2 * tau))
        for eq in (cases["D"], delayed):
            out = classify(self._problem([eq], [u], x, t))
            self.assertEqual((out.sub_solver, out.supported), ("D", True))
        out = classify(self._problem([sp.Eq(u.diff(t), u.diff(x, 2) - u + f(x, t - tau).diff(x))], [u], x, t))
        self.assertEqual((out.sub_solver, out.supported), ("D", False))
        # Free boundary: sub-solver F for every form of the boundary conditions.
        problem = self._problem([cases["L"]], [u], x, t)
        problem.boundary = dict(conditions=[], speed=sp.Integer(0), values=[None])
        out = classify(problem)
        self.assertEqual((out.sub_solver, out.supported), ("F", True))


class EvansCalibration(unittest.TestCase):
    """Family A1 at q = 0, a = 4: the roots 3, 0, +-2i are known in closed form."""

    @classmethod
    def setUpClass(cls):
        problem = load_problem(PROBLEMS / "han_A1.py")
        problem.parameters = {k: {"q": 0.0, "a": 4.0}[str(k)] for k in problem.parameters}
        cls.system = LocalSystem(problem)
        cls.wave = exact_a1_wave(cls.system)
        zero = [0.0] * 4
        cls.asym = Asymptotics(cls.system, zero, zero, 0.0, 30.0)
        cls.engine = Evans.build(cls.system, cls.wave, cls.asym, 18.0, 18.0, 0.12)

    def test_known_roots(self):
        for exact in (3.0, 0.0, 2j, -2j):
            root = locate(self.engine, exact + 0.03, 0.02)
            self.assertLess(abs(root - exact), 2e-5)
            self.assertEqual(winding(self.engine, circle(exact, 0.1), 12)["count"], 1)

    def test_analytic(self):
        def D(lam):
            log_abs, phase, _ = self.engine(lam)
            return np.exp(log_abs) * phase
        lam, h = 1.3 + 0.7j, 1e-4
        dx = (D(lam + h) - D(lam - h)) / (2 * h)
        dy = (D(lam + 1j * h) - D(lam - 1j * h)) / (2j * h)
        self.assertLess(abs(dx - dy) / abs(dx), 2e-6)

    def test_empty_disk_and_certificate(self):
        out = winding(self.engine, circle(1.5 + 1.0j, 0.4), 12)
        self.assertEqual((out["count"], out["gamma_count"]), (0, 0))

    def test_subdivision_finds_every_root(self):
        # Independent root search: no seeds, only winding numbers of the Evans function.
        exact = [0.0, 3.0, 2j, -2j]
        found = isolate(self.engine, (-0.03, 5.0, -3.0, 3.0))
        self.assertEqual(len(found), 4)
        for z in exact:
            self.assertLess(min(abs(z - f) for f in found), 2e-5)
        rest = isolate(self.engine, (-0.03, 5.0, -3.0, 3.0), known=[0.0, 3.0])
        self.assertEqual(sorted(round(f.imag, 3) for f in rest), [-2.0, 2.0])

    def test_real_on_the_real_axis(self):
        # Parity index: D is real for real lambda and changes sign at the simple roots 0 and 3.
        phases = [self.engine(x)[1] for x in (-0.02, 1.0, 4.0)]
        for ph in phases:
            self.assertLess(abs(ph.imag), 1e-8)
        signs = [np.sign(ph.real) for ph in phases]
        self.assertEqual((signs[0] * signs[1], signs[1] * signs[2]), (-1.0, -1.0))


class WaveSolver(unittest.TestCase):
    def test_nagumo_speed_and_profile(self):
        problem = load_problem(PROBLEMS / "nagumo_front.py")
        system = LocalSystem(problem)
        wave = solve_wave(problem, system)
        self.assertTrue(wave.info["speed_is_unknown"])
        self.assertLess(abs(wave.c - 0.5 / np.sqrt(2)), 1e-7)
        z = np.linspace(-10, 10, 401)
        U = wave.Y(z)[0]
        shifts = np.linspace(-3, 3, 6001)
        exact = 1 / (1 + np.exp((z[:, None] - shifts[None, :]) / np.sqrt(2)))
        self.assertLess(np.abs(exact - U[:, None]).max(axis=0).min(), 2e-3)   # up to translation

    def test_cubic_pulse(self):
        problem = load_problem(PROBLEMS / "cubic_pulse.py")
        system = LocalSystem(problem)
        wave = solve_wave(problem, system)
        self.assertLess(abs(wave.c), 1e-7)
        self.assertLess(abs(np.abs(wave.Y(np.linspace(-5, 5, 2001))[0]).max() - 1.0), 1e-5)

    def test_a_guess_with_a_kink_is_rejected(self):
        problem = load_problem(PROBLEMS / "cubic_pulse.py")
        problem.wave["guess"] = {"u": sp.exp(-sp.Abs(sp.Symbol("z")))}
        with self.assertRaises(WaveError):
            reference_from_guess(problem, LocalSystem(problem))


class Continuation(unittest.TestCase):
    def test_nagumo_speed_along_the_branch(self):
        # c(a) = (1 - 2a)/sqrt(2) at every point of the branch.
        problem = load_problem(PROBLEMS / "nagumo_front.py")
        system = LocalSystem(problem)
        wave = solve_wave(problem, system)
        branch = trace(system, problem.wave["left"], problem.wave["right"], wave, "a", until=0.45)
        self.assertEqual(branch.status, "reached")
        self.assertGreater(len(branch.points), 4)
        for point in branch.points:
            self.assertLess(abs(point.c - (1 - 2 * point.value) / np.sqrt(2)), 1e-7)
        self.assertAlmostEqual(branch.points[-1].value, 0.45, places=12)
        self.assertEqual(branch.folds, [])

    @slow
    def test_fhn_label_changes_at_the_fold(self):
        # One branch, fast pulse -> fold in eps -> slow pulse.  A real eigenvalue crosses zero at the fold.
        path = PROBLEMS / "fhn_fast_pulse.py"
        problem = load_problem(path)
        system = LocalSystem(problem)
        wave = solve_wave(problem, system)
        branch = trace(system, problem.wave["left"], problem.wave["right"], wave, "epsilon",
                       direction=+1, until=0.01)
        self.assertEqual((branch.status, len(branch.folds)), ("reached", 1))
        fold = branch.folds[0]
        self.assertGreater(fold["value"], 0.01)
        slow = []
        for point in branch.points:
            record = analyse(path, parameters=dict(epsilon=point.value), start=point.wave)
            before = point.c > fold["speed"]
            self.assertEqual(record["label"], "STABLE" if before else "UNSTABLE", record["reason"])
            if not before:
                slow.append((point.c, record["roots"][0]["value"][0]))
        self.assertGreaterEqual(len(slow), 2)
        (c1, l1), (c2, l2) = sorted(slow, reverse=True)[:2]      # the two points nearest to the fold
        crossing = c1 + l1 * (c1 - c2) / (l2 - l1)               # speed at which the eigenvalue is zero
        self.assertLess(abs(crossing - fold["speed"]), 2e-3)


class FreeBoundary(unittest.TestCase):
    """u_t = u_xx - u on x < s(t), u = 1 and ds/dt = kappa (u_x - 1) at x = s(t): closed forms."""

    def test_linear_stefan_problem_in_closed_form(self):
        def second(kappa):
            m = -1 / (1 + kappa)
            return (1 - m * m) / m, kappa * (1 - m)          # speed and eigenvalue of the second wave
        # (file, kappa, label, speed, unstable eigenvalue)
        cases = [("stefan_linear.py", -3.0, "UNSTABLE", 0.0, 3.0),          # kappa (kappa + 2)
                 ("stefan_linear.py", 1.0, "STABLE", 0.0, None),            # no eigenvalue
                 ("stefan_linear.py", -1.5, "STABLE", 0.0, None),           # eigenvalue -0.75
                 ("stefan_linear_moving.py", -3.0, "STABLE", second(-3.0)[0], None),
                 ("stefan_linear_moving.py", -1.5, "UNSTABLE", *second(-1.5)),
                 ("stefan_linear_moving.py", -2.3, "STABLE", second(-2.3)[0], None)]
        self.assertAlmostEqual(second(-3.0)[0], 1.5)
        self.assertAlmostEqual(second(-1.5)[1], 1.5)
        for name, kappa, label, speed, eigenvalue in cases:
            with self.subTest(file=name, kappa=kappa):
                record = analyse(PROBLEMS / name, parameters=dict(kappa=kappa))
                self.assertEqual(record["label"], label, record["reason"])
                self.assertEqual(record["classification"]["sub_solver"], "F")
                self.assertLess(abs(record["wave"]["speed"] - speed), 1e-7)
                if eigenvalue is None:
                    self.assertEqual((record["decided_in_pass"], record["pass_1"]["W"]), (2, 0))
                    self.assertTrue(all(c["passed"] for c in record["pass_2"]["checks"]))
                else:
                    root = record["roots"][0]
                    self.assertLess(abs(root["value"][0] - eigenvalue), 1e-3)
                    self.assertLess(abs(root["value"][0] - eigenvalue), root["error"])

    def test_stable_eigenvalue_of_a_moving_wave(self):
        # kappa = -2.3: the second wave moves with c = 0.5308 and has the eigenvalue kappa (1 - m) < 0.
        kappa = -2.3
        m = -1 / (1 + kappa)
        problem = load_problem(PROBLEMS / "stefan_linear_moving.py", dict(kappa=kappa))
        system = LocalSystem(problem)
        wave = solve_semi_wave(problem, system)
        z = np.linspace(-8.0, 0.0, 401)
        self.assertLess(np.abs(wave.Y(z)[0] - np.exp(m * z)).max(), 1e-7)
        asym = BoundaryAsymptotics(system, problem.wave["left"], wave, 5.0)
        engine = build_engine(system, wave, asym, wave.L, 0.05)
        self.assertLess(abs(locate(engine, -0.5, 0.02) - kappa * (1 - m)), 1e-5)
        self.assertEqual(winding(engine, circle(1.0, 0.9), 16)["count"], 0)

    def test_fisher_stefan_semi_wave(self):
        # Speed for small kappa: c / kappa = (1 - 0.63923 kappa) / sqrt(3) + O(kappa^2).
        record = analyse(PROBLEMS / "fisher_stefan.py", parameters=dict(kappa=0.02))
        self.assertEqual(record["label"], "STABLE", record["reason"])
        self.assertLess(abs(record["wave"]["speed"] / 0.02 - (1 - 0.63923 * 0.02) / np.sqrt(3)), 5e-4)
        record = analyse(PROBLEMS / "fisher_stefan.py")
        self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
        self.assertTrue(all(c["passed"] for c in record["pass_2"]["checks"]))
        self.assertEqual(record["region"]["source"], "energy estimate")
        self.assertTrue(0 < record["wave"]["speed"] < 2)


class FreeBoundaryGeneral(unittest.TestCase):
    """Robin and Neumann conditions, the speed of the boundary inside a condition, components without diffusion."""

    @staticmethod
    def feedback_reference(a, b, c):
        """stefan_feedback.py with speed c: kappa and the characteristic function (independent of twstab).

        Modes exp(sigma z), u-component 1, w-component b / (lam + 1 - c sigma):
        (sigma^2 + c sigma - 1 - lam)(c sigma - 1 - lam) = a b,  Re sigma > 0.
        """
        def modes(lam):
            poly = np.polymul([1.0, c, -1.0 - lam], [c, -1.0 - lam]).astype(complex)
            poly[-1] -= a * b
            s = np.roots(poly)
            s = s[s.real > 0]
            return s, b / (lam + 1 - c * s)

        m, v = modes(0.0)
        A = np.linalg.solve(np.array([[1.0, 1.0], v]), np.array([1.0, 0.0], dtype=complex))   # u = 1, w = 0
        dU, d2U, dW = (A * m).sum().real, (A * m ** 2).sum().real, (A * v * m).sum().real
        kappa = c / (dU - 1)

        def char(lam):
            s, vq = modes(lam)
            M = np.array([[dU, 1, 1], [dW, vq[0], vq[1]], [kappa * d2U - lam, kappa * s[0], kappa * s[1]]])
            return np.linalg.det(M) / (lam * (s[0] - s[1]))
        return kappa, char

    def check(self, cases):
        for name, parameters, label, speed, eigenvalue in cases:
            with self.subTest(file=name, parameters=parameters):
                record = analyse(PROBLEMS / name, parameters=parameters)
                self.assertEqual(record["label"], label, record["reason"])
                self.assertEqual(record["region"]["source"], "energy estimate")
                self.assertLess(abs(record["wave"]["speed"] - speed), 1e-7)
                if eigenvalue is None:
                    self.assertEqual((record["decided_in_pass"], record["pass_1"]["W"]), (2, 0))
                    self.assertTrue(all(c["passed"] for c in record["pass_2"]["checks"]))
                else:
                    root = record["roots"][0]
                    self.assertLess(abs(root["value"][0] - eigenvalue), 1e-3)
                    self.assertLess(abs(root["value"][0] - eigenvalue), root["error"])

    def test_robin_and_neumann_conditions_in_closed_form(self):
        mu = -1 + np.sqrt(2.5)                                   # second wave, kappa = 1.5: m = 0.5
        second = (mu - 0.5) * (mu + 2.0)
        self.assertAlmostEqual(second, 0.209431, places=6)
        self.check([("stefan_robin.py", dict(kappa=6.0, r=0.0), "UNSTABLE", 0.0, 3.0),
                    ("stefan_robin.py", dict(kappa=9.0, r=1.0), "UNSTABLE", 0.0, 3.0),
                    ("stefan_robin.py", dict(kappa=2.0, r=1.0), "STABLE", 0.0, None),      # eigenvalue -0.83
                    ("stefan_robin_moving.py", dict(kappa=6.0), "STABLE", -4.8, None),
                    ("stefan_robin_moving.py", dict(kappa=1.5), "UNSTABLE", 1.5, second),
                    ("stefan_robin_flux.py", None, "UNSTABLE", 0.0, 3.0),                  # ds/dt in the condition
                    ("stefan_mixed.py", dict(kappa2=12.0), "UNSTABLE", 0.0, 3.0),
                    ("stefan_mixed.py", dict(kappa2=1.0), "STABLE", 0.0, None)])

    def test_components_without_diffusion(self):
        mu = -1 + np.sqrt(2.5)
        self.check([("stefan_passive.py", None, "STABLE", 1.5, None),                      # inflow
                    ("stefan_passive_outflow.py", None, "UNSTABLE", -1.5, 1.5),            # no condition for w
                    ("stefan_robin_passive.py", None, "UNSTABLE", 1.5, (mu - 0.5) * (mu + 2.0))])

    def test_feedback_against_the_characteristic_determinant(self):
        far = dict(kind="semi", far=dict(u=0, w=0), width=1.2)
        # c = 1: one real unstable eigenvalue, the zero of the characteristic function.
        kappa, char = self.feedback_reference(-1.0, 0.5, 1.0)
        exact = brentq(lambda s: char(s).real, 0.5, 2.0)
        record = analyse(PROBLEMS / "stefan_feedback.py", parameters=dict(kappa=kappa), wave=dict(far, speed_guess=0.9))
        self.assertEqual(record["label"], "UNSTABLE", record["reason"])
        self.assertLess(abs(record["wave"]["speed"] - 1.0), 1e-7)
        root = record["roots"][0]
        self.assertLess(abs(root["value"][0] - exact), min(1e-4, root["error"]))
        # c = 2.5: stable.  The characteristic function has no zero in the search rectangle of the solver.
        kappa, char = self.feedback_reference(-1.0, 0.5, 2.5)
        record = analyse(PROBLEMS / "stefan_feedback.py", parameters=dict(kappa=kappa), wave=dict(far, speed_guess=2.25))
        self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
        self.assertLess(abs(record["wave"]["speed"] - 2.5), 1e-7)
        self.assertEqual(record["region"]["bounds"]["gamma0"], 1.0)
        x1, y1 = record["region"]["x_max"], record["region"]["y_max"]
        corners = [-0.05 - 1j * y1, x1 - 1j * y1, x1 + 1j * y1, -0.05 + 1j * y1, -0.05 - 1j * y1]
        contour = np.concatenate([np.linspace(p, q, 3000, endpoint=False) for p, q in zip(corners[:-1], corners[1:])])
        values = np.array([char(s) for s in np.append(contour, contour[0])])
        steps = np.angle(values[1:] / values[:-1])
        self.assertLess(np.abs(steps).max(), 1.0)
        self.assertEqual(round(steps.sum() / (2 * np.pi)), 0)

    def test_the_estimate_holds_on_computed_eigenfunctions(self):
        # Each inequality of the search bound, evaluated on eigenfunctions of the difference operator.
        # r = -0.5 makes the Robin boundary term positive (sigma_b > 0); the margins there are small.
        cases = [("stefan_linear.py", dict(kappa=-3.0), 3.0), ("stefan_robin.py", dict(kappa=9.0, r=-0.5), 7.0),
                 ("stefan_robin.py", dict(kappa=2.0, r=1.0), -0.8), ("stefan_robin_moving.py", dict(kappa=1.5), 0.2),
                 ("stefan_mixed.py", dict(kappa2=12.0), 3.0), ("stefan_robin_passive.py", None, 0.2),
                 ("stefan_feedback.py", None, 1.0)]
        nz = 2000
        for name, parameters, shift in cases:
            with self.subTest(file=name, parameters=parameters):
                problem = load_problem(PROBLEMS / name, parameters)
                system = LocalSystem(problem)
                wave = solve_semi_wave(problem, system)
                bounds = boundary_bounds(system, wave, coefficient_bounds(system, wave))
                self.assertTrue(bounds["applies"], bounds["reason"])
                A, B = fd_operator_boundary(system, wave, wave.L, nz)
                lam, vector = eigs(A, k=1, M=B, sigma=shift, tol=1e-10)
                lam, p = complex(lam[0]), vector[:, 0].reshape(system.n, nz + 1)
                self.assertLess(abs(lam - shift), 0.2 * max(1.0, abs(shift)))
                z = np.linspace(-wave.L, 0.0, nz + 2)[1:]
                h = z[1] - z[0]
                norm = lambda f: float(np.sqrt(np.trapezoid((np.abs(f) ** 2).sum(axis=0), z)))
                Hs = list(system.first_order_components)
                Ps = [j for j in range(system.n) if j not in Hs]
                gradient = system.stefan(wave.Y_plus)[1]
                eta = sum(gradient[system.offset[j]] * p[j, -1]
                          + (gradient[system.offset[j] + 1] * (3 * p[j, -1] - 4 * p[j, -2] + p[j, -3]) / (2 * h)
                             if system.mu[j] == 2 else 0.0) for j in range(system.n))
                scale = norm(p[Ps])
                t, e = norm(np.gradient(p[Ps], z, axis=1)) / scale, abs(eta) / scale
                estimate = boundary_estimate(bounds, max(0.1, 1e-3 - lam.real))
                self.assertLessEqual(lam.real, estimate["re"](t, e))
                self.assertLessEqual(abs(lam.imag), estimate["im"](t, e))
                self.assertLessEqual(e, estimate["e_max"](t))
                if Hs:
                    self.assertLessEqual(norm(p[Hs]) / scale, estimate["rho"](t, e))

    def test_input(self):
        # The default guess takes its boundary value from the condition: b / width = 1 for u_x = 1.
        problem = load_problem(PROBLEMS / "stefan_robin.py")
        system = LocalSystem(problem)
        self.assertAlmostEqual(reference_from_guess(problem, system).Y(np.array([0.0]))[0, 0], 1.2)
        self.assertEqual(problem.boundary["values"], [None])
        # The speed symbol is replaced by the Stefan expression.
        flux = load_problem(PROBLEMS / "stefan_robin_flux.py")
        self.assertFalse(flux.boundary["conditions"][0].has(sp.Symbol("sdot", real=True)))
        # More conditions than unknowns, or the speed symbol inside the Stefan expression: rejected.
        x, t, s = sp.Symbol("x"), sp.Symbol("t"), sp.Symbol("s")
        u = sp.Function("u")(x, t)
        for spec in (dict(conditions=[sp.Eq(u, 1), sp.Eq(u.diff(x), 0)], speed=u),
                     dict(conditions=[sp.Eq(u, 1)], speed=u + s, speed_symbol=s)):
            with self.assertRaises(ProblemError):
                problem_module._boundary(spec, [u], {}, x)
        # The bound does not apply to conditions that couple two components: UNCERTAIN, not a wrong region.
        problem = load_problem(PROBLEMS / "stefan_mixed.py")
        system = LocalSystem(problem)
        wave = solve_semi_wave(problem, system)
        jacobian = system.boundary_jacobian
        system.boundary_jacobian = lambda Y: jacobian(Y) + np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        bounds = boundary_bounds(system, wave, coefficient_bounds(system, wave))
        self.assertFalse(bounds["applies"])
        self.assertIn("one component each", bounds["reason"])


class CrossDiffusion(unittest.TestCase):
    """Nagumo front with a passive component and cross-diffusion: lambda_n = (s - n)^2/8 - 1 - c^2/4."""

    C2 = (1 - 2 * 0.25) ** 2 / 2                                  # c^2

    def test_unstable_front_with_three_known_eigenvalues(self):
        exact = [(5 - n) ** 2 / 8 - 1 - self.C2 / 4 for n in (0, 1, 2)]       # sigma = 15, s = 5
        self.assertEqual(exact, [2.09375, 0.96875, 0.09375])
        record = analyse(PROBLEMS / "cross_diffusion_front.py", parity_shortcut=False)
        self.assertEqual((record["label"], record["decided_in_pass"]), ("UNSTABLE", 1), record["reason"])
        self.assertEqual((record["pass_1"]["W"], record["pass_1"]["m0"]), (4, 1))
        found = sorted((r["value"][0] for r in record["roots"] if r["status"] == "unstable"), reverse=True)
        np.testing.assert_allclose(found, exact, atol=2e-5)
        # The symmetric part of A2 is indefinite (beta = 3): the bound needs the scaled variables.
        self.assertIn("scaling", record["region"]["bounds"])
        self.assertLess(max(exact), record["region"]["x_max"])
        fast = analyse(PROBLEMS / "cross_diffusion_front.py")
        self.assertEqual((fast["label"], fast["pass_1"]["shortcut"]), ("UNSTABLE", "parity"))

    def test_stable_front(self):
        record = analyse(PROBLEMS / "cross_diffusion_front.py", parameters=dict(sigma=3.0))   # s = 2
        self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
        self.assertLess(abs(record["wave"]["speed"] - np.sqrt(self.C2)), 1e-7)

    def sigma_for(self, eigenvalue):
        s = np.sqrt(8 * (1 + self.C2 / 4 + eigenvalue))
        return s * (s + 1) / 2

    def test_a_small_eigenvalue_in_the_origin_disk_is_resolved(self):
        # Largest eigenvalue +0.0005: inside the origin disk (radius 0.031), next to the translation zero.
        record = analyse(PROBLEMS / "cross_diffusion_front.py", parameters=dict(sigma=self.sigma_for(5e-4)))
        self.assertEqual(record["label"], "UNSTABLE", record["reason"])
        self.assertEqual(record["pass_1"]["m0"], 2)
        root = record["roots"][0]
        self.assertTrue(root["near_origin"])
        self.assertLess(abs(root["value"][0] - 5e-4), root["error"])
        self.assertLess(root["error"], 2e-5)

    @slow
    def test_an_eigenvalue_below_the_resolution_gives_uncertain(self):
        # Largest eigenvalue +1e-9: it cannot be separated from the translation zero.
        record = analyse(PROBLEMS / "cross_diffusion_front.py", parameters=dict(sigma=self.sigma_for(1e-9)))
        self.assertEqual(record["label"], "UNCERTAIN")
        self.assertEqual(record["pass_1"]["m0"], 2)


class Nonlocal(unittest.TestCase):
    """Convolution with K = exp(-|x|)/2, written as an unknown w with the constraint w_xx - w + g(u) = 0."""

    A = 0.25
    C = (1 - 2 * 0.25) / np.sqrt(2)

    def integral_form(self, m, Lg=15.0):
        """Largest eigenvalues of the linearisation in its integral form (no auxiliary unknown).

        p'' + c p' + (f'(U) - m B'(U)) p + m K * (g'(U) p): second-order differences and the
        trapezoid rule on 150 and 300 points, then Richardson extrapolation.
        """
        def top(n):
            z = np.linspace(-Lg, Lg, n + 2)[1:-1]
            h = z[1] - z[0]
            U = 1 / (1 + np.exp(z / np.sqrt(2)))
            fp = -3 * U ** 2 + 2 * (1 + self.A) * U - self.A
            Bp = 2 * U * (1 - U) * (1 - 2 * U)
            gp = (1 - 2 * U) * U * (1 - U) * (30 * U * (1 - U) - 2)
            M = (np.eye(n, k=1) - 2 * np.eye(n) + np.eye(n, k=-1)) / h ** 2
            M += self.C * (np.eye(n, k=1) - np.eye(n, k=-1)) / (2 * h) + np.diag(fp - m * Bp)
            M += m * 0.5 * np.exp(-np.abs(z[:, None] - z[None, :])) * h * gp[None, :]
            ev = np.linalg.eigvals(M)
            ev = ev[ev.imag >= 0]
            return ev[np.lexsort((ev.imag, -ev.real))][:4]
        coarse, fine = top(150), top(300)
        return fine + (fine - coarse) / 3

    def test_constraint_in_the_symbolic_layer(self):
        system = LocalSystem(load_problem(PROBLEMS / "nonlocal_core_front.py"))
        self.assertEqual((list(system.mass), system.mu, system.N), ([1.0, 0.0], [2, 2], 4))
        # Exact wave: u = Nagumo front, w = B(u).
        z = np.linspace(-9, 9, 181)
        U = 1 / (1 + np.exp(z / np.sqrt(2)))
        dU = -U * (1 - U) / np.sqrt(2)
        B, dB = U ** 2 * (1 - U) ** 2, 2 * U * (1 - U) * (1 - 2 * U) * dU
        Y = np.array([U, dU, B, dB])
        numerical = np.gradient(Y, z, axis=1)
        self.assertLess(np.abs(system.rhs(Y, self.C)[[0, 2]] - Y[[1, 3]]).max(), 1e-14)
        self.assertLess(np.abs(system.rhs(Y, self.C) - numerical)[:, 2:-2].max(), 2e-3)

    def test_essential_spectrum_with_a_convolution(self):
        # nonlocal_front.py: sup Re = -a - mu/2 (closed form, attained at kappa = 0 for mu >= -2).
        for mu, label in ((1.0, "STABLE"), (-0.4, "STABLE"), (-0.6, "UNSTABLE")):
            with self.subTest(mu=mu):
                record = analyse(PROBLEMS / "nonlocal_front.py", parameters=dict(mu=mu))
                self.assertAlmostEqual(record["essential"]["s_ess"], -self.A - mu / 2, places=9)
                self.assertEqual(record["label"], label, record["reason"])
                if label == "STABLE":
                    self.assertLess(abs(record["wave"]["speed"] - self.C), 1e-7)

    def test_point_spectrum_against_the_integral_form(self):
        path = PROBLEMS / "nonlocal_core_front.py"
        stable = analyse(path, parameters=dict(m=5.0))
        self.assertEqual((stable["label"], stable["decided_in_pass"]), ("STABLE", 2), stable["reason"])
        self.assertLess(self.integral_form(5.0)[1].real, -0.1)       # [0] is the translation eigenvalue
        for m, count in ((-20.0, 3), (-50.0, 5)):
            with self.subTest(m=m):
                record = analyse(path, parameters=dict(m=m), parity_shortcut=False)
                self.assertEqual(record["label"], "UNSTABLE", record["reason"])
                self.assertLess(abs(record["wave"]["speed"] - self.C), 1e-7)
                self.assertEqual((record["pass_1"]["W"], record["pass_1"]["complete_count"]), (count, True))
                self.assertIn("constraint_coupling", record["region"]["bounds"])
                found = sorted((complex(*r["value"]) for r in record["roots"] if r["status"] == "unstable"),
                               key=lambda v: -v.real)
                reference = [v for v in self.integral_form(m) if v.real > 0.1]
                self.assertEqual(len(found), len(reference))
                np.testing.assert_allclose(found, reference, atol=3e-3)


@slow
class MultiPulse(unittest.TestCase):
    """FitzHugh-Nagumo double pulses in the oscillatory wake of the fast pulse."""

    @classmethod
    def setUpClass(cls):
        cls.records = {D0: analyse(PROBLEMS / "fhn_double_pulse.py", parameters=dict(D0=D0))
                       for D0 in (51.0, 70.0, 90.0)}
        problem = load_problem(PROBLEMS / "fhn_fast_pulse.py")
        system = LocalSystem(problem)
        c = cls.records[51.0]["wave"]["primary_pulse"]["speed"]
        ev = np.linalg.eigvals(system.end_state(problem.wave["left"], c)[0])
        lead = ev[ev.real > 0][0]
        cls.alpha, cls.beta = lead.real, abs(lead.imag)            # the wake behaves like exp(alpha z) cos(beta z)

    def near_zero(self, D0):
        (root,) = [r for r in self.records[D0]["roots"] if r.get("near_origin")]
        return root

    def test_labels_alternate(self):
        # Theory: double pulses of a stable pulse are alternately stable and unstable.
        labels = [self.records[D0]["label"] for D0 in (51.0, 70.0, 90.0)]
        self.assertEqual(labels, ["STABLE", "UNSTABLE", "STABLE"])
        for D0 in (51.0, 90.0):
            self.assertTrue(all(c["passed"] for c in self.records[D0]["pass_2"]["checks"]))

    def test_distances_follow_the_tail_frequency(self):
        D = [self.records[D0]["wave"]["distances"][0] for D0 in (51.0, 70.0, 90.0)]
        period = np.pi / self.beta
        self.assertLess(abs(D[1] - D[0] - period), 0.03 * period)
        self.assertLess(abs(D[2] - D[1] - period), 0.005 * period)
        for D0, record in self.records.items():
            wave = record["wave"]
            self.assertLess(abs(wave["speed"] - wave["primary_pulse"]["speed"]), 1e-9)
            self.assertLess(abs(wave["speed_defects"][0]), 1e-8)

    def test_interaction_eigenvalue_by_two_independent_routes(self):
        # Route 1: zero of the Evans function next to the origin.  Route 2: -sigma'(D) of the reduction.
        # The reduction is asymptotic for weak interaction: the agreement improves with the distance.
        for D0, tolerance in ((70.0, 0.12), (90.0, 0.03)):
            root = self.near_zero(D0)
            predicted = self.records[D0]["wave"]["predicted_interaction_eigenvalues"][0]
            self.assertGreater(abs(root["value"][0]), 10 * root["error"])
            self.assertLess(abs(root["value"][0] - predicted), tolerance * abs(predicted))
        # Theory: successive interaction eigenvalues shrink by exp(-alpha pi / beta) and change sign.
        ratio = self.near_zero(90.0)["value"][0] / self.near_zero(70.0)["value"][0]
        self.assertLess(abs(-ratio - np.exp(-self.alpha * np.pi / self.beta)), 0.03 * abs(ratio))

    def test_triple_pulse(self):
        # Each gap behaves like a double pulse: gaps 51 and 90 are both stable distances.
        record = analyse(PROBLEMS / "fhn_triple_pulse.py")
        self.assertEqual((record["label"], record["wave"]["copies"]), ("STABLE", 3), record["reason"])
        (root,) = [r for r in record["roots"] if r.get("near_origin")]
        self.assertLess(abs(root["value"][0] - self.near_zero(90.0)["value"][0]), 2e-6)
        self.assertLess(max(abs(d) for d in record["wave"]["speed_defects"]), 1e-8)


class Delay(unittest.TestCase):
    """One discrete time delay: second engine (determinant of a banded matrix)."""

    def test_essential_spectrum_against_the_lambert_function(self):
        # Scalar end states: sup Re = A + W0(k tau exp(-A tau)) / tau with A = f'(u) - k.
        from scipy.special import lambertw

        def closed_form(a, k, tau):
            A = -a - k
            return A + lambertw(k * tau * np.exp(-A * tau), 0).real / tau

        record = analyse(PROBLEMS / "delay_nagumo_front.py")                       # k = 0.5, tau = 2
        self.assertAlmostEqual(record["essential"]["sides"]["right"]["s"], closed_form(0.25, 0.5, 2.0), places=9)
        self.assertAlmostEqual(record["essential"]["sides"]["left"]["s"], closed_form(0.75, 0.5, 2.0), places=9)
        unstable = analyse(PROBLEMS / "delay_nagumo_front.py", parameters=dict(k=-2.0, tau=2.0))
        self.assertEqual((unstable["label"], unstable["decided_in_pass"]), ("UNSTABLE", 1))
        self.assertAlmostEqual(unstable["essential"]["s_ess"], closed_form(0.25, -2.0, 2.0), places=8)
        self.assertGreater(unstable["essential"]["s_ess"], 1.6)

    @slow
    def test_monotone_delayed_front_is_stable(self):
        # Quasi-monotone case (k > 0): the front is stable for every delay (Smith and Zhao 2000).
        speeds = []
        for pars in (dict(tau=0.01), dict(tau=2.0), dict(tau=6.0), dict(k=2.0, tau=4.0)):
            with self.subTest(**pars):
                record = analyse(PROBLEMS / "delay_nagumo_front.py", parameters=pars)
                self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
                self.assertEqual(record["classification"]["sub_solver"], "D")
                self.assertTrue(all(c["passed"] for c in record["pass_2"]["checks"]))
                speeds.append(record["wave"]["speed"])
        self.assertLess(abs(speeds[0] - 0.5 / np.sqrt(2)), 3e-3)                  # tau -> 0: the Nagumo speed
        self.assertTrue(speeds[0] > speeds[1] > speeds[2] > 0)                    # the delay slows the front

    def test_translation_zero_converges_with_fourth_order(self):
        problem = load_problem(PROBLEMS / "delay_nagumo_front.py")
        system = delay.DelaySystem(problem)
        wave = delay.solve_wave(problem, system)
        zeros = [abs(locate(delay.DelayEngine(system, wave, wave.L, wave.L, h), 0.01, 0.005))
                 for h in (0.2, 0.1)]
        self.assertLess(zeros[0], 1e-6)
        self.assertTrue(8 < zeros[0] / zeros[1] < 32)

    @staticmethod
    def kink_residual(lam, sigma, tau):
        """Distance of lambda from the roots of  lambda + 1 = (s - n)^2/2,  s(s + 1) = 2 sigma exp(-lambda tau)."""
        s = (-1 + np.sqrt(1 + 8 * sigma * np.exp(-lam * tau) + 0j)) / 2
        return min(abs(lam + 1 - (branch - n) ** 2 / 2) for branch in (s, -1 - s) for n in range(8)
                   if (branch - n).real > 0)

    @slow
    def test_standing_kink_against_the_scalar_equation(self):
        from scipy.optimize import brentq

        def real_root(sigma, tau, n):
            def f(lam):
                s = (-1 + np.sqrt(1 + 8 * sigma * np.exp(-lam * tau))) / 2
                return lam + 1 - (s - n) ** 2 / 2
            return brentq(f, -0.9, 10.0)

        record = analyse(PROBLEMS / "delay_kink_passive.py", parity_shortcut=False)      # sigma = 6, tau = 1
        self.assertEqual(record["label"], "UNSTABLE", record["reason"])
        self.assertLess(abs(record["wave"]["speed"]), 1e-9)
        self.assertTrue(record["pass_1"]["complete_count"])
        found = sorted((r["value"][0] for r in record["roots"] if r["value"][1] == 0), reverse=True)
        np.testing.assert_allclose(found, [real_root(6.0, 1.0, n) for n in (0, 1, 2)], atol=1e-4)
        for r in record["roots"]:                              # every located zero solves the scalar equation
            self.assertLess(self.kink_residual(complex(*r["value"]), 6.0, 1.0), 1e-3)
        self.assertTrue(any(r["value"][1] > 1 for r in record["roots"]))            # a root created by the delay
        stable = analyse(PROBLEMS / "delay_kink_passive.py", parameters=dict(sigma=0.5))
        self.assertEqual((stable["label"], stable["decided_in_pass"]), ("STABLE", 2), stable["reason"])

    @slow
    def test_oscillatory_instability_by_delayed_negative_feedback(self):
        record = analyse(PROBLEMS / "delay_kink_passive.py", parameters=dict(sigma=-6.0, tau=2.0))
        self.assertEqual(record["label"], "UNSTABLE", record["reason"])
        unstable = [complex(*r["value"]) for r in record["roots"] if r["status"] == "unstable"]
        self.assertTrue(unstable and all(z.imag > 0.5 for z in unstable))           # complex pairs only
        for z in unstable:
            self.assertLess(self.kink_residual(z, -6.0, 2.0), 1e-3)

    @slow
    def test_moving_front_against_the_generator_matrix(self):
        # The front moves: the delayed term is shifted by c tau.  Reference: independent discretisation
        # (second-order differences, Chebyshev points in the delay), Richardson in the space step.
        path = PROBLEMS / "delay_front_passive.py"
        record = analyse(path, parity_shortcut=False)                               # tau = 1
        self.assertEqual(record["label"], "UNSTABLE", record["reason"])
        self.assertLess(abs(record["wave"]["speed"] - 0.5 / np.sqrt(2)), 1e-7)
        found = sorted((r["value"][0] for r in record["roots"] if r["status"] == "unstable"), reverse=True)
        problem = load_problem(path)
        system = delay.DelaySystem(problem)
        wave = delay.solve_wave(problem, system)
        tops = []
        for nz in (400, 800):
            ev = fd_probe(delay.generator_matrix(system, wave, wave.L, wave.L, nz, M=14), [1.0, 0.3], k=6)
            tops.append(np.unique(np.round(ev[ev.real > 0.02].real, 9))[::-1])
        reference = tops[1] + (tops[1] - tops[0]) / 3
        self.assertEqual(len(found), len(reference))
        np.testing.assert_allclose(found, reference, atol=2e-4)
        # tau -> 0: the closed form of the problem without delay, (s - n)^2/8 - 1 - c^2/4 with s = 5.
        small = analyse(path, parameters=dict(tau=1e-3), parity_shortcut=False)
        near = sorted((r["value"][0] for r in small["roots"] if r["status"] == "unstable"), reverse=True)
        np.testing.assert_allclose(near, [2.09375, 0.96875, 0.09375], atol=0.01)


class DelaysSeveral(unittest.TestCase):
    """Several discrete delays and distributed delays (quadrature into discrete delays)."""

    FACTOR = dict(two=lambda lam, q: q["omega"] * np.exp(-lam * q["tau1"]) + (1 - q["omega"]) * np.exp(-lam * q["tau2"]),
                  gamma=lambda lam, q: (1 + q["T"] * lam) ** (-q["p"]),
                  uniform=lambda lam, q: (1 - np.exp(-lam * q["tau"])) / (lam * q["tau"]))

    @classmethod
    def residual(cls, lam, kind, q):
        """Distance of lambda from the roots of  lambda + 1 = (s - n)^2/2,  s(s + 1) = 2 sigma E(lambda)."""
        s = (-1 + np.sqrt(1 + 8 * q["sigma"] * cls.FACTOR[kind](lam, q) + 0j)) / 2
        return min(abs(lam + 1 - (branch - n) ** 2 / 2) for branch in (s, -1 - s) for n in range(8)
                   if (branch - n).real > 0)

    @classmethod
    def real_roots(cls, kind, q, count=2):
        def f(lam, n):
            s = (-1 + np.sqrt(1 + 8 * q["sigma"] * cls.FACTOR[kind](lam, q))) / 2
            return lam + 1 - (s - n) ** 2 / 2
        return [brentq(f, 1e-3, 10.0, args=(n,)) for n in range(count)]

    def kink(self, kind, parameters, label):
        record = analyse(PROBLEMS / f"delay_kink_{kind}.py", parameters=parameters, parity_shortcut=False)
        self.assertEqual(record["label"], label, record["reason"])
        self.assertEqual(record["classification"]["sub_solver"], "D")
        self.assertAlmostEqual(record["essential"]["s_ess"], -1.0, places=9)
        q = record["parameters"]
        if label == "STABLE":
            self.assertEqual(record["decided_in_pass"], 2)
            self.assertTrue(all(c["passed"] for c in record["pass_2"]["checks"]))
            return record
        self.assertTrue(record["pass_1"]["complete_count"])
        found = sorted((r["value"][0] for r in record["roots"] if r["status"] == "unstable"), reverse=True)
        np.testing.assert_allclose(found, self.real_roots(kind, q), atol=1e-4)
        for r in record["roots"]:                              # every located zero solves the scalar equation
            self.assertLess(self.residual(complex(*r["value"]), kind, q), 1e-3)
        return record

    def test_two_discrete_delays_against_the_scalar_equation(self):
        record = self.kink("two", None, "UNSTABLE")
        self.assertEqual(record["classification"]["details"]["delays"], 2)
        self.kink("two", dict(sigma=0.5), "STABLE")

    def test_distributed_delay_against_the_scalar_equation(self):
        record = self.kink("uniform", None, "UNSTABLE")                      # kernel 1 / tau on (0, tau)
        self.assertEqual(record["classification"]["details"]["distributed_delays"][0]["nodes"], 16)
        self.kink("uniform", dict(sigma=0.5), "STABLE")
        record = self.kink("gamma", None, "UNSTABLE")                        # exp(-s): roots 1 and 0.28134
        self.assertLess(abs(record["roots"][0]["value"][0] - 1.0), 1e-5)
        self.assertNotIn("quadrature_rebuilds", record)                      # a standing wave: no shift

    @slow
    def test_gamma_kernel_of_order_two_and_a_stable_case(self):
        self.kink("gamma", dict(p=2.0, T=0.5), "UNSTABLE")
        self.kink("gamma", dict(sigma=0.5), "STABLE")

    @slow
    def test_distributed_delay_of_a_moving_front_by_two_routes(self):
        # The exponential kernel is also a local system (linear chain): engine 1 gives the reference.
        first = analyse(PROBLEMS / "delay_front_distributed.py", parity_shortcut=False)
        second = analyse(PROBLEMS / "chain_front_passive.py", parity_shortcut=False)
        self.assertEqual((first["label"], second["label"]), ("UNSTABLE", "UNSTABLE"))
        self.assertEqual((first["classification"]["sub_solver"], second["classification"]["sub_solver"]), ("D", "L"))
        self.assertLess(abs(first["wave"]["speed"] - second["wave"]["speed"]), 1e-8)
        roots = [sorted((r["value"][0] for r in rec["roots"] if r["status"] == "unstable"), reverse=True)
                 for rec in (first, second)]
        self.assertEqual(len(roots[0]), 2)
        np.testing.assert_allclose(roots[0], roots[1], atol=2e-5)
        for pars in (None, dict(k=1.0, T=1.0)):
            with self.subTest(parameters=pars):
                first = analyse(PROBLEMS / "delay_nagumo_distributed.py", parameters=pars)
                second = analyse(PROBLEMS / "chain_nagumo.py", parameters=pars)
                self.assertEqual((first["label"], second["label"]), ("STABLE", "STABLE"), first["reason"])
                self.assertTrue(all(c["passed"] for c in first["pass_2"]["checks"]))
                self.assertLess(abs(first["wave"]["speed"] - second["wave"]["speed"]), 1e-8)
                self.assertLess(abs(first["essential"]["s_ess"] - second["essential"]["s_ess"]), 1e-8)

    def test_quadrature_of_a_kernel(self):
        rule = delay._quadrature(lambda s: np.exp(-s), 0.0, np.inf)                 # mean delay 1: frequency 12
        self.assertLess(rule["nodes"].size, 200)
        self.assertAlmostEqual(rule["frequency"], 12.0, places=9)
        self.assertLess(abs(rule["decay"] - 1.0), 0.05)
        self.assertLess(abs(rule["weights"].sum() - 1.0), 1e-13)                    # the integral is kept
        for mu in (0.5, 3j, 12j, -0.2 + 5j, 2 - 9j):
            self.assertLess(abs((rule["weights"] * np.exp(-mu * rule["nodes"])).sum() - 1 / (1 + mu)), 1e-6)
        finer = delay._quadrature(lambda s: np.exp(-s), 0.0, np.inf, frequency=40.0)
        self.assertGreater(finer["nodes"].size, rule["nodes"].size)
        self.assertLess(abs((finer["weights"] * np.exp(-40j * finer["nodes"])).sum() - 1 / (1 + 40j)), 1e-6)
        box = delay._quadrature(lambda s: 0.5 + 0 * s, 0.0, 2.0)                    # 1 / tau on (0, tau)
        self.assertEqual((box["nodes"].size, box["decay"]), (16, np.inf))
        with self.assertRaises(StructureError):
            delay._quadrature(lambda s: 1 / (1 + s), 0.0, np.inf)                   # not integrable

    def test_essential_spectrum_with_several_delays(self):
        # Scalar end state: lambda = A + k1 exp(-lambda tau1) + k2 exp(-lambda tau2); k1, k2 > 0: the
        # rightmost root is real (at wavenumber 0).
        A, k1, k2, taus = -1.5, 0.4, 0.7, np.array([1.0, 2.5])
        a_k = np.array([[[A]], [[0.3]], [[1.0]]])
        exact = brentq(lambda s: s - A - k1 * np.exp(-s * taus[0]) - k2 * np.exp(-s * taus[1]), -1.5, 1.0)
        s, sides = delay.essential_bound((a_k, np.array([[[k1]], [[k2]]])), (a_k, np.array([[[k1]], [[k2]]])), 0.3, taus)
        self.assertLess(abs(s - exact), 1e-11)
        self.assertTrue(sides["left"]["corrected_by_newton"])
        # A long delay with a very small coefficient: the root near A counts, not the eigenvalues of
        # the discretised history (they lie near -4 / tau and approximate nothing).
        k, tau = 1e-30, 40.0
        exact = brentq(lambda s: s - A - k * np.exp(-s * tau), A, A + 0.01)
        s, _ = delay.essential_bound((a_k, np.array([[[k]]])), (a_k, np.array([[[k]]])), 0.3, np.array([tau]))
        self.assertLess(abs(s - exact), 1e-9)
        self.assertLess(s, -1.4)

    def test_input_errors(self):
        x, t = sp.symbols("x t", real=True)
        s = sp.Symbol("s", positive=True)
        f = sp.Function("u")
        u = f(x, t)

        def problem(term):
            return Problem(path=Path("memory"), name="memory", x=x, t=t, unknowns=[u], parameters={}, wave={},
                           equations=[sp.Eq(u.diff(t), u.diff(x, 2) - u + term)])
        mixed = sp.Integral(sp.exp(-s) * f(x, t - s) * u, (s, 0, sp.oo))            # a present value inside
        sum_form = sp.Integral(sp.exp(-s) * f(x, t - s) + sp.exp(-2 * s) * f(x, t - s) ** 2, (s, 0, sp.oo))
        for term in (mixed, sum_form):
            out = classify(problem(term))
            self.assertEqual((out.sub_solver, out.supported), ("D", False))
        good = classify(problem(sp.Integral(sp.exp(-s) * f(x, t - s) ** 2, (s, 0, sp.oo))))
        self.assertEqual((good.sub_solver, good.supported), ("D", True))
        both = classify(problem(sp.Integral(sp.exp(-s) * f(x, t - s), (s, 0, sp.oo))
                                + sp.Integral(sp.exp(-s ** 2) * f(x - s, t), (s, -sp.oo, sp.oo))))
        self.assertEqual((both.sub_solver, both.supported), ("N", False))
        with self.assertRaises(StructureError):                                    # a kernel without a mean delay
            delay.DelaySystem(problem(sp.Integral(f(x, t - s) / (1 + s) ** 2, (s, 0, sp.oo))))


class GeneralKernel(unittest.TestCase):
    """Convolution integrals kept as integrals: band-determinant engine."""

    C0 = 0.5 / np.sqrt(2)

    @slow
    def test_one_equation_two_routes(self):
        # Exponential kernel.  Route 1: auxiliary unknown + Evans engine (nonlocal_core_front.py).
        # Route 2: the integral itself + band determinant (kernel_core_front.py).
        for m in (5.0, -20.0):
            with self.subTest(m=m):
                first = analyse(PROBLEMS / "nonlocal_core_front.py", parameters=dict(m=m), parity_shortcut=False)
                second = analyse(PROBLEMS / "kernel_core_front.py", parameters=dict(m=m), parity_shortcut=False)
                self.assertEqual(second["classification"]["sub_solver"], "N")
                self.assertEqual(first["label"], second["label"], second["reason"])
                self.assertLess(abs(first["wave"]["speed"] - second["wave"]["speed"]), 5e-5)
                self.assertEqual(first["pass_1"]["W"], second["pass_1"]["W"])
                roots = [sorted(r["value"][0] for r in rec["roots"] if r["status"] == "unstable")
                         for rec in (first, second)]
                np.testing.assert_allclose(roots[0], roots[1], atol=2e-3)

    @slow
    def test_gaussian_kernel_monotone_front(self):
        # mu > 0: comparison principle, the front is stable (Bates, Fife, Ren and Wang 1997; Chen 1997).
        # Small sigma: c / c0 = sqrt(1 + mu sigma^2 / 2) + O(sigma^4).
        defect = {}
        for sigma in (0.15, 0.3):
            record = analyse(PROBLEMS / "kernel_gauss_front.py", parameters=dict(sigma=sigma))
            self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
            self.assertTrue(all(c["passed"] for c in record["pass_2"]["checks"]))
            defect[sigma] = record["wave"]["speed"] / self.C0 - np.sqrt(1 + sigma ** 2 / 2)
        self.assertLess(abs(defect[0.15]), 2e-5)
        self.assertTrue(8 < defect[0.3] / defect[0.15] < 32)                       # remainder of fourth order

    def test_essential_instability_at_a_finite_wavenumber(self):
        # mu = -6, sigma = 1: sup Re = |mu| - a - (2/sigma^2)(1 + ln(|mu| sigma^2 / 2)) at kappa != 0.
        record = analyse(PROBLEMS / "kernel_gauss_front.py", parameters=dict(mu=-6.0, sigma=1.0))
        self.assertEqual((record["label"], record["decided_in_pass"]), ("UNSTABLE", 1))
        self.assertAlmostEqual(record["essential"]["s_ess"], 6 - 0.25 - 2 * (1 + np.log(3)), places=6)
        self.assertGreater(record["essential"]["sides"]["right"]["kappa"], 1.0)

    @slow
    def test_gaussian_kernel_against_the_probe_matrix(self):
        # No closed form: second-order differences on two grids and Richardson extrapolation.
        path = PROBLEMS / "kernel_gauss_core.py"
        stable = analyse(path, parameters=dict(m=5.0))
        self.assertEqual((stable["label"], stable["decided_in_pass"]), ("STABLE", 2), stable["reason"])
        for m, is_complex in ((-9.0, False), (-15.0, True)):
            with self.subTest(m=m):
                record = analyse(path, parameters=dict(m=m), parity_shortcut=False)
                self.assertEqual(record["label"], "UNSTABLE", record["reason"])
                (root,) = [complex(*r["value"]) for r in record["roots"] if r["status"] == "unstable"]
                self.assertEqual(root.imag > 0.1, is_complex)
                problem = load_problem(path, dict(m=m))
                system = kernel.KernelSystem(problem)
                wave = kernel.solve_wave(problem, system)
                tops = []
                for nz in (400, 800):
                    ev = fd_probe(kernel.probe_matrix(system, wave, wave.L, wave.L, nz), [1.0, 0.3], k=6)
                    ev = ev[(ev.real > 0.05) & (ev.imag >= 0)]
                    tops.append(ev[np.argmax(ev.real)])
                self.assertLess(abs(root - (tops[1] + (tops[1] - tops[0]) / 3)), 5e-4)


class NoDiffusion(unittest.TestCase):
    """Convolution equations without a diffusion term: u_t = G(u, K * g(u))."""

    def test_front_of_the_nonlocal_phase_transition_equation(self):
        # u_t = J * u - u + k u (1 - u)(u - a).  Known: the front is stable (Chen 1997); the essential
        # spectrum is  i c kappa - 1 + f'(u) + Jhat(kappa)  with supremum f'(u) and the limit f'(u) - 1.
        record = analyse(PROBLEMS / "nonlocal_ac_front.py")
        self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
        self.assertEqual(record["classification"]["sub_solver"], "N")
        self.assertFalse(record["classification"]["details"]["diffusion"])
        self.assertTrue(all(c["passed"] for c in record["pass_2"]["checks"]))
        sides = record["essential"]["sides"]
        self.assertAlmostEqual(sides["right"]["s"], -0.25, places=9)             # -k a
        self.assertAlmostEqual(sides["left"]["s"], -0.75, places=9)              # -k (1 - a)
        self.assertAlmostEqual(sides["right"]["limit_at_infinity"], -1.25, places=12)
        self.assertEqual(record["region"]["source"], "resolvent estimate")
        self.assertGreater(record["wave"]["speed"], 0.1)                         # the state 1 invades for a < 1/2

    @slow
    def test_standing_front(self):
        record = analyse(PROBLEMS / "nonlocal_ac_front.py", parameters=dict(a=0.5))
        self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
        self.assertLess(abs(record["wave"]["speed"]), 1e-8)                      # c = 0 if and only if a = 1/2
        self.assertEqual(record["region"]["source"], "numerical range")

    @staticmethod
    def bump_reference(wave, q, n=1601):
        """Largest eigenvalues of  -1 + K * (S'(U) .)  from the trapezoidal rule, in the symmetric form."""
        z = np.linspace(-wave.L, wave.L, n)
        h = z[1] - z[0]
        s = 1 / (1 + np.exp(-q["beta"] * (wave.Y(z)[0] - q["theta"])))
        root = sparse.diags(np.sqrt(q["beta"] * s * (1 - s)))                    # sqrt(S'(U))
        gauss = lambda x, w: np.exp(-x ** 2 / (2 * w ** 2)) / (w * np.sqrt(2 * np.pi))
        d = np.arange(-int(14 / h), int(14 / h) + 1)
        K = sparse.diags(list(h * (q["A"] * gauss(d * h, 1.0) - q["B"] * gauss(d * h, 2.0))), list(d), shape=(n, n))
        return np.sort(eigsh((root @ K @ root).tocsc(), k=3, which="LA", return_eigenvectors=False))[::-1] - 1.0

    def bump(self, parameters):
        problem = load_problem(PROBLEMS / "neural_field_bump.py", parameters)
        wave = kernel.solve_wave(problem, kernel.KernelSystem(problem))
        reference = self.bump_reference(wave, {str(k): v for k, v in problem.parameters.items()})
        record = analyse(PROBLEMS / "neural_field_bump.py", parameters=parameters, parity_shortcut=False)
        self.assertLess(abs(record["wave"]["speed"]), 1e-10)
        self.assertEqual(record["region"]["source"], "numerical range")
        return record, reference

    def test_narrow_bump_of_a_neural_field_is_unstable(self):
        # Amari (1977): the narrow bump is unstable.  Reference: an independent matrix (trapezoidal rule).
        record, reference = self.bump(None)
        self.assertLess(abs(reference[1]), 1e-4)                                  # translation: tests the profile
        self.assertEqual(record["label"], "UNSTABLE", record["reason"])
        (root,) = [r for r in record["roots"] if r["status"] == "unstable"]
        self.assertLess(abs(root["value"][0] - reference[0]), max(root["error"], 2e-4))
        self.assertGreater(reference[0], 0.3)
        # Essential spectrum: -1 + S'(0) max Khat, at a wavenumber different from zero.
        s0 = 1 / (1 + np.exp(8.0 * 0.55))
        kappa2 = np.log(2.0) / 1.5
        exact = -1 + 8.0 * s0 * (1 - s0) * (2 * np.exp(-kappa2 / 2) - np.exp(-2 * kappa2))
        self.assertAlmostEqual(record["essential"]["s_ess"], exact, places=8)
        self.assertAlmostEqual(record["essential"]["sides"]["left"]["kappa"], np.sqrt(kappa2), places=4)

    @slow
    def test_wide_bump_of_a_neural_field_is_stable(self):
        # Amari (1977): the wide bump is stable.  Its largest eigenvalue below the translation zero is negative.
        record, reference = self.bump(dict(bump=3.27))
        self.assertLess(abs(reference[0]), 1e-4)                                  # translation
        self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
        self.assertTrue(all(c["passed"] for c in record["pass_2"]["checks"]))
        stable = [r["value"][0] for r in record["roots"] if r["status"] == "stable"]
        self.assertTrue(stable and abs(max(stable) - reference[1]) < 2e-3)
        self.assertLess(reference[1], -0.1)

    @slow
    def test_moving_front_by_two_routes(self):
        # Exponential kernel = Green's function of 1 - d^2/dx^2: the same equations as a local system
        # with auxiliary unknowns go through the first engine (the rectangle is given there).
        first = analyse(PROBLEMS / "nonlocal_ac_passive.py", parity_shortcut=False)
        second = analyse(PROBLEMS / "nonlocal_ac_passive_local.py", parity_shortcut=False)
        self.assertEqual((first["label"], second["label"]), ("UNSTABLE", "UNSTABLE"), first["reason"])
        self.assertEqual((first["classification"]["sub_solver"], second["classification"]["sub_solver"]), ("N", "L"))
        self.assertEqual(first["region"]["source"], "decay from the equation")
        self.assertLess(abs(first["wave"]["speed"] - second["wave"]["speed"]), 1e-5)
        roots = [sorted((r["value"][0] for r in rec["roots"] if r["status"] == "unstable"), reverse=True)
                 for rec in (first, second)]
        self.assertEqual((len(roots[0]), len(roots[1])), (2, 2))
        np.testing.assert_allclose(roots[0], roots[1], atol=2e-4)
        for r, exact in zip(sorted(first["roots"], key=lambda r: -r["value"][0]), roots[1]):
            self.assertLess(abs(r["value"][0] - exact), r["error"])

    def test_structure(self):
        # Diffusion in one component only, together with a convolution: not supported.
        x, t, y = sp.symbols("x t y", real=True)
        u, v = sp.Function("u"), sp.Function("v")
        conv = sp.Integral(sp.exp(-(x - y) ** 2) * v(y, t), (y, -sp.oo, sp.oo))
        problem = Problem(path=Path("memory"), name="memory", x=x, t=t, unknowns=[u(x, t), v(x, t)], parameters={},
                          wave={}, equations=[sp.Eq(u(x, t).diff(t), u(x, t).diff(x, 2) - u(x, t) + conv),
                                              sp.Eq(v(x, t).diff(t), u(x, t) - v(x, t))])
        with self.assertRaises(StructureError):
            kernel.KernelSystem(problem)
        # The search rectangle is refused when the local coefficient is not below -delta along the wave
        # and the wave moves: no estimate applies.
        problem = load_problem(PROBLEMS / "nonlocal_ac_front.py")
        system = kernel.KernelSystem(problem)
        wave = kernel.solve_wave(problem, system)
        self.assertEqual(kernel.region(system, wave, 0.05)[1], "resolvent estimate")
        self.assertIsNone(kernel.region(system, wave, 0.74))                     # -delta < sup of a0 = -0.729


class HigherOrder(unittest.TestCase):
    """Parabolic equations of order 4 and 6 (the same engine; the search rectangle has its own estimate)."""

    def check(self, name, parameters, label, s_ess, unstable=(), stable=(), speed=0.0):
        record = analyse(PROBLEMS / name, parameters=parameters, parity_shortcut=False)
        self.assertEqual(record["label"], label, record["reason"])
        self.assertEqual(record["region"]["source"], "energy estimate")
        self.assertAlmostEqual(record["essential"]["s_ess"], s_ess, places=8)
        self.assertLess(abs(record["wave"]["speed"] - speed), 1e-8)
        for status, exact in (("unstable", unstable), ("stable", stable)):
            found = sorted((r for r in record["roots"] if r["status"] == status), key=lambda r: -r["value"][0])
            self.assertEqual(len(found), len(exact))
            for r, value in zip(found, sorted(exact, reverse=True)):
                self.assertLess(abs(r["value"][0] - value), max(r["error"], 1e-5))
                self.assertLess(abs(r["value"][0] - value), 1e-3)
        if label == "STABLE":
            self.assertEqual(record["decided_in_pass"], 2)
            self.assertTrue(all(c["passed"] for c in record["pass_2"]["checks"]))
        return record

    def test_fourth_order_kink_in_closed_form(self):
        # L = -A^2 + alpha A: eigenvalues 0 and -9/4 - 3 alpha / 2, essential spectrum up to -4 - 2 alpha.
        record = self.check("fourth_order_kink.py", None, "STABLE", -4.0)
        self.assertEqual(record["classification"]["details"]["orders"], [4])
        self.check("fourth_order_kink.py", dict(alpha=-1.8), "UNSTABLE", -0.4, unstable=[0.45])

    def test_moving_front_with_a_fourth_order_component(self):
        # Eigenvalues -b^2 + alpha b with b = (5 - n)^2 / 8 - 1 - c^2 / 4 (sigma = 15, a = 0.25).
        c = 0.5 / np.sqrt(2)
        b = [(5 - n) ** 2 / 8 - 1 - c * c / 4 for n in range(5)]
        q = lambda alpha: [-x * x + alpha * x for x in b]
        self.assertAlmostEqual(q(1.0)[2], 0.084961, places=6)
        record = self.check("fourth_order_passive.py", None, "UNSTABLE", -0.25, speed=c,
                            unstable=[v for v in q(1.0) if v > 0])
        self.assertEqual(record["classification"]["details"]["orders"], [2, 4])
        self.assertEqual([g["order"] for g in record["region"]["bounds"]["general"]], [2, 4])
        self.check("fourth_order_passive.py", dict(alpha=-0.4), "STABLE", -0.25, speed=c, stable=[max(q(-0.4))])

    def test_sixth_order_kink_in_closed_form(self):
        # L = A^3 + beta A: eigenvalues 0 and -27/8 - 3 beta / 2, essential spectrum up to -8 - 2 beta.
        self.check("sixth_order_kink.py", None, "STABLE", -8.0)
        self.check("sixth_order_kink.py", dict(beta=-3.0), "UNSTABLE", -2.0, unstable=[1.125])

    def test_extended_fisher_kolmogorov_kink(self):
        # gamma -> 0: the Allen-Cahn kink.  0 < gamma <= 1/8: one monotone kink, stable.
        for gamma in (1e-3, 0.1):
            with self.subTest(gamma=gamma):
                self.check("efk_kink.py", dict(gamma=gamma), "STABLE", -2.0)

    def test_the_estimate_holds_on_computed_eigenfunctions(self):
        # Re(lambda) <= c0 + sum over the groups of ( -d T^2 + sum of w T^gamma ), T from the eigenfunction.
        for name, shift in (("fourth_order_kink.py", 0.0), ("fourth_order_passive.py", 0.08),
                            ("sixth_order_kink.py", 0.0)):
            with self.subTest(file=name):
                problem = load_problem(PROBLEMS / name)
                system = LocalSystem(problem)
                wave = solve_wave(problem, system)
                bounds = coefficient_bounds(system, wave)
                self.assertTrue(bounds["applies"], bounds["reason"])
                nz = 1500
                A = fd_operator(system, wave, wave.L, wave.L, nz)
                lam, vector = eigs(A, k=1, sigma=shift + 0.01, tol=1e-10)
                lam, p = complex(lam[0]), vector[:, 0].reshape(system.n, nz)
                z = np.linspace(-wave.L, wave.L, nz + 2)[1:-1]
                norm = lambda f: float(np.sqrt(np.trapezoid((np.abs(f) ** 2).sum(axis=0), z)))
                total = bounds["c0"]
                for group in bounds["general"]:
                    q = p[group["components"]]
                    for _ in range(group["order"] // 2):
                        q = np.gradient(q, z, axis=1)
                    T = norm(q) / norm(p)
                    total += -group["d"] * T * T + sum(w * T ** power for power, w in group["re"])
                self.assertLessEqual(lam.real, total)
                self.assertGreater(lam.real, -1.0)

    def test_refusals_and_stencils(self):
        x, t = sp.symbols("x t", real=True)
        u = sp.Function("u")(x, t)

        def bounds(rhs):
            problem = Problem(path=Path("memory"), name="memory", x=x, t=t, equations=[sp.Eq(u.diff(t), rhs)],
                              unknowns=[u], parameters={}, wave={})
            system = LocalSystem(problem)
            zero = np.zeros(system.N)
            flat = Wave(c=0.3, L=5.0, N=system.N, Y_minus=zero, Y_plus=zero, mesh=np.linspace(-5, 5, 3),
                        interpolant=lambda z: np.zeros((system.N, np.size(z))), info=dict(profile_error=0.0))
            return coefficient_bounds(system, flat)
        self.assertIn("not positive definite", bounds(u.diff(x, 4) - u)["reason"])          # backward in time
        self.assertIn("even order", bounds(-u.diff(x, 3) + u.diff(x, 2) - u)["reason"])      # dispersive leading term
        good = bounds(-u.diff(x, 4) + u.diff(x, 3) - u)
        self.assertTrue(good["applies"])
        # c0 = -1; the first and the third derivative give no real part: no eigenvalue with Re >= -0.1.
        self.assertAlmostEqual(search_region(good, 0.1)[0], 1.1 * 0.4)
        # Centred differences of order 5 and 6 (second-order accurate).
        s = np.linspace(-3, 3, 601)
        D = _difference_matrices(s.size, s[1] - s[0], 6)
        symbol = sp.Symbol("s")
        for order in (5, 6):
            exact = sp.lambdify(symbol, sp.diff(sp.exp(-symbol ** 2), symbol, order), "numpy")(s)
            self.assertLess(np.abs(D[order] @ np.exp(-s ** 2) - exact)[30:-30].max(), 5e-3 * np.abs(exact).max())


class SecondOrderInTime(unittest.TestCase):
    '''Equations with u_tt: the eigenvalue problem is quadratic in lambda.'''

    C0 = 0.5 / np.sqrt(2)                                         # speed of the Nagumo front, a = 0.25

    def test_hyperbolic_front_speed_and_stability(self):
        # tau u_tt + u_t = u_xx + f(u): c = c0 / sqrt(1 + c0^2 tau) (Hadeler 1988); stable (Gallay and Raugel 1997).
        # Essential spectrum: (-1 + sqrt(1 - 4 tau m)) / (2 tau) if 4 tau m < 1, else -1 / (2 tau); m = 0.25 at u = 0.
        for tau, s_ess in ((2.0, -0.25), (0.5, -1 + np.sqrt(0.5))):
            with self.subTest(tau=tau):
                record = analyse(PROBLEMS / "hyperbolic_front.py", parameters=dict(tau=tau))
                self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
                self.assertTrue(all(c["passed"] for c in record["pass_2"]["checks"]))
                self.assertLess(abs(record["wave"]["speed"] - self.C0 / np.sqrt(1 + self.C0 ** 2 * tau)), 1e-8)
                self.assertAlmostEqual(record["essential"]["s_ess"], s_ess, places=8)
                self.assertEqual(record["region"]["source"], "energy estimate")
                self.assertTrue(record["region"]["bounds"]["quadratic"])

    def test_standing_pulse_in_closed_form(self):
        # tau lambda^2 + lambda = 3: lambda = (-1 + sqrt(1 + 12 tau)) / (2 tau).
        for tau in (1.0, 0.2):
            with self.subTest(tau=tau):
                record = analyse(PROBLEMS / "hyperbolic_pulse.py", parameters=dict(tau=tau), parity_shortcut=False)
                self.assertEqual(record["label"], "UNSTABLE", record["reason"])
                (root,) = [r for r in record["roots"] if r["status"] == "unstable"]
                exact = (-1 + np.sqrt(1 + 12 * tau)) / (2 * tau)
                self.assertLess(abs(root["value"][0] - exact), max(root["error"], 1e-4))
                self.assertLess(abs(record["wave"]["speed"]), 1e-9)
        self.assertAlmostEqual(record["essential"]["s_ess"], (-1 + np.sqrt(1 - 0.8)) / 0.4, places=8)   # tau = 0.2

    def test_damping_that_depends_on_the_state(self):
        # (1 + beta u^2) u_t: no closed form.  Reference: the companion matrix on two grids, extrapolated.
        record = analyse(PROBLEMS / "hyperbolic_pulse.py", parameters=dict(beta=1.0), parity_shortcut=False)
        (root,) = [r for r in record["roots"] if r["status"] == "unstable"]
        problem = load_problem(PROBLEMS / "hyperbolic_pulse.py", dict(beta=1.0))
        system = LocalSystem(problem)
        wave = solve_wave(problem, system)
        values = []
        for nz in (800, 1600):
            A, B = fd_companion(system, wave, wave.L, wave.L, nz)
            ev = fd_probe(A, [1.05], k=2, mass=B)
            values.append(ev[np.argmin(np.abs(ev - 1.05))].real)
        reference = values[1] + (values[1] - values[0]) / 3
        self.assertLess(abs(root["value"][0] - reference), 2e-4)
        self.assertLess(reference, (-1 + np.sqrt(13)) / 2 - 0.1)                   # more damping: slower growth

    def passive_roots(self, sigma, tau):
        s = (-1 + np.sqrt(1 + 8 * sigma)) / 2
        mu = np.array([(s - n) ** 2 / 8 - 1 - self.C0 ** 2 / 4 for n in range(int(np.ceil(s)))])
        return (-1 + np.sqrt(1 + 4 * tau * mu[mu > 0] / (1 + self.C0 ** 2 * tau))) / (2 * tau)

    def test_moving_front_with_a_passive_component(self):
        # (1 + c0^2 tau)(tau lambda^2 + lambda) = mu_n, mu_n the eigenvalues of the problem of first order in time.
        exact = self.passive_roots(15.0, 1.0)
        np.testing.assert_allclose(exact, [0.952966, 0.554093, 0.077350], atol=1e-6)
        record = analyse(PROBLEMS / "hyperbolic_front_passive.py", parity_shortcut=False)
        self.assertEqual(record["label"], "UNSTABLE", record["reason"])
        self.assertLess(abs(record["wave"]["speed"] - 1 / 3), 1e-8)                # c0 / sqrt(1 + c0^2) = 1/3
        found = sorted((r for r in record["roots"] if r["status"] == "unstable"), key=lambda r: -r["value"][0])
        self.assertEqual(len(found), 3)
        for r, value in zip(found, exact):
            self.assertLess(abs(r["value"][0] - value), max(r["error"], 2e-5))

    @slow
    def test_stable_cases_and_a_mixed_system(self):
        record = analyse(PROBLEMS / "hyperbolic_front_passive.py", parameters=dict(sigma=3.0))
        self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
        self.assertTrue(all(c["passed"] for c in record["pass_2"]["checks"]))
        # One component of second order in time, one of first order: lambda = (3 - n)^2 / 2 - 1 for the v-block.
        record = analyse(PROBLEMS / "hyperbolic_kink_mixed.py", parity_shortcut=False)
        self.assertEqual(record["label"], "UNSTABLE", record["reason"])
        found = sorted((r["value"][0] for r in record["roots"] if r["status"] == "unstable"), reverse=True)
        np.testing.assert_allclose(found, [3.5, 1.0], atol=1e-3)
        stable = analyse(PROBLEMS / "hyperbolic_kink_mixed.py", parameters=dict(sigma=0.5))
        self.assertEqual((stable["label"], stable["decided_in_pass"]), ("STABLE", 2), stable["reason"])

    def test_energy_identities_on_a_computed_eigenfunction(self):
        # The two identities behind the search rectangle (DESIGN 2.12), with w = lambda p - c p':
        #   x (||w||^2 + <D p', p'>) + Re <gamma w, w> = Re <b p' + a p, w>,
        #   ||w||^2 - <D p', p'> = 2 x Re <w, p> + Re <gamma w, p> - Re <b p' + a p, p>.
        problem = load_problem(PROBLEMS / "hyperbolic_front_passive.py")
        system = LocalSystem(problem)
        wave = solve_wave(problem, system)
        nz = 3000
        A, B = fd_companion(system, wave, wave.L, wave.L, nz)
        lam, vector = eigs(A, k=1, M=B, sigma=0.56, tol=1e-12)
        lam, p = complex(lam[0]), vector[:system.n * nz, 0].reshape(system.n, nz)
        self.assertLess(abs(lam - self.passive_roots(15.0, 1.0)[1]), 1e-3)
        z = np.linspace(-wave.L, wave.L, nz + 2)[1:-1]
        Y, c = wave.Y(z), wave.c
        a, gamma = system.coefficients(Y, c), system.damping(Y, c)
        D, b = a[2] + c * c * np.eye(2)[None], a[1] - c * gamma
        dot = lambda f, g: np.trapezoid((f * g.conj()).sum(axis=0), z)
        apply = lambda M, f: np.einsum("mij,jm->im", M, f)
        dp = np.gradient(p, z, axis=1)
        w = lam * p - c * dp
        force = apply(b, dp) + apply(a[0], p)
        x, energy = lam.real, dot(w, w).real
        first = x * (energy + dot(apply(D, dp), dp).real) + dot(apply(gamma, w), w).real - dot(force, w).real
        second = (energy - dot(apply(D, dp), dp).real
                  - (2 * x * dot(w, p).real + dot(apply(gamma, w), p).real - dot(force, p).real))
        self.assertLess(abs(first), 2e-3 * energy)
        self.assertLess(abs(second), 2e-3 * energy)
        # The rectangle of the estimate contains the eigenvalue.
        bounds = coefficient_bounds(system, wave)
        x_max, y_max = search_region(bounds, 0.1)
        self.assertTrue(lam.real < x_max and abs(lam) < y_max)

    def test_symbolic_layer_and_refusals(self):
        x, t = sp.symbols("x t", real=True)
        u, v = sp.Function("u")(x, t), sp.Function("v")(x, t)

        def system(equations, unknowns):
            return LocalSystem(Problem(path=Path("memory"), name="memory", x=x, t=t, equations=equations,
                                       unknowns=unknowns, parameters={}, wave={}))
        wave_eq = sp.Eq(2 * u.diff(t, 2) + (1 + u ** 2) * u.diff(t), u.diff(x, 2) - u)
        s = system([wave_eq], [u])
        self.assertTrue(s.quadratic)
        self.assertEqual((list(s.inertia), s.orders), ([1.0], [2]))
        Y = np.array([[0.5], [0.0]])
        np.testing.assert_allclose(s.damping(Y, 0.3)[0], [[(1 + 0.25) / 2]])          # N1 = (1 + u^2) / tau
        np.testing.assert_allclose(s.coefficients(Y, 0.3)[2, 0], [[0.5 - 0.09]])       # D / tau - c^2
        M0, M1, M2 = s.pencil(Y, 0.3)
        np.testing.assert_allclose(M2[0], [[0, 0], [1 / 0.41, 0]])                     # lambda^2 p / (D - c^2)
        np.testing.assert_allclose(M1[0], [[0, 0], [0.625 / 0.41, -0.6 / 0.41]])       # N1 p - 2 c p'
        # First order in time: the pencil has two matrices and the damping is the mass matrix.
        first = system([sp.Eq(u.diff(t), u.diff(x, 2) - u)], [u])
        self.assertFalse(first.quadratic)
        self.assertEqual(len(first.pencil(Y, 0.3)), 2)
        np.testing.assert_allclose(first.damping(Y, 0.3)[0], [[1.0]])
        # Mixed system: no estimate of the rectangle (u_tt is needed in every component).
        mixed = system([wave_eq, sp.Eq(v.diff(t), v.diff(x, 2) - v)], [u, v])
        self.assertEqual(list(mixed.inertia), [1.0, 0.0])
        flat = Wave(c=0.3, L=5.0, N=mixed.N, Y_minus=np.zeros(4), Y_plus=np.zeros(4), mesh=np.linspace(-5, 5, 3),
                    interpolant=lambda zz: np.zeros((4, np.size(zz))), info=dict(profile_error=0.0))
        self.assertIn("in every component", coefficient_bounds(mixed, flat)["reason"])
        # Essential spectrum of the mixed system: the wave part gives -N1 / 2 = -0.25, the heat part -1.
        end = (mixed.coefficients(np.zeros(4), 0.3)[:, 0], mixed.damping(np.zeros(4), 0.3)[0], mixed.inertia)
        self.assertAlmostEqual(quadratic_essential_bound(end, end, 0.3)[0], -0.25, places=9)
        for bad in (sp.Eq(u.diff(t, 2) + u.diff(t), u.diff(x, 2) + u.diff(x, t) - u),          # mixed derivative
                    sp.Eq((1 + u ** 2) * u.diff(t, 2) + u.diff(t), u.diff(x, 2) - u),          # inertia depends on u
                    sp.Eq(u.diff(t, 3), u.diff(x, 2) - u)):                                    # order three
            with self.assertRaises(StructureError):
                system([bad], [u])


class WeightedSpace(unittest.TestCase):
    '''Waves with essential spectrum in Re > 0: unstable in L2; a second label in a weighted space.'''

    def kpp(self, speed):
        return analyse(PROBLEMS / "fisher_kpp.py", wave=dict(kind="front", left={"u": 1}, right={"u": 0}, speed=speed))

    def test_fisher_kpp_fronts(self):
        # Weight exp(r z) ahead of the front: sup Re = r^2 - c r + 1.  Negative for r between
        # (c -+ sqrt(c^2 - 4)) / 2 if c > 2 (Sattinger 1976); the best rate is c / 2.
        record = self.kpp(2.5)
        self.assertEqual((record["label"], record["decided_in_pass"]), ("UNSTABLE", 1))      # the label in L2
        w = record["weighted"]
        self.assertEqual((w["label"], w["decided_in_pass"]), ("STABLE", 2), w["reason"])
        self.assertTrue(all(c["passed"] for c in w["pass_2"]["checks"]))
        self.assertEqual(w["ends"]["left"]["rate"], 0.0)                                      # u = 1 needs no weight
        self.assertAlmostEqual(w["ends"]["right"]["rate"], 1.25, places=6)
        np.testing.assert_allclose(w["ends"]["right"]["interval"], [0.5, 2.0], atol=1e-6)
        self.assertAlmostEqual(w["s_ess"], 1 - 2.5 ** 2 / 4, places=9)
        self.assertFalse(w["translation_mode_in_the_space"])            # the front decays more slowly than the weight grows
        self.assertIn("in the weighted space", record["reason"])
        critical = self.kpp(2.0)["weighted"]                            # c = 2: the best weight touches the axis
        self.assertEqual(critical["label"], "UNCERTAIN")
        self.assertLess(abs(critical["s_ess"]), 1e-9)
        slow = self.kpp(1.5)["weighted"]                                # c < 2: unstable in every weight
        self.assertEqual(slow["label"], "UNSTABLE")
        self.assertAlmostEqual(slow["s_ess"], 1 - 1.5 ** 2 / 4, places=9)

    def test_pushed_front(self):
        # a = -0.25: rates between -sqrt(2) a and 1 / sqrt(2); supremum -a - c^2 / 4 at c / 2; the translation mode stays.
        record = analyse(PROBLEMS / "nagumo_pushed.py")
        self.assertEqual(record["label"], "UNSTABLE")
        self.assertAlmostEqual(record["essential"]["s_ess"], 0.25, places=9)
        w = record["weighted"]
        self.assertEqual((w["label"], w["decided_in_pass"]), ("STABLE", 2), w["reason"])
        c = 1.5 / np.sqrt(2)
        self.assertAlmostEqual(w["wave"]["speed"], c, places=12)
        np.testing.assert_allclose(w["ends"]["right"]["interval"], [np.sqrt(2) / 4, 1 / np.sqrt(2)], atol=1e-6)
        self.assertAlmostEqual(w["s_ess"], 0.25 - c * c / 4, places=9)
        self.assertTrue(w["translation_mode_in_the_space"])

    def test_eigenvalues_in_the_weighted_space(self):
        # v-block: (s - n)^2 / 8 - 1 - c^2 / 4 with c^2 = 9 / 8; sigma = 15: 1.84375 and 0.71875.
        record = analyse(PROBLEMS / "pushed_front_passive.py", parity_shortcut=False)
        w = record["weighted"]
        self.assertEqual(w["label"], "UNSTABLE", w["reason"])
        found = sorted((r for r in w["roots"] if r["status"] == "unstable"), key=lambda r: -r["value"][0])
        self.assertEqual(len(found), 2)
        for r, exact in zip(found, (1.84375, 0.71875)):
            self.assertLess(abs(r["value"][0] - exact), max(r["error"], 1e-5))
        stable = analyse(PROBLEMS / "pushed_front_passive.py", parameters=dict(sigma=3.0))["weighted"]
        self.assertEqual((stable["label"], stable["decided_in_pass"]), ("STABLE", 2), stable["reason"])

    def test_the_weight_as_a_change_of_the_operator(self):
        # (d - a)^2 = d^2 - 2 a d + a^2 - a'.
        a_k = np.array([[[3.0]], [[2.0]], [[1.0]]])
        np.testing.assert_allclose(weighted.weighted_coefficients(a_k, 0.5, 0.25)[:, 0, 0],
                                   [3.0 - 0.5 * 2.0 + (0.25 - 0.25), 2.0 - 2 * 0.5, 1.0])
        # Scalar end state u_t = u_xx + c u_x + m u: supremum r^2 - c r + m.
        end = weighted.find_rate(np.array([[[1.0]], [[2.5]], [[1.0]]]), [], np.ones(1))
        self.assertAlmostEqual(end["supremum"], 1 - 2.5 ** 2 / 4, places=9)
        self.assertAlmostEqual(end["without_weight"], 1.0, places=9)
        # The first-order system is shifted by the rate: Q' = (M + a I) Q.
        problem = load_problem(PROBLEMS / "fisher_kpp.py")
        system = LocalSystem(problem)
        shifted = weighted.WeightedSystem(system)
        Y = np.array([[0.3], [-0.1], [0.7], [0.0]])                                # state, rate, slope of the rate
        np.testing.assert_allclose(shifted.first_order(Y, 2.5)[0][0],
                                   system.first_order(Y[:2], 2.5)[0][0] + 0.7 * np.eye(2))
        # The option switches the second label off.
        self.assertNotIn("weighted", analyse(PROBLEMS / "fisher_kpp.py", weighted=False))


class ConservationLaw(unittest.TestCase):
    """Equations in conservation form: every constant is an equilibrium; the integrated operator is analysed."""

    @staticmethod
    def difference_operator_eigenvalue(name, parameters, guess):
        """Real eigenvalue of the integrated operator near `guess`, by differences (Richardson, two meshes)."""
        problem = load_problem(PROBLEMS / name, parameters)
        cons = ConservativeSystem(problem, LocalSystem(problem))
        wave = solve_wave(cons.reduced_problem, cons.reduced)
        values = []
        for nz in (3000, 6000):
            ev = fd_probe(fd_operator(cons, wave, wave.L, wave.L, nz), [guess], k=4)
            real = ev[np.abs(ev.imag) < 1e-6]
            values.append(real[np.argmin(np.abs(real - guess))].real)
        return values[1] + (values[1] - values[0]) / 3

    def test_flux_recovery(self):
        U = list(sp.symbols("U0:5"))
        total = lambda e: sum(sp.diff(e, U[k]) * U[k + 1] for k in range(4))
        burgers = antiderivative([U[2] - U[0] * U[1]], [U[:3]], [2])
        self.assertEqual(sp.expand(burgers[0] - (U[1] - U[0] ** 2 / 2)), 0)
        self.assertIsNone(antiderivative([U[1] ** 2], [U[:3]], [2]))                 # not a total derivative
        self.assertIsNone(antiderivative([U[0] * (1 - U[0])], [U[:3]], [2]))         # a reaction term
        potential = -U[2] - U[0] + U[0] ** 3                                         # Cahn-Hilliard: two integrations
        flux = antiderivative([total(total(potential))], [U], [4])
        self.assertEqual(sp.expand(flux[0] - total(potential)), 0)
        self.assertEqual(sp.expand(antiderivative(flux, [U], [3])[0] - potential), 0)
        a, b = list(sp.symbols("a0:3")), list(sp.symbols("b0:3"))                    # a system with a coupled flux
        both = antiderivative([b[1] + a[2], a[1] * b[0] + a[0] * b[1] + b[2]], [a, b], [2, 2])
        self.assertEqual([sp.expand(both[0] - (b[0] + a[1])), sp.expand(both[1] - (a[0] * b[0] + b[1]))], [0, 0])

    def test_burgers_shocks(self):
        # c = (u- + u+) / 2 from the jump condition.  With D = u- - u+ the integrated operator is conjugate
        # to d^2/dz^2 - D^2 / 16: rates -D/4 and +D/4, supremum -D^2 / 16, no eigenvalue.
        for left, right in ((2, 0), (1, -1), (4, 0)):
            record = analyse(PROBLEMS / "burgers_shock.py",
                             wave=dict(kind="front", left={"u": left}, right={"u": right}, width=1.0))
            self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
            self.assertEqual(record["qualifier"], "algebraic decay")
            cons, D = record["conservation"], left - right
            self.assertAlmostEqual(cons["speed"], (left + right) / 2, places=12)
            self.assertEqual((cons["depth"], cons["operator"]), (1, "integrated"))
            self.assertAlmostEqual(cons["flux_constant"][0], left * right / 2, places=12)      # q = c u - u^2 / 2
            w = record["weighted"]
            self.assertEqual(w["label"], "STABLE", w["reason"])
            np.testing.assert_allclose(w["rates"], [-D / 4, D / 4], atol=1e-6)
            self.assertAlmostEqual(w["s_ess"], -D * D / 16, places=9)
            np.testing.assert_allclose(w["ends"]["right"]["interval"], [0, D / 2], atol=1e-6)
            self.assertEqual(record["roots"], [])

    def test_refusals(self):
        burgers = lambda **wave: analyse(PROBLEMS / "burgers_shock.py", wave=dict(kind="front", width=1.0, **wave))
        expansive = burgers(left={"u": 0}, right={"u": 2})                 # u- < u+: a rarefaction, no shock profile
        self.assertEqual(expansive["label"], "UNCERTAIN")
        self.assertTrue(expansive["reason"].startswith("wave:"), expansive["reason"])
        wrong = burgers(left={"u": 2}, right={"u": 0}, speed=0.7)          # the jump condition gives 1
        self.assertEqual(wrong["label"], "UNCERTAIN")
        self.assertIn("jump condition", wrong["reason"])
        pulse = analyse(PROBLEMS / "cahn_hilliard_pulse.py",
                        wave=dict(kind="pulse", rest={"u": -0.8}, amplitude={"u": 0.75}, width=1.6))
        self.assertEqual(pulse["label"], "UNSTABLE")                       # no speed given: the standing pulse
        self.assertEqual(pulse["wave"]["speed"], 0.0)
        moving = analyse(PROBLEMS / "gkdv_soliton.py",                     # a soliton has no standing member
                         wave=dict(kind="pulse", rest={"u": 0}, amplitude={"u": 1.5}, width=1.5))
        self.assertEqual(moving["label"], "UNCERTAIN")
        self.assertIn("give wave['speed']", moving["reason"])

    def test_kuramoto_sivashinsky_is_essentially_unstable(self):
        # lambda(kappa) = kappa^2 - kappa^4 - i (u - c) kappa: supremum 1/4 at kappa = 1 / sqrt(2).  No wave is needed.
        record = analyse(PROBLEMS / "ks_shock.py")
        self.assertEqual((record["label"], record["decided_in_pass"]), ("UNSTABLE", 1), record["reason"])
        self.assertAlmostEqual(record["essential"]["s_ess"], 0.25, places=6)
        self.assertAlmostEqual(record["essential"]["sides"]["left"]["kappa"], 1 / np.sqrt(2), places=3)
        self.assertFalse(record["wave"]["computed"])
        self.assertNotIn("qualifier", record)

    def test_cahn_hilliard_kink_is_neutrally_stable(self):
        # u_t = -(u_xx + u - u^3)_xx: two integrations; no transport at the end states, so no weight exists.
        record = analyse(PROBLEMS / "cahn_hilliard_kink.py")
        self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
        self.assertEqual((record["conservation"]["depth"], record["conservation"]["speed"]), (2, 0.0))
        neutral = record["neutral"]
        self.assertEqual(record["qualifier"], f"neutral: no spectrum with Re(lambda) > {neutral['threshold']:.3g}")
        self.assertEqual((neutral["counts"], neutral["resolved"]), ([0, 0, 0], [True, True, True]))
        self.assertEqual(neutral["region"]["source"], "energy estimate")
        self.assertNotIn("weighted", record)

    @slow
    def test_cahn_hilliard_pulse_is_unstable(self):
        record = analyse(PROBLEMS / "cahn_hilliard_pulse.py")
        self.assertEqual((record["label"], record["decided_in_pass"]), ("UNSTABLE", 1), record["reason"])
        (root,) = record["roots"]
        self.assertLess(abs(root["value"][1]), 1e-9)
        # The Dirichlet truncation of the difference operator costs about 4e-5 here (the tails decay slowly).
        reference = self.difference_operator_eigenvalue("cahn_hilliard_pulse.py", None, root["value"][0])
        self.assertLess(abs(root["value"][0] - reference), 2e-4)

    def test_korteweg_de_vries_solitary_waves_are_neutrally_stable(self):
        # Pego and Weinstein 1992: no spectrum with a positive real part for p < 4.
        for p in (1, 2):
            record = analyse(PROBLEMS / "gkdv_soliton.py", parameters=dict(p=p))
            self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
            self.assertTrue(record["qualifier"].startswith("neutral: no spectrum with Re(lambda) > "))
            self.assertEqual((record["conservation"]["speed"], record["conservation"]["depth"]), (1.0, 1))
            self.assertEqual(record["neutral"]["region"]["source"], "given in the problem file")
        # A dispersive equation has no estimate of the search rectangle: without the option there is no label.
        blind = analyse(PROBLEMS / "gkdv_soliton.py", region=None, default_region=False)
        self.assertEqual(blind["label"], "UNCERTAIN")
        self.assertIn("give options['region']", blind["reason"])

    @slow
    def test_supercritical_korteweg_de_vries_is_unstable(self):
        # p > 4: one real positive eigenvalue (Pego and Weinstein 1992).
        record = analyse(PROBLEMS / "gkdv_soliton.py", parameters=dict(p=5))
        self.assertEqual((record["label"], record["decided_in_pass"]), ("UNSTABLE", 1), record["reason"])
        (root,) = record["roots"]
        reference = self.difference_operator_eigenvalue("gkdv_soliton.py", dict(p=5), root["value"][0])
        self.assertLess(abs(root["value"][0] - reference), 2e-3)
        self.assertLess(abs(root["value"][0] - reference), root["error"])


class ConservationSystems(unittest.TestCase):
    """Shocks of systems and of equations of higher order: the determinant on the closed half-plane."""

    @staticmethod
    def profile(name, parameters=None):
        problem = load_problem(PROBLEMS / name, parameters)
        cons = ConservativeSystem(problem, LocalSystem(problem))
        return cons, solve_wave(cons.reduced_problem, cons.reduced)

    def test_splitting_that_is_continuous_up_to_the_axis(self):
        # P'' - b P' = lambda P as a system: eigenvalues (b +- sqrt(b^2 + 4 lambda)) / 2.  The slow one is
        # -lambda / b + ...: it is unstable for Re(lambda) > 0 if and only if b < 0.
        M1 = np.array([[0.0, 0.0], [1.0, 0.0]])
        for b in (1.0, -1.0):
            for lam in (0.0, 1e-12, 1e-9j, -1e-9j, 1e-4j, 0.3j, 0.5, 2.0 + 1.0j):
                M = np.array([[0.0, 1.0], [lam, b]], dtype=complex)
                k, Pu, ev_u, ev_s = split_touching(M, M1)
                root = np.sqrt(b * b + 4 * lam + 0j)
                fast, slow = (b + np.sign(b) * root) / 2, (b - np.sign(b) * root) / 2
                unstable, stable = (fast, slow) if b > 0 else (slow, fast)
                self.assertEqual(k, 1, (b, lam))
                np.testing.assert_allclose(Pu @ np.array([1.0, unstable]), [1.0, unstable], atol=1e-9)
                np.testing.assert_allclose(Pu @ np.array([1.0, stable]), [0.0, 0.0], atol=1e-9)
                self.assertLess(abs(ev_u[0] - unstable), 1e-9)
        # Away from the axis the result is that of split().
        M = np.array([[0.0, 1.0], [0.7 + 0.2j, 1.0]])
        np.testing.assert_allclose(split_touching(M, M1)[1], split(M)[1], atol=1e-12)
        # No transport (b = 0): the slow eigenvalues cannot be assigned.
        with self.assertRaises(SlowModeError):
            split_touching(np.array([[0.0, 1.0], [0.0, 0.0]]), M1)

    def test_burgers_shock_by_two_methods(self):
        # A/B: the weighted space (DESIGN 2.13) and the determinant on the closed half-plane give the same label.
        for left, right in ((2, 0), (1, -1), (4, 0)):
            spec = dict(kind="front", left={"u": left}, right={"u": right}, width=1.0)
            weights = analyse(PROBLEMS / "burgers_shock.py", wave=spec)
            closed = analyse(PROBLEMS / "burgers_shock.py", wave=spec, conservation_method="axis")
            for record in (weights, closed):
                self.assertEqual((record["label"], record["qualifier"]), ("STABLE", "algebraic decay"), record["reason"])
            self.assertIn("weighted", weights)
            self.assertNotIn("weighted", closed)
            data = closed["closed_half_plane"]
            self.assertEqual((data["counts"], data["resolved"], data["threshold"]), ([0, 0, 0], [True] * 3, 0.0))
            self.assertEqual(closed["conservation"]["method"], "determinant on the closed half-plane")
            D = left - right
            np.testing.assert_allclose(closed["conservation"]["transport"]["left"], [-D / 2], atol=1e-12)
            np.testing.assert_allclose(closed["conservation"]["transport"]["right"], [D / 2], atol=1e-12)

    def test_shock_of_a_system_with_a_passive_field(self):
        # Characteristic speeds relative to the shock: 1, 2 on the left; -1, 2 on the right.  Both signs
        # at one end state: no exponential weight exists.
        record = analyse(PROBLEMS / "burgers_passive_shock.py")
        self.assertEqual((record["label"], record["qualifier"], record["decided_in_pass"]),
                         ("STABLE", "algebraic decay", 2), record["reason"])
        cons = record["conservation"]
        self.assertAlmostEqual(cons["speed"], 1.0, places=12)
        np.testing.assert_allclose(cons["transport"]["left"], [-2, -1], atol=1e-12)
        np.testing.assert_allclose(cons["transport"]["right"], [-2, 1], atol=1e-12)
        self.assertNotIn("weighted", record)
        self.assertEqual(record["closed_half_plane"]["counts"], [0, 0, 0])
        self.assertEqual(record["closed_half_plane"]["splitting_dimension"], 2)
        # The passive field passes through the shock with the speed of the shock: no transport.
        resonant = analyse(PROBLEMS / "burgers_passive_shock.py", parameters=dict(a=1.0),
                           wave=dict(kind="front", left={"u": 2, "v": 0}, right={"u": 0, "v": 0}, width=1.0))
        self.assertNotEqual(resonant["label"], "STABLE")

    def test_chemotaxis_and_gas_dynamics_shocks(self):
        # Keller-Segel after the Hopf-Cole transformation: c = chi u- v- / (u+ - u-), stable (Li and Wang 2011).
        record = analyse(PROBLEMS / "chemotaxis_shock.py")
        self.assertEqual((record["label"], record["qualifier"]), ("STABLE", "algebraic decay"), record["reason"])
        vm = -np.sqrt(0.8 / (0.5 + 1.0 / 0.8))
        self.assertAlmostEqual(record["conservation"]["speed"], vm / (0.2 - 1.0), places=12)
        self.assertLess(record["conservation"]["jump_residual"], 1e-12)
        signs = [np.sign(record["conservation"]["transport"][side]).tolist() for side in ("left", "right")]
        self.assertEqual(signs, [[-1, 1], [1, 1]])                       # a Lax shock: three characteristics enter
        # p-system with artificial viscosity: c^2 = -(p+ - p-) / (v+ - v-), stable.
        record = analyse(PROBLEMS / "p_system_shock.py")
        self.assertEqual((record["label"], record["qualifier"]), ("STABLE", "algebraic decay"), record["reason"])
        self.assertAlmostEqual(record["conservation"]["speed"], np.sqrt(1 - 2 ** -1.4), places=12)
        # End states that violate the jump condition of the second equation are refused.
        bad = analyse(PROBLEMS / "chemotaxis_shock.py", wave=dict(kind="front", left={"u": 1.0, "v": vm},
                                                                 right={"u": 0.2, "v": 0.1}, width=2.0))
        self.assertEqual(bad["label"], "UNCERTAIN")
        self.assertIn("jump condition", bad["reason"])

    def test_thin_film_waves_with_opposite_labels(self):
        # One equation, one pair of end states, two compressive waves (Bertozzi, Muench, Shearer and Zumbrun 2001).
        low = analyse(PROBLEMS / "thin_film_shock.py")
        self.assertEqual((low["label"], low["qualifier"]), ("STABLE", "algebraic decay"), low["reason"])
        self.assertAlmostEqual(low["conservation"]["speed"], 0.336 + 0.1 - (0.336 ** 2 + 0.0336 + 0.01), places=12)
        high = analyse(PROBLEMS / "thin_film_shock.py", parameters=dict(A=0.3))
        self.assertEqual((high["label"], high["decided_in_pass"]), ("UNSTABLE", 1), high["reason"])
        self.assertEqual(high["closed_half_plane"]["counts"], [1, 1, 1])
        (root,) = high["roots"]
        ridges = []
        for A in (0.08, 0.3):
            cons, wave = self.profile("thin_film_shock.py", dict(A=A))
            ridges.append(wave.Y(np.linspace(-wave.L, wave.L, 8001))[0].max())
        self.assertLess(ridges[0], 0.45)
        self.assertGreater(ridges[1], 0.57)
        # The eigenvalue is far below the size of the search rectangle; the difference operator confirms it.
        values = []
        for nz in (3000, 6000):
            ev = fd_probe(fd_operator(cons, wave, wave.L, wave.L, nz), [root["value"][0]], k=4)
            values.append(ev[np.argmin(np.abs(ev - root["value"][0]))].real)
        self.assertLess(abs(root["value"][0] - values[1]), 1e-6)
        self.assertGreater(high["closed_half_plane"]["region"]["x_max"], 1e4 * root["value"][0])
        # Below the window of film thicknesses there is one wave, and it is stable.
        thin = analyse(PROBLEMS / "thin_film_shock.py", parameters=dict(hinf=0.3))
        self.assertEqual((thin["label"], thin["qualifier"]), ("STABLE", "algebraic decay"), thin["reason"])


class Taxis(unittest.TestCase):
    """Transport towards a component without diffusion: first-order form in flux variables."""

    @staticmethod
    def system(name, parameters=None):
        problem = load_problem(PROBLEMS / name, parameters)
        return problem, TaxisSystem(problem, LocalSystem(problem))

    def test_flux_form_reproduces_the_dispersion_relation(self):
        problem, system = self.system("haptotaxis_front.py")
        self.assertEqual((system.diffusive, system.first_order_components, system.N, system.mu), ([0], [1], 3, [2, 1]))
        # Spatial eigenvalues nu of M0 + lambda M1 at an end state are the roots of
        # det(a0 + nu a1 + nu^2 a2 - lambda) = 0, with the coefficients a_k of the original equations.
        c = 0.41
        for u in (problem.wave["left"], problem.wave["right"]):
            M0, M1, a = system.end_state(u, c)
            for lam in (0.7, -0.1 + 0.9j, 2.0 - 0.3j):
                for nu in np.linalg.eigvals(M0 + lam * M1):
                    symbol = a[0] + nu * a[1] + nu * nu * a[2] - lam * np.eye(2)
                    self.assertLess(abs(np.linalg.det(symbol)), 1e-9 * (1 + abs(nu)) ** 4)
        # The coefficient of m_xx in the equation of u is -chi u: not zero behind the front.
        self.assertAlmostEqual(system.end_state(problem.wave["left"], c)[2][2, 0, 1], -1.0, places=12)
        self.assertEqual(system.top_matrix_condition(problem.wave["left"], 0.0), np.inf)       # W' = -g / c
        # Equations without this structure are refused.
        for name in ("nagumo_front.py", "fhn_fast_pulse.py", "cross_diffusion_front.py"):
            other = load_problem(PROBLEMS / name)
            with self.assertRaises(StructureError):
                TaxisSystem(other, LocalSystem(other))

    def test_profile_satisfies_the_original_equations(self):
        problem, system = self.system("haptotaxis_front.py")
        wave = solve_wave(problem, system)
        base, z = system.base, np.linspace(-wave.L, wave.L, 2001)
        jets = system._jets(*wave.Y(z), wave.c, *system.theta)
        low, top = jets[:base.N], jets[base.N:]
        H = base._H(*low, wave.c, *base.theta).reshape(2, 2, z.size)
        residual = np.einsum("ijm,jm->im", H, top) + base._r(*low, wave.c, *base.theta)
        self.assertLess(np.abs(residual).max(), 1e-7)
        self.assertGreater(wave.c, 0.5 / np.sqrt(2))                    # haptotaxis speeds the invasion up
        # Without taxis the front is the Nagumo front: c = (1 - 2 a) / sqrt(2).
        problem, system = self.system("haptotaxis_front.py", dict(chi=1e-7))
        self.assertLess(abs(solve_wave(problem, system).c - 0.5 / np.sqrt(2)), 1e-6)

    def test_haptotaxis_front_is_stable(self):
        record = analyse(PROBLEMS / "haptotaxis_front.py")
        self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
        self.assertEqual(record["taxis"], dict(with_diffusion=["u"], without_diffusion=["m"], form="flux variables"))
        self.assertTrue(all(c["passed"] for c in record["pass_2"]["checks"]))      # includes the difference probe
        self.assertEqual((record["pass_1"]["W"], record["pass_1"]["m0"]), (1, 1))
        sides = record["essential"]["sides"]
        self.assertAlmostEqual(record["essential"]["s_ess"], -0.25, places=9)       # -a at the state u = 0
        # Limit of the branch without diffusion: g_m - g_u B / eps = -(mu + k) - k m chi / eps, and -mu.
        self.assertAlmostEqual(sides["left"]["limit_at_infinity"], -1.5 - 1.0 / 3.0, places=9)
        self.assertAlmostEqual(sides["right"]["limit_at_infinity"], -0.5, places=9)
        # No estimate bounds the search rectangle; a zero speed is not possible.
        blind = analyse(PROBLEMS / "haptotaxis_front.py", region=None, default_region=False)
        self.assertEqual(blind["label"], "UNCERTAIN")
        self.assertIn("taxis", blind["reason"])
        still = analyse(PROBLEMS / "haptotaxis_front.py",
                        wave=dict(kind="front", left={"u": 1, "m": 1 / 3}, right={"u": 0, "m": 1}, width=2.0))
        self.assertEqual(still["label"], "UNCERTAIN")
        self.assertIn("zero speed", still["reason"])

    def test_passive_component_against_a_scalar_reference(self):
        # v-block: lambda r = d r'' + c r' + (sigma U (1 - U) - 1) r.  With r = exp(-c z / 2) s (d = 1):
        # s'' + (sigma U (1 - U) - 1 - c^2 / 4) s = lambda s, a symmetric tridiagonal matrix.
        record = analyse(PROBLEMS / "haptotaxis_front_passive.py", parity_shortcut=False)
        self.assertEqual(record["label"], "UNSTABLE", record["reason"])
        found = sorted((r["value"][0] for r in record["roots"] if r["status"] == "unstable"), reverse=True)
        problem, system = self.system("haptotaxis_front.py")
        wave = solve_wave(problem, system)
        values = []
        for nz in (4000, 8000):
            z = np.linspace(-wave.L, wave.L, nz + 2)[1:-1]
            h = z[1] - z[0]
            U = wave.Y(z)[0]
            A = sparse.diags([np.full(nz - 1, 1 / h ** 2), 30.0 * U * (1 - U) - 1 - wave.c ** 2 / 4 - 2 / h ** 2,
                              np.full(nz - 1, 1 / h ** 2)], [-1, 0, 1], format="csc")
            values.append(np.sort(eigsh(A, k=6, sigma=6.0, which="LM", return_eigenvectors=False))[::-1])
        reference = values[1] + (values[1] - values[0]) / 3
        reference = reference[reference > 0]
        self.assertEqual(len(found), len(reference))
        np.testing.assert_allclose(found, reference, atol=2e-5)
        # The box count: every located root (one of them is stable, just left of the axis) and the translation zero.
        self.assertEqual(record["pass_1"]["W"], len(record["roots"]) + 1)

    def test_aggregation_at_small_scales(self):
        # Signal produced by the cells, no diffusion: at u = m = 1 a root of the dispersion relation
        # tends to mu (chi / eps - 1) as kappa -> infinity.
        record = analyse(PROBLEMS / "chemotaxis_aggregation.py")
        self.assertEqual((record["label"], record["decided_in_pass"]), ("UNSTABLE", 1), record["reason"])
        self.assertAlmostEqual(record["essential"]["s_ess"], 0.5, places=9)
        self.assertAlmostEqual(record["essential"]["sides"]["left"]["limit_at_infinity"], 0.5, places=9)
        self.assertFalse(record["wave"]["computed"])
        calm = analyse(PROBLEMS / "chemotaxis_aggregation.py", parameters=dict(chi=0.5))
        self.assertEqual((calm["label"], calm["decided_in_pass"]), ("STABLE", 2), calm["reason"])
        self.assertAlmostEqual(calm["essential"]["sides"]["left"]["limit_at_infinity"], -0.25, places=9)
        # The limit as a formula: B = A0_HH - A0_HP A2_PP^(-1) A2_PH.
        a = np.array([[[-0.75, 0.0], [0.5, -0.5]], [[0.3, 0.0], [0.0, 0.3]], [[1.0, -2.0], [0.0, 0.0]]])
        self.assertAlmostEqual(high_wavenumber_limit(a, [1]), 0.5, places=12)


class NeutralMode(unittest.TestCase):
    """Essential spectrum on the imaginary axis (no damping): only Re(lambda) > r0 is examined."""

    def test_klein_gordon_pulse_in_closed_form(self):
        # u_tt = u_xx - u + 2 u^3, U = sech: lambda^2 = 3 at rest.  A boost with the speed c divides the
        # growth rate in the co-moving frame by the Lorentz factor: lambda = sqrt(3 (1 - c^2)).
        for c in (0.0, 0.6):
            gamma = 1 / np.sqrt(1 - c * c)
            record = analyse(PROBLEMS / "klein_gordon_pulse.py",
                             wave=dict(kind="pulse", rest={"u": 0}, amplitude={"u": 1.1}, width=1.2 / gamma, speed=c))
            self.assertEqual((record["label"], record["decided_in_pass"]), ("UNSTABLE", 1), record["reason"])
            self.assertEqual(record["wave"]["speed"], c)
            self.assertLess(abs(record["essential"]["s_ess"]), 1e-9)
            (root,) = record["roots"]
            self.assertLess(abs(root["value"][0] - np.sqrt(3) / gamma), min(root["error"], 1e-4))
            self.assertNotIn("qualifier", record)

    def test_phi4_kink_is_neutrally_stable(self):
        # All of the spectrum lies on the imaginary axis: 0, +-i sqrt(3/2), |Im| >= sqrt(2).
        for c in (0.0, 0.5):
            record = analyse(PROBLEMS / "phi4_kink.py", wave=dict(kind="front", left={"u": -1}, right={"u": 1}, speed=c,
                                                                  width=float(np.sqrt(2 * (1 - c * c)))))
            self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
            neutral = record["neutral"]
            self.assertAlmostEqual(neutral["threshold"], 0.02, places=6)           # 0.01 x |f'(+-1)|
            self.assertEqual(record["qualifier"], "neutral: no spectrum with Re(lambda) > 0.02")
            self.assertEqual((neutral["counts"], neutral["resolved"]), ([0, 0, 0], [True, True, True]))
        record = analyse(PROBLEMS / "phi4_kink.py", neutral_threshold=0.1)          # the threshold is an option
        self.assertEqual(record["qualifier"], "neutral: no spectrum with Re(lambda) > 0.1")
        blind = analyse(PROBLEMS / "phi4_kink.py", region=None, default_region=False)   # no estimate without damping
        self.assertEqual(blind["label"], "UNCERTAIN")
        self.assertIn("give options['region']", blind["reason"])

    def test_wave_at_a_given_speed(self):
        # With damping only one speed has a front: sqrt(tau)-corrected Nagumo speed (hyperbolic_front.py).
        problem = load_problem(PROBLEMS / "hyperbolic_front.py")
        system = LocalSystem(problem)
        found = solve_wave(problem, system)
        spec = dict(kind="front", left={"u": 1}, right={"u": 0}, width=1.5)
        given = analyse(PROBLEMS / "hyperbolic_front.py", wave=dict(spec, speed=found.c))
        self.assertEqual((given["label"], given["decided_in_pass"]), ("STABLE", 2), given["reason"])
        self.assertLess(abs(given["wave"]["unfolding_parameter"]), 1e-8)
        wrong = analyse(PROBLEMS / "hyperbolic_front.py", wave=dict(spec, speed=found.c + 0.1))
        self.assertEqual(wrong["label"], "UNCERTAIN")
        self.assertIn("there is no wave with the given speed", wrong["reason"])
        # Without damping every speed below 1 has a kink, U = tanh(z / sqrt(2 (1 - c^2))).
        problem = load_problem(PROBLEMS / "phi4_kink.py",
                               wave=dict(kind="front", left={"u": -1}, right={"u": 1}, speed=0.5, width=1.2))
        kink = solve_wave_at_speed(problem, LocalSystem(problem))
        z = np.linspace(-3, 3, 13)
        profile = kink.interpolant(z)[0]
        shift = brentq(lambda s: kink.interpolant(np.array([s]))[0, 0], -2, 2)
        np.testing.assert_allclose(profile, np.tanh((z - shift) / np.sqrt(1.5)), atol=1e-7)

    def test_rounding_of_the_dispersion_curves_is_not_an_instability(self):
        # Schroedinger-type symbol: eigenvalues +-i (1 + kappa^2).  The computed real parts are rounding
        # errors of the order eps kappa^2; at kappa = 1e4 they exceed a tolerance that ignores the wavenumber.
        a = np.array([[[0.0, 1.0], [-1.0, 0.0]], np.zeros((2, 2)), [[0.0, -1.0], [1.0, 0.0]]])
        s, sides = essential_bound(a, a)
        self.assertLess(abs(s), axis_tolerance(sides, 1.0))
        self.assertLess(axis_tolerance(sides, 1.0), 1e-5)
        b = a.copy()
        b[0] += 1e-4 * np.eye(2)                                            # a true growth rate is still found
        s, sides = essential_bound(b, b)
        self.assertGreater(s, axis_tolerance(sides, 1.0))
        # The Schroedinger soliton: the rounding errors are not taken for an essential instability.
        record = analyse(PROBLEMS / "nls_soliton.py")
        self.assertLess(abs(record["essential"]["s_ess"]), 1e-6)
        self.assertEqual(record["label"], "STABLE", record["reason"])
        self.assertTrue(record["wave"]["computed"])

    def test_default_search_rectangle(self):
        # Without an estimate and without options['region']: a rule, and a printed qualifier with STABLE.
        record = analyse(PROBLEMS / "phi4_kink.py", region=None)
        self.assertEqual(record["label"], "STABLE", record["reason"])
        self.assertEqual(record["qualifier"],
                         "neutral: no spectrum with Re(lambda) > 0.02; default search rectangle 8 x 16")
        self.assertEqual(record["neutral"]["region"]["source"], "default rule (no estimate)")
        record = analyse(PROBLEMS / "klein_gordon_pulse.py", region=None)
        self.assertEqual(record["label"], "UNSTABLE", record["reason"])
        self.assertNotIn("qualifier", record)                              # UNSTABLE needs no qualifier
        self.assertLess(abs(record["roots"][0]["value"][0] - np.sqrt(3)), 1e-4)
        record = analyse(PROBLEMS / "haptotaxis_front.py", region=None)
        self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
        self.assertTrue(record["qualifier"].startswith("default search rectangle"))
        self.assertEqual(record["region"]["source"], "default rule (no estimate)")
        record = analyse(PROBLEMS / "haptotaxis_front_passive.py", region=None)
        self.assertEqual(record["label"], "UNSTABLE")
        self.assertNotIn("qualifier", record)
        record = analyse(PROBLEMS / "gkdv_soliton.py", parameters=dict(p=5), region=None)
        self.assertEqual(record["label"], "UNSTABLE")
        self.assertLess(abs(record["roots"][0]["value"][0] - 0.6354), 2e-3)
        # A rectangle of the file takes precedence, and the label then has no such qualifier.
        self.assertEqual(analyse(PROBLEMS / "haptotaxis_front.py").get("qualifier"), None)

    def test_a_given_speed_that_the_wave_does_not_have(self):
        # The Nagumo front has the speed 0.35355.  The index says that the speed is isolated, so it is
        # an unknown; a wave with another speed than the given one is not accepted.
        spec = dict(kind="front", left={"u": 1}, right={"u": 0}, width=1.5)
        wrong = analyse(PROBLEMS / "nagumo_front.py", wave=dict(spec, speed=0.3))
        self.assertEqual(wrong["label"], "UNCERTAIN")
        self.assertIn("not the given speed 0.3", wrong["reason"])
        right = analyse(PROBLEMS / "nagumo_front.py", wave=dict(spec, speed=0.5 / np.sqrt(2)))
        self.assertEqual((right["label"], right["decided_in_pass"]), ("STABLE", 2), right["reason"])


class RotationSymmetry(unittest.TestCase):
    """Waves with a frequency, u = exp(omega t J) U(x - c t): Schroedinger and Ginzburg-Landau type."""

    PULSE = dict(kind="pulse", rest={"u": 0, "v": 0})

    @staticmethod
    def frame(name, parameters=None, wave=None):
        problem = load_problem(PROBLEMS / name, parameters, wave=wave)
        frame = RotatingFrame(problem, LocalSystem(problem))
        return frame, frame.solve_wave()

    @staticmethod
    def difference_operator_eigenvalue(frame, wave, guess):
        """Eigenvalue near `guess` of the linearisation in the rotating frame, by differences (Richardson)."""
        values = []
        for nz in (2000, 4000):
            ev = fd_probe(fd_operator(frame.system, wave, wave.L, wave.L, nz), [guess], k=6)
            values.append(ev[np.argmin(np.abs(ev - guess))])
        return values[1] + (values[1] - values[0]) / 3

    @staticmethod
    def hocking_stewartson(mu=-1.0, alpha=0.3, beta=1.0):
        """Exact pulse R sech(kappa x)^(1 + i delta) exp(i omega t) of the cubic Ginzburg-Landau equation."""
        q = (1 + 1j * beta) / (1 + 1j * alpha)
        d = (-3 * q.real + np.sqrt(9 * q.real ** 2 + 8 * q.imag ** 2)) / (2 * q.imag)
        k2 = -mu / (1 - d * d - 2 * alpha * d)
        return dict(delta=d, kappa=np.sqrt(k2), omega=k2 * (alpha * (1 - d * d) + 2 * d),
                    R=np.sqrt(k2 * (2 - d * d) / q.real))

    @staticmethod
    def intensity(frame, wave, z):
        """|psi|^2 on the wave, as a function of the distance from its centroid."""
        zz = np.linspace(-wave.L, wave.L, 20001)
        s = (wave.Y(zz)[frame.system.offset][list(frame.pair)] ** 2).sum(axis=0)
        centre = np.trapezoid(zz * s, zz) / np.trapezoid(s, zz)
        return (wave.Y(z + centre)[frame.system.offset][list(frame.pair)] ** 2).sum(axis=0)

    def test_detection_and_refusals(self):
        pairs = lambda name: rotation_pairs(LocalSystem(load_problem(PROBLEMS / name)))
        self.assertEqual(pairs("nls_soliton.py"), [(0, 1)])
        self.assertEqual(pairs("cgl_pulse_passive.py"), [(0, 1)])            # the passive component does not rotate
        for name in ("nagumo_front.py", "fhn_fast_pulse.py", "cross_diffusion_front.py", "hyperbolic_front.py"):
            self.assertEqual(pairs(name), [], name)
        plain = analyse(PROBLEMS / "nagumo_front.py", wave=dict(kind="front", left={"u": 1}, right={"u": 0},
                                                               speed_guess=0.3, frequency=1.0, width=1.5))
        self.assertEqual(plain["label"], "UNCERTAIN")
        self.assertIn("do not commute with the rotation", plain["reason"])
        dark = analyse(PROBLEMS / "nls_soliton.py", wave=dict(kind="front", left={"u": -1, "v": 0},
                                                             right={"u": 1, "v": 0}, speed=0.0, frequency=1.0))
        self.assertEqual(dark["label"], "UNCERTAIN")                         # end states of non-zero amplitude
        self.assertIn("not invariant under the rotation", dark["reason"])
        both = analyse(PROBLEMS / "nls_soliton.py", wave=dict(self.PULSE, speed=0.0, frequency=1.0, frequency_guess=1.0,
                                                             guess={"u": "1.3/cosh(z)", "v": 0}))
        self.assertIn("not both", both["reason"])

    def test_schroedinger_solitons(self):
        # Cubic equation: no growing mode (Vakhitov-Kolokolov).  A Galilean boost gives the same wave.
        for c in (0.0, 0.6):
            record = analyse(PROBLEMS / "nls_soliton.py", parameters=dict(c=c))
            self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
            self.assertTrue(record["qualifier"].startswith("neutral: no spectrum with Re(lambda) > "))
            self.assertEqual(record["symmetry"]["rotation"], ["u", "v"])
            self.assertAlmostEqual(record["symmetry"]["frequency"], 1 + c * c / 4, places=12)
            self.assertEqual(record["wave"]["speed"], c)
            for d in record["wave"]["unfolding_parameters"].values():       # both values are given: nothing to unfold
                self.assertLess(abs(d), 1e-10)
            frame, wave = self.frame("nls_soliton.py", dict(c=c))
            z = np.linspace(-5, 5, 41)
            np.testing.assert_allclose(self.intensity(frame, wave, z), 2 / np.cosh(z) ** 2, atol=1e-7)
        # Other members of the family: amplitude sqrt(2 omega).
        frame, wave = self.frame("nls_soliton.py", dict(omega=2.5))
        self.assertAlmostEqual(self.intensity(frame, wave, np.zeros(1))[0], 5.0, places=7)

    def test_supercritical_schroedinger_soliton_is_unstable(self):
        # Septic equation (sigma = 3 > 2): one real positive eigenvalue; the same at rest and with a boost.
        values = []
        for c in (0.0, 0.6):
            record = analyse(PROBLEMS / "nls_soliton.py", parameters=dict(q=1.0, c=c))
            self.assertEqual((record["label"], record["decided_in_pass"]), ("UNSTABLE", 1), record["reason"])
            (root,) = record["roots"]
            self.assertLess(abs(root["value"][1]), 1e-9)
            values.append(root["value"][0])
        self.assertLess(abs(values[0] - values[1]), 2e-3)
        frame, wave = self.frame("nls_soliton.py", dict(q=1.0))
        self.assertAlmostEqual(self.intensity(frame, wave, np.zeros(1))[0], 4 ** (1 / 3), places=7)
        reference = self.difference_operator_eigenvalue(frame, wave, values[0])
        self.assertLess(abs(values[0] - reference), 2e-3)

    def test_a_family_of_waves_needs_given_values(self):
        guess = dict(self.PULSE, guess={"u": "1.3/cosh(z/1.1)", "v": 0})
        family = analyse(PROBLEMS / "nls_soliton.py", wave=dict(guess, speed_guess=0.0, frequency_guess=1.0))
        self.assertEqual(family["label"], "UNCERTAIN")                       # the iteration returns some member
        self.assertIn("is not isolated (a family of waves)", family["reason"])
        missing = analyse(PROBLEMS / "nls_soliton.py", wave=dict(guess, speed=0.0))
        self.assertEqual(missing["label"], "UNCERTAIN")
        self.assertIn("needs wave['frequency'] or wave['frequency_guess']", missing["reason"])

    def test_hocking_stewartson_pulse_in_closed_form(self):
        exact = self.hocking_stewartson()
        guess = dict(self.PULSE, guess={"u": "1.4/cosh(1.1*z)", "v": 0})
        z = np.linspace(-5, 5, 41)
        for values in (dict(speed=0.0, frequency_guess=1.2), dict(speed_guess=0.0, frequency_guess=1.2),
                       dict(speed_guess=0.0, frequency=exact["omega"]), dict(speed=0.0, frequency=exact["omega"])):
            frame, wave = self.frame("cgl_pulse.py", wave=dict(guess, **values))
            self.assertAlmostEqual(wave.info["frequency"], exact["omega"], places=9, msg=str(values))
            self.assertLess(abs(wave.c), 1e-9)
            np.testing.assert_allclose(self.intensity(frame, wave, z),
                                       exact["R"] ** 2 / np.cosh(exact["kappa"] * z) ** 2, atol=1e-7)
        # The isolated pulse does not exist at another frequency.
        wrong = analyse(PROBLEMS / "cgl_pulse.py", wave=dict(guess, speed=0.0, frequency=1.3))
        self.assertEqual(wrong["label"], "UNCERTAIN")
        self.assertIn("there is no wave with the given speed 0 and frequency 1.3", wrong["reason"])

    def test_hocking_stewartson_pulse_is_unstable(self):
        record = analyse(PROBLEMS / "cgl_pulse.py", parity_shortcut=False)
        self.assertEqual((record["label"], record["decided_in_pass"]), ("UNSTABLE", 1), record["reason"])
        first = record["pass_1"]
        self.assertEqual((first["W"], first["m0"]), (3, 2))                  # one root and the two symmetry zeros
        (root,) = record["roots"]
        frame, wave = self.frame("cgl_pulse.py")
        reference = self.difference_operator_eigenvalue(frame, wave, root["value"][0])
        self.assertLess(abs(root["value"][0] - reference), max(root["error"], 2e-4))
        # Real coefficients: the pulse of cubic_pulse.py scaled; frequency 0 and the eigenvalue -3 mu.
        real = analyse(PROBLEMS / "cgl_pulse.py", parameters=dict(alpha=0.0, beta=0.0))
        self.assertLess(abs(real["symmetry"]["frequency"]), 1e-10)
        self.assertLess(abs(real["roots"][0]["value"][0] - 3.0), real["roots"][0]["error"])

    def test_passive_component_on_the_pulse_in_closed_form(self):
        # w-block: Poeschl-Teller, lambda_n = -m + kappa^2 (s - n)^2 with s (s + 1) = sigma R^2 / kappa^2.
        exact = self.hocking_stewartson()
        s = (-1 + np.sqrt(1 + 4 * 3.8 * exact["R"] ** 2 / exact["kappa"] ** 2)) / 2
        closed = [-1 + exact["kappa"] ** 2 * (s - n) ** 2 for n in (0, 1)]
        record = analyse(PROBLEMS / "cgl_pulse_passive.py", parity_shortcut=False)
        self.assertEqual(record["label"], "UNSTABLE", record["reason"])
        self.assertEqual((record["pass_1"]["W"], record["pass_1"]["m0"]), (5, 2))
        found = [r for r in record["roots"] if r["status"] == "unstable"]
        self.assertEqual(len(found), 3)                                     # two of the w-block, one of the pulse
        for value in closed:
            nearest = min(found, key=lambda r: abs(r["value"][0] - value))
            self.assertLess(abs(nearest["value"][0] - value), min(nearest["error"], 5e-4))

    def test_dissipative_solitons_of_the_cubic_quintic_equation(self):
        # Two pulses for one set of parameters.  The low one is the threshold (unstable); the high one is stable.
        high = analyse(PROBLEMS / "cq_cgl_pulse.py")
        self.assertEqual((high["label"], high["decided_in_pass"]), ("STABLE", 2), high["reason"])
        self.assertEqual((high["pass_1"]["W"], high["pass_1"]["m0"]), (2, 2))  # the translation zero and the rotation zero
        self.assertIn("2 symmetry zeros", high["pass_1"]["reason"])
        self.assertTrue(all(c["passed"] for c in high["pass_2"]["checks"]))
        self.assertTrue(high["wave"]["frequency_is_unknown"])
        low = analyse(PROBLEMS / "cq_cgl_pulse.py", parameters=dict(a0=1.0, w0=0.5))
        self.assertEqual((low["label"], low["decided_in_pass"]), ("UNSTABLE", 1), low["reason"])
        self.assertLess(low["wave"]["frequency"], 0.2)
        self.assertGreater(high["wave"]["frequency"], 1.5)

    @slow
    def test_hopf_instability_of_the_dissipative_soliton(self):
        # Larger gain: a complex pair has crossed the axis (pulsating solitons, Soto-Crespo et al. 2000).
        pars = dict(eps=0.66, a0=2.6, w0=2.6)
        record = analyse(PROBLEMS / "cq_cgl_pulse.py", parameters=pars)
        self.assertEqual((record["label"], record["decided_in_pass"]), ("UNSTABLE", 1), record["reason"])
        (root,) = [r for r in record["roots"] if r["status"] == "unstable"]
        value = complex(*root["value"])
        self.assertGreater(value.imag, 0.5)
        frame, wave = self.frame("cq_cgl_pulse.py", pars)
        reference = self.difference_operator_eigenvalue(frame, wave, value)
        self.assertLess(abs(value - reference), 5e-4)
        self.assertGreater(reference.real, 0.01)


class Survey(unittest.TestCase):
    """Input = equations and parameter values only.  Output = a list of waves with labels."""

    @staticmethod
    def kinds(result):
        return {tuple(e["state"].values()): e["kind"] for e in result["equilibria"]}

    def test_equilibria(self):
        for name, expected in (("nagumo.py", [[0.0], [0.25], [1.0]]),
                               ("allen_cahn.py", [[-1.0], [0.0], [1.0]]),
                               ("fitzhugh_nagumo.py", [[0.0, 0.0]]),
                               ("cross_diffusion.py", [[0.0, 0.0], [0.25, 0.0], [1.0, 0.0]])):
            system = LocalSystem(load_problem(EQUATIONS / name))
            found = find_equilibria(system, -100.0, 100.0)
            np.testing.assert_allclose(np.array(found), np.array(expected), atol=1e-10)

    def test_nagumo(self):
        result = survey(EQUATIONS / "nagumo.py")
        self.assertEqual(self.kinds(result), {(0.0,): "stable", (0.25,): "unstable", (1.0,): "stable"})
        waves = {w["kind"]: w for w in result["waves"]}
        self.assertEqual(sorted(waves), ["front", "pulse"])
        self.assertEqual(waves["front"]["label"], "STABLE")
        self.assertLess(abs(abs(waves["front"]["speed"]) - 0.5 / np.sqrt(2)), 1e-7)
        # Standing pulse on u = 0: its maximum is the positive root of  u^2/4 - (1 + a) u/3 + a/2 = 0.
        self.assertEqual(waves["pulse"]["label"], "UNSTABLE")
        maximum = 2 * (1.25 / 3 - np.sqrt(1.25 ** 2 / 9 - 0.125))
        self.assertLess(abs(waves["pulse"]["extremum"]["u"] - maximum), 1e-4)
        self.assertEqual([(f["state"]["u"], f["label"]) for f in result["families"]], [(0.25, "UNSTABLE")])
        # There is no pulse on u = 1 for a < 1/2: the search must report "not found", not a wave.
        self.assertEqual([(c["kind"], c["left"]["u"]) for c in result["not_found"]], [("pulse", 1.0)])

    def test_fisher_kpp(self):
        # u = 0 is unstable: one statement for all fronts.  The dip on u = 1 reaches u = -1/2 exactly.
        result = survey(EQUATIONS / "fisher_kpp.py")
        self.assertEqual(self.kinds(result), {(0.0,): "unstable", (1.0,): "stable"})
        self.assertEqual([(f["state"]["u"], f["label"]) for f in result["families"]], [(0.0, "UNSTABLE")])
        (pulse,) = result["waves"]
        self.assertEqual((pulse["kind"], pulse["label"]), ("pulse", "UNSTABLE"))
        self.assertLess(abs(pulse["extremum"]["u"] + 0.5), 1e-4)

    def test_two_mirror_pulses(self):
        # u_t = u_xx - u + 2 u^3: the pulses sech(z) and -sech(z), both with the eigenvalue 3.
        result = survey(EQUATIONS / "cubic.py", parity_shortcut=False)
        self.assertEqual(sorted(round(w["extremum"]["u"], 3) for w in result["waves"]), [-1.0, 1.0])
        for w in result["waves"]:
            self.assertEqual(w["label"], "UNSTABLE")
            self.assertLess(abs(w["record"]["roots"][0]["value"][0] - 3.0), 3e-4)
        self.assertEqual(len(result["families"]), 2)

    def test_free_boundary(self):
        result = survey(EQUATIONS / "fisher_stefan.py")
        (wave,) = result["waves"]
        self.assertEqual((wave["kind"], wave["label"]), ("semi", "STABLE"))
        self.assertLess(abs(wave["speed"] - 0.36437072), 1e-6)
        # Neumann condition; a component without diffusion (speed 1 and eigenvalue 0.972616 by construction).
        for name, speed, eigenvalue in (("stefan_robin.py", 0.0, 3.0), ("stefan_feedback.py", 1.0, 0.972616)):
            (wave,) = survey(EQUATIONS / name)["waves"]
            self.assertEqual((wave["kind"], wave["label"]), ("semi", "UNSTABLE"))
            self.assertLess(abs(wave["speed"] - speed), 1e-5)
            self.assertLess(abs(wave["record"]["roots"][0]["value"][0] - eigenvalue), 1e-3)

    def test_fitzhugh_nagumo_both_pulses(self):
        # The generic guesses reach the slow (unstable) pulse.  It has one real unstable eigenvalue, so
        # its branch is followed; beyond a fold lies the fast pulse, which is stable.
        result = survey(EQUATIONS / "fitzhugh_nagumo.py", double_pulses=0)
        self.assertEqual(len(result["equilibria"]), 1)
        slow, fast = result["waves"]
        self.assertEqual((slow["label"], slow["found_by"]), ("UNSTABLE", "direct search"))
        self.assertLess(abs(slow["speed"] - 0.38702890), 1e-6)
        self.assertEqual((fast["label"], fast["found_by"]), ("STABLE", "continuation through a fold"))
        self.assertLess(abs(fast["speed"] - 0.43491843), 1e-6)
        self.assertEqual(fast["record"]["decided_in_pass"], 2)
        # Without the partner search only the unstable pulse is listed.
        alone = survey(EQUATIONS / "fitzhugh_nagumo.py", partners=False)
        self.assertEqual([w["label"] for w in alone["waves"]], ["UNSTABLE"])

    def test_haptotaxis_front_from_the_equations_alone(self):
        result = survey(EQUATIONS / "haptotaxis.py")
        (wave,) = result["waves"]
        self.assertEqual((wave["kind"], wave["label"]), ("front", "STABLE"), wave["reason"])
        self.assertAlmostEqual(abs(wave["speed"]), 0.4111473, places=5)
        self.assertTrue(wave["qualifier"].startswith("default search rectangle"))       # no estimate for taxis
        self.assertEqual(wave["record"]["taxis"]["without_diffusion"], ["m"])
        self.assertEqual([f["label"] for f in result["families"]], ["UNSTABLE"])

    def test_undamped_wave_equations_list_standing_waves(self):
        # No damping: the end states are marginal and the waves form a family in the speed.
        result = survey(EQUATIONS / "phi4.py")
        self.assertEqual([e["kind"] for e in result["equilibria"]], ["marginal", "unstable", "marginal"])
        (kink,) = result["waves"]
        self.assertEqual((kink["kind"], kink["label"], kink["speed"]), ("front", "STABLE", 0.0), kink["reason"])
        self.assertTrue(kink["qualifier"].startswith("neutral: no spectrum with Re(lambda) > 0.02; default search"))
        self.assertIn("family of waves in the speed", kink["note"])
        result = survey(EQUATIONS / "klein_gordon.py")
        self.assertEqual([(w["kind"], w["label"]) for w in result["waves"]], [("pulse", "UNSTABLE")] * 2)
        np.testing.assert_allclose(sorted(w["extremum"]["u"] for w in result["waves"]), [-1.0, 1.0], atol=1e-4)
        root = result["waves"][0]["record"]["roots"][0]
        self.assertLess(abs(root["value"][0] - np.sqrt(3)), max(root["error"], 1e-4))

    def test_conservation_laws_need_end_states(self):
        result = survey(EQUATIONS / "burgers.py")
        self.assertEqual(result["waves"], [])
        self.assertIn("give the end states", result["note"])

    @slow
    def test_rotation_symmetry_lists_waves_with_frequency_zero_only(self):
        # Travelling waves only: a wave of a rotation-symmetric equation is listed if its frequency is 0.
        result = survey(EQUATIONS / "ginzburg_landau_real.py")
        self.assertEqual(result["rotation"], dict(components=["u", "v"], frequency=0.0))
        self.assertTrue(result["equilibria"][0]["kind"].startswith("circle of equilibria"))
        (pulse,) = result["waves"]
        self.assertEqual((pulse["kind"], pulse["label"]), ("pulse", "UNSTABLE"), pulse["reason"])
        self.assertAlmostEqual(pulse["extremum"]["u"], np.sqrt(2), places=4)
        self.assertEqual(pulse["record"]["symmetry"]["frequency"], 0.0)
        # The pulses of the cubic-quintic equation rotate: they are no travelling waves and are not listed.
        rotating = survey(EQUATIONS / "cq_cgl.py")
        self.assertEqual(rotating["waves"], [])
        self.assertTrue(any("there is no wave with the given frequency 0" in reason
                            for c in rotating["not_found"] for reason in c["reasons"]))

    @slow
    def test_double_pulses_are_listed(self):
        # Two copies of the stable fast pulse at the three smallest admissible distances; the labels alternate.
        result = survey(EQUATIONS / "fitzhugh_nagumo.py")
        doubles = [w for w in result["waves"] if w.get("copies") == 2]
        np.testing.assert_allclose([w["distance"] for w in doubles], [51.385, 70.630, 90.205], atol=2e-3)
        self.assertEqual([w["label"] for w in doubles], ["STABLE", "UNSTABLE", "STABLE"])
        self.assertTrue(any("multi-pulse made of copies of the unstable pulse" in (f.get("text") or "")
                            and f["label"] == "UNSTABLE" for f in result["families"]))
        self.assertEqual(len(result["waves"]), 5)

    @slow
    def test_systems_and_the_second_engine(self):
        # (file, speed of the front, label of the front)
        for name, speed, label in (("allen_cahn.py", 0.0, "STABLE"),
                                   ("cross_diffusion.py", 0.5 / np.sqrt(2), "UNSTABLE"),
                                   ("nonlocal_auxiliary.py", 0.5 / np.sqrt(2), "STABLE"),
                                   ("delay_nagumo.py", 0.17971874, "STABLE"),
                                   ("delay_nagumo_distributed.py", 0.18314646, "STABLE"),
                                   ("kernel_gauss.py", 0.43009894, "STABLE"),
                                   ("nonlocal_ac.py", 0.23325583, "STABLE"),
                                   ("efk.py", 0.0, "STABLE"),
                                   ("hyperbolic_nagumo.py", 0.31622777, "STABLE")):
            with self.subTest(file=name):
                result = survey(EQUATIONS / name)
                fronts = [w for w in result["waves"] if w["kind"] == "front"]
                self.assertEqual(len(fronts), 1)                   # the mirror image is not listed twice
                self.assertEqual(fronts[0]["label"], label, fronts[0]["reason"])
                self.assertLess(abs(abs(fronts[0]["speed"]) - speed), 1e-6)
                self.assertEqual(len(result["families"]), 1)


    @slow
    def test_neural_field_without_diffusion_lists_a_stable_pulse(self):
        # Only the equation u_t = -u + K * S(u) is given.  On the rest state the search finds the narrow
        # bump (unstable) and the wide bump (stable), as in Amari (1977).
        result = survey(EQUATIONS / "neural_field.py")
        self.assertEqual({round(state[0], 3): kind for state, kind in self.kinds(result).items()},
                         {0.0: "stable", 0.615: "unstable", 0.948: "stable"})
        on_rest = sorted((w["extremum"]["u"], w["label"]) for w in result["waves"]
                         if w["kind"] == "pulse" and abs(w["left"]["u"]) < 1e-9)
        self.assertEqual([label for _, label in on_rest], ["UNSTABLE", "STABLE"])
        fronts = [w for w in result["waves"] if w["kind"] == "front"]
        self.assertEqual([w["label"] for w in fronts], ["STABLE"])


class ListInput(unittest.TestCase):
    """A list of equations in, a list of labels out (batch.py; DESIGN 4.1 and 4.3)."""

    def setUp(self):
        self.x, self.t = sp.symbols("x t", real=True)
        self.u = sp.Function("u")(self.x, self.t)

    def test_one_result_for_each_entry_in_the_order_of_the_list(self):
        x, t, u = self.x, self.t, self.u
        a = sp.Symbol("a", positive=True)
        problems = [dict(equations=sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u) * (u - a)), parameters={a: 0.25},
                         wave=dict(kind="front", left={u: 1}, right={u: 0})),
                    "not an equation",
                    dict(name="burgers", equations=[sp.Eq(u.diff(t) + (u ** 2 / 2).diff(x), u.diff(x, 2))],
                         wave=dict(kind="front", left={u: 2}, right={u: 0})),
                    sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u))]
        seen = []
        results = solve(problems, on_result=seen.append)
        self.assertEqual([r["name"] for r in results], ["eqn1", "eqn2", "burgers", "eqn4"])
        self.assertEqual(seen, results)
        json.dumps(results)                                    # plain JSON types
        (front,) = results[0]["waves"]
        self.assertEqual((front["wave"], front["label"], front["qualifier"]),
                         ("front (u = 1) -> (u = 0)", "STABLE", None))
        self.assertAlmostEqual(front["speed"], 0.5 / np.sqrt(2), places=7)
        self.assertEqual(results[1]["waves"], [])              # the list goes on after an entry that cannot be read
        self.assertTrue(results[1]["note"].startswith("input error"))
        self.assertEqual(lines(results[1]), [f"eqn2: no label ({results[1]['note']})"])
        self.assertEqual(lines(results[2]), ["burgers: front (u = 2) -> (u = 0), c = 1: STABLE (algebraic decay)"])
        # Fisher-KPP from the equation alone: every wave that ends at the unstable state u = 0 is unstable.
        self.assertIn(("every wave that ends at (u = 0)", "UNSTABLE", None),
                      [(w["wave"], w["label"], w["speed"]) for w in results[3]["waves"]])
        self.assertIsNone(results[3]["note"])

    def test_x_and_t_are_read_from_the_equations(self):
        xi, s, y = sp.symbols("xi s y", real=True)
        w = sp.Function("w")
        delayed = sp.Eq(w(xi, s).diff(s), w(xi, s).diff(xi, 2) + w(xi, s - 2) - w(xi, s) ** 3)
        convolution = sp.Eq(w(xi, s).diff(s), sp.Integral(sp.exp(-(xi - y) ** 2) * w(y, s), (y, -sp.oo, sp.oo))
                            - w(xi, s) ** 3)
        second_order = sp.Eq(w(xi, s).diff(s, 2), w(xi, s).diff(xi, 2) - w(xi, s))
        for equation in (delayed, convolution, second_order):
            problem = load_problem(dict(equations=equation))
            self.assertEqual((problem.x, problem.t, problem.wave, problem.name), (xi, s, None, "entry"))
        with self.assertRaises(ProblemError):                  # no time derivative: t is not determined
            load_problem(dict(equations=sp.Eq(w(xi, s).diff(xi, 2), w(xi, s))))

    def test_an_entry_that_fails_does_not_stop_the_list(self):
        x, t, u = self.x, self.t, self.u
        entry = dict(equations=sp.Eq(u.diff(t), u.diff(x, 2) + u * (1 - u) * (u - 0.25)),
                     wave=dict(kind="front", left={u: 1}, right={u: 0}))
        answers = [RuntimeError("boom"), dict(label="STABLE", reason="r", wave=dict(computed=True, speed=0.25))]
        with mock.patch("twstab.batch.analyse", side_effect=answers):
            first, second = solve([entry, entry])
        self.assertEqual((first["waves"], first["note"]), ([], "solver error: RuntimeError: boom"))
        self.assertEqual([(w["label"], w["speed"]) for w in second["waves"]], [("STABLE", 0.25)])

    def test_the_example_list_follows_the_input_format(self):
        entries = read_list(ROOT / "examples" / "example_list.py")
        self.assertEqual(len(entries), 11)
        problems = [load_problem(batch._entry(item, k)) for k, item in enumerate(entries, start=1)]
        self.assertEqual([p.name for p in problems], [f"eqn{k}" for k in range(1, 11)] + ["fitzhugh_nagumo"])
        self.assertEqual([k for k, p in enumerate(problems, start=1) if p.wave is not None], [3, 9, 10])
        with self.assertRaises(ProblemError):                  # a file of one problem is not a list
            read_list(PROBLEMS / "nagumo_front.py")

    def test_a_named_pulse_of_a_conservation_law_needs_no_guess(self):
        # Rest state and speed only.  The amplitude of the guess comes from the constant solutions of the
        # reduced profile equation.  Pego and Weinstein 1992: neutral for p = 2, unstable for p = 5.
        x, t, u = self.x, self.t, self.u
        p = sp.Symbol("p", positive=True)
        entries = [dict(equations=sp.Eq(u.diff(t) + u ** p * u.diff(x) + u.diff(x, 3), 0), parameters={p: value},
                        wave=dict(kind="pulse", rest={u: 0}, speed=1.0)) for value in (2, 5)]
        (subcritical,), (supercritical,) = (r["waves"] for r in solve(entries))
        self.assertEqual((subcritical["label"], subcritical["speed"]), ("STABLE", 1.0), subcritical["reason"])
        self.assertTrue(subcritical["qualifier"].startswith("neutral: no spectrum with Re(lambda) > "))
        self.assertEqual((supercritical["label"], supercritical["qualifier"]), ("UNSTABLE", None),
                         supercritical["reason"])

    @slow
    def test_named_waves_without_a_guess_use_the_search(self):
        # The default guess of a named wave fails for these two; the search finds the waves.  The labels are
        # those of the problem files, which give a guess.
        entries = []
        for name in ("haptotaxis_front", "klein_gordon_pulse"):
            ns = runpy.run_path(str(PROBLEMS / f"{name}.py"))
            wave = {k: v for k, v in ns["wave"].items() if k in ("kind", "left", "right", "rest", "speed")}
            entries.append(dict(name=name, equations=ns["equations"], parameters=ns["parameters"], wave=wave,
                                options=dict(equilibria_range=3.0)))
        front, pulses = (r["waves"] for r in solve(entries))
        self.assertEqual([(w["label"], round(abs(w["speed"]), 6)) for w in front], [("STABLE", 0.411147)])
        self.assertEqual([w["label"] for w in pulses], ["UNSTABLE", "UNSTABLE"])      # the pulse and its mirror image

    @slow
    def test_the_example_list(self):
        results = solve(read_list(ROOT / "examples" / "example_list.py"))
        labels = {r["name"]: [w["label"] for w in r["waves"]] for r in results}
        self.assertEqual(labels, EXAMPLE_LABELS)
        self.assertTrue(all(r["note"] is None for r in results))


class Pipeline(unittest.TestCase):
    def test_fisher_is_essentially_unstable(self):
        record = analyse(PROBLEMS / "fisher_kpp.py")
        self.assertEqual((record["label"], record["decided_in_pass"]), ("UNSTABLE", 1))
        self.assertFalse(record["wave"]["computed"])

    def test_nagumo_is_stable_after_the_audit(self):
        record = analyse(PROBLEMS / "nagumo_front.py")
        self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
        self.assertTrue(all(c["passed"] for c in record["pass_2"]["checks"]))

    def test_cubic_pulse_is_unstable(self):
        record = analyse(PROBLEMS / "cubic_pulse.py")
        self.assertEqual((record["label"], record["decided_in_pass"]), ("UNSTABLE", 1), record["reason"])
        unstable = [r for r in record["roots"] if r["status"] == "unstable"]
        self.assertEqual(len(unstable), 1)
        self.assertLess(abs(complex(*unstable[0]["value"]) - 3.0), 3e-4)
        self.assertLess(abs(complex(*unstable[0]["value"]) - 3.0), unstable[0]["error"])
        self.assertEqual(record["pass_1"]["shortcut"], "parity")      # one real root: odd parity

    def test_a1_matches_version_1(self):
        record = analyse(PROBLEMS / "han_A1.py")
        self.assertEqual(record["label"], "UNSTABLE", record["reason"])
        self.assertLess(abs(record["wave"]["speed"]), 1e-6)
        unstable = [complex(*r["value"]) for r in record["roots"] if r["status"] == "unstable"]
        self.assertEqual(len(unstable), 1)                    # one root of a conjugate pair is stored
        self.assertLess(abs(unstable[0] - (4.469956 + 1.108045j)), 2e-4)
        self.assertEqual(record["pass_1"]["W"], 3)

    @slow
    def test_a1_other_parameter_points_of_version_1(self):
        # (q, a): unstable roots and the zero count in the box reported by Version_1.
        reference = {(1.0, 4.0): ([4.912131, 1.652243], 3), (1.0, 6.0): ([4.978320], 2),
                     (0.5, 4.0): ([4.129011, 1.113986], 3), (0.0, 4.0): ([3.0], 4)}
        for (q, a), (roots, count) in reference.items():
            with self.subTest(q=q, a=a):
                record = analyse(PROBLEMS / "han_A1.py", parameters=dict(q=q, a=a), parity_shortcut=False)
                self.assertEqual(record["label"], "UNSTABLE", record["reason"])
                found = sorted(r["value"][0] for r in record["roots"] if r["status"] == "unstable")
                self.assertEqual(len(found), len(roots))
                np.testing.assert_allclose(found, sorted(roots), atol=3e-4)
                self.assertEqual(record["pass_1"]["W"], count)
                self.assertTrue(record["pass_1"]["complete_count"])
                self.assertEqual(record["pass_1"]["parity"]["odd"], len(roots) % 2 == 1)

    def test_a1_with_the_independent_root_search(self):
        # Same case, but the seeds come from the subdivision of the box and not from the difference probe.
        record = analyse(PROBLEMS / "han_A1.py", parameters=dict(q=1.0, a=4.0), parity_shortcut=False,
                         seeds="evans")
        found = sorted(r["value"][0] for r in record["roots"] if r["status"] == "unstable")
        np.testing.assert_allclose(found, [1.652243, 4.912131], atol=3e-4)
        self.assertEqual((record["pass_1"]["W"], record["pass_1"]["complete_count"]), (3, True))

    @slow
    def test_graded_mesh_against_the_uniform_mesh(self):
        # A/B: mesh_cap = 1 is the uniform mesh of M1.
        records = [analyse(PROBLEMS / "han_A1.py", mesh_cap=cap) for cap in (1.0, 4.0)]
        roots = [complex(*r["roots"][0]["value"]) for r in records]
        self.assertEqual(records[0]["pass_1"]["W"], records[1]["pass_1"]["W"])
        self.assertLess(abs(roots[0] - roots[1]), 1e-4)
        self.assertLess(records[1]["pass_1"]["cells_fine"], 0.5 * records[0]["pass_1"]["cells_fine"])

    def test_fhn_fast_pulse_is_stable(self):
        record = analyse(PROBLEMS / "fhn_fast_pulse.py")
        self.assertEqual((record["label"], record["decided_in_pass"]), ("STABLE", 2), record["reason"])
        self.assertTrue(all(c["passed"] for c in record["pass_2"]["checks"]))
        # Energy bound of the partly parabolic system, by hand: x_max = 1.1 (sup df/du + 4/(3 gamma)).
        self.assertEqual(record["region"]["source"], "energy estimate")
        self.assertAlmostEqual(record["region"]["x_max"], 1.1 * ((1 - 0.1 + 0.01) / 3 + 4 / 3), places=3)

    def test_fhn_slow_pulse_is_unstable(self):
        # Same equations and parameters as the fast pulse; the wave is reached by continuation through a fold.
        record = analyse(PROBLEMS / "fhn_slow_pulse.py")
        self.assertEqual(record["label"], "UNSTABLE", record["reason"])
        self.assertEqual(len(record["wave"]["continuation"]["folds"]), 1)
        self.assertLess(record["wave"]["speed"], 0.41)
        shortcut = record["roots"][0]["value"][0]
        # The full procedure (box count, difference probe) must give the same root.
        full = analyse(PROBLEMS / "fhn_slow_pulse.py", parity_shortcut=False)
        self.assertEqual((full["label"], full["pass_1"]["W"], full["pass_1"]["m0"]), ("UNSTABLE", 2, 1))
        self.assertLess(abs(full["roots"][0]["value"][0] - shortcut), 1e-6)
        self.assertLess(shortcut, full["region"]["x_max"])


if __name__ == "__main__":
    unittest.main()
