"""E2 exact moving interface wave and independent boundary transfer check."""
from fractions import Fraction

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import brentq

from .contract import ROOT, digest
from .families import rational


def interface_wave(gamma, beta, seed):
    gamma, beta = rational(gamma), rational(beta)
    if gamma not in (1, 2) or beta not in (-4, -8):
        raise ValueError("outside the fixed E2 pilot grid")
    problem = dict(family_id="E2", structure_id="E2.two-field-interface-feedback", template_version="1",
                   parameters=dict(gamma=str(gamma), beta=str(beta), speed="3/2"),
                   equations=["u_t=u_xx-u+gamma*(v-u^2)", "v_t=v_xx-(5/2)*v"],
                   boundary=dict(domain="x<s(t)", values=[1, 1],
                                 velocity="s'=3/2+3*(u_x-1/2)+beta*(v_x-1)"),
                   target_wave=dict(profile="(exp(z/2),exp(z))", frame="z=x-s(t)",
                                    interface="s(t)=3*t/2", endstate=[0, 0]),
                   eigenproblem=dict(space="L2((-inf,0);C^2) x C (interface displacement)",
                                     domain="H2 Dirichlet for both fields; free scalar displacement",
                                     stability_question="Is there spectrum in the open right half-plane?"))
    key = digest(problem)
    return dict(**problem, schema_version="e2-interface-0.1", seed=seed, instance_id=f"E2-{key[:20]}",
                specification_hash=key, calibration=False,
                validity_checks=dict(profile="PASS", linearisation="PASS", solver_dispatch="PASS",
                                     proof_difficulty="UNVERIFIED"),
                novelty_assessment=dict(status="UNCERTAIN", grouping=problem["structure_id"],
                                        training_admission="HOLD"), references=[])


def validate_interface_wave(item):
    p = item["parameters"]
    if type(item["seed"]) is not int or item != interface_wave(p["gamma"], p["beta"], item["seed"]):
        raise ValueError("E2 differs from the canonical specification")
    return item


def boundary_transfer(lam, gamma, beta, length, tolerance):
    """Solve the exact-profile Dirichlet response to a unit interface velocity."""
    speed = 1.5
    nu = 2 * (2.5 + lam) / (speed + np.sqrt(speed**2 + 4 * (2.5 + lam)))
    # nu-1 = lam/(nu+1+c), including the analytic limit at lam=0.
    slope_v = -1 / (nu + 1 + speed)

    def rhs(z, state):
        u = np.exp(z / 2)
        v_response = (-z * np.exp(z) / (2 + speed) if lam == 0 else
                      -np.exp(z) * np.expm1(lam * z / (nu + 1 + speed)) / lam)
        return np.vstack((state[1], -speed * state[1] + (lam + 1 + 2 * gamma * u) * state[0]
                          - gamma * v_response - u / 2))

    def boundary(left, right):
        return [left[0], right[0]]

    grid = np.linspace(-length, 0, 161)
    solution = solve_bvp(rhs, boundary, grid, np.zeros((2, grid.size)), tol=tolerance, max_nodes=10000)
    if not solution.success:
        raise ArithmeticError(f"independent interface transfer BVP failed: {solution.message}")
    return 3 * solution.y[1, -1] + beta * slope_v - 1


def verify_positive_root(centre, gamma, beta):
    radius = min(0.25, centre / 2)
    left, right = centre - radius, centre + radius
    rows = []
    for length, tol in ((14, 1e-6), (22, 1e-8), (30, 1e-10)):
        def response(lam):
            return boundary_transfer(lam, gamma, beta, length, tol)

        values = [response(left), response(right)]
        if values[0] * values[1] >= 0:
            raise ArithmeticError("independent transfer check did not bracket the reported real root")
        root = float(brentq(response, left, right, xtol=1e-11))
        rows.append(dict(length=length, tolerance=tol, bracket=[left, right], endpoint_values=values, root=root))
    return rows


def solve_interface_wave(item, analyse):
    p = validate_interface_wave(item)["parameters"]
    gamma, beta = (float(Fraction(p[key])) for key in ("gamma", "beta"))
    native = analyse(ROOT / "solver" / "problems" / "interface_feedback_wave.py",
                     parameters=dict(gamma=gamma, beta=beta))
    native["file"] = "solver/problems/interface_feedback_wave.py"
    result = dict(label="UNCERTAIN", evidence_grade=None, open_rhp_count=None, native=native,
                  reason="The run did not provide a corroborated positive root or a full-domain stability certificate.")
    wave = native.get("wave", {})
    if not wave.get("computed") or abs(wave.get("speed", np.inf) - 1.5) > 1e-5:
        result["reason"] = "Native profile computation did not recover the designated advancing speed."
        return result
    if native["label"] != "UNSTABLE":
        return result
    for witness in native.get("roots", []):
        real, imag = witness["value"]
        error = witness["error"]
        if witness["status"] != "unstable" or abs(imag) > max(error, 1e-9) or real <= max(1e-5, 10 * error):
            continue
        try:
            rows = verify_positive_root(real, gamma, beta)
        except ArithmeticError as exc:
            result["reason"] = str(exc)
            continue
        fine = rows[-1]["root"]
        discrepancy = abs(fine - complex(real, imag))
        variation = max(abs(fine - row["root"]) for row in rows[:-1])
        margin = max(error, discrepancy, variation)
        result["independent_transfer"] = dict(refinements=rows, native_root=witness["value"],
                                               native_difference=discrepancy, observed_error_scale=margin)
        if fine > max(1e-5, 10 * margin) and discrepancy < 1e-3:
            result.update(label="UNSTABLE", evidence_grade="CONVERGED_NUMERICAL",
                          reason="Native spectral result and independent exact-profile boundary transfer agree.")
            return result
    return result
