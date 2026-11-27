"""D2 negative delayed feedback on an exact standing Allen–Cahn kink."""
from fractions import Fraction

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from .contract import ROOT, digest
from .d2_certificate import certify_delay
from .families import rational


SUPPORT_WIDTH = Fraction(1, 4)


def shifted_delay_kink(sigma, tau, epsilon, seed):
    sigma, tau, epsilon = map(rational, (sigma, tau, epsilon))
    if not 0 < sigma <= 8 or not 1 <= tau <= 2 or not 0 <= epsilon <= Fraction(3, 4):
        raise ValueError("outside the reviewed D2 parameter range")
    parameters = dict(sigma=str(sigma), tau=str(tau), epsilon=str(epsilon), width=str(SUPPORT_WIDTH))
    problem = dict(family_id="D2", structure_id="D2.shifted-negative-memory-kink", template_version="1",
                   parameters=parameters,
                   equations=["u_t=u_xx+u-u^3",
                              "v_t=v_xx-v-(sigma/width)*(1-u^2)*(1+epsilon*u)*integral(v(x,t-s),s=tau..tau+width)"],
                   target_wave=dict(profile="(tanh(x/sqrt(2)),0)", speed=0, phase="u(0)=0, u_x(0)>0",
                                    endstates=[[-1, 0], [1, 0]]),
                   eigenproblem=dict(space="L2(R;C) x C([-(tau+width),0];L2(R;C))",
                                     domain="u perturbation in H2; C1 history with H2 endpoint and generator compatibility",
                                     characteristic="lambda-partial_x^2+1+sigma*H(lambda)*(1-U^2)*(1+epsilon*U)",
                                     delay_transform="H(lambda)=integral(exp(-lambda*s),s=tau..tau+width)/width"),
                   stability_question="Is there open-right-half-plane spectrum on the recorded space?")
    key = digest(problem)
    control = epsilon == 0 or sigma * (1 + epsilon) < 1
    return dict(**problem, schema_version="d2-delay-0.1", seed=seed, instance_id=f"D2-{key[:20]}",
                specification_hash=key, calibration=control,
                validity_checks=dict(profile="PASS", linearisation="PASS", solver_dispatch="PASS",
                                     proof_difficulty="NOT_APPLICABLE" if control else "UNVERIFIED"),
                novelty_assessment=dict(status="NOT_NOVEL" if control else "UNCERTAIN",
                                        grouping=problem["structure_id"], training_admission="HOLD"),
                references=["https://doi.org/10.3934/dcdsb.2015.20.1855"])


def validate_shifted_delay(item):
    parameters = item["parameters"]
    candidate = shifted_delay_kink(parameters["sigma"], parameters["tau"], parameters["epsilon"], item["seed"])
    if type(item["seed"]) is not int or candidate != item:
        raise ValueError("D2 differs from the canonical specification")
    return item


def laplace_average(lam, tau):
    scaled = float(SUPPORT_WIDTH) * lam
    return np.exp(-tau * lam) * (-np.expm1(-scaled) / scaled if scaled != 0 else 1)


def spatial_match(lam, sigma, tau, epsilon, halfwidth, step, tolerance):
    """Continuous shooting from both tails with the exact profile and memory transform."""
    kappa = np.sqrt(1 + lam)
    delayed = sigma * laplace_average(lam, tau)

    def tail(direction):
        # Use distance from the corresponding endpoint as the integration coordinate.
        # Divide out the free decaying mode, leaving a bounded two-component IVP.
        def rhs(distance, state):
            x = direction * (distance - halfwidth)
            u = np.tanh(x / np.sqrt(2))
            potential = delayed * (1 - u**2) * (1 + epsilon * u)
            value, derivative = state
            return derivative, potential * value - 2 * kappa * derivative

        sol = solve_ivp(rhs, (0, halfwidth), np.array([1, 0], dtype=complex), method="DOP853",
                        max_step=step, rtol=tolerance, atol=tolerance * 0.01)
        if not sol.success:
            raise ArithmeticError("continuous tail shooting failed")
        return sol.y[:, -1]

    left, left_derivative = tail(1)
    right, right_derivative = tail(-1)
    return 2 * kappa * left * right + left_derivative * right + left * right_derivative


def verify_root(start, sigma, tau, epsilon):
    evidence = []
    for halfwidth, step, tolerance in ((10, 0.4, 1e-8), (16, 0.25, 1e-10), (22, 0.18, 1e-12)):
        def characteristic(coordinates):
            value = spatial_match(complex(*coordinates), sigma, tau, epsilon, halfwidth, step, tolerance)
            return [value.real, value.imag]

        solved = root(characteristic, [start.real, start.imag], tol=2e-10)
        value = complex(*solved.x)
        residual = float(np.linalg.norm(characteristic(solved.x)))
        if not np.isfinite(residual) or residual > 1e-8:
            raise ArithmeticError("continuous shooting did not resolve the characteristic zero")
        evidence.append(dict(halfwidth=halfwidth, max_step=step, tolerance=tolerance,
                             root=[value.real, value.imag], matching_residual=residual))
        start = value
    return evidence


def solve_shifted_delay(item, analyse):
    parameters = validate_shifted_delay(item)["parameters"]
    exact = {name: Fraction(parameters[name]) for name in ("sigma", "tau", "epsilon")}
    settings = {name: float(value) for name, value in exact.items()}
    native = analyse(ROOT / "solver" / "problems" / "shifted_delay_kink.py", parameters=settings)
    native["file"] = "solver/problems/shifted_delay_kink.py"
    result = dict(label="UNCERTAIN", open_rhp_count=None, evidence_grade=None, native=native,
                  reason="No independently confirmed positive nonreal spectral witness.")
    if exact["sigma"] * (1 + exact["epsilon"]) < 1 and native["label"] != "UNSTABLE":
        result.update(label="STABLE", open_rhp_count=0, evidence_grade="RIGOROUSLY_CERTIFIED",
                      reason="The history resolvent has a uniform Neumann-series bound on the closed right half-plane.",
                      certificate=dict(resolvent_perturbation_bound=str(exact["sigma"] * (1 + exact["epsilon"])),
                                       scope="entire closed right half-plane of the delayed block",
                                       report="docs/a1-d2-comparison.md"))
        return result

    wave = native.get("wave", {})
    if (not wave.get("computed") or wave.get("left") != [-1, 0] or wave.get("right") != [1, 0]
            or abs(wave.get("speed", np.inf)) > 1e-7 or wave.get("profile_error", np.inf) > 1e-6):
        result["reason"] = "Native solver did not recover the designated standing front to the pilot tolerance."
        return result

    if native["label"] != "UNSTABLE":
        try:
            certificate = certify_delay(**exact)
        except (ArithmeticError, ValueError) as error:
            result["certificate_failure"] = str(error)
            result["reason"] = f"The full-domain D2 certificate did not close: {error}"
        else:
            result.update(label=certificate["label"], open_rhp_count=certificate["open_rhp_count"],
                          evidence_grade=certificate["evidence_grade"], certificate=certificate,
                          reason="Interval Wronskian winding and an exterior resolvent bound cover the open right half-plane.")
            return result

    for witness in native.get("roots", []):
        if witness["status"] != "unstable" or witness.get("winding_fine") != 1 or witness.get("winding_coarse") != 1:
            continue
        value = complex(*witness["value"])
        if min(value.real, abs(value.imag)) <= max(1e-5, 10 * witness["error"]):
            continue
        try:
            checks = verify_root(value, **settings)
        except ArithmeticError as error:
            result["reason"] = str(error)
            continue
        fine = complex(*checks[-1]["root"])
        variation = max(abs(fine - complex(*entry["root"])) for entry in checks[:-1])
        error = max(witness["error"], abs(fine - value), variation)
        result["independent_shooting"] = dict(refinements=checks, native_root=witness["value"],
                                              native_difference=abs(fine - value), observed_error_scale=error)
        if (native["label"] == "UNSTABLE" and abs(fine - value) < 1e-3
                and min(fine.real, abs(fine.imag)) > max(1e-5, 10 * error)):
            result.update(label="UNSTABLE", evidence_grade="CONVERGED_NUMERICAL",
                          reason="Native time-delay solver and continuous exact-profile shooting corroborate a growing oscillatory mode.")
            return result
    return result
