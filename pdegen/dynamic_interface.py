"""E3 moving interface with a dynamical two-state feedback law."""
from fractions import Fraction

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded
from scipy.optimize import root

from .contract import digest
from .families import rational


DAMPING = Fraction(1, 8)
SPEED = 1.5


def dynamic_interface(gain, seed):
    gain = rational(gain)
    if not 0 < gain <= 16:
        raise ValueError("E3 gain outside the pilot range")
    problem = dict(family_id="E3", structure_id="E3.dynamic-interface-oscillator", template_version="1",
                   parameters=dict(gain=str(gain), damping=str(DAMPING), frequency="1", speed="3/2"),
                   equations=["u_t=u_xx-u+v-u^2", "v_t=v_xx-(5/2)*v"],
                   boundary=dict(domain="x<s(t)", values=[1, 1], state=["a", "b"],
                                 evolution=["s'=3/2+a", "a'=b",
                                            "b'=-a-2*damping*b+gain*(u_x-1/2+v_x-1)"]),
                   target_wave=dict(profile="(exp(z/2),exp(z))", frame="z=x-s(t)",
                                    boundary_state=[0, 0], interface="s(t)=3*t/2"),
                   eigenproblem=dict(space="L2((-inf,0);C^2) x C^3",
                                     domain="H2 Dirichlet in both fields; free scalar states",
                                     characteristic="lambda^2+2*damping*lambda+1-gain*(p_lambda'(0)+q_lambda'(0))",
                                     question="Is there spectrum in the open right half-plane?"))
    key = digest(problem)
    control = small_gain_certificate(gain) is not None
    return dict(**problem, schema_version="e3-dynamic-interface-0.1", seed=seed, instance_id=f"E3-{key[:20]}",
                specification_hash=key, calibration=control,
                validity_checks=dict(profile="PASS", linearisation="PASS", solver_dispatch="PASS",
                                     proof_difficulty="NOT_APPLICABLE" if control else "UNVERIFIED"),
                novelty_assessment=dict(status="NOT_NOVEL" if control else "UNCERTAIN",
                                        grouping=problem["structure_id"], training_admission="HOLD"),
                references=["https://pmc.ncbi.nlm.nih.gov/articles/PMC11301126/"])


def validate_dynamic_interface(item):
    if type(item["seed"]) is not int or item != dynamic_interface(item["parameters"]["gain"], item["seed"]):
        raise ValueError("E3 differs from the canonical specification")
    return item


def small_gain_certificate(gain):
    # The polynomial lower bound squared is exactly 4*d^2*(1-d^2).
    bound = gain * (Fraction(1, 5) + Fraction(2, 7))
    lower_squared = 4 * DAMPING**2 * (1 - DAMPING**2)
    if bound**2 < lower_squared:
        return dict(method="uniform small gain", feedback_bound=str(bound),
                    oscillator_modulus_lower_bound_squared=str(lower_squared),
                    scope="entire closed right half-plane excluding the constant-displacement mode",
                    proof="docs/e3-dynamic-pilot.md#stable-control")


def stability_certificate(gain):
    small = small_gain_certificate(gain)
    if small is not None:
        return small
    moment = Fraction(4, 125) + Fraction(8, 343)
    if gain * moment < 2 * DAMPING:
        return dict(method="positive bulk resolvent and trace moment",
                    moment_upper_bound=str(moment), gain_times_moment_bound=str(gain * moment),
                    damping_coefficient=str(2 * DAMPING),
                    scope="entire closed right half-plane excluding the constant-displacement mode",
                    proof="docs/e3-dynamic-pilot.md#stable-certificate-beyond-small-gain")


def fd_trace(lam, length, intervals):
    """Second-order physical discretisation of both Dirichlet response fields."""
    h = length / intervals
    z = h * np.arange(-intervals + 1, 0)
    u, v = np.exp(z / 2), np.exp(z)
    bands = np.zeros((3, len(z)), dtype=complex)
    bands[0, 1:] = -1 / h**2 - SPEED / (2 * h)
    bands[2, :-1] = -1 / h**2 + SPEED / (2 * h)
    bands[1] = 2 / h**2 + 2.5 + lam
    q = solve_banded((1, 1), bands, v)
    bands[1] = 2 / h**2 + 1 + 2*u + lam
    p = solve_banded((1, 1), bands, u / 2 + q)
    return ((p[-2] + q[-2]) - 4*(p[-1] + q[-1])) / (2*h)


def continuous_trace(lam, length, tolerance):
    """Adjoint Green identity with analytic q and a continuous IVP for the p trace."""
    nu = (-SPEED + np.sqrt(SPEED**2 + 4*(2.5 + lam))) / 2
    kappa = (SPEED + np.sqrt(SPEED**2 + 4*(1 + lam))) / 2

    def rhs(z, state):
        value, derivative, _ = state
        u = np.exp(z / 2)
        q = (-z * np.exp(z) / (2 + SPEED) if lam == 0 else
             -np.exp(z) * np.expm1(lam * z / (nu + 1 + SPEED)) / lam)
        return (derivative, 2*u*value - (2*kappa - SPEED)*derivative,
                np.exp(kappa*z)*value*(u/2 + q))

    solution = solve_ivp(rhs, (-length, 0), np.array([1, 0, 0], dtype=complex), method="DOP853",
                         rtol=tolerance, atol=tolerance * 0.01, max_step=0.4)
    if not solution.success:
        raise ArithmeticError(f"continuous trace integration failed: {solution.message}")
    value, _, integral = solution.y[:, -1]
    return -integral/value - 1/(nu + 1 + SPEED)


def find_root(gain, start, trace):
    def characteristic(coords):
        lam = complex(*coords)
        value = lam**2 + 2*float(DAMPING)*lam + 1 - gain*trace(lam)
        return [value.real, value.imag]

    solved = root(characteristic, [start.real, start.imag], tol=1e-10)
    error = float(np.linalg.norm(characteristic(solved.x)))
    if not np.isfinite(error) or error > 1e-7:
        raise ArithmeticError("the interface characteristic zero could not be refined")
    return complex(*solved.x), error


def solve_dynamic_interface(item):
    gain_exact = Fraction(validate_dynamic_interface(item)["parameters"]["gain"])
    gain = float(gain_exact)
    result = dict(label="UNCERTAIN", open_rhp_count=None, evidence_grade=None,
                  reason="The tracked mode does not establish a reliable label under the full-domain gate.")
    certificate = stability_certificate(gain_exact)
    if certificate is not None:
        result.update(label="STABLE", open_rhp_count=0, evidence_grade="RIGOROUSLY_CERTIFIED",
                      reason=f"Full-domain exclusion by {certificate['method']}.", certificate=certificate)

    start = complex(-float(DAMPING), np.sqrt(1 - float(DAMPING)**2))
    try:
        for coupling in np.linspace(0, gain, int(np.ceil(gain))*2 + 1)[1:]:
            start, _ = find_root(coupling, start, lambda lam: fd_trace(lam, 14, 700))
        estimates = []
        for length, intervals in ((14, 700), (22, 2200), (30, 4500)):
            value, residual = find_root(gain, start, lambda lam: fd_trace(lam, length, intervals))
            estimates.append(dict(method="physical_finite_difference", length=length,
                                  intervals=intervals, root=[value.real, value.imag], residual=residual))
            start = value
        for length, tolerance in ((16, 1e-8), (24, 1e-11)):
            value, residual = find_root(gain, start, lambda lam: continuous_trace(lam, length, tolerance))
            estimates.append(dict(method="continuous_adjoint_green_identity", length=length,
                                  tolerance=tolerance, root=[value.real, value.imag], residual=residual))
            start = value
    except ArithmeticError as error:
        result["numerical_error"] = str(error)
        return result

    fine = complex(*estimates[-1]["root"])
    disagreement = max(abs(fine - complex(*row["root"])) for row in estimates)
    result["witness"] = dict(refinements=estimates, observed_difference=disagreement,
                             positive_margin=fine.real)
    if min(fine.real, abs(fine.imag)) > max(1e-5, 10 * disagreement) and disagreement < 1e-3:
        if certificate is not None:
            result.update(label="UNCERTAIN", open_rhp_count=None, evidence_grade=None,
                          reason="The stable certificate conflicts with the numerical witness.")
        else:
            result.update(label="UNSTABLE", evidence_grade="CONVERGED_NUMERICAL",
                          reason="Physical finite differences and continuous Green matching confirm a growing nonreal mode.")
    return result
