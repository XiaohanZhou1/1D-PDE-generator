"""B2 numerical periodic rolls of the cubic Swift–Hohenberg equation."""
from fractions import Fraction

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import root

from .contract import digest, schema_check, unique
from .families import rational


def periodic_roll(q, amplitude, seed):
    q, amplitude = rational(q), rational(amplitude)
    if not Fraction(4, 5) <= q <= Fraction(6, 5) or not Fraction(1, 20) <= amplitude <= Fraction(2, 5):
        raise ValueError("B2 requires 4/5 <= q <= 6/5 and 1/20 <= amplitude <= 2/5")
    problem = dict(
        family_id="B2", structure_id="B2.swift-hohenberg", template_version="1",
        parameters=dict(q=str(q), first_cosine_amplitude=str(amplitude)),
        equations=dict(form="u_t=r*u-(1+partial_x^2)^2*u-u^3", variables=["u"],
                       r="branch value selected by q and first_cosine_amplitude"),
        target_wave=dict(form="stationary_roll", speed=0, period="2*pi/q",
                         phase="even in theta=q*x with positive first cosine coefficient",
                         symmetry="U(theta+pi)=-U(theta)",
                         branch="primary branch from (r=(1-q^2)^2,U=0) at fixed q"),
        eigenproblem=dict(form="selfadjoint_fourth_order", potential="-3*U^2",
                          function_space="unweighted_L2_R_C", operator_domain="H4_R_C",
                          bloch_domain="H4_per_0_L_C", bloch_wavenumber="[-q/2,q/2]",
                          fiber="r-(1+(partial_x+i*xi)^2)^2-3*U^2"),
        stability_question="Compare the co-periodic spectrum and the full-line Bloch spectrum.",
        neutral_mode_policy="Identify U_x at xi=0; retain uncertainty without a full exclusion proof.")
    key = digest(problem)
    return dict(**problem, schema_version="periodic-0.1", seed=seed,
                instance_id=f"B2-{key[:20]}", specification_hash=key, calibration=True,
                validity_checks=dict(profile="UNVERIFIED", period="UNVERIFIED", phase="UNVERIFIED",
                                     linearisation="PASS", phase_mode="UNVERIFIED",
                                     solver_dispatch="PASS", proof_difficulty="NOT_APPLICABLE"),
                novelty_assessment=dict(status="NOT_NOVEL", grouping="B2.swift-hohenberg",
                                        training_admission="HOLD"),
                references=["https://doi.org/10.1007/s002200050230"])


def validate_periodic_roll(item):
    schema_check("periodic-instance", item)
    parameters = item["parameters"]
    if item != periodic_roll(parameters["q"], parameters["first_cosine_amplitude"], item["seed"]):
        raise ValueError("B2 record differs from the canonical problem or specification hash")
    return item


def periodic_batch(seed, values=None):
    if values is None:
        values = (("1", "1/5"), ("11/10", "1/5"), ("11/10", "2/5"))
    return unique([periodic_roll(q, amplitude, seed) for q, amplitude in values])


def continue_profile(q, amplitude, highest_mode, step_count):
    modes = np.arange(1, highest_mode + 1, 2)
    theta = np.arange(8 * highest_mode) * (2 * np.pi / (8 * highest_mode))
    basis = np.cos(modes[:, None] * theta)
    projection = 2 * basis / theta.size
    dispersion = (1 - (q * modes)**2)**2
    state = np.zeros(len(modes))
    steps = []
    for fraction in np.arange(1, step_count + 1) / step_count:
        first = amplitude * fraction
        if not steps:
            state[0] = dispersion[0] + 0.75 * first**2

        def evaluate(candidate):
            r = candidate[0]
            coefficients = np.r_[first, candidate[1:]]
            profile = coefficients @ basis
            residual = (r - dispersion) * coefficients - projection @ profile**3
            jacobian = np.diag(r - dispersion) - (3 * projection * profile**2) @ basis.T
            jacobian[:, 0] = coefficients
            return residual, jacobian

        solution = root(evaluate, state, jac=True, tol=1e-11)
        state = solution.x
        error = float(np.max(np.abs(evaluate(state)[0])))
        if not np.isfinite(error) or error > 1e-11:
            raise ArithmeticError(f"B2 continuation failed at first cosine amplitude {first:.6g}")
        steps.append(dict(amplitude=float(first), r=float(state[0]), projected_residual=error))

    coefficients = np.r_[amplitude, state[1:]]
    # This shifted grid resolves all harmonics through the third power.
    theta = (np.arange(1024) + 0.5) * (2 * np.pi / 1024)
    basis = np.cos(modes[:, None] * theta)
    profile = coefficients @ basis
    residual = ((state[0] - dispersion) * coefficients) @ basis - profile**3
    derivative = -q * modes * coefficients
    sine = np.sin(modes[:, None] * theta)
    phase_residual = ((state[0] - dispersion) * derivative) @ sine - 3 * profile**2 * (derivative @ sine)
    return dict(modes=modes.tolist(), cosine_coefficients=coefficients.tolist(), r=float(state[0]),
                steps=steps, residual_max=float(np.max(np.abs(residual))),
                phase_residual_max=float(np.max(np.abs(phase_residual))))


def fourier_fiber(profile, q, xi, cutoff):
    coefficients = np.zeros(2 * profile["modes"][-1] + 1)
    center = profile["modes"][-1]
    modes = np.array(profile["modes"])
    coefficients[center + modes] = coefficients[center - modes] = np.array(profile["cosine_coefficients"]) / 2
    square = np.convolve(coefficients, coefficients)
    indices = np.arange(-cutoff, cutoff + 1)
    difference = indices[:, None] - indices
    potential = np.pad(square, max(0, 2 * cutoff - 2 * center))
    diagonal = profile["r"] - (1 - (q * indices + xi)**2)**2
    matrix = np.diag(diagonal) - 3 * potential[potential.size // 2 + difference]
    return eigh(matrix, subset_by_index=(2 * cutoff - 2, 2 * cutoff), eigvals_only=True)[::-1]


def physical_fiber(profile, q, xi, nodes):
    """Centered twisted physical stencil for an independent eigenvalue check."""
    theta = 2 * np.pi * np.arange(nodes) / nodes
    u = np.array(profile["cosine_coefficients"]) @ np.cos(np.outer(profile["modes"], theta))
    h = 2 * np.pi / (q * nodes)
    shift = np.roll(np.eye(nodes), -1, axis=1)
    forward = np.exp(1j * xi * h)
    laplace = (forward * shift + forward.conjugate() * shift.T - 2 * np.eye(nodes)) / h**2
    symmetric = np.eye(nodes) + laplace
    matrix = np.diag(profile["r"] - 3 * u**2) - symmetric @ symmetric
    return float(eigh(matrix, subset_by_index=(nodes - 1, nodes - 1), eigvals_only=True)[0])


def solve_periodic_roll(item):
    parameters = validate_periodic_roll(item)["parameters"]
    q = float(Fraction(parameters["q"]))
    amplitude = float(Fraction(parameters["first_cosine_amplitude"]))
    coarse = continue_profile(q, amplitude, 11, 8)
    fine = continue_profile(q, amplitude, 19, 16)
    delta = np.array(fine["cosine_coefficients"])
    delta[:len(coarse["modes"])] -= coarse["cosine_coefficients"]
    profile_difference = float(np.sum(abs(delta)))
    r_difference = abs(fine["r"] - coarse["r"])
    profile_ok = (max(coarse["residual_max"], fine["residual_max"]) < 1e-9
                  and max(coarse["phase_residual_max"], fine["phase_residual_max"]) < 1e-8
                  and max(profile_difference, r_difference) < 1e-9)
    evidence = dict(label="UNCERTAIN", open_rhp_count=None, evidence_grade=None,
                    reason="Periodic profile or translation residual failed the refinement checks.",
                    profile=dict(coarse=coarse, fine=fine, coefficient_l1_difference=profile_difference,
                                 r_difference=r_difference, checks_passed=profile_ok))
    if not profile_ok:
        return evidence

    relative_xi = np.linspace(-0.5, 0.5, 129)
    fibers = q * relative_xi
    # Include finer samples near the translation mode for the curvature diagnostic.
    near_zero = q * np.array([1 / 512, 1 / 1024, 1 / 2048])
    coarse_scan = np.array([fourier_fiber(coarse, q, xi, 12) for xi in fibers[::2]])
    fine_scan = np.array([fourier_fiber(fine, q, xi, 20) for xi in fibers])
    change = float(np.max(abs(coarse_scan - fine_scan[::2])))
    zero = len(fibers) // 2
    origin = fine_scan[zero]
    simple_phase = abs(origin[0]) < 1e-8 and origin[1] < -1e-6
    curvature = None
    if simple_phase:
        curvature = [float((fourier_fiber(fine, q, xi, 20)[0] - origin[0]) / xi**2)
                     for xi in near_zero]
    # The signed grid is symmetric. Restrict tie-breaking to positive xi.
    positive_peak = zero + 1 + int(np.argmax(fine_scan[zero + 1:, 0]))
    # A same-period instability can also supply the positive witness.
    peak = zero if origin[0] > fine_scan[positive_peak, 0] else positive_peak
    xi = float(fibers[peak])
    growth = float(fine_scan[peak, 0])
    coarse_growth = float(fourier_fiber(coarse, q, xi, 12)[0])
    evidence.update(reason="The scan does not certify exclusion between Bloch samples or near the phase mode.",
                    bloch=dict(fine_fibers=fibers.tolist(), fine_top_three=fine_scan.tolist(),
                               coarse_fibers=fibers[::2].tolist(), coarse_top_three=coarse_scan.tolist(),
                               fourier_cutoffs=[12, 20], shared_grid_max_difference=change,
                               coperiodic_top_three=origin.tolist(), simple_phase_observed=bool(simple_phase),
                               near_zero_fibers=near_zero.tolist(), phase_band_growth_over_xi_squared=curvature,
                               continuous_zone_excluded=False, peak_xi=xi, peak_growth=growth,
                               coarse_peak_growth=coarse_growth))
    if min(growth, coarse_growth) > max(1e-6, 10 * abs(growth - coarse_growth), 10 * change):
        fd = [physical_fiber(fine, q, xi, n) for n in (64, 128, 256)]
        errors = [abs(value - growth) for value in fd]
        confirmed = min(fd) > 0 and errors[-1] < min(growth / 10, 0.4 * errors[0] + 1e-8)
        evidence["bloch"]["physical_check"] = dict(nodes=[64, 128, 256], top_eigenvalues=fd,
                                                  absolute_errors=errors, confirmed=bool(confirmed))
        if confirmed:
            evidence.update(label="UNSTABLE", evidence_grade="CONVERGED_NUMERICAL",
                            reason="A positive Bloch eigenvalue converged in Fourier and physical discretisations.")
        else:
            evidence["reason"] = "Independent physical discretisation did not confirm the positive witness."
    return evidence
