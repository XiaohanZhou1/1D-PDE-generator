"""C2 Hamiltonian polarization spectrum of a two-field solitary wave."""
from fractions import Fraction
from itertools import combinations
from math import ceil

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.sparse import bmat, diags
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from .extension_contract import outcome, specification, validate_instance
from .families import rational


PAIRS = list(combinations(range(4), 2))
SCOUT_SHIFTS = (.07+1.2j, .07+1.6j, .07+1.9j)
EXPANDED_SHIFTS = tuple(complex(real, imag) for imag in
                        (0, .4, .75, .95, 1.03, 1.15, 1.4, 1.75, 2.1, 2.5, 3, 4)
                        for real in (.04, .25, .8))


def polarization_wave(alpha, beta, seed):
    alpha, beta = rational(alpha), rational(beta)
    if not 0 < beta <= alpha <= 6:
        raise ValueError("outside the reviewed C2 parameter range")
    problem = dict(family_id="C2", structure_id="C2.hamiltonian-polarization", template_version="1",
                   parameters=dict(alpha=str(alpha), beta=str(beta)),
                   equations=["i*psi_t=-psi_xx+psi-|psi|^2*psi-alpha*|chi|^2*psi-beta*chi^2*conj(psi)",
                              "i*chi_t=-chi_xx+chi-alpha*|psi|^2*chi-beta*psi^2*conj(chi)"],
                   target_wave=dict(profile="(sqrt(2)*sech(x),0)", speed=0),
                   eigenproblem=dict(space="complexification of real L2(R;R^4)",
                                     domain="H2(R;C^4)", question="Is there spectrum with Re(lambda)>0?"))
    return specification(problem, seed, calibration=alpha+beta < Fraction(1, 2),
                         references=("https://www.sciencedirect.com/science/article/abs/pii/S0022039623004618",))


def validate_polarization(item):
    return validate_instance(item, polarization_wave)


def physical_root(alpha, beta, length, intervals, shift, scout=False):
    h = 2*length/intervals
    x = h*np.arange(1, intervals) - length
    decay = np.exp(-2*abs(x))
    squared = 8*decay/(1+decay)**2
    operators = [diags((-1/h**2, 1+2/h**2-depth*squared, -1/h**2), (-1, 0, 1),
                       shape=(len(x), len(x)), format="csc") for depth in (alpha-beta, alpha+beta)]
    matrix = bmat([[None, operators[0]], [-operators[1], None]], format="csc", dtype=complex)
    values, vectors = eigs(matrix, k=4, sigma=shift, v0=np.cos(np.arange(matrix.shape[0])))
    if scout:
        eligible = np.where((values.real > .001) & (values.imag > .001))[0]
        choice = (eligible[np.argmin(abs(values[eligible]-shift))] if eligible.size
                  else int(np.argmin(abs(values-shift))))
    else:
        choice = int(np.argmin(abs(values-shift)))
    value, vector = values[choice], vectors[:, choice]
    residual = float(np.linalg.norm(matrix@vector-value*vector))
    blocks = vector.reshape(2, len(x))
    parity = "even" if np.linalg.norm(blocks-blocks[:, ::-1]) < np.linalg.norm(blocks+blocks[:, ::-1]) else "odd"
    tail = float(np.linalg.norm(blocks[:, abs(x) > 0.8*length]))
    return value, dict(length=length, intervals=intervals, root=[float(value.real), float(value.imag)],
                       residual=residual, parity=parity, tail_l2=tail)


def parity_determinant(lam, alpha, beta, parity, length, tolerance):
    k1, k2 = np.sqrt(1+1j*lam), np.sqrt(1-1j*lam)
    first, second = np.array((1, 1j, -k1, -1j*k1)), np.array((1, -1j, -k2, 1j*k2))
    initial = np.array([first[i]*second[j]-first[j]*second[i] for i, j in PAIRS])

    def rhs(x, exterior):
        squared = 2/np.cosh(x)**2
        minus, plus = 1-(alpha-beta)*squared, 1-(alpha+beta)*squared
        matrix = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [plus, lam, 0, 0], [-lam, minus, 0, 0]])
        wedge = np.zeros((4, 4), dtype=complex)
        for (i, j), value in zip(PAIRS, exterior, strict=True):
            wedge[i, j], wedge[j, i] = value, -value
        derivative = matrix@wedge+wedge@matrix.T+(k1+k2)*wedge
        return np.array([derivative[i, j] for i, j in PAIRS])

    solution = solve_ivp(rhs, (length, 0), initial, method="DOP853", rtol=tolerance,
                         atol=.01*tolerance, max_step=.5)
    if not solution.success:
        raise ArithmeticError("C2 exterior shooting failed")
    return solution.y[5 if parity == "even" else 0, -1]


def exact_polarization_label(alpha, beta):
    minus, plus = alpha-beta, alpha+beta
    proof = "docs/bcf-extension.md#c2-exact-spectral-checks"
    if plus <= 1 or (minus, plus) == (1, 3):
        method = "NONNEGATIVE_HAMILTONIAN_FORM" if plus <= 1 else "SCALAR_NLS_EQUIVALENCE"
        result = outcome("STABLE", "RIGOROUSLY_CERTIFIED", 0,
                         "The Hamiltonian form excludes growth, using the scalar NLS constraint where needed.")
        result["certificate"] = dict(method=method, depths=[str(minus), str(plus)],
                                     scope="entire open right half-plane", proof=proof)
        if plus < Fraction(1, 2):
            result["certificate"]["polarization_form_lower_bound"] = str(1-2*plus)
        return result
    for parity, threshold in (("even", 1), ("odd", 3)):
        if minus < threshold < plus:
            result = outcome("UNSTABLE", "RIGOROUSLY_CERTIFIED",
                             reason="A sign change of the lowest selfadjoint pencil eigenvalue proves a positive real mode.")
            result["certificate"] = dict(method="SELFADJOINT_PENCIL_SIGN_CHANGE", parity=parity,
                                         threshold=threshold, depths=[str(minus), str(plus)],
                                         positive_pencil_parameter=str((2*beta+1)**2), proof=proof,
                                         scope="existence of a positive real eigenvalue")
            return result
    return None


def verify_oscillatory_root(alpha, beta, start, first):
    parity = first["parity"]
    continuous = []
    for length, tolerance in ((16, 1e-8), (26, 1e-11)):
        def characteristic(coords):
            value = parity_determinant(complex(*coords), alpha, beta, parity, length, tolerance)
            return value.real, value.imag

        solved = root(characteristic, [start.real, start.imag], tol=1e-10)
        residual = float(np.linalg.norm(characteristic(solved.x)))
        if not np.isfinite(residual) or residual > 1e-7:
            return False, dict(scout=first, continuous=continuous,
                               rejection="Continuous exterior shooting did not resolve the candidate.")
        start = complex(*solved.x)
        continuous.append(dict(length=length, tolerance=tolerance, root=[start.real, start.imag],
                               determinant_residual=residual))

    if min(start.real, start.imag) <= .001:
        return False, dict(scout=first, continuous=continuous,
                           rejection="The continued root does not clear the growth and oscillation floors.")

    decay = min(np.sqrt(1+1j*start).real, np.sqrt(1-1j*start).real)
    fd = [first]
    for length, step in ((max(55, min(500, ceil(8/decay))), .025),
                         (max(85, min(640, ceil(10/decay))), .0125)):
        _, evidence = physical_root(alpha, beta, length, round(2*length/step), start)
        fd.append(evidence)

    error = max(abs(start-complex(*row["root"])) for row in (fd[-2], fd[-1], continuous[0]))
    passed = (min(start.real, start.imag) > max(.001, 10*error) and fd[-1]["tail_l2"] < .04
              and fd[-1]["residual"] < 1e-7 and error < .005
              and all(row["parity"] == parity for row in fd))
    witness = dict(physical=fd, continuous=continuous, far_field_decay=float(decay),
                   discrepancy=float(error), parity=parity)
    if not passed:
        witness["rejection"] = "The candidate did not clear the discrepancy, localization, residual or parity gate."
    return passed, witness


def solve_polarization(item):
    p = validate_polarization(item)["parameters"]
    alpha, beta = (Fraction(p[key]) for key in ("alpha", "beta"))
    exact = exact_polarization_label(alpha, beta)
    if exact:
        return exact

    alpha, beta = float(alpha), float(beta)
    result = outcome(reason="No verified oscillatory witness or applicable exact spectral certificate.")
    result.update(scout=[], rejected_witnesses=[])
    checked = []
    for shifts in (SCOUT_SHIFTS, EXPANDED_SHIFTS):
        roots = []
        for shift in shifts:
            try:
                value, evidence = physical_root(alpha, beta, 36, 1440, shift, scout=True)
            except ArpackNoConvergence:
                result["scout"].append(dict(shift=[shift.real, shift.imag], status="NO_CONVERGENCE"))
                continue
            result["scout"].append(dict(shift=[shift.real, shift.imag], **evidence))
            if min(value.real, value.imag) > .001 and all(abs(value-root) > 1e-5 for root in checked):
                roots.append((value, evidence))
                checked.append(value)
        for value, evidence in sorted(roots, key=lambda pair: pair[0].real, reverse=True):
            try:
                passed, witness = verify_oscillatory_root(alpha, beta, value, evidence)
            except (ArithmeticError, ArpackNoConvergence) as error:
                passed, witness = False, dict(scout=evidence, rejection=str(error))
            if passed:
                result.update(outcome("UNSTABLE", "CONVERGED_NUMERICAL",
                                      reason="A nonreal positive root agrees in physical differences and continuous parity shooting."),
                              witness=witness)
                return result
            result["rejected_witnesses"].append(witness)
    return result
