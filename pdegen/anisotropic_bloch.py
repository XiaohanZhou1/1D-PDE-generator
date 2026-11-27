"""B3 anisotropic non-self-adjoint Bloch rolls with an exact profile."""
from fractions import Fraction

import numpy as np
from scipy.linalg import eig, eigvals
from scipy.sparse import block_diag, coo_matrix, csc_matrix, diags, kron
from scipy.sparse.linalg import eigs

from .b3_certificate import certify_roll
from .extension_contract import outcome, specification, validate_instance
from .families import rational


D = np.diag([1, 2])
J = np.array([[0, -1], [1, 0]])


def anisotropic_roll(damping, skew, seed):
    damping, skew = rational(damping), rational(skew)
    if not 2 <= damping <= 8 or not 0 < skew <= 4:
        raise ValueError("outside the reviewed B3 parameter range")
    problem = dict(family_id="B3", structure_id="B3.anisotropic-skew-rotation", template_version="1",
                   parameters=dict(damping=str(damping), skew=str(skew)),
                   equations=["X_t=D*(X_xx+X)+(1-|X|^2)*(damping*I+skew*J)*X",
                              "D=diag(1,2)", "J=[[0,-1],[1,0]]"],
                   target_wave=dict(profile="(cos(x),sin(x))", period="2*pi", speed=0),
                   eigenproblem=dict(space="unweighted L2(R;C^2)", domain="H2(R;C^2)",
                                     bloch_zone="[-1/2,1/2]", question="Is there spectrum with Re(lambda)>0?"))
    return specification(problem, seed, references=("https://doi.org/10.1137/18M1174751",))


def validate_roll(item):
    return validate_instance(item, anisotropic_roll)


def potential(damping, skew, nodes):
    x = 2*np.pi*np.arange(nodes)/nodes
    u = np.column_stack((np.cos(x), np.sin(x)))
    v = u @ (damping*np.eye(2) + skew*J).T
    return -2*v[:, :, None]*u[:, None, :]


def fourier_matrix(damping, skew, xi, cutoff):
    modes = np.arange(-cutoff, cutoff+1)
    nodes = 8*cutoff+8
    coefficients = np.fft.fft(potential(damping, skew, nodes), axis=0)/nodes
    convolution = coefficients[modes[:, None]-modes].transpose(0, 2, 1, 3).reshape(2*len(modes), -1)
    return np.kron(np.diag(1-(modes+xi)**2), D) + convolution


def physical_matrix(damping, skew, xi, nodes):
    h = 2*np.pi/nodes
    positions = np.arange(nodes)
    forward = coo_matrix((np.ones(nodes), (positions, (positions+1) % nodes)), shape=(nodes, nodes)).tocsc()
    phase = np.exp(1j*xi*h)
    differential = (phase*forward + phase.conjugate()*forward.T)/h**2 + (1-2/h**2)*diags(np.ones(nodes))
    blocks = [csc_matrix(block) for block in potential(damping, skew, nodes)]
    return (kron(differential, csc_matrix(D)) + block_diag(blocks)).tocsc()


def solve_roll(item):
    params = validate_roll(item)["parameters"]
    exact_parameters = tuple(Fraction(params[key]) for key in ("damping", "skew"))
    damping, skew = map(float, exact_parameters)
    fibers = np.linspace(0, .5, 65)
    growth = [float(max(eigvals(fourier_matrix(damping, skew, xi, 12)).real)) for xi in fibers]
    index = int(np.argmax(growth))
    xi = float(fibers[index])
    result = outcome()
    result["scan"] = dict(fibers=fibers.tolist(), growth=growth, coperiodic_growth=growth[0],
                          reflected_by_conjugation=True)
    if growth[index] <= 1e-6:
        try:
            certificate = certify_roll(*exact_parameters)
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
            result["certificate_failure"] = str(error)
            return result
        if certificate:
            result.update(outcome("STABLE", "RIGOROUSLY_CERTIFIED", 0,
                                  reason="An exact-checked Fourier energy inequality covers every Bloch fiber."),
                          certificate=certificate)
        else:
            result["certificate_failure"] = "No verified positive metric on the complete Bloch interval."
        return result
    matrix = fourier_matrix(damping, skew, xi, 20)
    values, left, right = eig(matrix, left=True, right=True)
    selected = int(np.argmax(values.real))
    witness = values[selected]
    conditioning = float(1/abs(np.vdot(left[:, selected], right[:, selected])))
    coarse = eigvals(fourier_matrix(damping, skew, xi, 12))
    differences = [float(min(abs(coarse-witness)))]
    checks = []
    for nodes in (96, 192, 384):
        operator = physical_matrix(damping, skew, xi, nodes)
        values, vectors = eigs(operator, k=1, sigma=witness+1e-6, v0=np.ones(operator.shape[0]))
        value, vector = values[0], vectors[:, 0]
        residual = float(np.linalg.norm(operator@vector-value*vector))
        differences.append(float(abs(value-witness)))
        checks.append(dict(nodes=nodes, root=[float(value.real), float(value.imag)], residual=residual))
    result["witness"] = dict(xi=xi, root=[float(witness.real), float(witness.imag)], condition=conditioning,
                             fourier_cutoffs=[12, 20], physical=checks, discrepancies=differences)
    if (conditioning < 100 and max(row["residual"] for row in checks) < 1e-7
            and differences[-1] < differences[1] / 2 + 1e-8
            and witness.real > max(1e-6, 10 * max(differences))):
        result.update(outcome("UNSTABLE", "CONVERGED_NUMERICAL",
                              reason="A positive Bloch growth rate agrees under Fourier and physical discretisation."))
    return result
