"""F3 space-dependent time switching with an exact forced periodic orbit."""
from fractions import Fraction

import numpy as np
from scipy.linalg import eig, eigvals, expm

from .extension_contract import outcome, specification, validate_instance
from .f3_certificate import certify_switch
from .families import rational


def switched_orbit(gain, seed):
    gain = rational(gain)
    if not 0 < gain <= 2:
        raise ValueError("outside the reviewed F3 gain range")
    problem = dict(family_id="F3", structure_id="F3.alternating-spatial-shear", template_version="1",
                   parameters=dict(gain=str(gain)),
                   equations=["X_t=X_xx-(f_xx/f)*X+J*X-(X-Xstar)/4+gain*R(t)*S(x,t)*R(t)^T*(X-Xstar)",
                              "f=2+cos(x)", "R(t) is rotation by t, J=R'(0)",
                              "S[0,1]=1/2+cos(x) in first half-period, S[1,0]=1/2+sin(x) in second; other entries zero"],
                   target_wave=dict(profile="(2+cos(x))*(cos(t),sin(t))", phase="prescribed time phase",
                                    space_period="2*pi", time_period="2*pi"),
                   eigenproblem=dict(space="unweighted L2(R;C^2)", domain="H2 for each evolution segment",
                                     bloch_zone="[-1/2,1/2]",
                                     question="Does the full-line time monodromy have spectrum outside the unit disk?"))
    return specification(problem, seed, calibration=gain < Fraction(1, 3),
                         references=("https://doi.org/10.1070/IM1983v021n01ABEH001642",))


def validate_switch(item):
    return validate_instance(item, switched_orbit)


def period_map(bulk, first, second, gain):
    zero = np.zeros_like(bulk)
    upper = np.block([[bulk, gain*first], [zero, bulk]])
    lower = np.block([[bulk, zero], [gain*second, bulk]])
    return expm(np.pi*lower) @ expm(np.pi*upper)


def fourier_map(gain, xi, cutoff):
    modes = np.arange(-cutoff, cutoff+1)
    nodes = 8*cutoff+8
    x = 2*np.pi*np.arange(nodes)/nodes
    differences = modes[:, None]-modes

    def convolution(values):
        return (np.fft.fft(values)/nodes)[differences]

    bulk = -np.diag((modes+xi)**2+.25) + convolution(np.cos(x)/(2+np.cos(x)))
    return period_map(bulk, convolution(.5+np.cos(x)), convolution(.5+np.sin(x)), gain)


def physical_map(gain, xi, nodes):
    h = 2*np.pi/nodes
    x = h*np.arange(nodes)
    f = 2+np.cos(x)
    midpoint = 2+np.cos(x+h/2)
    forward = np.roll(np.eye(nodes), 1, axis=1)
    gradient = midpoint[:, None]*(np.exp(1j*xi*h)*forward-np.eye(nodes))/(h*f)
    bulk = -gradient.conj().T@gradient - .25*np.eye(nodes)
    return period_map(bulk, np.diag(.5+np.cos(x)), np.diag(.5+np.sin(x)), gain)


def solve_switch(item):
    gain_exact = Fraction(validate_switch(item)["parameters"]["gain"])
    gain = float(gain_exact)
    if item["calibration"]:
        result = outcome("STABLE", "RIGOROUSLY_CERTIFIED",
                         reason="A uniform energy estimate bounds every Bloch monodromy by a strict contraction.")
        result["certificate"] = dict(energy_decay_rate=str(Fraction(1, 4)-3*gain_exact/4),
                                     scope="all spatial frequencies and all Bloch fibers",
                                     proof="docs/bcf-extension.md#f3")
        return result

    fibers = (0., .0625, .125, .25, .375, .5)
    radii = [float(max(abs(eigvals(fourier_map(gain, xi, 10))))) for xi in fibers]
    xi = fibers[int(np.argmax(radii))]
    result = outcome()
    result["scan"] = dict(fibers=list(fibers), spectral_radii=radii, reflected_by_conjugation=True)
    if max(radii) <= 1.000001:
        try:
            certificate = certify_switch(gain_exact)
        except (ArithmeticError, ValueError) as error:
            result["certificate_failure"] = str(error)
            return result
        if certificate:
            result.update(outcome("STABLE", "RIGOROUSLY_CERTIFIED",
                                  reason="Certified powers of the monodromy contract across the full Bloch zone."),
                          certificate=certificate)
        else:
            result["certificate_failure"] = "The power-norm estimate did not certify the full Bloch cover."
        return result

    values, left, right = eig(fourier_map(gain, xi, 18), left=True, right=True)
    index = int(np.argmax(abs(values)))
    witness = values[index]
    condition = float(1/abs(np.vdot(left[:, index], right[:, index])))
    coarse = eigvals(fourier_map(gain, xi, 10))
    differences = [float(min(abs(coarse-witness)))]
    checks = []
    for nodes in (40, 80, 120):
        values = eigvals(physical_map(gain, xi, nodes))
        value = values[int(np.argmin(abs(values-witness)))]
        differences.append(float(abs(value-witness)))
        checks.append(dict(nodes=nodes, multiplier=[float(value.real), float(value.imag)]))
    result["witness"] = dict(xi=xi, multiplier=[float(witness.real), float(witness.imag)], condition=condition,
                             fourier_cutoffs=[10, 18], physical=checks, discrepancies=differences)
    if (condition < 100 and differences[-1] < differences[1] / 2 + 1e-8
            and abs(witness)-1 > max(1e-5, 10*max(differences))):
        result.update(outcome("UNSTABLE", "CONVERGED_NUMERICAL",
                              reason="A unit-disk-exterior multiplier is reproduced by independent spatial discretisations."))
    return result
