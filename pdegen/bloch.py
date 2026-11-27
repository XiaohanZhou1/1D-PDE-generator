"""B1 Eckhaus calibration for periodic rolls of real Ginzburg–Landau."""
from fractions import Fraction
from math import sqrt

from .contract import digest, schema_check, unique
from .families import rational


def roll(q2, seed):
    q2 = rational(q2)
    if not 0 < q2 < 1:
        raise ValueError("B1 requires 0 < q² < 1 for a nonconstant roll")
    # With a=1-q² the residual is (-q²+1-a)U and the gauge
    # mode satisfies the differentiated profile equation.
    a = 1-q2
    problem = dict(
        family_id="B1",structure_id="B1.real-ginzburg-landau",template_version="1",
        parameters=dict(q_squared=str(q2)),
        equations=dict(form="X_t=X_xx+X-(X^T X)X",variables=["u","v"],diffusion="identity_2"),
        target_wave=dict(form="rotating_roll",amplitude_squared=str(a),phase="U(0)=(sqrt(a),0)",
                         speed=0,period="2*pi/sqrt(q_squared)",profile="sqrt(a)*(cos(q*x),sin(q*x))"),
        eigenproblem=dict(form="identity_second_order_plus_periodic_potential",
                          potential="q_squared*I-2*U*U^T",
                          function_space="unweighted_L2_R_C2",operator_domain="H2_R_C2",
                          bloch_domain="H2_per_0_L_C2",bloch_wavenumber="[-q/2,q/2)",
                          gauge="periodic_rotation_R(q*x)",sideband_shift="k=m*q+xi",bands=2),
        stability_question="Compare co-periodic and full-line spectral stability.",
        neutral_mode_policy="Record the phase mode and identify the quartic Eckhaus edge.")
    key = digest(problem)
    return dict(**problem,schema_version="bloch-0.1",seed=seed,instance_id=f"B1-{key[:20]}",
                specification_hash=key,calibration=True,
                validity_checks=dict(profile="PASS",period="PASS",phase="PASS",diffusion="PASS",
                                     linearisation="PASS",phase_mode="PASS",solver_dispatch="PASS",
                                     proof_difficulty="NOT_APPLICABLE"),
                novelty_assessment=dict(status="NOT_NOVEL",grouping="B1.eckhaus",
                                        training_admission="HOLD"),
                references=["https://doi.org/10.1137/18M1174751"])


def validate_roll(item):
    schema_check("bloch-instance",item)
    if item != roll(item["parameters"]["q_squared"],item["seed"]):
        raise ValueError("B1 record differs from the canonical problem or specification hash")
    return item


def bloch_batch(seed, values=None):
    if values is None:
        values = ("1/4","9/25","1/3")
    return unique([roll(q2,seed) for q2 in values])


def exact_labels(q2):
    """Sign of the upper band for k != 0 equals sign(6q²-2-k²)."""
    whole_line = ("UNSTABLE" if q2>Fraction(1,3) else
                  "MARGINAL" if q2==Fraction(1,3) else "STABLE")
    coperiodic = ("UNSTABLE" if q2>Fraction(2,5) else
                  "MARGINAL" if q2==Fraction(2,5) else "STABLE")
    return whole_line,coperiodic


def bands(q2, k, np):
    a = 1-q2
    k2 = k*k
    # Rationalized upper branch avoids cancellation near zero.
    radical = np.sqrt(a*a+4*q2*k2)
    return k2*(6*q2-2-k2)/(radical+a+k2),-k2-a-radical


def physical_fd(q2, xi, n, np):
    """Hermitian Bloch matrix in physical components, independent of the rotating gauge."""
    from scipy.linalg import eigh
    q = sqrt(q2)
    theta = 2*np.pi*np.arange(n)/n
    r = np.column_stack((np.cos(theta),np.sin(theta)))
    h = 2*np.pi/(q*n)
    shift = np.roll(np.eye(n),1,axis=1)
    forward = np.exp(1j*xi*h)
    laplace = (forward*shift+forward.conjugate()*shift.T-2*np.eye(n))/(h*h)
    matrix = np.kron(laplace,np.eye(2))
    potentials = q2*np.eye(2)-2*(1-q2)*r[:,:,None]*r[:,None,:]
    indices = 2*np.arange(n)[:,None,None]+np.arange(2)
    matrix[indices.swapaxes(1,2),indices] += potentials
    return eigh(matrix,subset_by_index=(2*n-3,2*n-1),eigvals_only=True)[::-1]


def solve_roll(item):
    import numpy as np
    exact = Fraction(item["parameters"]["q_squared"])
    q2 = float(exact)
    q = sqrt(q2)
    whole,coperiodic = exact_labels(exact)
    cutoff = max(0.0,float(6*exact-2))
    xi = min(q/3,sqrt(cutoff)/2) if whole=="UNSTABLE" else q/3
    checks = []
    passed = True
    for fiber in (0.0,xi):
        k = np.arange(-4,5)*q+fiber
        predicted = np.sort(np.concatenate(bands(q2,k,np)))[-3:][::-1]
        observed = [physical_fd(q2,fiber,n,np) for n in (64,128,256)]
        errors = [float(np.max(abs(row-predicted))) for row in observed]
        # Check convergence of sampled fibers. The algebraic proof covers all k.
        ok = errors[-1] < 2e-4 and errors[-1] <= .4*errors[0]+1e-10
        if fiber != 0:
            ok = ok and 10*errors[-1] < abs(predicted[0])
        passed &= ok
        checks.append(dict(xi=fiber,nodes=[64,128,256],exact_top_three=predicted.tolist(),
                           finite_difference_top_three=[row.tolist() for row in observed],
                           max_absolute_error=errors,status="PASS" if ok else "UNVERIFIED"))
    return dict(label=whole if passed else "UNCERTAIN",open_rhp_count=None,
                evidence_grade="EXACT_LINEAR_SPECTRUM" if passed else None,
                reason={"UNSTABLE":"The upper Bloch band has a positive sideband.",
                        "STABLE":"The exact dispersion relation excludes positive Bloch spectrum.",
                        "MARGINAL":"The nonpositive phase band touches zero at fourth order."}[whole]
                       if passed else "Independent sample-fiber numerical check was inconclusive.",
                analytical_label=whole,coperiodic_label=coperiodic,
                instability_cutoff_squared=str(max(Fraction(0),6*exact-2)),
                phase_diffusion_coefficient=str((1-3*exact)/(1-exact)),
                numerical_fiber_checks=checks)
