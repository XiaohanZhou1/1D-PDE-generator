"""Exact C–F calibration specifications and continuum spectral checks."""
import numpy as np
from scipy.integrate import solve_ivp

from .contract import ROOT, digest, schema_check


CALIBRATION_FAMILIES = ("C1", "D1", "E1", "F1")
PROOF_REPORT = "docs/cf-calibration.md"


def specification(family, case, seed):
    if (family, case) not in {
        ("C1", "soliton"), ("D1", "small_gain"), ("D1", "positive_root"),
        ("E1", "advancing"), ("E1", "standing"), ("F1", "rotation"),
    }:
        raise ValueError("unknown C–F calibration case")
    if family == "C1":
        structure = "C1.modified-kdv"
        parameters = dict(case=case, p=2, speed="1")
        equations = ["u_t+u^2*u_x+u_xxx=0"]
        target = dict(profile="sqrt(6)*sech(z)", frame="z=x-t", endstates=[0, 0])
        operator = dict(form="partial_z*(-partial_z^2+1-U^2)", space="L2(R;C)", domain="H3(R;C)")
        references = ["https://doi.org/10.1098/rsta.1992.0055"]
    elif family == "D1":
        structure = "D1.uniform-delay-kink"
        parameters = dict(case=case, sigma="1/2" if case == "small_gain" else "6", tau="2")
        equations = ["u_t=u_xx+u-u^3", "v_t=v_xx-v+sigma*(1-u^2)*integral(v(x,t-s),s=0..tau)/tau"]
        target = dict(profile="(tanh(x/sqrt(2)),0)", speed=0, endstates=[[-1, 0], [1, 0]])
        operator = dict(form="Allen-Cahn L2 block plus uniform-delay history generator",
                        space="L2(R;C) x C([-tau,0];L2(R;C))",
                        characteristic="lambda-partial_x^2+1-sigma*sech(x/sqrt(2))^2*H(lambda)",
                        delay_transform="H(lambda)=integral(exp(-lambda*s),s=0..tau)/tau")
        references = []
    elif family == "E1":
        structure = "E1.linear-stefan"
        moving = case == "advancing"
        parameters = dict(case=case, kappa="-3", speed="3/2" if moving else "0",
                          decay="1/2" if moving else "1")
        equations = ["u_t=u_xx-u for x<s(t)", "u(s(t),t)=1", "s'(t)=kappa*(u_x(s(t),t)-1)"]
        target = dict(profile="exp(m*z)", frame="z=x-s(t)", interface="s(t)=c*t", far_field=0)
        operator = dict(form="p_zz+c*p_z-p+kappa*U_z*p_z(0)",
                        space="L2((-inf,0);C) on the boundary-attached domain",
                        domain="H2((-inf,0);C) with p(0)=0", displacement="s_pert'=kappa*p_z(0)")
        references = []
    else:
        structure = "F1.rotating-reaction-diffusion"
        parameters = dict(case=case, diffusivities=[1, 2], period="2*pi")
        equations = ["X_t=diag(1,2)*X_xx+J*X+(1-|X|^2)*X", "J=[[0,-1],[1,0]]"]
        target = dict(profile="(cos(t),sin(t))", spatial_domain="R", phase="X(0)=(1,0)", period="2*pi")
        operator = dict(form="temporal monodromy on L2(R;C^2)",
                        fiber="Y_t=(J-2*X(t)*X(t)^T-k^2*diag(1,2))*Y for k in R")
        references = []
    problem = dict(family_id=family, structure_id=structure, template_version="1", parameters=parameters,
                   equations=equations, target_wave=target, eigenproblem=operator,
                   stability_question="Exclude Re(lambda)>0 in the recorded space, or exhibit unstable spectrum."
                   if family != "F1" else "Exclude |multiplier|>1 for the full-line temporal monodromy.")
    key = digest(problem)
    return dict(**problem, schema_version="cf-calibration-0.1", seed=seed,
                instance_id=f"{family}-{key[:20]}", specification_hash=key, calibration=True,
                validity_checks=dict(contract_target="PASS", solver_dispatch="PASS", proof_difficulty="NOT_APPLICABLE"),
                novelty_assessment=dict(status="NOT_NOVEL", grouping=structure, training_admission="HOLD"),
                references=references)


def calibration_batch(family, seed):
    cases = {"C1": ("soliton",), "D1": ("small_gain", "positive_root"),
             "E1": ("advancing", "standing"), "F1": ("rotation",)}
    return [specification(family, case, seed) for case in cases[family]]


def validate_calibration(item):
    schema_check("cf-calibration-instance", item)
    if item != specification(item["family_id"], item["parameters"]["case"], item["seed"]):
        raise ValueError("C–F record differs from the reviewed calibration specification")
    return item


def floquet_regression():
    """Compare numerical fundamental matrices against the analytic fiber bound."""
    period = 2 * np.pi
    diffusion = np.diag([1, 2])
    rotation = np.array([[0, -1], [1, 0]])
    checks = []
    for k in (0.0, 0.1, 0.5, 1.0, 2.0):
        def rhs(t, state):
            orbit = np.array([np.cos(t), np.sin(t)])
            matrix = rotation - 2 * np.outer(orbit, orbit) - k**2 * diffusion
            return (matrix @ state.reshape((2, 2))).ravel()

        estimates = []
        for tolerance in (1e-9, 1e-11):
            result = solve_ivp(rhs, (0, period), np.eye(2).ravel(), method="DOP853",
                               rtol=tolerance, atol=tolerance * 0.01)
            if not result.success:
                raise ArithmeticError("F1 numerical monodromy integration failed")
            estimates.append(result.y[:, -1].reshape((2, 2)))
        coarse, fine = estimates
        bound = float(np.exp(-period * k**2))
        checks.append(dict(k=k, bound=bound, matrix=fine.tolist(),
                           norm=float(np.linalg.norm(fine, ord=2)),
                           multiplier_moduli=sorted(np.abs(np.linalg.eigvals(fine)).tolist(), reverse=True),
                           refinement_difference=float(np.linalg.norm(fine - coarse, ord=2))))
    zero, *other = checks
    zero_error = float(np.linalg.norm(np.array(zero["matrix"]) - np.diag([np.exp(-2 * period), 1]), ord=2))
    passed = (zero_error < 1e-9 and all(row["norm"] <= row["bound"] + 1e-9 for row in other)
              and all(row["refinement_difference"] < 1e-8 for row in checks))
    return dict(status="PASS" if passed else "UNVERIFIED", zero_fiber_error=zero_error, samples=checks)


def certify_calibration(item):
    """Only dispatch the proofs for the exact canonical, reviewed specifications."""
    validate_calibration(item)
    family, case = item["family_id"], item["parameters"]["case"]
    label, count = "STABLE", 0
    if family == "C1":
        method = "Hamiltonian constrained coercivity and compact perturbation"
        reason = "Constraining to U-orthogonal perturbations excludes open-right-half-plane eigenvalues."
        coverage = "All Re(lambda)>0 in unweighted L2; essential spectrum lies on the imaginary axis."
        anchor = "c1--modified-kdv"
        details = dict(negative_directions=1, momentum_slope_at_c1="3",
                       neutral="translation kernel and speed generalized mode")
    elif family == "D1":
        anchor = "d1--uniform-distributed-delay"
        if case == "small_gain":
            method = "History resolvent and uniform small-gain bound"
            reason = "The delay perturbation of the resolvent has norm at most 1/2 in the closed right half-plane."
            coverage = "All Re(lambda)>0 on the history space and the Allen–Cahn L2 block."
            details = dict(maximum_resolvent_perturbation_norm="1/2", neutral="kink translation")
        else:
            label, count = "UNSTABLE", None
            method = "Selfadjoint test function and continuity on the real characteristic axis"
            reason = "The top of the characteristic operator crosses zero on (0,5) above its essential spectrum."
            coverage = "At least one positive real eigenvalue; total unstable count unproved."
            details = dict(positive_root_interval=[0, 5], neutral="kink translation")
    elif family == "E1":
        method = "Exact boundary determinant with Dirichlet resolvent"
        anchor = "e1--moving-boundary"
        coverage = "All Re(lambda)>0 on the half-line with the specified interface condition."
        if case == "advancing":
            reason = "The only candidate spatial root is -1/2 and fails left-decay admissibility."
            details = dict(candidate_spatial_root="-1/2", neutral="constant interface displacement")
        else:
            label, count = "UNSTABLE", 1
            reason = "The admissible spatial root 2 gives one simple eigenvalue lambda=3."
            details = dict(candidate_spatial_root="2", positive_eigenvalue="3",
                           neutral="constant interface displacement")
    else:
        method = "Uniform contractivity of all temporal monodromy fibers"
        reason = "The Hermitian part yields a monodromy norm bound exp(-2*pi*k^2) for every real k."
        coverage = "All real spatial frequencies and all |multiplier|>1 on full-line L2."
        anchor = "f1--temporal-floquet"
        count = None
        details = dict(neutral="time phase is a fiber multiplier at k=0",
                       multiplier_norm_bound="exp(-2*pi*k^2)", zero_fiber_multipliers=["1", "exp(-4*pi)"])

    result = dict(label=label, open_rhp_count=count, evidence_grade="RIGOROUSLY_CERTIFIED",
                  reason=reason, certificate=dict(method=method, coverage=coverage,
                                                  proof=f"{PROOF_REPORT}#{anchor}", **details))
    if family == "F1":
        try:
            result["numerical_check"] = floquet_regression()
        except ArithmeticError as error:
            result["numerical_check"] = dict(status="UNVERIFIED", reason=str(error))
    return result


def native_reference(item, analyse):
    """Run the bundled solver only on the identical reference specification."""
    family, case = item["family_id"], item["parameters"]["case"]
    if family == "F1":
        return dict(status="UNSUPPORTED", reason="The bundled solver has no temporal-monodromy interface.")
    files = {"C1": "gkdv_soliton.py", "D1": "delay_kink_uniform.py",
             "E1": "stefan_linear_moving.py" if case == "advancing" else "stefan_linear.py"}
    parameters = {"sigma": 0.5} if family == "D1" and case == "small_gain" else {}
    record = analyse(ROOT / "solver" / "problems" / files[family], parameters=parameters)
    # Keep the evidence portable across working directories.
    record["file"] = f"solver/problems/{files[family]}"
    return dict(status="RUN", record=record)
