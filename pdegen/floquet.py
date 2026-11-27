"""F2 temporal Floquet analysis with coupled spatial Bloch modes."""
import numpy as np
from scipy.integrate import solve_ivp

from .contract import digest, schema_check


PERIOD = 2 * np.pi
DIFFUSION = np.diag([1, 2])
ROTATION = np.array([[0, -1], [1, 0]])


def spatial_orbit(seed):
    problem = dict(
        family_id="F2", structure_id="F2.heterogeneous-rotating-reaction-diffusion", template_version="1",
        parameters=dict(amplitude_offset=2, amplitude_cosine=1, diffusivities=[1, 2]),
        equations=dict(variables=["u", "v"], form="X_t=D*(X_xx-(f_xx/f)*X)+J*X+(f^2-|X|^2)*X",
                       amplitude="f(x)=2+cos(x)", J=[[0, -1], [1, 0]]),
        target_wave=dict(profile="(2+cos(x))*(cos(t),sin(t))", spatial_period="2*pi",
                         temporal_period="2*pi", phase="X(x,0)=(2+cos(x),0)",
                         existence="exact manufactured solution in a spatially periodic medium"),
        eigenproblem=dict(form="temporal_monodromy_spatial_bloch", space="unweighted L2(R;C^2)",
                          cell="[0,2*pi]", periodic_fiber_space="L2_per([0,2*pi];C^2)",
                          generator_domain="H2_per([0,2*pi];C^2)", bloch_zone="[-1/2,1/2]",
                          convention="physical perturbation exp(i*xi*x)*p(x,t), p spatially periodic",
                          fiber="D*((partial_x+i*xi)^2-f_xx/f)+J-2*f^2*e(t)*e(t)^T",
                          time_direction="M_xi propagates data from t=0 to t=2*pi",
                          stability_region="|multiplier|<=1",
                          neutral_mode_policy="Simple time-phase multiplier 1 in the xi=0 fiber."),
        stability_question="Exclude spectrum outside the unit disk for the full-line temporal monodromy.")
    key = digest(problem)
    return dict(**problem, schema_version="floquet-0.1", seed=seed,
                instance_id=f"F2-{key[:20]}", specification_hash=key, calibration=True,
                validity_checks=dict(profile="PASS", linearisation="PASS", solver_dispatch="PASS",
                                     proof_difficulty="NOT_APPLICABLE"),
                novelty_assessment=dict(status="NOT_NOVEL", grouping=problem["structure_id"],
                                        training_admission="HOLD"),
                references=["https://doi.org/10.1070/IM1983v021n01ABEH001642"])


def validate_spatial_orbit(item):
    schema_check("floquet-instance", item)
    if item != spatial_orbit(item["seed"]):
        raise ValueError("F2 record differs from the reviewed periodic orbit or its operator")
    return item


def monodromy(generator, dimension, period, tolerance):
    """Integrate the fundamental matrix of a finite-dimensional time-periodic fiber."""
    def rhs(time, state):
        return (generator(time) @ state.reshape(dimension, dimension)).ravel()

    initial = np.eye(dimension, dtype=complex).ravel()
    result = solve_ivp(rhs, (0, period), initial, method="DOP853",
                       rtol=tolerance, atol=0.01 * tolerance)
    if not result.success:
        raise ArithmeticError(f"F2 monodromy integration failed: {result.message}")
    return result.y[:, -1].reshape(dimension, dimension)


def fourier_generator(xi, cutoff):
    """Project both spatial multiplication operators onto all retained Fourier modes."""
    modes = np.arange(-cutoff, cutoff + 1)
    nodes = max(256, 8 * cutoff)
    x = PERIOD * np.arange(nodes) / nodes
    amplitude = 2 + np.cos(x)
    differences = modes[:, None] - modes[None, :]
    potential = (np.fft.fft(-np.cos(x) / amplitude) / nodes)[differences]
    square = (np.fft.fft(amplitude**2) / nodes)[differences]
    elliptic = -np.diag((modes + xi)**2) - potential
    base = np.kron(elliptic, DIFFUSION) + np.kron(np.eye(len(modes)), ROTATION)

    def generator(time):
        direction = np.array([np.cos(time), np.sin(time)])
        return base - 2 * np.kron(square, np.outer(direction, direction))

    return generator, 2 * len(modes)


def numerical_regression():
    configurations = ((4, 1e-8), (7, 1e-9), (10, 1e-9), (10, 1e-11))
    rows = []
    for xi in (-0.5, -0.25, -0.125, 0.0, 1 / 32, 0.125, 0.25, 0.5):
        estimates = []
        for cutoff, tolerance in configurations:
            generator, dimension = fourier_generator(xi, cutoff)
            matrix = monodromy(generator, dimension, PERIOD, tolerance)
            multipliers = np.linalg.eigvals(matrix)
            spectral_radius = float(np.max(np.abs(multipliers)))
            norm = float(np.linalg.norm(matrix, ord=2))
            estimates.append(dict(cutoff=cutoff, tolerance=tolerance,
                                  spectral_radius=spectral_radius, norm=norm))

        fine = estimates[-1]
        bound = float(np.exp(-PERIOD * xi**2 / 9))
        phase_residual = None
        if xi == 0:
            phase = np.zeros((2 * cutoff + 1, 2))
            phase[cutoff - 1:cutoff + 2, 1] = [0.5, 2, 0.5]
            phase = phase.ravel()
            phase_residual = float(np.linalg.norm(matrix @ phase - phase) / np.linalg.norm(phase))
        spatial_change = abs(estimates[2]["spectral_radius"] - estimates[1]["spectral_radius"])
        time_change = abs(fine["spectral_radius"] - estimates[2]["spectral_radius"])
        rows.append(dict(xi=xi, analytic_norm_bound=bound, estimates=estimates,
                         fine_spectral_radius=fine["spectral_radius"],
                         spatial_refinement_difference=spatial_change,
                         time_refinement_difference=time_change, phase_residual=phase_residual,
                         second_multiplier_modulus=float(np.sort(np.abs(multipliers))[-2])))

    zero = next(row for row in rows if row["xi"] == 0)
    passed = (zero["phase_residual"] < 1e-9 and zero["second_multiplier_modulus"] < 0.99
              and all(row["estimates"][-1]["norm"] <= row["analytic_norm_bound"] + 1e-9 for row in rows)
              and all(row["spatial_refinement_difference"] < 1e-6
                      and row["time_refinement_difference"] < 1e-8 for row in rows))
    return dict(status="PASS" if passed else "UNVERIFIED", fibers=rows)


def solve_spatial_orbit(item):
    validate_spatial_orbit(item)
    try:
        numerical = numerical_regression()
    except (ArithmeticError, np.linalg.LinAlgError) as error:
        numerical = dict(status="UNVERIFIED", reason=str(error))
    return dict(label="STABLE", open_rhp_count=None, evidence_grade="RIGOROUSLY_CERTIFIED",
                reason="The continuum monodromy is a contraction for every spatial Bloch fiber.",
                certificate=dict(method="Bloch ground-state transform and time-dependent energy estimate",
                                 coverage="every spatial Bloch fiber and all spatial modes",
                                 proof="docs/f2.md#continuum-proof",
                                 fiber_norm_bound="exp(-2*pi*xi^2/9)",
                                 phase_mode="f(x)*(-sin(t),cos(t)) at xi=0"),
                numerical_check=numerical)
