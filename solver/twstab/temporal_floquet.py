"""Temporal Floquet action on Bloch envelopes for the bundled F2 benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ORBIT_VERSION = "f2-sampled-orbit-0.1"
REQUEST_VERSION = "twstab-temporal-floquet-request-0.1"
RESULT_VERSION = "twstab-temporal-floquet-result-0.1"
# The sampled coefficients and phase checks below implement this exact F2 specification.
F2_SPECIFICATION_HASH = "29331e74ab259059d7844a2ce9c08f3a08dda7a08ce1ba0ef019f8c68f110b0e"
PHASE = "positive first component, zero second component at t=0"
PERIOD = 2 * np.pi
DIFFUSION = np.diag([1, 2])
ROTATION = np.array([[0, -1], [1, 0]])


def spatial_data(nodes):
    x = PERIOD * np.arange(nodes) / nodes
    amplitude = 2 + np.cos(x)
    return amplitude, -np.cos(x) / amplitude


def nonlinear_rhs(state, amplitude, potential):
    modes = np.fft.fftfreq(len(amplitude), d=1 / len(amplitude))
    laplace = np.fft.ifft(-modes[:, None]**2 * np.fft.fft(state, axis=0), axis=0).real
    growth = amplitude**2 - np.sum(state**2, axis=1)
    return ((laplace - potential[:, None] * state) @ DIFFUSION
            + state @ ROTATION.T + growth[:, None] * state)


class SampledPeriodMap:
    """Validate the orbit packet and propagate one Fourier envelope at a time."""

    def __init__(self, record):
        if (record["schema_version"] != ORBIT_VERSION or record["family_id"] != "F2"
                or record["specification_hash"] != F2_SPECIFICATION_HASH
                or record["periods"] != dict(space="2*pi", time="2*pi")
                or record["phase_condition"] != PHASE):
            raise ValueError("sampled orbit does not match the F2 specification")
        grid = record["grid"]
        nodes, snapshots = grid["spatial_nodes"], grid["time_samples"]
        if (type(nodes) is not int or type(snapshots) is not int or nodes < 16 or snapshots < 8
                or grid["space_origin"] != 0 or grid["time_origin"] != 0
                or grid["layout"] != "time, space, component; periodic grids omit endpoints"):
            raise ValueError("invalid F2 sampled-orbit grid")
        samples = np.asarray(record["samples"], dtype=float)
        endpoint = np.asarray(record["integrated_endpoint"], dtype=float)
        if (samples.shape != (snapshots, nodes, 2) or endpoint.shape != (nodes, 2)
                or not np.isfinite(samples).all() or not np.isfinite(endpoint).all()):
            raise ValueError("invalid F2 orbit sample array")

        self.amplitude, self.potential = spatial_data(nodes)
        self.nodes = nodes
        self.time_modes = np.fft.fftfreq(snapshots, d=1 / snapshots)
        self.coefficients = np.fft.fft(samples, axis=0) / snapshots
        phase_target = np.column_stack((self.amplitude, np.zeros_like(self.amplitude)))
        phase = float(np.max(np.abs(samples[0] - phase_target)))
        closure = float(np.max(np.abs(endpoint - samples[0])))
        self.checks = dict(phase_error_max=phase, closure_error_max=closure)
        if phase > 1e-8 or closure > 1e-8:
            raise ValueError("F2 phase or independently integrated periodic closure failed")
        residual = 0.0
        for time in PERIOD * np.arange(2 * snapshots) / (2 * snapshots):
            multiplier = np.exp(1j * time * self.time_modes)
            derivative = ((1j * self.time_modes * multiplier) @ self.coefficients.reshape(snapshots, -1))
            state = self.orbit(time)
            rhs = nonlinear_rhs(state, self.amplitude, self.potential)
            error = float(np.max(np.abs(derivative.real.reshape(nodes, 2) - rhs)))
            if not np.isfinite(error):
                raise ValueError("F2 interpolated orbit has a nonfinite PDE residual")
            residual = max(residual, error)
        self.checks["sampled_pde_residual_max"] = residual
        if residual > 1e-7:
            raise ValueError(f"F2 interpolated orbit PDE residual {residual:.3g} exceeds 1e-7")

    def orbit(self, time):
        weights = np.exp(1j * time * self.time_modes)
        return (weights @ self.coefficients.reshape(len(weights), -1)).real.reshape(self.nodes, 2)

    def apply(self, xi, cutoff, vector, tolerance=1e-10):
        """Apply M_xi to one vector ordered by Fourier mode then component."""
        if (type(cutoff) is not int or cutoff < 1 or 4 * cutoff + 1 > self.nodes
                or not np.isfinite(xi) or not -0.5 <= xi <= 0.5
                or not np.isfinite(tolerance) or tolerance <= 0):
            raise ValueError("invalid F2 Bloch, projection or integration parameter")
        modes = np.arange(-cutoff, cutoff + 1)
        vector = np.asarray(vector, dtype=complex)
        if vector.shape != (2 * len(modes),) or not np.isfinite(vector).all():
            raise ValueError("invalid Fourier envelope vector")
        laplace = -(modes + xi)**2

        def rhs(time, state):
            coefficients = state.reshape(len(modes), 2)
            padded = np.zeros((self.nodes, 2), dtype=complex)
            padded[modes] = coefficients
            envelope = np.fft.ifft(padded, axis=0) * self.nodes
            orbit = self.orbit(time)
            growth = self.amplitude**2 - np.sum(orbit**2, axis=1)
            radial = np.sum(orbit * envelope, axis=1)
            product = (-self.potential[:, None] * (envelope @ DIFFUSION)
                       + growth[:, None] * envelope - 2 * radial[:, None] * orbit)
            projected = (np.fft.fft(product, axis=0) / self.nodes)[modes]
            return (laplace[:, None] * (coefficients @ DIFFUSION)
                    + coefficients @ ROTATION.T + projected).ravel()

        integrated = solve_ivp(rhs, (0, PERIOD), vector, t_eval=[PERIOD], method="DOP853",
                               rtol=tolerance, atol=0.01 * tolerance)
        if not integrated.success:
            raise ArithmeticError(f"F2 period-map action failed: {integrated.message}")
        return integrated.y[:, -1]


def _decode_vector(value):
    if not isinstance(value, list) or any(
        not isinstance(pair, list) or len(pair) != 2
        or any(type(part) not in (int, float) for part in pair)
        for pair in value
    ):
        raise ValueError("each Fourier coefficient must be a [real, imaginary] pair")
    return np.array([complex(*pair) for pair in value])


def apply_actions(orbit, request):
    """Apply a versioned batch of Bloch perturbations to one sampled orbit."""
    if not isinstance(request, dict) or set(request) != {"schema_version", "actions"}:
        raise ValueError("expected an object with schema_version and actions")
    if request["schema_version"] != REQUEST_VERSION:
        raise ValueError("unsupported temporal Floquet request version")
    actions = request["actions"]
    if not isinstance(actions, list) or not actions:
        raise ValueError("actions must be a nonempty list")
    period_map = SampledPeriodMap(orbit)
    seen = set()
    results = []
    for action in actions:
        fields = {"id", "xi", "cutoff", "vector"}
        if not isinstance(action, dict) or not fields <= action.keys() <= fields | {"tolerance"}:
            raise ValueError("expected id, xi, cutoff, vector and optional tolerance")
        ident, xi, cutoff = action["id"], action["xi"], action["cutoff"]
        if not isinstance(ident, str) or not ident.strip() or ident in seen:
            raise ValueError("action IDs must be nonempty and unique")
        seen.add(ident)
        tolerance = action.get("tolerance", 1e-10)
        if type(xi) not in (int, float) or type(tolerance) not in (int, float):
            raise ValueError("xi and tolerance must be real numbers")
        try:
            vector = _decode_vector(action["vector"])
            output = period_map.apply(xi, cutoff, vector, tolerance=tolerance)
        except (ValueError, ArithmeticError) as error:
            raise ValueError(f"action {ident!r}: {error}") from error
        results.append(dict(id=ident, xi=xi, cutoff=cutoff, tolerance=tolerance,
                            vector=[[float(value.real), float(value.imag)] for value in output]))
    return dict(schema_version=RESULT_VERSION, family_id="F2", operation="apply_monodromy",
                status="PASS", specification_hash=orbit["specification_hash"],
                orbit_checks=period_map.checks, actions=results)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="twstab floquet", description=__doc__)
    parser.add_argument("--orbit", type=Path, required=True, help="phase-selected F2 orbit JSON")
    parser.add_argument("--request", type=Path, required=True, help="Bloch action request JSON")
    parser.add_argument("--out", type=Path, required=True, help="new output JSON file")
    args = parser.parse_args(argv)
    try:
        orbit = args.orbit.read_bytes()
        request = args.request.read_bytes()
        result = apply_actions(json.loads(orbit), json.loads(request))
        result.update(orbit_sha256=hashlib.sha256(orbit).hexdigest(),
                      request_sha256=hashlib.sha256(request).hexdigest())
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    except (OSError, KeyError, TypeError, ValueError, ArithmeticError) as error:
        parser.error(str(error))
    print(f"{len(result['actions'])} temporal Floquet actions -> {args.out}")


if __name__ == "__main__":
    main()
