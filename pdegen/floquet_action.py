"""Sampled-orbit exchange and matrix-free F2 period-map verification."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .contract import write_json
from solver.twstab.temporal_floquet import (ORBIT_VERSION, PERIOD, PHASE, SampledPeriodMap,
                                           nonlinear_rhs, spatial_data)

from .floquet import fourier_generator, monodromy, spatial_orbit


def capture_orbit():
    """Integrate the physical nonlinear PDE discretisation and sample its trajectory."""
    nodes, snapshots = 64, 24
    amplitude, potential = spatial_data(nodes)
    initial = np.column_stack((amplitude, np.zeros_like(amplitude)))

    def rhs(_, state):
        return nonlinear_rhs(state.reshape(nodes, 2), amplitude, potential).ravel()

    times = PERIOD * np.arange(snapshots + 1) / snapshots
    integration = dict(method="DOP853", rtol=1e-13, atol=1e-15)
    integrated = solve_ivp(rhs, (0, PERIOD), initial.ravel(), t_eval=times, **integration)
    if not integrated.success:
        raise ArithmeticError(f"sampled orbit integration failed: {integrated.message}")
    values = integrated.y.T.reshape(snapshots + 1, nodes, 2)
    return dict(schema_version=ORBIT_VERSION, family_id="F2",
                specification_hash=spatial_orbit(0)["specification_hash"],
                periods=dict(space="2*pi", time="2*pi"), phase_condition=PHASE,
                grid=dict(space_origin=0, time_origin=0, spatial_nodes=nodes, time_samples=snapshots,
                          layout="time, space, component; periodic grids omit endpoints"),
                capture=dict(spatial_method="Fourier collocation", **integration),
                samples=values[:-1].tolist(), integrated_endpoint=values[-1].tolist())


def verify_action(record):
    """Compare sampled matrix-free actions with the exact-orbit dense backend."""
    operator = SampledPeriodMap(record)
    cutoff = 6
    dimension = 2 * (2 * cutoff + 1)
    phase_grid = nonlinear_rhs(operator.orbit(0), operator.amplitude, operator.potential)
    modes = np.arange(-cutoff, cutoff + 1)
    phase = (np.fft.fft(phase_grid, axis=0)[modes] / operator.nodes).ravel()
    low = np.zeros(dimension, dtype=complex)
    low[2 * cutoff:2 * cutoff + 2] = [1, 0.5j]
    mixed = np.random.default_rng(20261124).normal(size=(2, dimension))
    probes = dict(low_mode=low, mixed_modes=mixed[0] + 1j * mixed[1])
    checks = []
    phase_error = None
    for xi in (0.0, -0.25, 0.25, 0.5):
        generator, _ = fourier_generator(xi, cutoff)
        reference = monodromy(generator, dimension, PERIOD, 1e-12)
        fiber_probes = {**probes, "time_phase": phase} if xi == 0 else probes
        for name, vector in fiber_probes.items():
            value = operator.apply(xi, cutoff, vector, tolerance=1e-11)
            coarse = operator.apply(xi, cutoff, vector, tolerance=1e-8)
            scale = np.linalg.norm(vector)
            error = float(np.linalg.norm(value - reference @ vector) / scale)
            refinement = float(np.linalg.norm(value - coarse) / scale)
            if name == "time_phase":
                phase_error = float(np.linalg.norm(value - vector) / scale)
            checks.append(dict(xi=xi, probe=name, error_per_input_norm=error,
                               refinement_difference_per_input_norm=refinement))
    passed = (phase_error < 1e-8 and all(row["error_per_input_norm"] < 1e-8
                                        and row["refinement_difference_per_input_norm"] < 1e-7 for row in checks))
    return dict(status="PASS" if passed else "UNVERIFIED", orbit_checks=operator.checks,
                cutoff=cutoff, action_checks=checks, phase_action_error=phase_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--out", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--orbit", required=True, type=Path)
    verify.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        write_json(args.out, capture_orbit())
        return
    try:
        raw = args.orbit.read_bytes()
        record = json.loads(raw)
        result = verify_action(record)
    except (OSError, KeyError, TypeError, ValueError, ArithmeticError) as error:
        parser.error(str(error))
    result.update(input_sha256=hashlib.sha256(raw).hexdigest(), specification_hash=record["specification_hash"])
    write_json(args.out, result)
    print(f"{result['status']}: {len(result['action_checks'])} period-map actions")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
