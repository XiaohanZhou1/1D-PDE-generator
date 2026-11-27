# F2 sampled orbit and period-map action

This test exercises exchange of a phase-selected numerical orbit and an application of the time-period map. It uses the PDE from [F2](f2.md), with its prescribed spatially varying coefficients. Starting from the known F2 phase condition, the capture step integrates the nonlinear PDE on 64 Fourier collocation nodes and writes 24 snapshots over one orbit period. The packet separately stores the independently integrated final state at \(t=T\). This is a numerical orbit exchange test for a known orbit. Numerically continuing a new orbit is a separate step.

## Input and application

The packet identifies the F2 specification by its hash. It records the space and time periods, the phase condition, grid origins, axis order, integration settings, the real samples and the endpoint. Samples have shape \((n_t,n_x,2)\). Both grids are periodic and omit their repeated endpoints. The backend uses a periodic Fourier interpolant in time, compares the independently integrated endpoint with the initial snapshot and checks the PDE residual at every snapshot and temporal midpoint. It rejects a mismatched specification, wrong phase, open orbit, inconsistent sample array or a residual above tolerance.

The Python entry point is:

~~~python
period_map = SampledPeriodMap(packet)
output = period_map.apply(xi, cutoff, vector)
~~~

The complex vector has length \(2(2\,\texttt{cutoff}+1)\), ordered by spatial Fourier mode from \(-\texttt{cutoff}\) to \(+\texttt{cutoff}\), with the two components adjacent within each mode. It represents the periodic Bloch envelope. The exponent convention is \(e^{i\xi x}\). The output has the same layout and represents the envelope at \(t=T\). The implementation integrates the linearised equation for that vector using FFT projection and the time-interpolated snapshots in the full reaction Jacobian. This path never constructs a dense monodromy matrix. Its coefficients come from the numerical snapshots and the F2 PDE.

## Reproduce

~~~powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m pdegen.floquet_action capture --out scratch/f2-action/orbit.json
python -m pdegen.floquet_action verify --orbit scratch/f2-action/orbit.json --out scratch/f2-action/verification.json
python -m unittest discover -s tests -v
~~~

The verification compares nine actions with the dense Fourier monodromy for the exact orbit at the same spatial cutoff. It tests the zero fiber, a positive and a negative Bloch parameter, and the zone edge. Probes include a low mode, mixed complex modes and the time-phase vector. It compares two temporal tolerances and requires the time-phase vector to return to itself. The test scope is exchange and operator action. The full-line stability certificate is the analytic F2 result. [Orbit packet](../evidence/f2-action/orbit.json) · [Verification](../evidence/f2-action/verification.json)

The bundled Jim solver does not expose this operation. The packet, Fourier vector convention and reference verifier document a concrete interface to test against a Jim backend when it is available.
