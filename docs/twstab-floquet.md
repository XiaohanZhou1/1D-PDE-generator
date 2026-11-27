# twstab temporal Floquet entry point

The bundled solver exposes a temporal Floquet action for the sampled F2 calibration orbit. An action takes a Fourier representation of a periodic Bloch envelope at \(t=0\) and returns the same representation at \(t=T\). The solver integrates the variational equation for each requested vector with the supplied orbit samples. [F2 sampled-orbit format](f2-action.md)

## Reproduce the nine reference actions

Run from the repository root in the Python environment installed from requirements.txt.

~~~powershell
$env:OPENBLAS_NUM_THREADS = '1'
Push-Location solver
python -m twstab floquet --orbit ../evidence/f2-action/orbit.json --request ../evidence/f2-native/request.json --out ../scratch/f2-native/result.json
Pop-Location
~~~

Choose a new output path for every run. The solver preserves existing output files and writes a response only after all requested actions finish. [Request fixture](../evidence/f2-native/request.json) · [Reference response](../evidence/f2-native/result.json)

## Request and response format

- The orbit file uses version `f2-sampled-orbit-0.1`. The solver checks the bound specification hash, time phase, periodic closure, array layout and sampled PDE residual.
- The request has schema version `twstab-temporal-floquet-request-0.1` and a nonempty array `actions`. Each action has a unique `id`, Bloch wavenumber `xi` in \([-1/2,1/2]\), integer Fourier `cutoff`, complex `vector` and optional `tolerance` for integration. The default tolerance is \(10^{-10}\).
- The vector contains \(2(2\,\texttt{cutoff}+1)\) entries, each encoded as a real–imaginary pair. Order the spatial modes from \(-\texttt{cutoff}\) to \(+\texttt{cutoff}\), with the two components adjacent within each mode. The orbit grid needs at least \(4\,\texttt{cutoff}+1\) nodes for the Fourier projection. The physical perturbation is \(e^{i\xi x}\) times this periodic envelope.
- The response uses version `twstab-temporal-floquet-result-0.1`. It records the input hashes, target specification hash and orbit checks, and returns each output vector with its original action ID in request order. The status `PASS` describes successful propagation. The continuum spectral certificate for F2 is documented separately. [F2 stability proof](f2.md#continuum-proof)

Python callers with `solver/` on their import path can use `twstab.temporal_floquet.apply_actions(orbit, request)` for JSON-shaped data or `SampledPeriodMap(orbit).apply(xi, cutoff, vector)` for a NumPy vector.

This local extension uses the reviewed F2 PDE, phase selector and period convention. A general orbit from another equation requires a linearisation adapter and a matching orbit validation rule. The supplied verifier checks four Bloch parameters, including zero and the zone edge, and compares the returned vectors with the dense exact-orbit monodromy at the same spatial cutoff. At zero it also checks the time-phase vector returns to itself. [CLI regression](../tests/test_twstab_floquet.py)
