# Unified A1, D2 and E3 generator batch

The batch entry point constructs canonical PDE instances, deduplicates them by mathematical specification hash, runs the existing family-specific evidence backends and exports one standalone prompt for every reliable label. Prompts use the same question and blind protocol. The first run repeats reviewed witness points and calibration controls as a pipeline check. Its yield is conditional on this mix.

## Run

Use the dependencies in requirements.txt and a fresh output directory. For A1, provide the original compatible spectral solver release with run.py and benchmark.json. The A1 adapter checks the release identity. D2 uses the bundled solver and independent exact-profile shooting. E3 uses interval counting, with the existing validated label gate applied to any fallback. This first implementation runs cases serially because the legacy adapter updates its import context and launches a subprocess.

~~~powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m pdegen.batch_workflow --legacy-solver C:/path/to/spectral_solver --out scratch/unified-batch
~~~

The default seed is 20261126. A different `--seed` changes the A1 point selection. D2 and E3 use fixed default grids. Use `--plan` with a JSON list to supply explicit family parameters. Use exact rational strings for nonintegers.

~~~json
[
  {"family": "E3", "parameters": {"gain": 5}},
  {"family": "E3", "parameters": {"gain": 8}},
  {"family": "D2", "parameters": {"sigma": 3, "tau": 1, "epsilon": "3/4"}}
]
~~~

A plan with no dispatchable A1 case runs without the external release. Parameters outside the reviewed mathematical template ranges are rejected during construction. Runtime capability holds and unresolved labels are retained in the internal manifest. The `--preset bcf` option runs the B3, C2 and F3 extension batch through these same gates. Each extension family can also appear in a mixed JSON plan. [BCF parameters and verification](bcf-extension.md)

## Outputs and review

F4 is available through `--plan` in both batch runners. Its plan rows use exact `stiffness` and `modulation` parameters. [F4 model, proof and pilot](f4-parametric-resonance.md)

- `blind/` contains numbered Markdown files. Each contains a single PDE with its parameters, target wave, space, question and generic proof instructions. Share this folder alone with reviewers.
- `internal/manifest.json` joins exact instance IDs, parameters, label evidence, audit decisions, sampling groups, duplicate relationships, prompt hashes and recorded blind attempts. Canonical inputs and evidence files are stored under `internal/`. Reliable unstable evidence includes both validated analytic labels and converged numerical witnesses already checked by the family backend. Stable labels require full-domain evidence.
- `--reviews` can point to a JSON list of observations in the format of the shipped review file. Records join by full specification hash. A single-question attempt with no hints and no numerical tools that ended without a complete proof meets the agreed blind threshold. A completed qualifying single-question proof is recorded separately. A grouped result retains its scope in the internal record.
