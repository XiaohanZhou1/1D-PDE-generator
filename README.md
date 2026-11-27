# 1D PDE generator and travelling-wave solver

This repository generates reproducible 1D PDE spectral problems and checks their labels with family-specific solvers and certificates. The resumable batch supports A1, B3, C2, D2, E3, F3 and F4. The A5 travelling-wave workflow uses the bundled twstab 2.18.0 solver and its fixed-speed unfolding and continued-origin contour. [Bundled solver and provenance](solver/UPSTREAM.md)

## Run a small batch

With the dependencies installed, run the checked-in public example from the repository root:

~~~powershell
python -m pdegen.resumable_batch --plan examples/public-batch-plan.json --out scratch/public-batch --workers 2 --retries 1
~~~

Rerun the same command to resume from its checkpoint. `internal/manifest.json` records per-case labels and evidence. `blind/` contains standalone prompts for reliable labels. Supply an observation file explicitly with `--reviews` to join blind results. The public example covers B3 through F4 and uses the bundled solver for D2. An A1 batch also takes `--legacy-solver` with the compatible original solver archive. [Batch format](docs/batch-workflow.md) · [Checkpoint and replay](docs/resumable-batch.md)

## Run generator → solver

From the repository root, with Python 3.14 on Windows:

~~~powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python -m pdegen.workflow --seed 20261124 --out scratch/a5
~~~

On Linux or macOS use the environment's `bin/python`. Choose a fresh output directory for each run. The workflow writes `instances/`, `results/`, a manifest linking the two, and an environment record with solver source hashes. The five reference cases have numerical open-right-half-plane count zero. Solver results retain their numerical evidence and unresolved outcomes. [A5 regression and controls](evidence/jim-a5-unfolding/)

| Start here | Contents |
|---|---|
| [Pilot report](docs/pilot.md) | Actual results, numerical margins, rejection and reproduction checks |
| [G2 report](docs/g2.md) | A5 generation, exact shock checks, solver results and review items |
| [Jim's A5 integration](docs/jim-a5.md) | Native twstab run, A5 origin continuation and comparison with G2 |
| [Jim's exact-profile interface patch](docs/jim-a5-formal.md) | Solver entry point, origin settings and five-case regression |
| [Fixed-speed unfolding](docs/jim-a5-unfolding.md) | Automatic conservative profile reconstruction and target-parameter checks |
| [G3 audit and B1 Bloch calibration](docs/g3-b1.md) | Audit decisions, reproducible samples and separate co-periodic and full-line spectra |
| [B2 numerical periodic waves](docs/b2.md) | Swift–Hohenberg continuation, full-zone Bloch sampling and an independently checked sideband witness |
| [C–F spectral calibration](docs/cf-calibration.md) | Six reviewed cases, continuum exclusion proofs and numerical solver comparisons |
| [F2 spatially nonuniform temporal Floquet](docs/f2.md) | Generator–solver exchange and continuum stability of a space-dependent periodic orbit |
| [F2 sampled orbit and period-map action](docs/f2-action.md) | Numerical orbit exchange, phase and closure checks, and propagation of individual Bloch perturbations |
| [twstab temporal Floquet command](docs/twstab-floquet.md) | Apply the F2 one-period map to a JSON batch of perturbations |
| [G4 proof-training readiness](docs/g4-review.md) | Audit of the A–F calibration evidence and one complete task prepared for blind review |
| [G4 measured label yield](docs/g4-label-pilot.md) | Thirty solves and separate label and preliminary difficulty pass rates |
| [A1 and D2 comparison](docs/a1-d2-comparison.md) | Common label gates, a new shifted-delay construction and exact-profile corroboration |
| [D2 resampling and E2 pilot](docs/d2-e-followup.md) | Fresh D2 samples, advancing interface waves and a comparison shortcut |
| [E3 dynamic-interface pilot](docs/e3-dynamic-pilot.md) | Dynamic interface states, whole-domain stable certificates and oscillatory witnesses |
| [E3 verified gain comparison](docs/e3-verified-stability.md) | Whole-half-plane counts at gains 5, 6, 8 and 12 using an interval winding certificate |
| [Unified A1, D2 and E3 generator batch](docs/batch-workflow.md) | Canonical inputs, shared label gates, review tracking and standalone blind prompts |
| [B3, C2 and F3 extensions](docs/bcf-extension.md) | Anisotropic Bloch rolls, Hamiltonian polarization and alternating spatial shear |
| [F4 smooth parametric resonance](docs/f4-parametric-resonance.md) | Continuous-time stiffness modulation and a verified Floquet trace threshold |
| [Resumable process-isolated batch](docs/resumable-batch.md) | Per-case checkpoints and retries, verified on an interrupted A–F run |
| [Mathematical construction](docs/mathematics.md) | A1 equations, admissibility, operator derivation, proof shortcuts and novelty review |
| [Generator–Solver contract](docs/interface.md) | Wire format, supported capabilities, evidence rules and decisions for Jim |
| [Instance schema](schemas/instance.json) · [Result schema](schemas/result.json) | JSON Schema 2020-12 |
| [Pilot manifest](examples/manifest.json) · [A5 fixture](examples/fixtures/a5-mu3.json) | Reproducible inputs, with exact rational coefficients |
| [Numerical evidence](evidence/) | One result directory per mathematical instance, including the held candidate |

## Reproduce the original benchmark

The earlier G0–G2 reference runs used a separate solver. To replay those runs, extract the original `spectral_solver.zip` and pass the directory containing its `run.py` and `benchmark.json`. The generator and checked-in evidence tests can run without that archive.

~~~powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -m pdegen.generate --seed 20261124 --out scratch/repeat
.\.venv\Scripts\python -m pdegen.show scratch/repeat/fixtures/a5-mu3.json
.\.venv\Scripts\python -m pdegen.adapter --examples scratch/repeat --solver C:/path/to/spectral_solver --out scratch/run-01
.\.venv\Scripts\python -m pdegen.generate --family A5 --seed 20261124 --out scratch/g2
.\.venv\Scripts\python -m pdegen.adapter --examples scratch/g2 --solver C:/path/to/spectral_solver --out scratch/run-g2
~~~

On Linux or macOS use the environment’s `bin/python`. Select a fresh output directory for each solver run. Reusing an existing instance directory raises an error to preserve its evidence. Generation for a fixed seed reproduces the specification files byte for byte. Elapsed times, environment records, compressed files and floating-point diagnostics can vary across runs. Use the manifest to enumerate a generation batch.

## Project scope

G0–G2 establish the contract and the original A1/A5 workflows. G3 adds shared validity records and deduplication. G4 adds batch evidence and independent review intake across A–F. A1, B3 and C2 cover local waves, Bloch rolls and Hamiltonian polarization. D2 and E3 cover distributed delay and moving interfaces. F3 and F4 cover switched and smoothly modulated temporal Floquet dynamics.

The next G4 run targets PDE problems with reliable spectral labels and reproducible evidence. Reference solutions are optional. Difficulty review checks for direct reductions and known shortcuts. Keep the family, branch and proof-mechanism group together when eventually assigning dataset splits. [Current G4 acceptance criteria](docs/g4-review.md#current-deliverable) · [Generator–Solver contract and decision list](docs/interface.md#decisions-for-jim)
