# 1D PDE generator and travelling-wave solver

This repository bundles the G2 A5 generator and Jim's twstab 2.18.0 solver. The end-to-end entry point generates reproducible conservative shock specifications, reconstructs each wave using the fixed-speed unfolding interface, and computes its spectrum with the A5 continued-origin contour. [Bundled solver and provenance](solver/UPSTREAM.md)

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

G0 supplies the draft contract and A1/A5 fixtures. G1 supplies the original A1 template, compensated parameter variation and one structural feedback rule. G2 generates the original A5 shock branch with identity viscosity, zero unfolding parameter and variable coupling. The solver bridge enables fixed-parameter numerical counts on its reviewed subset. G3 would generalise the audit gates. G4 and G5 would measure dataset yield and decide whether to scale.

The first review in this repository is the [Generator–Solver contract and decision list](docs/interface.md#decisions-for-jim). Keep the family, branch and proof-mechanism group together when eventually assigning dataset splits.
