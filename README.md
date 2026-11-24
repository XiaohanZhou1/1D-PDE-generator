# 1D PDE generator · G0–G1

This repository builds exact travelling-wave problems and tests whether the supplied Module A solver can solve the **same** spectral problem. The checked-in pilot has four A1 candidates, one A5 interface fixture and the raw evidence from four fresh solver runs. The draft contract is ready for Jim’s review.

| Start here | Contents |
|---|---|
| [Pilot report](docs/pilot.md) | Actual results, numerical margins, rejection and reproduction checks |
| [Mathematical construction](docs/mathematics.md) | A1 equations, admissibility, operator derivation, proof shortcuts and novelty review |
| [Generator–Solver contract](docs/interface.md) | Wire format, supported capabilities, evidence rules and decisions for Jim |
| [Instance schema](schemas/instance.json) · [Result schema](schemas/result.json) | JSON Schema 2020-12 |
| [Pilot manifest](examples/manifest.json) · [A5 fixture](examples/fixtures/a5-mu3.json) | Reproducible inputs, with exact rational coefficients |
| [Numerical evidence](evidence/) | One result directory per mathematical instance, including the held candidate |

## Reproduce

Run from the repository root. The tested environment is Windows x64 with Python 3.14.4. The solver is an external input. Extract the supplied `spectral_solver.zip` and pass the directory containing its `run.py` and `benchmark.json`. The generator and the checked-in evidence tests can run without the ZIP.

~~~powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -m pdegen.generate --seed 20261124 --out scratch/repeat
.\.venv\Scripts\python -m pdegen.show scratch/repeat/fixtures/a5-mu3.json
.\.venv\Scripts\python -m pdegen.adapter --examples scratch/repeat --solver C:/path/to/spectral_solver --out scratch/run-01
~~~

On Linux or macOS use the environment’s `bin/python`. Select a fresh output directory for each solver run. Reusing an existing instance directory raises an error to preserve its evidence. Generation for a fixed seed reproduces the specification files byte for byte. Elapsed times, environment records, compressed files and floating-point diagnostics can vary across runs. Use the manifest to enumerate a generation batch.

## Project scope

G0 supplies the draft contract and A1/A5 fixtures. G1 supplies the original A1 template, compensated parameter variation and one structural feedback rule. The solver bridge enables fixed-point numerical counts on its reviewed subset. The A5 case exercises the conservative interface at one fixed parameter. G2 would add an admissible A5 generator. G3 would generalise the audit gates. G4 and G5 would measure dataset yield and decide whether to scale.

The first review in this repository is the [Generator–Solver contract and decision list](docs/interface.md#decisions-for-jim). Keep the family, branch and proof-mechanism group together when eventually assigning dataset splits.
