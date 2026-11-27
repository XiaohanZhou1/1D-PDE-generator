# G4 measured label yield

The pilot ran 30 distinct PDE specifications with seed 20261126. It measures the yield under the label standard in [current G4 criteria](g4-review.md#current-deliverable). The question scored here is the binary spectral label in the recorded space. For B2 the scored label is full-line stability, with co-periodic diagnostics retained.

## Observed yield

| Family | Attempted | Reliable labels | Label and preliminary difficulty screen |
|---|---:|---:|---:|
| A1 | 4 | 3 | 2 |
| A5 | 5 | 0 | 0 |
| B1 | 1 | 1 | 0 |
| B2 | 20 | 8 | 0 |
| **Total** | **30** | **12 (40.0%)** | **2 (6.7%)** |

The accepted labels consist of 11 unstable witnesses and one analytically stable calibration. Of the 18 other cases, 12 B2 profiles had no independently confirmed positive witness or full-zone exclusion, five A5 cases had numerical stability evidence without a full-domain certificate, and one A1 feedback variant was held by the legacy adapter. The 12 unresolved B2 cases remain in the denominator. The A1 controls and candidates have the Evans and independently refined physical finite-difference checks of the original solver. The B2 witnesses have both Fourier and twisted physical finite-difference checks.

## Difficulty interpretation

Nine numerical label passes and the one exact stability pass are calibration examples. The B2 grid tests a standard Swift–Hohenberg roll template with a known stability literature. The run uses B2 as a backend calibration. [Barker et al., Swift–Hohenberg appendix](https://arxiv.org/abs/1203.3795)

The two provisional A1 passes are finite-coupling pulses at `(q,a,beta)=(1,2,0)` and `(125/128,129/64,0)`. The decoupled seed argument does not immediately decide these parameters. They use the same A1 proof family, and the published local audit's novelty assessment concerned the full diagram. Difficulty for the binary label remains provisional. No independent review or final training admission was performed in this run. [A1 construction and reductions](mathematics.md#proof-difficulty-and-novelty)

The 40% figure is specific to this deliberately calibration-heavy mix. The 6.7% figure is a preliminary screening yield concentrated in one family, so neither figure estimates production yield for 1,000 difficult questions. The next comparison should include a structurally different mechanism with independently verifiable instability witnesses.

## Reproduce

This run uses the bundled solver for A5 and the pinned original spectral solver for A1. Extract the original reference archive to a directory containing `run.py` and `benchmark.json`. The command uses a deterministic grid for B2 and seeded samplers for A1 and A5. Choose a fresh output directory. About 22 seconds elapsed across the four family runs in the reference environment.

~~~powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m pdegen.label_pilot --legacy-solver C:/path/to/spectral_solver --out scratch/g4-label-pilot
~~~
