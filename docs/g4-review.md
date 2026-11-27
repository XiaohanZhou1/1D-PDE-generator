# G4 proof-training readiness pre-pilot

## Current deliverable

The next G4 run produces labelled PDE problems. Each deliverable records the PDE, parameters, target wave and phase, perturbation space, spectral question, label and reproducible evidence. A reference proof is optional. Prioritize cases for which a short proof or a direct reduction has not been found in the difficulty review. Record that finding with the scope of the review. It does not establish that the problem is open or novel.

Keep difficulty and label confidence independent. For `UNSTABLE`, look for a positive spectral witness with a resolved margin and an independent cross-check. Its scope determines what the label supports. For `STABLE`, retain the agreed standard of full coverage of the relevant spectral domain by an analytic argument or verified error bounds. A certificate can establish this without a written reference solution. Where the evidence is incomplete, keep `UNCERTAIN` outside the reliably labelled yield. Count that yield by family and mechanism rather than counting parameter variants as distinct mechanisms.

The material below records the earlier proof-focused pre-pilot. The D follow-up later found a shorter uniform count proof using a norm perturbation bound, which makes that case useful as a difficulty benchmark.

## Original pre-pilot

This pre-pilot asks whether the workflow can produce a self-contained proof task with a checkable answer and substantive reasoning steps. The staged G4 run in the brief would use a larger batch. This report covers a frozen set of 22 existing specifications and one newly written interval-level task. Its three solver probes are checks of that one task. No independent blind review has been run.

## Existing evidence

| Finding | Cases | Interpretation |
|---|---:|---|
| Analytic calibration | 10 | Complete answer at the requested scope and a known or elementary reduction |
| Numerical evidence only | 6 | Five numerically stable A5 records and one B2 sideband witness |
| Unresolved | 2 | B2 runs with inconclusive full-zone stability evidence |
| Specification only | 3 | A1 mathematical validation, with no solve in this manifest |
| Solver dispatch hold | 1 | A1 normal-feedback variant outside the enabled solver capability |
| Proof-training admitted | 0 | All existing cases retain their G3 hold |

The script verifies audit, instance and result hashes and finds 22 distinct specifications in the selected manifests. The distribution is 11 stable, four unstable, one marginal, two uncertain and four without a solver label. Stable numerical A5 records remain on hold under the continuum exclusion standard. The ten analytic calibrations cover distinct lessons, but grouping by proof mechanism prevents a roll or a threshold parameter change from counting as an independent proof. The A1 full multiplicity diagram and the A5 interval-wide finite-frequency exclusion still lack reference proofs.

## Candidate and evidence

The D task changes the spatial feedback by an exact profile-preserving asymmetric term and uses two weighted uniform memory windows. It asks for the entire closed right half-plane spectrum and algebraic multiplicities, uniformly over \(0\le\varepsilon\le1/8\). The unperturbed spatial case at \(\varepsilon=0\) provides a solvable comparison. The full claim requires treating the history-space resolvent, nonreal spectrum, a uniform eigenvalue count and the absence of generalized eigenvectors.

| \(\varepsilon\) | Finite-difference first positive root | Second positive root | Solver comparison |
|---|---:|---:|---|
| 0 | 1.067043 | 0.325088 | Pass |
| 1/16 | 1.067654 | 0.325007 | Pass |
| 1/8 | 1.069479 | 0.324773 | Pass |

The finite differences independently discretize the spatial characteristic pencil on two domains and re-solve the two real eigenvalue crossings. The bundled solver finds roots in agreement to within \(10^{-5}\) across the three samples. Measured runtimes were about 1.45 s per probe with a single OpenBLAS thread in the reference environment, including both runs and post-comparison. The numbers establish reproducibility of the sampled cases. The analytic proof supplies the scope and counts the whole spectrum in the specified region.

## Difficulty and novelty review

On 25 November 2026, the search covered localized delayed kink operators, uniform or mixed distributed delays, sine/Laplace transform bounds and the Pöschl–Teller spectrum. The closest in-repository control supplies the solvable symmetric characteristic equation and an instability witness. Its exact spatial reduction does not directly give the count with the asymmetric feedback. Infeld and Hull supply the classical comparison-operator factorisation. Bernard and Crauste study scalar equations with distributed delay and stability criteria at a fixed mean, with no spatial operator or count for this kink. The ingredients for the new proof are standard. The search and the in-repo similarity leave proof novelty **UNCERTAIN**. [Symmetric in-repo control](../solver/problems/delay_kink_uniform.py) · [Infeld and Hull](https://doi.org/10.1103/RevModPhys.23.21) · [Bernard and Crauste](https://doi.org/10.3934/dcdsb.2015.20.1855)

The prompt is ready for a blind attempt. Give the reviewer only the prompt. Ask for a proof and any shortcut they find, then compare it against the four obligations in the rubric. Record whether the shortcut solves the full task or only a weaker one, where the proof attempt stalls, and whether this proof mechanism merits reuse. The task stays on hold pending this review. The pilot establishes one reference-backed candidate and no accepted training yield. Cost per admitted training task is undefined. A scale-up estimate requires independent review and more than one proof mechanism.

## Reproduce

~~~powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m pdegen.g4_pilot --out scratch/g4-review
~~~

Use a fresh output directory. The command audits the selected frozen manifests and reruns the new candidate's native solver and independent finite differences on three fixed parameter values.
