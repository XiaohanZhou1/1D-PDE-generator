# Executed G0–G1 pilot · 24 November 2026

**Four A1 candidates passed exact target and operator checks. Three went through the attached solver. The fourth changed the operator and was held before numerical dispatch.** A separate fixed A5 shock exercised the G0 interface. The evidence below was freshly computed from exported specifications. The results shipped in the input ZIP were used for comparison only.

| Target and parameters | Role | Native outcome | Open-RHP count | Positive witness or neutral treatment | Adapter time |
|---|---|---|---:|---|---:|
| A1, q=0, a=4 | Analytic regression | UNSTABLE | 1 | Real root 2.99999991. Translation and ±2i account for the other three outer roots. | 6.23 s |
| A1, q=1, a=2 | Coupled reference | UNSTABLE | 2 | 4.46995634 ± 1.10804544i | 4.79 s |
| A1, q=127/128, a=257/128 | New dyadic sample | UNSTABLE | 2 | 4.45687838 ± 1.10257261i | 4.68 s |
| Same sample with beta=−1/4 | Structural mate | NOT_RUN | — | The extra term changes the y-row Jacobian. Label and evidence grade are null. | <0.01 s |
| A5, mu=3, rho=0 | Interface fixture only | STABLE_MOD_TRANSLATION | 0 | Simple continued translation allowed at the essential edge | 2.92 s |

All four completed results have native grade **CONVERGED_NUMERICAL**. Stable/unstable refers to the specified wave and convention. The examples establish no parameter-uniform conclusion. The new and reference pairs support the same observed mechanism. Their proximity gives a useful transport test for the interface, with little additional proof diversity.

## Numerical checks and retained artifacts

For the new A1 sample, the outer grid extends to Re λ = 250 and |Im λ| = 616 with left edge −0.04. Its winding is three, including translation. At the finest grid each positive witness has unit local count, and the two observed error scales are about 0.0007982. The native driver used four Evans (L,h) pairs, four finite-difference pairs, adjoint conditioning, far-field graph checks, and two asymmetric exterior extensions. The neutral finite-difference residual fell from 0.13642 to 0.00862 as h decreased from 0.12 to 0.03. The largest sampled relative change on its outer contour was 0.04222. These quantities measure numerical convergence. Continuous tail, parameter and contour error bounds remain open. [New A1 result](../evidence/A1-49dce71cae67b4b4a998/result.json)

Across the completed integrations, the maximum sampled profile defect was $1.78\times10^{-15}$, and the maximum reaction/flux Jacobian discrepancy was $2.14\times10^{-14}$ (rounded up). The fixed shock passed the conservative and integrated counts, outgoing obstruction check, continued-origin test and independent shooting gates. The native runtimes for the seed, reference, new point and shock were 5.27, 4.13, 4.04 and 2.27 seconds, respectively. Each evidence folder retains the precise settings, stdout, native case, environment, diagnostics and compressed raw contours. The held record identifies the missing capability. [A5 result](../evidence/A5-7675e3315ff5c3041444/result.json) · [Held feedback record](../evidence/A1-08cb57a942e347825a1a/result.json)

## Reproduction and negative tests

The commands below use repository-relative paths for the actual workspace run. They follow extraction of the original ZIP under `../../work/spectral_solver`. The local Python virtual environment inherited the installed solver dependencies and loaded jsonschema 4.26.0 and its dependencies from cached wheels. The portable environment setup is in the README.

~~~powershell
../../work/env/Scripts/python -m pdegen.generate --out examples
../../work/env/Scripts/python -m pdegen.adapter --solver ../../work/spectral_solver --examples examples --out ../../work/verified-run
../../work/env/Scripts/python -m unittest discover -s tests -v
~~~

The final verified run was copied into `evidence/`. A separate `python -m unittest -v` in the untouched solver directory passed its six upstream tests. All 12 generator and evidence tests passed. An independent generation in a fresh directory produced six byte-identical JSON files, including the manifest and A5 fixture.

* Exact row cancellations and translation at all rectangle corners for zero and signed feedback, phase, endstates, positive diffusion, the integrated A5 identity and endpoint fluxes.
* The row-two product rule and normal Jacobian agree with independently evaluated directional derivatives of the full physical PDE for arbitrary perturbation jets. Corrupting the profile, removing the derivative flux, altering a compensation sign, or destroying diffusion positivity causes rejection.
* A feedback preserving the wave while changing its far field fails the analytic far-field gate. Schema and semantic validation reject code-like monomials, missing operator terms, stale hashes, unsupported weights and incorrect branch or unfolding data.
* Equivalent rational inputs deduplicate across seeds. A deterministic reconstruction matches the shipped batch. Unsupported physics yields no numerical label. A missing solver yields a retained failure record.
* An independent reader recalculates **76 winding numbers** from the saved unit phases, including closing edges, across all four refinements, and checks the maximum phase increment against the native limit. Recounting the sampled contours checks serialization and arithmetic. It does not establish what happens between sample points.

## Yield and limitations

For G1 only, attempted 4, exact wave and admissibility 4/4, duplicate specifications 0/4, dispatchable 3/4, solver completion 3/3 dispatched, native numerical labels 3/4 attempted. Among the three numerical results there were three unstable, zero stable, zero marginal and zero uncertain. The fourth is a capability hold. There are two A1 structures, one explicit branch and **zero independently reviewed novel-proof admissions**. There was no per-case manual intervention in dispatch. The neighbourhood, scouting policy and structural rule were selected by hand. The A5 fixture adds one successful round-trip outside those denominators.

The G1 adapter time summed to 15.70 s including the held case, or 5.23 s per native numerical label. This excludes installation, design, tests, exact-record generation and pre-dispatch schema validation. End-to-end production cost and cost per trusted proof admission are undefined here. The final run used one BLAS thread, Windows x64, Python 3.14.4, NumPy 2.4.4 and SciPy 1.18.1. Peak RAM and CPU-time profiles were not measured. No inference about production yield follows from this selected neighbourhood.

The principal open scientific decision is whether to pursue a certified count at one coupled point or a finite-parameter collision theorem. Jim’s interface and numerical-policy decisions are listed in the [contract](interface.md#decisions-for-jim). An external review of novelty and of numerical admission remains pending.

## Input provenance

| Supplied input | SHA-256 |
|---|---|
| Development brief | `c999b6da1f77191a96a0b3a22a032c0d4f8a803aa1f0f789a2d97492d45b1b01` |
| Five-problem design v1 | `f0b133d5dfb2994ae5ec9ebe7520481801a4c2915c3c21780d5b15f1be4330c9` |
| Independent audit 1 | `7066f7cab8aab21d46eeec519f764a063d1cac177df0706861f20f45ab09771a` |
| Supplied solver ZIP | `952af0e51f7c31f9aff3e7f7b5268d557fd5c17ba65d963fa1d4567b111f7a24` |

Historical prompts, previous commands and reported runs inside the supporting documents are source material. The design supplies the equations, the audit supplies claims and reductions to check, and the current development brief supplies the roadmap. The solver input archive and its historical results remain external to this repository.
