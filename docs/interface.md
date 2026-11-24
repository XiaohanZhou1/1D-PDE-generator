# Generator–Solver contract · 0.1-draft

## Input record

The [instance schema](../schemas/instance.json) specifies the required fields and matrix shapes. The mathematical object is a PDE, exact parameter tuple, designated wave and speed, spectral space/domain, and a precise question. `pdegen.contract.validate` checks the schema and independently rebuilds the canonical record from the named template before dispatch. A schema pass alone establishes wire shape. The semantic check enforces the particular template, target, coefficient values, bounds, task and capability set.

| Field group | Interpretation |
|---|---|
| Identity | `schema_version` versions the wire language. `generator_version` identifies the implementation. `family_id`, `structure_id`, and `template_version` select the reviewed construction. The integer seed records the sampler recipe. |
| Laboratory equations | Ordered state variables, exact parameters, diagonal vectors D and A, reaction vector F, and differential-flux vector G. Each PDE is given by `D*X_xx-A*X_x+F+dx(G)`. D and A act diagonally. The A5 physical flux equals −G. |
| Target | `wave_id` and `branch_id` select a phase-pinned, exact full-line solution. The record supplies its speed, profile, first two derivatives and both endstates in left/right order. |
| Spectral problem | Use the comoving coordinate z and the complexified unweighted space. D, B and C are in output-row/input-column order for `Dp''+Bp'+Cp`. Limiting B,C matrices at both ends and the common D define the Fourier symbols. A5 additionally requests conservative flux variables, integrated cross-checks and continuation at the origin. |
| Question and checks | A1 asks for the fixed-tuple open-RHP count with algebraic multiplicity. A5 asks for binary stability under its specified shock convention. Checks have PASS, FAIL, UNVERIFIED or NOT_APPLICABLE. A mathematically valid target may fail solver dispatch. |
| Review metadata | Expected mechanisms, family grouping, difficulty screen and provenance travel with the internal record. Use the equation/target/space/question fields or `pdegen.show` for a proof-facing view. Solver centres, roots and evidence belong in the results directory. |

### Restricted coefficients and identity

A polynomial is a JSON object mapping a sorted product of allowed scalar names to an exact rational coefficient string. For example, `{"s*s":"18"}` represents $18s^2$, `{"1":"-3/2"}` is a constant, and `{}` is zero. Physical equations use the declared state variables. Wave and operator coefficients use s = sech(z), t = tanh(z). Omitted monomials are zero. Multiplication is commutative. Rationals are reduced with positive denominator. Integers use no slash. No general parser, function calls or evaluation of source code is required. Phase, question, and symbolic-form text select reviewed conventions. Template-bound validation catches a changed string even when the JSON type matches.

The mathematical fingerprint is SHA-256 of compact, key-sorted, ASCII-escaped JSON with no NaNs. The exact field list `MATH_KEYS` is versioned in [contract.py](../pdegen/contract.py). It covers the template, equations, normalized wave/branch, parameters, space, operator, endstates, task and neutral convention. It excludes seed, timestamps, diagnostics and editorial review. `instance_id` combines the family with the first 20 hex digits. Result linkage uses the full digest as well. Algebraically identical sparse polynomials and rational spellings intern to the same form on generation. Both pulse and shock have a single canonical phase. Import accepts only the canonical representation. Arbitrary translations, parameter scalings, state-variable changes and metamorphic equivalence across templates have no general matcher. Nearby L1 samples share a group and must stay on the same side of an evaluation split.

## Capabilities and dispatch

| Model or request | Generator | Adapter to supplied solver 1.0.0 |
|---|---|---|
| Original A1, original five benchmark tuples | Exact A1 on its admissible rectangle | Native A1 model, seeded local contours, independent full contour, finite differences and neutral checks |
| Original A1, sampled neighbourhood of (1,2) | Seeded dyadic parameters with exact binary64 representation | Scout the upper root by coarse physical-space shift-invert at the new parameter. Use its conjugate and exact translation as additional centres. Full native gates still apply. |
| Other admissible A1 parameters | Exact target and operator can be exported | Hold pending a reviewed scouting policy. Exact parameters that round in binary64 are also held. |
| A1 with beta nonzero | Exact candidate with a changed second reaction row | NOT_RUN with reason. Native physics, physical-space check and high-frequency envelope require extension. |
| A5, mu=3 and rho=0 | Fixed G0 fixture | Native conservative and integrated formulations with origin continuation |
| Parameter-uniform theorem, Jordan certification, G2 sampling, A2–A4, weighted or mixed-order systems | Roadmap or specialist review | Outside this bridge |

The adapter pins the native source files and default settings by SHA-256. A change of solver release requires a compatibility review. Before calling the native CLI it reconstructs the input and compares off-wave reactions or fluxes and on-wave coefficients at seven points to the native model. These samples are regression checks. The independent derivation and polynomial identities supply the mathematical justification. The native CLI receives one case through a dedicated config file using `--config` and a fresh `--output` folder. It returns native case, settings, environment, diagnostics and contour samples. The bridge compares the returned family, parameters and settings and links their record to the input hash. Scouting hints are numerical search data.

## Result semantics

The [result schema](../schemas/result.json) links each record by full input hash and ID. `NOT_RUN` records a capability hold, `FAILED` records a bridge or process error, and `COMPLETED` means that the native driver returned a result. A native numerical failure can return a completed `UNCERTAIN` record with diagnostics. The result includes observed timing, configuration digest, artifact paths relative to its own directory, completeness, numerical label, evidence grade and post-solver disposition.

* A positive, independently checked witness can support **UNSTABLE** with a null total count. Record $N_+$ only after successful global accounting. An outer contour may include axis roots. Its winding alone is not $N_+$.
* **STABLE_MOD_TRANSLATION** requires complete exclusion under the declared convention and the allowed neutral structure. For A5, the continued simple zero lies at the essential edge.
* **MARGINAL** requires a resolved extra neutral mechanism under the declared convention after positive spectrum is excluded. **UNCERTAIN** records insufficient evidence.
* **EXPLORATORY** and **CONVERGED_NUMERICAL** are numerical evidence grades. **RIGOROUSLY_CERTIFIED** requires validated error bounds for the stated claim. The attached driver produces the first two grades. All pilot records have proof-training disposition **HOLD**.

The native diagnostic files store high-frequency search bounds, exterior checks, refinements, asymmetrically extended domains, neutral residuals, root-disk counts, independent eigenpairs, conditioning and tail norms. Root error scales are empirical. A failure reason and the artifacts produced before a failure are retained in its directory. Runtime covers dispatch after input validation. Environment metadata report BLAS thread count. Per-process peak memory is unmeasured in this adapter.

## Decisions for Jim

1. **Contract freeze.** Confirm field meanings, exact-to-native parameter policy, version negotiation, hash payload, and whether to adopt the restricted polynomial representation or register model callbacks. Decide how a reviewed new release advertises capabilities.
2. **A1 structural feedback.** Decide whether to support the normal-feedback row. It needs a native reaction/Jacobian update, revised high-frequency envelope and independent physical-space check. Agree who owns those before sending the first such case to numerical admission.
3. **Scouting and coverage.** Approve an extension beyond the small neighbourhood, including cluster contours when roots approach each other, sign margins, and retry policy when a contour or exterior graph is ill-conditioned.
4. **Neutral modes and evidence.** Confirm per-task count, stability and marginal semantics. Agree on tail/integration/contour error control and rank or Jordan-chain checks for claims of certification. Current tolerances are those in the supplied benchmark.
5. **Proof admission.** Choose whether the next scientific task is a certifiable finite-coupling count or a collision theorem, assign an independent expert to screen shortcuts, and decide how to evaluate a proof. The two adjacent nonreal-pair samples belong to one mechanism group.
6. **Solver distribution.** The generator is hosted in the public [1D-PDE-generator repository](https://github.com/XiaohanZhou1/1D-PDE-generator). Agree how collaborators should obtain the pinned solver release and who will maintain compatibility checks when it changes.
