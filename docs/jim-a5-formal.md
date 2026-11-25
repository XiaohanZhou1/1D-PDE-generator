# Jim's A5 exact-profile and origin interface

This patch targets the supplied twstab 2.18.0 archive, SHA-256 `fe1520170f6a5ad8b8904441772c8045aa5935cfa5e8a6eef78b4f01c0d7ce66`. Apply the patch once from the extracted directory containing `twstab/`, and run the generator adapter against that directory.

~~~powershell
git apply --check C:/path/to/jim-twstab-a5.patch
git apply C:/path/to/jim-twstab-a5.patch
~~~

`wave['exact_profile']` accepts a mapping from component names to SymPy expressions in `z` for a second-order conservative front with an explicit wave speed. The solver verifies the symbolic travelling-wave residual and the two far-field limits, then evaluates the profile directly on the numerical mesh. A mismatched profile returns `UNCERTAIN` with a reason. `wave['L']` selects the half-length, with default 20 in exact mode.

For A5 use `options['conservation_method']='axis'`, `options['conservation_origin_radius']=0.08`, and `options['region']=(xmax,ymax)` with a justified high-frequency enclosure. The adapter computes the enclosure from exact polynomial coefficient bounds and the integrated energy estimate. Opting in to origin continuation requires a two-component, identity-viscosity, saddle-to-saddle front with independent outgoing directions. The saddle directions are continued from the open right half-plane, and the radius is checked against the first far-field branch point. The solver runs the common Evans engine on a detoured outer contour and an origin circle at radii 0.04 and 0.08. It uses two spatial step sizes and compares with a shorter domain. The numerical label is `STABLE` with qualifier `numerical stability modulo translation` when the outer and origin counts both account for exactly one continued zero in every run.

The continued zero is anchored at the origin by the exact profile. In integrated variables its bounded state is \(Z=\bar Q+d\). The two independent outgoing directions determine \(d\) so both limits lie in the continued slow spaces. The derivative \(Z'=\bar Q'\) is the localized translation mode of the original system. The counts give numerical evidence for a simple continued zero and no other zeros in the closed right half-plane. Continuous numerical error bounds and uniformity in \(\mu\) remain open. [A5 mathematical and numerical background](jim-a5.md)

## Reproduction and evidence

Use the Python dependencies in the [README](../README.md) plus SymPy 1.14.0. Run from the generator root after applying the solver patch. Select fresh output paths.

~~~powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m pdegen.jim_a5 --solver C:/path/to/solver --examples examples/g2 --profile formal --half-length 20 --out scratch/jim-L20
python -m pdegen.jim_a5 --solver C:/path/to/solver --examples examples/g2 --profile formal --half-length 24 --out scratch/jim-L24
python -m pdegen.jim_a5 --solver C:/path/to/solver --examples examples/g2 --profile endpoints --out scratch/jim-endpoints
~~~

The patched public analysis entry returned numerical `STABLE` with open-right-half-plane count zero for \(\mu=1,3,6,5/2,29/8\) at both half-lengths. Each recorded outer and origin winding was one. The separate endpoint-only run supplied the two end states and known zero speed, with no profile or symbolic initial guess. All five BVPs still reached the 40,000-node cap and returned `UNCERTAIN`. This preserves the automatic reconstruction failure as a regression case. [Formal run, half-length 20](../evidence/jim-a5-formal/L20/) · [Half-length 24](../evidence/jim-a5-formal/L24/) · [Endpoint-only run](../evidence/jim-a5-formal/endpoints/)
