# Fixed-speed unfolding interface for conservative fronts

The solver patch adds a named flux parameter to the automatic profile solve for two-component, second-order conservative saddle-to-saddle fronts with identity viscosity. It targets the supplied twstab 2.18.0 archive and includes the previous exact-profile and origin patch. Apply it to a fresh extraction of the original solver, whose SHA-256 is `fe1520170f6a5ad8b8904441772c8045aa5935cfa5e8a6eef78b4f01c0d7ce66`.

~~~powershell
git apply --check C:/path/to/jim-twstab-a5-unfolding.patch
git apply C:/path/to/jim-twstab-a5-unfolding.patch
~~~

## Solver input

The calling problem declares `rho` as a SymPy symbol in the PDE. Its entry in `parameters` gives the physical target. The wave field supplies the end states and selects the same target and a numerical initial guess for the unfolding.

~~~python
parameters = {rho: 0.0}
wave = dict(
    kind="front", left=left, right=right, speed=0,
    unfolding=dict(parameter="rho", target=0.0, guess=0.02),
)
~~~

The solver fixes the Rankine–Hugoniot speed and requires the selected parameter to preserve the jump at the given end states. It rejects purely rescaling parameters, and updates the endpoint projection directions during collocation. The BVP uses the regular profile initial guess and integral phase condition. A5's initial functions are generated from the end states, while the `guess` inside `unfolding` initializes the scalar parameter. The first solve is subsequently refined.

Admission requires that the fitted parameter return within `1e-7 * max(1,abs(target))` of the target. The solver checks the actual spline against the original target PDE and its reduced first integral, examines the tails, and checks that the parameter variation has a normal component on the computed orbit. The labeler uses the original target operator for spectrum. For A5 the adapter also supplies the energy enclosure and enables the existing continued-origin contour. Root counts and PDE residuals are numerical.

## Reproduce

Run from the generator root with the normal dependencies and SymPy 1.14.0. The supplied verification driver builds the A5 input with the original `rho*(1-u**2)` flux perturbation, using the exact G2 specification for the target where rho is zero. No analytic starting profile is passed to the BVP. Phase alignment in the independent check translates the recovered profile to its zero of \(u\) before comparing it against the known profile on \([-8,8]\).

~~~powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m pdegen.jim_a5_unfolding --solver C:/path/to/solver --examples examples/g2 --out scratch/jim-unfolding
~~~

The driver also checks the separate triangular saddle-to-saddle front, wrong targets at rho=0.05 for each flux, and the original p-system fixed-speed path. The second shock verifies profile recovery. Its outgoing directions are parallel, so the A5 origin rule is outside its scope. [Run records](../evidence/jim-a5-unfolding/) · [Prior structural audit](a5-bvp-audit.md)

## Results

All five A5 profiles were recovered from the end states and the default front guess. The fitted parameters had absolute values below 8e-14. The largest phase-aligned profile error was 1.51e-11, the largest first-integral residual was 1.29e-9, and the largest second-order target PDE residual was 2.73e-6. All five cases returned numerical stability modulo translation with open-right-half-plane count zero. The 60 outer and origin contour checks each resolved a winding of one.

The triangular control recovered the profile with phase-aligned error 1.05e-11. Its default spectral label remained UNCERTAIN at the origin. Both wrong-target runs were rejected before spectrum after returning a fitted value near zero. The original p-system fixed-speed path still recovered its profile and returned STABLE. The tests also reject parameters that change the jump and parameters that only rescale the entire reduced field.
