# A5 integration with Jim's twstab 2.18.0

The adapter converts the five canonical G2 A5 records into SymPy equations and supplies their exact end states, zero speed and exact profile as the native initial guess. It reconstructs the native profile and operator coefficients independently of the G2 exporter. Across five cases and 401 sample locations the maximum discrepancy in the original and integrated operators was 1.42e-14. Each result retains the G2 specification hash and the native result. The companion environment records include hashes of the supplied solver source files.

| mu | Native wave BVP | Native axis contour with exact profile | A5 continued-origin run | Prior G2 open-RHP count |
|---:|---|---|---|---:|
| 1 | Node limit | Unresolved | 0 additional zeros | 0 |
| 3 | Node limit | Unresolved | 0 additional zeros | 0 |
| 6 | Node limit | Unresolved | 0 additional zeros | 0 |
| 5/2 | Node limit | Unresolved | 0 additional zeros | 0 |
| 29/8 | Node limit | Unresolved | 0 additional zeros | 0 |

The supplied release returned **UNCERTAIN** in both native workflows for all five cases. The reduced A5 profile joins two saddles at a fixed Rankine–Hugoniot speed. The native index-zero profile solve uses a free speed in the reduced BVP, whereas the supplied G2 profile and its unfolding are prescribed by the flux. The five failures occurred at the 40,000-node cap. The continued run uses the exact G2 profile and calls the same native Evans collocation and winding functions. The adapter supplies the A5-specific exterior graphs and uses contours that avoid the origin.

## Origin and search region

For nonzero eigenvalues the integrated operator is \(Z''-JZ'\), where \(J\) is the physical flux Jacobian. The continuation at zero admits an integrated bounded state associated with the jump. At zero \(Z'=\bar Q'\) gives \(Z=\bar Q+d\), and the two independent outgoing characteristic directions determine the constant \(d\) so the end limits lie in the continued slow subspaces. Thus the continued zero can persist even though the primitive of translation is not an \(L^2\) eigenfunction. The default contour passes through zero, and the recorded windings remained unresolved. The A5 pilot continues the same far-field square-root sheets from the right half-plane across a left semicircle. The nearest far-field branch point is at \(-1/4\), outside both origin circles of radius 0.04 and 0.08.

Writing \(M\) for a uniform bound on \(\|J\|_2\), the integrated energy estimate bounds right-half-plane eigenvalues by \(\Re\lambda\le M^2/4,\ |\Im\lambda|\le M^2\). The adapter bounds each entry over \(\tanh z\in[-1,1]\) by its maximum on a rational grid plus its derivative bound times the maximum distance to a grid point. The sum of the squared entry bounds controls \(M^2\).

The outer and origin-circle windings each gave one on both mesh sizes and a shorter domain, for both radii and both nominal half-lengths 20 and 24. The outer region includes the entire origin disk, so the counts account for a single continued zero and give zero additional open-right-half-plane roots numerically. The original conservative origin analysis and independent G2 runs provide the translation interpretation. These runs are fixed-parameter numerical evidence. Continuous contour error bounds and uniformity in the parameter remain open. [A5 mathematical convention](mathematics.md#a5-contract-fixture) · [G2 results](g2.md)

## Reproduce

Use the Python environment in the README with SymPy 1.14.0 installed. The executed environment used Python 3.14, SciPy 1.18.1, and one BLAS thread. Extract Jim's attachment to a local folder and pass the directory containing `twstab/`. The source archive has SHA-256 `fe1520170f6a5ad8b8904441772c8045aa5935cfa5e8a6eef78b4f01c0d7ce66`.

~~~powershell
python -m pip install sympy==1.14.0
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -m pdegen.jim_a5 --solver C:/path/to/solver --examples examples/g2 --profile native --out scratch/jim-native
python -m pdegen.jim_a5 --solver C:/path/to/solver --examples examples/g2 --profile continued --half-length 20 --out scratch/jim-L20
python -m pdegen.jim_a5 --solver C:/path/to/solver --examples examples/g2 --profile continued --half-length 24 --out scratch/jim-L24
~~~

For Jim's review, the two changes to discuss are an explicit exact-profile entry point and a caller-selectable treatment of the continued integrated determinant at the origin. The supplied solver release is an external input.
