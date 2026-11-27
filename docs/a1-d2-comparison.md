# A1 and D2 label comparison

This is a targeted comparison of a previously tested A1 batch and a new D2 generator template. The D2 grid was chosen after exploratory scouting of the symmetric model. These rates are conditional on the selected parameter ranges.

| Family | Attempted | Reliable labels | Label and preliminary difficulty |
|---|---:|---:|---:|
| A1 | 4 | 3 | 2 |
| D2 | 6 | 6 | 4 |
| Total | 10 | 9 | 6 |

A1 reproduces the earlier A1 batch at the same seed. Its decoupled case is a calibration, two finite-coupling cases clear the preliminary difficulty screen, and the existing normal-feedback variant is outside the pinned A1 adapter. D2 includes two controls and four asymmetric cases. Eight accepted labels are unstable and one is stable. The six joint passes cover two distinct generator families. Blinded difficulty review is pending.

## D2 construction and evidence

The exact front remains \((U,0)\), \(U=\tanh(x/\sqrt2)\). The v equation adds negative feedback
\[
-\frac{\sigma}{d}(1-u^2)(1+\varepsilon u)\int_\tau^{\tau+d}v(x,t-s)\,ds,\qquad d=\frac14.
\]
For every sample, the exact profile is admissible and the feedback vanishes at each endstate. The linearisation is block diagonal since \(v=0\) on the front. The u block retains the simple translation mode and has no positive spectrum. For the delayed block at \(\Re\lambda\ge 0\), solving in the history variable reduces the resolvent to a scalar spatial characteristic pencil. The free endpoint operator is invertible there and the delayed potential is an exponentially decaying compact perturbation from \(H^2\) to \(L^2\). Thus a spatial \(L^2\) mode yields a history-space eigenvalue.

The delay transform is \(H(\lambda)=d^{-1}\int_\tau^{\tau+d}e^{-\lambda s}\,ds\). For real \(\lambda\ge0\), \(H(\lambda)>0\) and the endpoint pencil
\[
\lambda-\partial_x^2+1+\sigma H(\lambda)(1-U^2)(1+\varepsilon U)
\]
is strictly positive. Any unstable spectral witness therefore has nonzero imaginary part. The earlier D1 sign argument excludes nonreal roots for a different choice of memory support and feedback sign. For D2, the shifted transform and negative feedback allow the imaginary part to balance the frequency. This fact permits oscillatory instability and leaves its existence to be established at the prescribed parameters. At \(\varepsilon=0\) the spatial potential has a Pöschl–Teller reduction. The asymmetric profile factor changes that potential. Scalar distributed-delay results give context for the negative feedback mechanism. Their scalar homogeneous equation does not directly decide this localized spatial pencil. [Bernard and Crauste](https://doi.org/10.3934/dcdsb.2015.20.1855)

The small-gain control has \(\sigma(1+\varepsilon)=7/8\). For \(\Re\lambda\ge0\), the \(L^2\) norm of the free resolvent is at most 1, \(|H(\lambda)|\le1\), and the delayed multiplication factor has norm at most \(7/8\) after including \(\sigma\). The Neumann-series inverse and history resolvent give full-domain exclusion in the delayed block. This control retains the front's neutral translation mode and gives a `STABLE` label under the open-right-half-plane convention.

For the unstable cases, bundled twstab uses a band determinant and a history discretization. An independent check integrates a continuous scalar spatial equation from both tails. It uses the exact target and analytic delay transform in place of a computed profile or delay quadrature. Three domain and integration refinements track the Wronskian zero. Both real and imaginary parts clear ten times the observed error scale. These are converged numerical instability witnesses.

| \(\sigma\) | \(\tau\) | \(\varepsilon\) | Refined positive-imaginary witness |
|---:|---:|---:|---|
| 3 | 1 | 3/4 | \(0.00672343+1.98206211i\) |
| 3 | 3/2 | 3/4 | \(0.09910570+1.46745124i\) |
| 5 | 1 | 3/4 | \(0.37282286+2.06620072i\) |
| 5 | 3/2 | 3/4 | \(0.35864894+1.51075080i\) |

The maximum native-to-shooting discrepancy on these four cases is \(3.43\times10^{-6}\). The near-threshold case \((\sigma,\tau)=(3,1)\) is especially useful for blind review. At \(\varepsilon=0\), the corresponding symmetric scalar root has real part about \(-0.042429\). Shooting tracks its sign change on the sampled sequence \(0,1/4,1/2,3/4\). That is numerical evidence for this branch. The complete spectrum along the parameter path and a bifurcation theorem remain open in this pilot. The sampled tracking calls `verify_root` at the listed epsilons, initially seeded with \(-0.04+1.96i\), and passes the fine root forward.

## D2 full right-half-plane certificate

Write \(W=(1-U^2)(1+\varepsilon U)\), \(g=\sigma H(\lambda)\). For \(\Re\lambda\ge0\), \(|H(\lambda)|\le1\) and \(0\le W\le1+\varepsilon^2/4\). To see the spatial bound, on \(U\le0\) use \(W\le1\), and on \(U\ge0\) use \(W\le1+\varepsilon U-U^2\). The free endpoint operator \(\lambda+1-\partial_x^2\) has inverse norm at most \(1/|\lambda+1|\), by Fourier transform. Thus the endpoint pencil is invertible wherever \(|\lambda+1|>\sigma(1+\varepsilon^2/4)\), by a Neumann series. Take an integer wall \(R>\sigma(1+\varepsilon^2/4)\). The exterior of \([0,R]+i[-R,R]\) in the closed right half-plane is covered by the same strict bound.

For the remaining rectangle, put \(\kappa=\sqrt{1+\lambda}\) with positive real part and \(p=\kappa/\sqrt2\). For the left decaying solution, set \(t=(1+U)/2\) and write \(f_-(x)=e^{\kappa x}h_-(t)\), with \(h_-(0)=1\). Its equation reduces to
\[
t(1-t)h_-''+(1+2p-2t)h_-'=2g(1-\varepsilon+2\varepsilon t)h_-.
\]
The coefficients of \(h_-=\sum_{n\ge0}a_nt^n\), with \(a_{-1}=0,\ a_0=1\), obey
\[
(n+1)(n+1+2p)a_{n+1}
=\bigl[n(n+1)+2g(1-\varepsilon)\bigr]a_n+4g\varepsilon a_{n-1}.
\]
The right decaying solution is \(f_+(x)=e^{-\kappa x}h_+((1-U)/2)\) and uses the same recurrence with \(-\varepsilon\). On a compact neighborhood of the rectangle, \(\Re(2p)>1\) and \(g\) is bounded. The recurrence then bounds all sufficiently late coefficients uniformly in \(\lambda\) by a constant, using the majorant below with a neighborhood bound on \(|g|\). The series and its spatial derivatives converge normally at \(t=1/2\), giving analytic dependence on \(\lambda\). The normalized solutions represent the unique decaying lines at their respective ends. The analytic characteristic at \(x=0\) is the Wronskian \(f_-'f_+-f_-f_+'\), given by
\[
E(\lambda)=2\kappa h_-h_++\frac{\sqrt2}{4}(h_-'h_++h_-h_+'),\qquad t=\frac12.
\]
A zero gives an \(H^2\) endpoint eigenfunction. Conversely, any such eigenfunction must lie in both decaying lines. The endpoint pencil is Fredholm of index zero by compactness of the decaying multiplier relative to the free resolvent. Solving the first-order inhomogeneous history equation with its endpoint value gives the pencil as the only obstruction to the bounded history resolvent. Consequently zero winding suffices to exclude the full open-right-half-plane spectrum, including for the specified history space.

For \(\Re\lambda\ge0\), the recurrence denominator has modulus at least \((n+1)(n+2)\), since \(\Re(2p)\ge\sqrt2>1\). To bound both spatial tails, replace \(2g(1\mp\varepsilon)\) and \(\pm4g\varepsilon\) in the numerator by \(C=2\sigma(1+\varepsilon)\) and \(S=4\sigma\varepsilon\). This yields a nonnegative rational recurrence majorant. If \(C+S\le2(N+1)\), the maximum \(B\) of its orders \(N-1,N\) bounds every later coefficient. The errors at \(t=1/2\) in value and \(t\)-derivative are bounded by \(B2^{-N}\) and \(B(N+2)2^{1-N}\). In the delay transform, the factor \(e^{-\tau\lambda}\) and the average series \(\sum_{j\ge0}(-d\lambda)^j/(j+1)!\) are enclosed by Taylor polynomials with geometric remainder bounds. The exponential uses scaling and squaring. All enclosures are outward rounded.

The checker subdivides the four contour edges until each Wronskian image box excludes zero and contains both approximated endpoint values. Convexity gives a homotopy between the exact contour image and the certified polygon, so exact oriented crossings give the winding. The saved image boxes and exterior norm margin cover the entire half-plane. The front's translation mode remains at \(\lambda=0\) in the independent scalar kink block. [Certificate implementation](../pdegen/d2_certificate.py)

## Reproduce

Extract the pinned original solver archive into a directory containing `run.py` and `benchmark.json`. The D2 side uses the bundled Jim solver. Use a fresh output directory.

~~~powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m pdegen.compare_a1_d2 --legacy-solver C:/path/to/spectral_solver --out scratch/a1-d2
~~~
