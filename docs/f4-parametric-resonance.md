# F4 smooth parametric resonance

This extension adds a second Floquet mechanism to the batch generator. Take \(f(x)=2+\cos x+\sin(2x)/4\), \(r(t)=(\cos 2t,\sin 2t)^\top\), and \(X_*=fr\). The prescribed forcing \(fr'\) makes \(X_*\) an exact \(\pi\)-periodic solution. The linearized reaction uses a smoothly modulated stiffness
\[
k(t)=a+h\left(\cos(2t)+\tfrac14\sin(4t)\right)
\]
and equal diagonal damping \(\gamma=1/8\). The reviewed parameter rectangle is \(3/4\le a\le5/4\), \(1/4\le h\le3/4\), with exact rational inputs. F4 is recorded as its own construction group. The initial grid samples around the first parametric resonance.

## Full-line spectral criterion

Let \(L_f=\partial_{xx}-f''/f\) on \(H^2(\mathbb R)\). The function \(f\) is smooth, bounded away from zero, and periodic. The bounded real potential gives selfadjointness on this domain. For compactly supported test functions the ground-state form identity is
\[
\langle u,L_fu\rangle=-\int_{\mathbb R} f^2\left|(u/f)'\right|^2\,dx .
\]
Density gives \(L_f\le0\). The same identity on a periodic cell has \(\partial_x+i\xi\) in the squared term and holds uniformly on the entire Bloch zone. In addition, \(L_ff=0\) and cutoffs \(\eta(x/R)f\), normalized in \(L^2\), have residual \(O(R^{-1})\). Hence \(0\in\sigma(L_f)\) on the full line and \(1\in\sigma(e^{\pi L_f})\).

The scalar spatial generator acts identically on both components and commutes with every temporal reaction matrix. Uniqueness of the evolution gives the factorization
\[
M=e^{\pi L_f}\otimes P,\qquad P=e^{-\gamma\pi}\Phi(\pi),
\qquad
\Phi'=\begin{pmatrix}0&1\\-k(t)&0\end{pmatrix}\Phi,\quad \Phi(0)=I .
\]
The spectral-radius formula applied to powers gives \(r(M)\le r(P)\), since \(\|e^{n\pi L_f}\|\le1\). An eigenvector of \(P\) times the spatial cutoff vectors gives the corresponding approximate spectral point for \(M\). Thus \(r(M)=r(P)\). This proves the full-Bloch and full-line conclusions through the exact spatial factor.

Liouville's formula gives \(\det\Phi(\pi)=1\), and its trace \(T\) is real. For \(|T|\le2\), the undamped multipliers have modulus one. For \(|T|>2\), the larger modulus is the positive number \(m>1\) with \(m+m^{-1}=|T|\). The threshold for the damped monodromy is therefore \(|T|=2\cosh(\gamma\pi)\). The strict comparisons below and above it prove stability and instability respectively. An unresolved interval comparison is held as `UNCERTAIN`.

## Interval certificate

The checker encloses \(\pi\) by the existing Machin-series interval. It halves the angles before using outward-rounded real Taylor polynomials and recovers the original values by double-angle identities. Each local time step has length \(\pi/32<1/10\). The normalized fundamental matrix on a complex disk of radius \(1/2\) about the real step start has infinity norm at most \(e^3<27\). Indeed \(|\cos(2z)|\le e<3\), \(|\sin(4z)|\le e^2<9\), and the generator infinity norm is below 6 throughout the disk in the reviewed parameter range. Integration along radial lines supplies the stated bound.

If \(k_j\) are the local Taylor coefficients of the stiffness, the ODE recurrence for one column is
\[
x_{n+1}=\frac{y_n}{n+1},\qquad
y_{n+1}=-\frac{\sum_{j=0}^n k_jx_{n-j}}{n+1}.
\]
Through degree \(N=18\), Cauchy's estimate bounds each omitted matrix-entry tail by \(27(1/5)^{N+1}/(1-1/5)\). Outward-rounded products propagate all local enclosures to \(\pi\). A positive even Taylor series encloses \(2\cosh(\pi/8)\). The machine evidence stores both intervals and the positive separation margin whenever a label is accepted.

The proof step being assessed in a blind review is the temporal trace comparison for the specified two-harmonic coefficient, after deriving the spatial factor. F3 instead requires handling spatial couplings inside the switched monodromy. F4 has an exact spatial separation and a finite-dimensional temporal threshold. The extension's blind difficulty and resistance to short analytical estimates are pending independent review.

## Reproduce

~~~powershell
python -m pdegen.resumable_batch --plan examples/public-batch-plan.json --out scratch/public-batch --workers 3 --retries 1
~~~
