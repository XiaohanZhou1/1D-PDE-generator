# E3 verified stability at gains 5 and 6

The same certificate counts the open-right-half-plane spectrum for all four gains.

| Gain | Verified eigenvalue count, with algebraic multiplicity | Label | Lower bound on contour distance from zero, in the maximum norm |
|---:|---:|---|---:|
| 5 | 0 | STABLE | \(1.3112\times10^{-2}\) |
| 6 | 0 | STABLE | \(8.4899\times10^{-4}\) |
| 8 | 2 | UNSTABLE | \(5.4079\times10^{-2}\) |
| 12 | 2 | UNSTABLE | \(7.7524\times10^{-2}\) |

The four cases share the same E3 bulk equations, moving profile, feedback-state equations and perturbation space. Only the feedback gain changes. All four retain a simple neutral displacement mode. The unstable counts refer to a conjugate pair, since there are no positive real roots. The difficulty assessment remains pending. [Machine-readable manifest](../evidence/e3-verified/manifest.json) · [E3 model and spectral reduction](e3-dynamic-pilot.md)

## Full-domain reduction

Use the boundary-attached coordinates and the notation of E3. For \(\Re\lambda\ge0\) the triangular bulk Dirichlet problem has a bounded resolvent into the bulk domain. Eliminate the bulk response and the oscillator velocity state to obtain the characteristic
\[
F_g(\lambda)=\lambda^2+\tfrac14\lambda+1+gR(\lambda),
\qquad R(\lambda)=-p_\lambda'(0)-q_\lambda'(0).
\]
For \(\lambda\ne0\), recovery of the displacement adds an invertible scalar block. The resolvent elimination is analytic and bounded on the contour, up to the identifiable displacement factor at zero, which is removed. It reduces each spectral point in the open right half-plane to a zero of \(F_g\) with the same algebraic multiplicity. In particular there is no essential spectrum there. The stable zero-frequency bulk response is positive and \(F_g(0)>0\), giving a simple displacement zero.

The positive bulk resolvent estimate from E3 gives \(|R(\lambda)|\le1/5+2/7\) on the entire closed right half-plane. For \(|\lambda|\ge3\), the reverse triangle inequality yields
\[
|F_g(\lambda)|\ge|\lambda|^2-\tfrac14|\lambda|-1-g(1/5+2/7)>0
\]
for the four gains, with the smallest tail margin \(199/140\) at gain 12. All possible unstable zeros are enclosed by the counterclockwise rectangle \([0,3]+i[-3,3]\). Its imaginary-axis edge is certified below.

## Infinite-domain boundary response with a series tail bound

Write \(c=3/2\), \(\kappa=(c+\sqrt{c^2+4(1+\lambda)})/2\) and \(\nu=(-c+\sqrt{c^2+4(5/2+\lambda)})/2\). Use the square roots with positive real part. The decaying adjoint homogeneous solution of the first bulk equation is
\[
\phi(z)=e^{\kappa z}\sum_{n\ge0} a_n e^{nz/2},\qquad
a_0=1,\quad a_n=\frac{2a_{n-1}}{(n/2)(2\kappa-c+n/2)}.
\]
The recurrence follows by substitution in \(\phi''-c\phi'-(1+\lambda+2e^{z/2})\phi=0\). Its series and derivatives converge for \(z\le0\). The normalization \(\phi(0)\) is nonzero. A zero would give a nontrivial homogeneous Dirichlet solution in the original bulk equation after the conjugation \(e^{-cz}\phi\), contradicting bulk invertibility.

Set \(K_n=\kappa+n/2\). Inserting \(\phi\) and the analytic second-field response \(q_\lambda=(e^z-e^{\nu z})/\lambda\) into the Green identity for \(p_\lambda'(0)\) yields
\[
R(\lambda)=
\frac{\sum_{n\ge0}a_n\left[\tfrac1{2(K_n+1/2)}
+\tfrac1{(\nu+1+c)(K_n+1)(K_n+\nu)}\right]}
{\sum_{n\ge0}a_n}
+\frac1{\nu+1+c}.
\]
The formula is valid at zero by continuity. The difference of the two exponential integrals cancels \(\lambda\) using \(\nu-1=\lambda/(\nu+1+c)\).

On \(\Re\lambda\ge0\), \(\Re\kappa\ge2\), \(\Re\nu\ge1\) and \(|a_n|\le8^n/(n!)^2\). The bracket in the numerator has modulus at most 2. The interval evaluation keeps terms through \(N=20\), and bounds the discarded denominator and numerator tails respectively by \(T\) and \(2T\), with
\[
T=\frac{8^{N+1}}{((N+1)!)^2}\frac1{1-8/(N+2)^2}.
\]
This error accounts for the infinite spatial half-line directly. The finite spatial grids from the earlier numerical pilot play no role in the certificate.

## Interval enclosure and winding

The implementation uses a \(2^{-96}\) grid for real intervals with Python integers. Multiplication and reciprocal use floor and ceiling division, and square root uses integer square roots. The principal complex square root is assembled from the real operations. The calculation rounds the series-tail disk outward to a square in both coordinates. Every contour segment is bisected until one rectangular enclosure of its entire image excludes zero and contains the two approximate image endpoints used by the polygon. Convexity gives a homotopy to the polygon avoiding zero on each segment, including between sampled endpoints. Integer oriented crossings of the positive real ray then give the exact winding. The tail bound and analytic resolvent reduction turn the winding into a full-domain count. [Related verified Evans-function methodology](https://arxiv.org/abs/1601.00837)

Each certificate stores the endpoint coordinates, full image box, exact lower separation margin and computed count. The conservative lower bounds in the table are rounded downward. [Certificate implementation](../pdegen/e3_certificate.py)

~~~powershell
python -m pdegen.e3_certificate --out scratch/e3-verified
~~~
