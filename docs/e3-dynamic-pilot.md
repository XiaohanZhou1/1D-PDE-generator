# E3 dynamic-interface pilot

E3 uses the E2 bulk PDE with \(\gamma=1\) and retains the exact advancing half-line wave \(U=e^{z/2}, V=e^z\), \(z=x-3t/2\le0\). Replace the instantaneous interface law with two real states \(a,b\):
\[
s'=3/2+a,\qquad a'=b,\qquad
b'=-a-2d b+g[(u_x-1/2)+(v_x-1)],\quad d=1/8.
\]
Here both derivatives are evaluated at the interface. The exact solution has \(a=b=0\) and \(u=v=1\) there. The perturbation space is complex unweighted \(L^2((-\infty,0);\mathbb C^2)\times\mathbb C^3\), with homogeneous Dirichlet \(H^2\) field domains and three scalar states for displacement and the two feedback variables.

Let \(L_u=-\partial_z^2-c\partial_z+1+2U\), \(L_v=-\partial_z^2-c\partial_z+5/2\), \(c=3/2\), and let \(L\) have diagonal \(L_u,L_v\) and upper coupling \(-1\). For unit interface velocity define \(w_\lambda=(p_\lambda,q_\lambda)=(\lambda+L)^{-1}(U',V')\). Set \(\ell(w)=-p'(0)-q'(0)\) and \(R(\lambda)=\ell(w_\lambda)\). The characteristic scalar is
\[
P(\lambda)+gR(\lambda),\qquad P(\lambda)=\lambda^2+2d\lambda+1.
\]
The two diagonal Dirichlet operators are coercive for \(\Re\lambda\ge0\); the triangular bulk resolvent is bounded into the field domain there. Elimination of the velocity state gives the scalar characteristic above. If the scalar is nonzero, reconstruct the bulk fields and oscillator states for arbitrary forcing, then reconstruct the displacement for \(\lambda\ne0\). A zero in the open right half-plane corresponds to an \(L^2\) eigenmode. The displacement alone contributes a neutral zero. Feedback dynamics make the characteristic frequency dependent. The bundled Jim solver's instantaneous interface law does not supply this oscillator. The E3 backend implements the specified bulk–boundary system directly.

## Stable control

At real \(\lambda\ge0\), the bulk resolvent preserves positivity. For complex \(\lambda=x+iy\), \(x\ge0\), the positive bulk evolution gives
\[
|(\lambda+L)^{-1}f|\le(x+L)^{-1}|f|\le L^{-1}|f|
\]
componentwise. This follows from the Laplace representation of the resolvent. The two Dirichlet diffusion evolutions preserve positivity and have an exponential bound; the triangular off-diagonal Duhamel integral is positive and retains exponential decay. For Dirichlet responses, the same pointwise comparison implies the corresponding one-sided derivative bound at zero.

Writing \(w_0=(p_0,q_0)\), one has \(q_0=-ze^z/(2+c)\) and \(0\le p_0\le -ze^{z/2}/5\) by the E2 supersolution comparison. Thus \(|R(\lambda)|\le R(0)\le 1/5+2/7\). On the closed right half-plane \(|P(\lambda)|\ge 2d\sqrt{1-d^2}\). To see this, factor \(P\) at \(-d\pm i\sqrt{1-d^2}\), minimize first over the real part at zero, then minimize the product over the imaginary part. At \(g=1/2\) the strict small-gain inequality follows, giving the stable calibration.

## Stable certificate beyond small gain

For \(y\ne0\), the resolvent identity and the same positivity bounds yield
\[
|\Im R(x+iy)|\le |R(x+iy)-R(x)|\le |y|\,\ell(L^{-2}(U',V')).
\]
Indeed, apply the resolvent identity to the forcing, dominate both factors by \(L^{-1}\), then compare the boundary derivatives of the resulting Dirichlet solutions. Let \((p_1,q_1)=L^{-2}(U',V')\). Put \(A=5/2\), \(B=7/2\) and \(m=1/2\). Then
\[
q_1=e^z(z^2/(2B^2)-z/B^3),\qquad
p_{1,*}=e^{mz}(mz^2/(2A^2)-mz/A^3).
\]
The free \(u\) operator applied to \(p_{1,*}\) is \(-mz e^{mz}/A\), the preceding supersolution for \(p_0\). The remaining contribution \(2U p_{1,*}\) dominates \(q_1\) coefficientwise in \(z^2\) and \(-z\), so \(0\le p_1\le p_{1,*}\). It follows that \(\ell(p_1,q_1)\le4/125+8/343\). If \(g(4/125+8/343)<2d\), the imaginary part of \(P+gR\) has the sign of \(y\). On the nonnegative real axis \(P>0\) and \(R>0\), so there are no zeros there for any positive gain. This excludes open-right-half-plane spectrum for \(g=1,4\), even though the simpler small-gain modulus bound is insufficient for these gains. The established bound also prevents a second zero at the displacement mode.

## Numerical pilot

A physical second-order finite difference solve of the two Dirichlet responses and an independent continuous adjoint Green identity give two approximations to the same scalar characteristic. The continuous method uses an analytic second-field response and computes the first-field trace by integration against the decaying adjoint solution. Both methods refine spatial truncation and resolution. The sampled zero is tracked from the free oscillator. The real axis exclusion above means any positive witness is oscillatory. A tracked negative real part supplies no full-domain stability certificate.

| Gain | Label | Difficulty disposition |
|---:|---|---|
| 1/2 | STABLE | Small-gain calibration |
| 1, 4 | STABLE | Analytic trace-moment argument found |
| 5, 6 | UNCERTAIN | Tracked pair alone does not cover the spectrum |
| 8, 12 | UNSTABLE | Preliminary pass; blind review pending |

Label yield is 5/7, including three stable cases. The two joint passes belong to one mechanism. The exploratory grid gives no population-level yield. The three stable cases are useful for validation and are not counted as difficult training candidates. The numerical checks do not assert a complete count or a bifurcation theorem.

~~~powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m pdegen.e3_pilot --out scratch/e3-dynamic
~~~
