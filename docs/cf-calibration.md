# C–F continuum spectral calibration

This pilot uses six fixed calibration cases. A stable result requires a proof excluding unstable spectrum on the entire operator or monodromy domain. Each claim has its own state space and neutral-mode convention. The certificates apply to the canonical specifications only. Training admission remains on hold because the mechanisms below are known or elementary.

Run from the repository root in the environment defined by requirements.txt.

~~~powershell
$env:OPENBLAS_NUM_THREADS = '1'
foreach ($family in 'C1','D1','E1','F1') {
    python -m pdegen.workflow --family $family --native-check --out "scratch/$family"
}
python -m unittest discover -s tests -v
~~~

The optional native result is diagnostic evidence. It retains the solver's own scope, qualifier and reason. The bundled solver's finite contours and numerical passes have no role in the continuum exclusion arguments below. F1 uses a separate two-dimensional monodromy integration at sampled spatial frequencies. [Reference evidence](../evidence/cf/)

## C1 — Modified KdV

For \(u_t+u^2u_x+u_{xxx}=0\), the positive soliton
\(U_c=\sqrt{6c}\operatorname{sech}(\sqrt c\,z)\) solves
\(-U_c''+cU_c-U_c^3/3=0\). The moving-frame linearisation is \(A=\partial_z L\), with \(L=-\partial_z^2+c-U_c^2\), on unweighted \(L^2(\mathbb R)\) with domain \(H^3\). The free Fourier multiplier \(ik(k^2+c)\) has spectrum on the imaginary axis and its variable-coefficient perturbation is compact from \(H^3\) to \(L^2\) by decay and local compactness. Thus every point in the open right half-plane is Fredholm of index zero and can be excluded by ruling out eigenfunctions.

The self-adjoint \(L\) has simple kernel spanned by \(U_c'\). This eigenfunction has one zero and the essential spectrum starts at \(c>0\), so Sturm oscillation gives one negative direction. Take \(h=-\partial_c U_c\). Then \(Lh=U_c\) and
\[
\langle h,Lh\rangle=-\frac{d}{dc}\frac12\|U_c\|_2^2=-3/\sqrt c<0.
\]
On \((U_c')^\perp\), the constraint \(v\perp U_c\) is the \(L\)-form orthogonality condition to the strictly negative \(h\). The index-one form is positive on that complement. More explicitly, write \(h=a e_-+h_+\) for the normalized negative eigenfunction with eigenvalue \(-\nu\), and the positive spectral subspace. Then \(\|L^{1/2}h_+\|^2<\nu|a|^2\). For \(v=b e_-+v_+\) in the complement, Cauchy–Schwarz in the positive form implies \(\nu|b|^2\le\|L^{1/2}v_+\|^2\|L^{1/2}h_+\|^2/(\nu|a|^2)\), proving positivity there. Restoring \(U_c'\) adds just the translation radical.

Since \(A^*U_c=0\), an eigenfunction with \(\lambda\ne0\) obeys \(v\perp U_c\). The identity \(\Re\langle Lv,\partial_z Lv\rangle=0\) forces the form of such an eigenfunction to vanish if \(\Re\lambda>0\). This would put it in the translation kernel, a contradiction. The speed derivative also satisfies \(A\partial_cU_c=-U_c'\), so the proof treats the neutral origin as a generalized mode. The reference case uses \(c=1\). This yields open-right-half-plane exclusion with imaginary-axis essential spectrum. [Pego and Weinstein, 1992](https://doi.org/10.1098/rsta.1992.0055)

## D1 — Uniform distributed delay

For the stationary kink \(U=\tanh(x/\sqrt2)\), \(v=0\), the linearisation is block diagonal. The Allen–Cahn block is \(A_u=\partial_x^2+1-3U^2\). Its essential spectrum is bounded above by \(-2\). The strictly positive \(U'\) spans the simple eigenfunction at zero and Sturm oscillation bounds the rest below zero. The delayed block has state space \(C([-\tau,0];L^2(\mathbb R))\) and current diffusion operator \(\partial_x^2-1\) with spatial domain \(H^2\). Put \(S=1-U^2=\operatorname{sech}^2(x/\sqrt2)\) and \(H(\lambda)=\tau^{-1}\int_0^\tau e^{-\lambda s}\,ds\). For a history resolvent equation with continuous right hand side \(g\), solve \(\lambda\phi-\phi'=g\) on \([-\tau,0]\) with unknown \(v=\phi(0)\). The resulting history is \(e^{\lambda\theta}v+\int_\theta^0 e^{\lambda(\theta-s)}g(s)\,ds\). The generator compatibility at zero gives the Schur complement
\[
\lambda-\partial_x^2+1-\sigma S H(\lambda)
\]
on \(v\), with a bounded forcing map from \(g\) for each fixed \(\lambda\). Inverting the complement and substituting into the history formula gives a bounded full resolvent.

For \(\Re\lambda\ge0\), \(|H(\lambda)|\le 1\), and the free resolvent norm is at most \(1/(1+\Re\lambda)\). With \(\sigma=1/2\) the Neumann series converges uniformly in the closed right half-plane. Thus the full linearisation has no open-right-half-plane spectrum.

For \(\sigma=6\) take \(\lambda\) real and let \(b(\lambda)\) be the supremum of the spectrum of \(\partial_x^2-1+6S H(\lambda)-\lambda\). By the variational formula and continuity of \(H\), \(b\) is continuous on \([0,5]\). At zero choose the test function \(S\). The identity \(S''=2S-3S^2\) gives \((\partial_x^2-1+6S)S=S+3S^2\), so \(b(0)>0\). At five, \(b(5)\le -6+6H(5)<0\). At a zero of \(b\) in \((0,5)\), the essential spectrum lies strictly below zero because the potential decays. A kernel vector gives an eigenhistory \(e^{\lambda\theta}v\). This argument proves existence of a growing mode. It leaves the total unstable count open.

## E1 — Moving boundary

For \(u_t=u_{xx}-u\) on \(x<s(t)\), \(u(s(t),t)=1\) and \(s'=\kappa(u_x(s(t),t)-1)\), use \(z=x-s(t)\). A base profile \(U=e^{mz}\) with \(s=ct\) satisfies \(m^2+cm-1=0\) and \(c=\kappa(m-1)\). At \(\kappa=-3\) the two cases are \((m,c)=(1/2,3/2)\) and \((1,0)\). In the attached frame,
\[
\lambda p=p''+cp'-p+\kappa U'p'(0),\qquad p(0)=0.
\]
The interface displacement evolves by \(\eta'=\kappa p'(0)\). This triangular extension adds the neutral constant displacement and leaves all \(\Re\lambda>0\) tests unchanged.

The Dirichlet generator \(\partial_z^2+c\partial_z-1\) has a resolvent for \(\Re\lambda>0\). This follows, for instance, from the coercive Dirichlet form with real part at least \(\|p'\|^2+(1+\Re\lambda)\|p\|^2\), followed by \(H^2\) regularity. The perturbation is rank one from the Dirichlet domain \(H^2\cap H^1_0\) to \(L^2\), since derivative trace is bounded there. Therefore only the scalar resolvent denominator can obstruct inversion there. The only spatial root \(\mu\) of \(\mu^2+c\mu-1=\lambda\) with \(\Re\mu>0\) enters the Dirichlet solution
\[
p=\frac{\kappa m p'(0)}{\lambda}\big(e^{mz}-e^{\mu z}\big).
\]
The derivative trace condition reduces to \(1=-\kappa m/(\mu+m+c)\). It forces \(\mu=-1/2\) for the advancing branch, which is inadmissible. The standing branch gives \(\mu=2\), hence \(\lambda=3\). The scalar denominator has a simple zero there since \(\mu'(\lambda)=1/(2\mu+c)\ne0\). Hence the standing branch has exactly one simple unstable eigenvalue.

## F1 — Temporal Floquet

Take \(X_t=D X_{xx}+JX+(1-|X|^2)X\), where \(D=\operatorname{diag}(1,2)\), and \(J\) is the standard real skew-symmetric rotation generator. The spatially homogeneous orbit \(X(t)=(\cos t,\sin t)^T\) has period \(T=2\pi\). Its variational equation on Fourier fiber \(k\in\mathbb R\) has matrix \(J-2X(t)X(t)^T-k^2D\). For every complex perturbation,
\[
\frac12\frac{d}{dt}|Y|^2=-2|X(t)^T Y|^2-k^2Y^*DY\le -k^2|Y|^2.
\]
Therefore \(\|M(k)\|\le e^{-Tk^2}\) for every real \(k\). Plancherel gives operator norm at most one for the full-line monodromy and a geometric resolvent for every multiplier of modulus greater than one. At \(k=0\) the rotating coordinate frame gives one radial multiplier \(e^{-2T}\) and one tangential multiplier 1. The tangential motion is a neutral fiber mode. Its spatially constant eigenvector is outside the full-line \(L^2\) space, and nearby fibers approach the unit modulus. The certificate concerns spectral exclusion outside the unit disk. The numerical integration exercises the monodromy convention and time phase at two tolerances.

## Scope of this pilot

C1 covers a solitary wave and its Hamiltonian constraint. D1 covers bounded uniform memory on a finite history interval. E1 covers two branches of the same linear one-phase Stefan model and retains the displacement convention. F1 covers temporal Floquet on a spatially homogeneous orbit and uses a uniform energy estimate over all Fourier frequencies. Generalizations to Krein collisions, delay-driven complex crossings, nonlinear interface coupling, or spatially structured time-periodic orbits require new proofs and solver capabilities before admission.
