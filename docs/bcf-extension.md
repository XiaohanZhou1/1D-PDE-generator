# B3, C2 and F3 extension pilot

The unified generator now accepts the three exact-profile families below. The BCF preset exercises three B3, three C2 and four F3 examples. Its parameter points come from exploratory scouting and calibration. The results measure this preset.

| Family | Attempted | Reliable labels | Held as UNCERTAIN |
|---|---:|---|---:|
| B3 | 3 | 1 stable and 2 unstable | 0 |
| C2 | 3 | 1 stable and 2 unstable | 0 |
| F3 | 4 | 2 stable and 2 unstable | 0 |

All six unstable results are independently corroborated numerical witnesses. The two stable controls have full-domain analytic bounds. Blind difficulty review is pending for the new witnesses. These constructions extend the generator. Their proof novelty is unverified.

## B3

Set \(D=\operatorname{diag}(1,2)\) and \(J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}\). The stationary roll \(U=(\cos x,\sin x)^T\) solves \(X_t=D(X_{xx}+X)+(1-|X|^2)(bI+\chi J)X\), since both terms vanish. The Bloch operator is
\[
D[(\partial_x+i\xi)^2+1]-2((bI+\chi J)U)U^T
\]
on periodic \(H^2\) over a \(2\pi\) cell. The same periodic frame that simplifies the isotropic real Ginzburg–Landau example leaves variable diffusion here, since the diffusion and rotation do not commute. Positive nonreal witnesses for \((b,\chi)=(3,1),(4,2)\) occur at nonzero Bloch parameter, with respective real parts about \(0.07927\) and \(0.01421\). The co-periodic sampled maximum was at numerical zero in both cases. Fourier and twisted finite differences agree, with separate conditioning and residual checks. Real coefficients reflect negative fibers by conjugation. The spectrum of the periodic full-line operator comes from the Bloch fibers, and a positive margin persists in a neighborhood of an isolated witness. The sampled scan for \((6,2)\) supports further study and supplies no continuous-zone exclusion. [Isotropic comparison](https://doi.org/10.1137/18M1174751)

### B3 full Bloch certificate

The certificate proves a dissipative energy estimate on each fiber, including the translation mode at \(\xi=0\). Write \(A(\xi)=A_0+\xi A_1+\xi^2 A_2\) in the Fourier basis. The potential coefficients at \(0,2,-2\) are \(-M\), \(-M\begin{pmatrix}1&-i\\-i&-1\end{pmatrix}/2\), and its conjugate, where \(M=bI+\chi J\). The diagonal differential part at index \(n\) is \((1-(n+\xi)^2)D\). All coefficient matrices are exact rational complex matrices for the generated inputs.

Use an energy metric \(P(\xi)\) which is identity on modes \(|n|>N\) and preserves the core and tail subspaces. For \(H=-A^*P-PA\), its tail block has the uniform form bound
\[
H_{\mathrm{tail},\mathrm{tail}}\ \ge\
2\bigl[(N+\tfrac12)^2-1-2(b+\chi)\bigr] I.
\]
This follows from \(D\ge I\) and the potential multiplication norm bound \(2(b+\chi)\). Only two layers of tail modes can couple to the core. Keep the exact core matrix and cross terms, and replace the interacting tail block by the lower bound times identity. A positive finite matrix gives the energy inequality for all modes, since the remaining tail contributes the same lower bound and has zero core coupling.

At zero, \(k=2U'\) and \(w=Mk/b\) are exact right and left nullvectors, with \(\langle w,k\rangle=4\). For the left identity, \(M^*w\) is a scalar multiple of \(k\), perpendicular to \(U\) pointwise, and the Fourier differential part vanishes on modes \(\pm1\). The affine metric \(P_0+\xi P_1\) is built with \(P_0 k=w\). The zeroth coefficient of the finite dissipative matrix annihilates \(k\); the coefficient of \(\xi\) on the kernel direction has zero quadratic form since differentiation in \(\xi\) gives a purely imaginary pairing of the two real functions. In a basis whose first coordinate is the translation direction, the first diagonal block is therefore divisible by \(\xi^2\) and the first cross blocks by \(\xi\). Both cancellations are checked exactly in realification. Dividing by these factors extends the scaled energy matrix continuously to zero. Positivity of this scaled matrix proves dissipation for positive small \(\xi\), and the unscaled form is nonnegative at zero.

The rest of \([0,1/2]\) is covered by adjoining closed rational subintervals, each using a constant core metric. Floating-point Lyapunov solves propose the metrics. Their rounded entries and the Bernstein coefficient matrices of the energy inequality are checked with rational arithmetic. For each matrix \(B\), a rounded square whitening matrix \(C\) proposes a congruence. The checker requires \(C^\mathsf T B C\) to be strictly diagonally dominant with positive diagonal using integer products. This implies positivity of \(B\), and Bernstein convexity transfers it to the whole subinterval. Checks also establish positivity of the metrics themselves, including both endpoints of the affine metric. A failed check leaves the case unresolved if the scan supplies no acceptable instability witness.

Conjugation covers \([-1/2,0]\). In each fiber, the constant principal Fourier operator has compact resolvent and the potential is bounded. The open right half-plane is Fredholm of index zero, and dissipativity excludes kernels there. In the energy norm, the resolvent bound is \(1/\Re\lambda\). The metrics are uniformly equivalent to the original norm because the cover is finite and the near-zero metric is continuous and positive on its closed interval. The direct integral therefore excludes the full right half-plane spectrum on the line.

The nine previously unresolved B3 cases all pass this certificate at cutoff \(N=6\). Each near-zero segment is \([0,1/8]\). Each remaining cover partitions \([1/8,1/2]\) into 7 to 34 closed subintervals. The prescribed neutral translation mode remains at zero.

## C2

The stationary target \((U,0)\), \(U=\sqrt2\operatorname{sech}x\), solves the two-field NLS equations in the rotating frame because \(-U''+U-U^3=0\). Their coupling is generated by the terms \(-\alpha|\psi|^2|\chi|^2-\tfrac\beta2(\chi^2\bar\psi^2+\bar\chi^2\psi^2)\) in the Hamiltonian density. At the target the linearization consists of a scalar NLS block and a polarization block. For complexified real and imaginary perturbations that second block has generator
\[
(v,w)\ \longmapsto\ (H_-w,-H_+v),\qquad
H_\pm=-\partial_x^2+1-(\alpha\pm\beta)U^2.
\]
The conjugate coupling makes the two Schrödinger operators distinct. The cases \((2,1/2),(5/2,1)\) give growing nonreal modes with real parts about \(0.06630\) and \(0.19728\). Real and Hamiltonian symmetries give the quartet. The witness is checked using Dirichlet finite differences on expanding domains, monitoring localization, and continuous half-line exterior shooting with even or odd conditions at zero. The count of other spectral points remains open. The related coupled NLS literature supplies methods for Hamiltonian spectral problems. Application to this specific conjugate coupling remains part of the difficulty review. [Coupled NLS comparison](https://www.sciencedirect.com/science/article/abs/pii/S0022039623004618)

For the stable control \((1/4,1/8)\), \(H_\pm\ge1/4\), so the quadratic Hamiltonian form excludes any open-right-half-plane eigenvalue in the polarization block. For the active scalar block, \(L_-=-\partial_x^2+1-U^2\) is nonnegative with kernel \(U\). The operator \(L_+=-\partial_x^2+1-3U^2\) has one negative eigenvalue and kernel spanned by \(U'\), by the nodal count of \(U'\). Differentiating the scalar soliton family \(U_\omega=\sqrt{2\omega}\operatorname{sech}(\sqrt\omega x)\) gives
\(\langle L_+^{-1}U,U\rangle=-\partial_\omega\|U_\omega\|_2^2/2|_{\omega=1}=-1\), with the inverse on the orthogonal complement of \(U'\). Thus \(L_+\)'s form is nonnegative on \(U^\perp\) with radical \(U'\), by removal of the single negative direction \(L_+^{-1}U\). If \(\Re\lambda>0\), the first component of a scalar eigenvector is in \(U^\perp\) by \(L_-U=0\). The Hamiltonian energy identity forces the constrained quadratic form to vanish, placing both components in their kernels and contradicting \(\lambda\ne0\). The far-field operators are constant skew Fourier multipliers. The decaying zeroth-order perturbations are compact from \(H^2\) into \(L^2\). The open right half-plane therefore has Fredholm index zero, and the eigenvalue exclusion proves the full-domain stability claim.

### C2 exact spectral checks

Let \(H_d=-\partial_x^2+1-2d\operatorname{sech}^2x\), and let \(d_\pm=\alpha\pm\beta\). The C2 polarization generator is \(J\mathcal K\), where \(J(v,w)=(w,-v)\) and \(\mathcal K=\operatorname{diag}(H_{d_+},H_{d_-})\). Its far-field operator is skew-adjoint, and the decaying potential is relatively compact by local Rellich compactness and a tail cutoff. Thus in the open right half-plane the operator minus \(\lambda\) is Fredholm of index zero. It suffices to test its kernel for stability.

The strictly positive function \(\operatorname{sech}x\) solves \(H_1 f=0\). Factoring \(H_1=(\partial_x+\tanh x)^*(\partial_x+\tanh x)\) gives \(H_d\ge0\) for \(d\le1\). For \(d_+\le1\), both diagonal blocks of \(\mathcal K\) are nonnegative. An eigenvector \(v\) with \(\Re\lambda>0\) has \(\langle \mathcal K v,v\rangle=0\) by the Hamiltonian identity, so \(\mathcal K v=0\), contradicting \(J\mathcal K v=\lambda v\). This includes \(d_+=1\). For \((d_-,d_+)=(1,3)\), the polarization block is identical to the scalar NLS block and the constrained proof above applies to both.

For even functions set \(d_*=1\), and for odd functions set \(d_*=3\). In the even sector the threshold mode is \(\operatorname{sech}x\). In the odd sector \(\operatorname{sech}x\tanh x\) solves \(H_3 f=0\) and is positive on the open half-line. The half-line Dirichlet ground-state identity gives \(H_3\ge0\) in that sector. If \(d_-<d_*<d_+\), comparison with the threshold gives \(P=H_{d_-}>0\) with a spectral gap in the chosen sector. Indeed its essential spectrum starts at 1 and its form is strictly positive on nonzero arguments. The same threshold function gives a negative expectation for \(H_{d_+}\).

On that parity sector, \(T(t)=H_{d_+}+tP^{-1}\) is selfadjoint on the common \(H^2\) domain for real \(t\ge0\). Its spectral infimum is continuous, by the bound \(|t-s|\|P^{-1}\|\) on the perturbation. At zero it is negative. For \(t=(2\beta+1)^2\), the inequality \(P+tP^{-1}\ge2\sqrt{t}\) and \(H_{d_+}=P-4\beta\operatorname{sech}^2x\) give \(T(t)\ge2\). Its essential spectrum is bounded below by 1 throughout, since the decaying potentials and the resolvent difference \(P^{-1}-(-\partial_x^2+1)^{-1}\) are relatively compact in the parity sector. At some intermediate \(t>0\) the bottom is zero, so there is a nonzero \(p\in H^2\) in the kernel. Then \(\lambda=\sqrt t>0\) and \((p,\lambda P^{-1}p)\) is a polarization eigenmode. This proves instability without assigning a total spectral count.

### C2 oscillatory search and rescan

The scout retains the original frequency shifts and expands to shifts covering smaller growth rates and additional frequencies. Candidate roots are independently continued in the exact-profile half-line exterior shooting problem at two truncation lengths. Physical finite differences use two spatial steps and extend the domain according to the slowest far-field decay rate of the continued mode, with recorded domain caps. Acceptance still requires real and imaginary parts above \(0.001\) and ten times the observed discrepancy, discrepancy below \(0.005\), fine residual below \(10^{-7}\), and fine tail norm below \(0.04\). Parity must also agree. These are numerical cross-checks. The scout is finite, and a miss outside an exact certificate still gives an uncertain result.

On the 15 unresolved C2 cases from the 100-case batch, the updated procedure admitted 15 labels. Eight stable labels have exact exclusion proofs, three unstable labels have positive-real-eigenvalue proofs, and four unstable labels have corroborated oscillatory numerical witnesses. Including the previously resolved calibration, C2 moves from 1/16 to 16/16 in this fixed sample. The historical batch records are preserved.

## F3

Use the prescribed orbit \(X_*=f(x)(\cos t,\sin t)^T,\ f=2+\cos x\), rotating by \(R(t)\). The generator adds two space-dependent shear terms, active on alternating half-periods, multiplying \(X-X_*\). They vanish on the exact orbit and are allowed to switch at half-periods. The orbit is phase-pinned by time-dependent coefficients. There is no neutral time-shift degree of freedom required by the problem.

In the rotating frame both half-periods have diagonal bulk \(A_\xi=(\partial_x+i\xi)^2-f''/f-1/4\). Their off-diagonal entries switch between the upper entry \(g(1/2+\cos x)\) and the lower entry \(g(1/2+\sin x)\). Each frozen generator in this frame is triangular and has spectrum in \(\Re\lambda\le-1/4\). The noncommuting product gives a positive Floquet witness for each of \(g=1,5/4\), with multiplier moduli about \(1.35551\) and \(1.91179\). Fourier and weighted physical differences independently discretize the monodromy. The witnesses occur in the zero Bloch fiber, also with positive growth on nearby sampled fibers. A sustained positive margin on an isolated fiber eigenvalue suffices for a full-line instability witness through the Bloch decomposition. [Floquet for periodic PDEs](https://doi.org/10.1070/IM1983v021n01ABEH001642)

For the \(g=1/4\) control, the ground-state identity for \((\partial_x+i\xi)^2-f''/f\) makes its quadratic form nonpositive for every fiber. The Hermitian part of either shear is bounded above by \(3g/4\). The resulting uniform decay rate \(1/4-3g/4=1/16\) bounds the full monodromy norm by \(e^{-\pi/8}<1\). The \(g=3/4\) case remains uncertain onward from the sampled scan. The dedicated solver uses the two segment propagators. The existing native F2 entry targets its own sampled PDE and remains separate.

### F3 full Bloch Floquet certificate

Write \(T_\xi=-i\partial_x+\xi\), \(V=\cos x/(2+\cos x)\) and \(A_\xi=-T_\xi^2+V-1/4\). The ground-state form identity with \(f=2+\cos x\) gives \(A_\xi\le-1/4\) for every real \(\xi\), including its Fourier compressions. In the rotating frame, each half-period generator is triangular with both diagonal entries \(A_\xi\) and one shear \(g(1/2+\cos x)\) or \(g(1/2+\sin x)\). The rotation is identity again at \(2\pi\). For complex \(\xi=u+iv\), the additional Hermitian part of the scalar bulk relative to \(A_u\) is \(v^2 I\). With \(s=3g/2\), the triangular evolution over time \(t\) therefore has operator norm at most \(e^{(-1/4+v^2)t}(1+st)\) by Duhamel's formula.

Project onto Fourier modes \(|n|\le N\), on each component. The coefficients of \(V\) at mode \(n\) are \(\delta_{n0}-(2/\sqrt3)(\sqrt3-2)^{|n|}\). The scalar core–tail coupling has norm at most \(\sqrt2/3<1/2\): the squared Frobenius sum at each edge is bounded by the square of \((2/\sqrt3)(2-\sqrt3)/(1-(2-\sqrt3)^2)=1/3\). The shear crosses each edge only at one pair of adjacent modes, with norm \(g/2\). Both directions of the total coupling are consequently bounded by \(b=(1+g)/2\). In the tail, the numerical range is bounded to the left of \(-a\), where \(a=(N+1/2)^2-1/12-s/2\), from the Fourier diagonal and \(V-1/4\le1/12\). This gives the high-mode semigroup estimate \(e^{-at}\) uniformly over \(|\xi|\le1/2\).

Put \(d=a-1/4\) and \(m(t)=e^{-t/4}(1+st)\). For one half-period, Duhamel's formula bounds the full and compressed core–core difference by
\[
\alpha(t)=\frac{b^2}{d}e^{-t/4}\left(t+st^2+\frac{s^2t^3}{6}\right).
\]
The two cross blocks are bounded by \(\delta(t)=b m(t)/d\). The tail–tail block is bounded by \(\gamma(t)=e^{-at}+(b/d)^2m(t)\). For example, a core-to-tail excursion costs one factor \(b/d\), by integrating the tail decay against \(m(t)\). In the core–core difference the return costs another \(b\); integrating \(m(t-r)m(r)\) gives the polynomial in \(\alpha(t)\). Applying the same cross bound to the adjoint controls tail-to-core evolution. In the tail–tail block, the direct tail evolution contributes \(e^{-at}\), and a core excursion gives the remaining term. The implementation bounds these quantities rationally at \(t=\pi\), using an enclosed \(\pi\) and the elementary exponential estimates in the checker.

For the product of the two segments, let \(f_1,f_2\) bound the two compressed segment norms on the Bloch subinterval. The exact infinite monodromy differs from the compressed monodromy, extended by zero, by at most
\[
\delta(\max(f_1,f_2)+\alpha+\gamma)
 + \max\bigl(\alpha(f_1+f_2)+\alpha^2+\delta^2,\ \delta^2+\gamma^2\bigr).
\]
This is the maximum row/column-sum bound for the matrix of operator block norms. It accounts for all Fourier modes.

At rational Bloch nodes, the compressed segment maps are enclosed by a scaled Taylor polynomial followed by squaring in fixed-point arithmetic. The checker encloses \(\sqrt3\) and \(\pi\), the latter by Machin's identity and alternating rational sums. It computes every rounded matrix product using exact signed integer arithmetic and accounts for all per-entry rounding in Frobenius bounds. The Taylor step has norm at most \(1/2\), with a geometric remainder bound. The squaring error is propagated against the analytic triangular norm estimate. The same routine bounds the errors in the assembled monodromy and its powers.

To cover a closed subinterval of width \(h\), apply Cauchy's estimate on complex disks of radius \(3/8\) to the finite segment maps and their product. If \(m\) bounds \(m(\pi)\), then their second derivatives have bounds \((128/3)m\) and \((128/3)m^2\), using \(e^{2\pi(3/8)^2}<3\). Multiply these by \(h^2/8\) for linear interpolation. For each \(\xi\), the distance to the nearer endpoint's rounded compressed monodromy is bounded by the spatial tail error, the interpolation error, half the difference of the endpoint matrices, and the rounded matrix error. With this total error \(\epsilon\), a bound \(B\) for the endpoint matrix and a verified upper bound \(B_p\) for its \(p\)-th power, the full monodromy obeys
\[
\|M_\xi^p\|\le B_p+(B+\epsilon)^p-B^p.
\]
The checker requires this to be strictly below one, for some \(1\le p\le5\) on each subinterval. The resulting finite cover of \([0,1/2]\) extends to negative frequencies by conjugation. The power inequalities on the finite cover give uniform resolvent bounds for \(|z|\ge1\) by a Neumann series in \(M_\xi^p/z^p\). Thus the direct-integral monodromy has its spectrum strictly inside the unit disk.

All seven formerly unresolved F3 cases passed, and the existing small-gain calibration remains covered by the elementary contraction estimate. [Certificate implementation](../pdegen/f3_certificate.py)

## Blinded artifacts and reproduction

The preset emits ten label-free prompts and one internal manifest. The prompts each ask about the recorded spectral domain. The F3 prompts ask about the unit disk for a temporal monodromy. Share prompts individually without the internal files. The three constructors also accept rows in the unified JSON plan. [Batch interface](batch-workflow.md)

~~~powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m pdegen.batch_workflow --preset bcf --out scratch/bcf-extension
~~~
