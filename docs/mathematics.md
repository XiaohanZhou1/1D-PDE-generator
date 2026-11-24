# Mathematical construction and proof review

## The A1 template

Use the real state $X=(u,y,v,w)^T$, laboratory coordinate $x$, comoving coordinate $z=x-t$, and the full real line. Set $R=v-u^2,\ S=w-u^3,\ P_j(u)=j^2u^j-j(j+1)u^{j+2}$. The source equations, including the row-two derivative, give

$$
 X_t=DX_{xx}-AX_x+F(X)+\partial_x G(X),\qquad G=(0,6u^3,0,0)^T,
$$
$$
 D=\operatorname{diag}(1,3,1,1+q/2),\qquad A=\operatorname{diag}(1+q/2,1,1-q/2,1+q).
$$
$$
 F=\begin{pmatrix}
 -P_1(u)+qy/2+qu(3R+2S)\\
 -3y+\beta u^2R\\
 -P_2(u)-quy+(6u^2-a)R-2S\\
 -(1+q/2)P_3(u)+3qu^2y+(6u^2-a+qu)S+2R
 \end{pmatrix}.
$$

The original design has $\beta=0$, $0\le q\le1,\ 2\le a\le6$. The regression seed is $(0,4,0)$. The optional structural template takes $\beta=\pm1/4$. These are conservative pilot choices, not an asserted maximal range. Changing $q$ moves diffusion, drift and reaction together. The coefficient of $P_3$ compensates the fourth diffusivity. The $y,uy,u^2y$ terms compensate comoving transport. Changing $a$ only damps the two normal defects.

### Target identity and admissibility

Let $s=\operatorname{sech}z,\ t=\tanh z$. Then $s'=-st,\ t'=s^2,\ s^2+t^2=1$ and $(s^j)''=P_j(s)$. The target $U=(s,s',s^2,s^3)^T$ has both endstates zero, speed one and unique scalar maximum $u(0)=1$. This pin fixes translation on the designated homoclinic branch. It makes no uniqueness assertion about other waves of the same PDE.

The precise identity is $DU''+(I-A)U'+F(U)+[G(U)]'=0$. On the wave $R=S=0$. Rows one, three and four cancel by the power identity and $(s^j)'=js^{j-1}s'$. Row two reduces to $3s'''-3s'+18s^2s'=0$. All feedbacks are autonomous state polynomials, including at both pulse tails. For $g=bR$, $J_g(U)=b(U)\nabla R(U)^T$, hence $J_g(U)U'=0$. Here the additional row is $\beta s^2(p_v-2sp_u)$. It acts on normal perturbations and vanishes at infinity. This establishes preservation of the target and a change of the operator. Its spectral effect remains open.

The implementation performs exact rational polynomial differentiation and substitution. It reduces monomials to degree at most one in $t$ through $t^2=1-s^2$. A zero remainder proves an identity for every real $z$. A sampled small residual alone would give weaker evidence. Every diffusivity is at least one and all on-wave coefficients and the derivatives used below are bounded. The closed whole-line realisation acts in complexified, unweighted $L^2(\mathbb R)^4$ with domain $H^2(\mathbb R)^4$. Uniform parabolicity and bounded lower-order coefficients suffice here for the linear realisation. Nonlinear global existence for arbitrary initial data is outside this check.

### Linearisation and the far field

For constant diagonal $D,A$ and speed $c$, differentiation gives
$$
 Lp=Dp''+Bp'+Cp,\qquad B=cI-A+J_G(U),\quad C=J_F(U)+[J_G(U)]'.
$$
Thus row two contains **both** $18s^2 p'_u$ and $36ss'p_u$. The exact translation test is $LU'=0$. The wire record contains every coefficient of both matrices. For $Y=(p,p')$ the eight-dimensional spatial system has upper blocks $0,I$, lower blocks $D^{-1}(\lambda I-C),-D^{-1}B$.

At either end $B_\infty=\mathrm{diag}(-q/2,0,q/2,-q)$. The upper $2\times2$ reaction block has rows $(-1,q/2),(0,-3)$. The lower block has rows $(-a,-2),(2,-a)$. The Fourier convention is $e^{ikz}$, giving $-k^2D+ikB_\infty+C_\infty$. The upper spectral real parts are $-1-k^2,-3-3k^2$. The Hermitian part of the lower symbol is $-aI-k^2\mathrm{diag}(1,1+q/2)$. The bound $\Re\lambda\le-1$ covers all real $k$. Equal limits and exponential convergence give no additional nonzero-index region. Counting spatial roots first for large positive real $\lambda$, then continuing without imaginary spatial roots, gives a $4+4$ splitting on the closed right half-plane. The gate matches the *computed* limit matrices to these hypotheses. The feedback has the same limits.

For a normalized localized eigenfunction set $r=\|p'\|_2$, $b=\sup_z\|B\|$, $d=\sup_z\|C\|$. Integration by parts gives $\Re\lambda\le-r^2+br+d,\ |\Im\lambda|\le br+d$. Consequently $\Re\lambda\ge0$ bounds both $r$ and the size of the search region. The attached solver implements entrywise envelopes for its original coefficients. A change of $C$ requires rechecking that numerical box, the physical-space discretisation and the native model even when the essential spectrum is unchanged.

## A5 contract fixture

Fix $\mu=3,\rho=0$. With $g=1-u^2,\ H=u^2/2+5u^3/24,\ h=v-H(u),\ k=\mu(1+u/2)$, the physical flux is $\mathcal F=(g+kh,\ H'g+(H'k+u)h+\rho g)^T$. The wire format takes $G=-\mathcal F,\ D=I,\ A=0,\ F=0$. On $U=(t,H(t))$, $U'=\mathcal F(U)$, $c=0,\ u(0)=0$, and both endpoint fluxes vanish. The endstates are $(-1,7/24),(1,17/24)$. This verifies the first integral and Rankine–Hugoniot relation. With $J=\mathcal F'(U)$, $Lp=p''-(Jp)'$. The native conservative system uses $j=p'-Jp$, so $p'=Jp+j,\ j'=\lambda p$.

The audited characteristic speeds are $(2,-1)$ on the left and $(-2,1)$ on the right. There is one incoming characteristic per side. The dispersion curves are $\lambda=-k^2-i b k$. The unequal spatial indices also include the strip $-(\Im\lambda)^2<\Re\lambda<-(\Im\lambda)^2/4$ in the essential spectrum. The open right half-plane has a $2+2$ splitting. Translation is at the essential edge. Simplicity here concerns the Evans zero continued from the right. The native origin test also checks outgoing-mode compatibility. A positive gap at zero would be an incorrect admission requirement. Extending the flux requires independent hyperbolicity, transversality, origin and finite-frequency checks.

## Proof difficulty and novelty

| Question | Shortcut or evidence | Remaining obligation and disposition |
|---|---|---|
| A1 at $q=0,\beta=0$ | The block triangular system has scalar pulse roots $3,0$ and rotated roots $4-a\pm2i,1-a\pm2i$. At $a=4$, the axis pair contributes zero to $N_+$. | **NOT_NOVEL**. Exact calibration. Local continuation preserves the root near 3. Binary instability near this control is inexpensive. |
| Original A1 at a fixed positive $q$ | Distinct entries of $B_\infty$ force any common constant invariant subspace to be coordinate-aligned. The derivative coupling and off-diagonal reactions couple all four coordinates. The endstate roots $-a\pm2i$ obstruct bounded similarity to a self-adjoint operator. | Fixed counts remain **UNCERTAIN** for proof novelty. Variable-frame reductions, comparison principles and a short index argument still merit specialist screening. |
| Original A1 across the rectangle | The existing solver archive contains sampled cuts and candidate collisions. Perturbation at the reducible seed only controls a neighbourhood. | The independent audit judged the **full diagram** LIKELY_NOVEL. Uniform root counts, exclusion of missed crossings, local multiplicity and a translation Jordan-chain argument remain to be proved. This pilot supplies no such theorem. |
| New normal feedback | The added row preserves $U$, changes $C$ and shares the far-field bound. | **UNCERTAIN**. Test its tangent-normal form and possible conjugacies. A new term supplies L2 structural variation. A genuinely different proof mechanism would require additional evidence. |
| A5 at one fixed $\mu$ | The conservative form, integrated nonzero spectrum and analytic origin calculation simplify the problem. | Fixed-point fixture. Uniform finite-frequency exclusion on $[1,6]$ has a different quantifier and remains open here. |

Independent literature check on 24 November 2026 covered four-field exact sech pulses, derivative coupling and rotating blocks, Maslov/gradient reductions, and singular fast–slow Evans factorisation. Beck et al. use gradient structure and a self-adjoint linearisation in their index argument. Finite-coupling A1 has nonreal essential spectrum. Their pulse-instability conclusion would also answer a weaker question than the requested multiplicity diagram. [Beck et al., primary paper](https://arxiv.org/abs/1705.03861)

Van Heijster, Doelman and Kaper construct fast and slow transmission factors in a singularly perturbed three-component system. The present rectangle supplies no uniform small scale or corresponding decomposition. Their analysis is a mechanism comparator. The search did not establish a general absence of another reduction. [Van Heijster, Doelman and Kaper, primary paper](https://research.wur.nl/en/publications/pulse-dynamics-in-a-three-component-system-stability-and-bifurcat/)

Post-solver review counts witnesses whose entire disks miss the real axis, compares them with controls and groups all A1 samples on the same explicit branch. No L3 discovery is claimed. A proof-benchmark candidate would need an explicit theorem statement, independent shortcut review, a proof or verifiable proof obligations, and a blind attempt by a reviewer. The source audit assigns A3 to precision calibration following its scalar Green-function reduction, reserves A4 for a weighted threshold/resolvent task, and holds A2 pending coupled homoclinic and exit verification. These use different admission gates.
