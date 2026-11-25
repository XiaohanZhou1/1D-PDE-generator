# Conservative front BVP verification

This experiment tests the choice of free parameter for the A5 automatic profile solve. It uses the five G2 specifications, a second two-component saddle-to-saddle shock, and Jim's p-system example as a positive control. It calls Jim's original reduced-profile BVP without an exact input or a supplied guess. The separate controlled test uses the same mesh, phase condition, projection boundary conditions and perturbed starting profile in both modes.

## What the structural check detects

For a second-order viscous conservation law with scalar constant viscosity \(dI\), the reduced profile problem has an auxiliary velocity \(\sigma\) of the form

$$
(d+\sigma)Q'=F(Q)-F(Q_-)-c_{\mathrm{RH}}(Q-Q_-).
$$

The physical speed \(c_{\mathrm{RH}}\) is fixed by the jump conditions before making this reduction. As long as \(d+\sigma>0\), the auxiliary parameter changes the rate of traversal of the same phase-plane orbits. The experiment reads the constant viscosity from the Jacobian of the native reduced right-hand side with respect to its first derivative variables. It records the profile index and flags cases in which Jim's rule releases a velocity that has this rescaling property.

For the controlled tests, use \(u'=g(u)+k(u)h,\ h'=uh+\rho g(u)\) with \(g=1-u^2\). The A5 case has \(h=v-u^2/2-5u^3/24,\ k=\mu(1+u/2)\). The triangular control uses \(h=v-u^2/2,\ k=0\) and a separately specified physical flux. The transformed field is checked against the native reduction with a bound on the polynomial discrepancy. In either case the first normal variation at \(\rho=0\) satisfies \((\operatorname{sech}z\,\dot h)'=\operatorname{sech}^3z\). Its integral is \(\pi/2\), giving a nonzero derivative of the signed mismatch. In the speed-only mode the normal forcing on \(h=0\) is zero.

The controlled BVP uses a finite interval with half-length 16 and a phase pinned by an integral against \(\operatorname{sech}^2z\). Its initial guess is deliberately given the wrong width and a nonzero normal component. Acceptance checks both that the fitted parameter returns to its prescribed value and that the spline satisfies the original, unmodified equation at 2,001 locations. A negative control evaluates the recovered wave against the flux with target \(\rho=0.05\), which has the same endstates and physical wave speed.

## Results

| Cases | Native reconstruction | Controlled free reduced speed | Controlled flux unfolding |
|---|---|---|---|
| Five A5 specifications | Index zero and the rescaling condition held in all cases. All hit the node cap. | All converged to nonzero auxiliary speeds. Residuals in the original equation were 0.0774 to 0.0819. | All accepted. Maximum absolute fitted rho below 5e-26, maximum profile error 7.4e-10 and maximum original-equation residual 2.16e-8. |
| Triangular saddle-to-saddle shock | Index zero and the rescaling condition held. The computed reduced speed 0.003005 failed the given-speed check. | Converged to a nonzero auxiliary speed. Residual in the original equation was 0.1613. | Accepted. Profile error 4.6e-10, original-equation residual 1.69e-8. |
| p-system shock | Index one. The fixed-speed solve succeeded with 1,871 nodes. | — | — |

The polynomial discrepancies from Jim's native field were at most 5.6e-16 on the specified coefficient bound. In the negative control the recovered profiles had original-equation residuals of approximately 0.05 at the deliberately mismatched target. The proposed target-value and residual checks reject this mismatch. [Machine-readable results](../evidence/jim-a5-bvp-audit.json)

## Reproduce

Use the generator environment and SymPy 1.14.0. Pass the directory containing Jim's `twstab/`.

~~~powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m pdegen.a5_bvp_audit --solver C:/path/to/solver --examples examples/g2 --out scratch/a5-bvp-audit.json
~~~

The verification experiment is separate from the production BVP. A generalized solver interface would need a caller-specified parameter, its target value, and a checked rule for jump consistency and for how the endstate subspaces depend on that parameter. The solver would check the returned parameter, profile residual and tails before dispatching spectral analysis. The p-system control already uses the fixed-speed path.
