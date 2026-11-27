# D2 resampling and E2 interface pilot

The grids were fixed before the scored runs. D2 uses four parameter sets disjoint from the first D2 comparison. E uses the existing E1 advancing control and four new E2 cases. The tests score binary spectral labels and preliminary difficulty separately.

| Batch | Attempted | Label evidence passes | Label and preliminary difficulty |
|---|---:|---:|---:|
| D2 fresh points | 4 | 3 | 3 |
| E1 analytic control | 1 | 1 | 0 |
| E2 coupled interface | 4 | 2 | 0 |
| Total | 9 | 6 | 3 |

The three D2 witnesses were reproduced by the native solver and independent continuous shooting on the exact profile. The largest absolute disagreement was \(2.87\times10^{-6}\). The fourth case, with \((\sigma,\tau,\varepsilon)=(2,3/2,3/4)\), received a native numerical stability label and remains `UNCERTAIN` under the full-domain requirement. The three candidates belong to one mechanism and independent difficulty review is still pending.

## Exact E2 problem

On \(x<s(t)\), take
\[
u_t=u_{xx}-u+\gamma(v-u^2),\qquad v_t=v_{xx}-(5/2)v,\qquad u=v=1\text{ at }x=s(t),
\]
with
\[
s'=3/2+3(u_x-1/2)+\beta(v_x-1).
\]
The advancing profile \((U,V)=(e^{z/2},e^z)\), \(z=x-3t/2\le0\), satisfies \(V-U^2=0\) and the two moving-frame equations. Both terms in the interface law involving traces vanish on this profile. In a preliminary unscored run, a generic initial guess for \((\gamma,\beta)=(1,-8)\) found a wave with speed about \(4.83\). The scored native runs start from the exact profile and \(c=3/2\). The runner checks the recovered wave speed before using its spectra.

The boundary-attached linearization on Dirichlet perturbations \((p,q)\) has forced terms \(\eta U'\) and \(\eta V'\), where \(\eta=3p'(0)+\beta q'(0)\). The interface displacement obeys \(\dot d=\eta\). For \(\lambda>0\), a unit interface velocity gives
\[
q_\lambda(z)=\frac{e^z-e^{\nu z}}{\lambda},\qquad
\nu^2+(3/2)\nu-5/2-\lambda=0,\quad \nu>0,
\]
and \(p_\lambda\) solves a scalar Dirichlet problem with an exponentially variable coefficient. The eigenvalue condition is
\[
T(\lambda)=3p_\lambda'(0)+\beta q_\lambda'(0)-1=0.
\]
The independent check uses this exact \(q\), a continuous collocation BVP for \(p\), and three domain-length and tolerance resolutions. Both \(\beta=-8\) candidates have independent positive witnesses, with a maximum native-to-transfer disagreement below \(6\times10^{-6}\). The two \(\beta=-4\) runs have native numerical stability labels, so they remain `UNCERTAIN`. The E1 advancing control retains its analytic stability certificate. The present E2 comparison checks real positive witnesses. [Known E1 certificate](cf-calibration.md#e1--moving-boundary)

## E2 shortcut found during screening

At \(\lambda=0\), \(q_0=-ze^z/(2+c)\) and \(p_*=-ze^{z/2}/5\) is a supersolution for \(p_0\). Indeed,
\[
(-\partial_z^2-c\partial_z+1+2\gamma U)p_*
=U'+(2\gamma/5)(-ze^z)\ \ge\ U'+\gamma q_0.
\]
The maximum principle on the half-line gives \(0\le p_0\le p_*\), hence \(p_0'(0)\ge -1/5\). With \(\beta=-8\), \(T(0)\ge -3/5+16/7-1>0\). For real \(\lambda\ge0\) the Dirichlet response \(p_\lambda\) is nonnegative, so \(p_\lambda'(0)\le0\). Since \(q_\lambda'(0)=-1/(\nu+1+c)\to0\), \(T(\lambda)<0\) for sufficiently large \(\lambda\). Continuity gives a positive real root. Thus both numerical E2 label passes have a short analytic instability argument. Their difficulty gate records `SHORTCUT_FOUND`. The two opposing interface coefficients alone did not prevent an elementary comparison bound. This test therefore adds a new E solver calibration and gives no new E examples passing both gates.

## Reproduce

~~~powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m pdegen.followup_pilot --out scratch/d2-e-followup
~~~

Use the normal Python dependencies and a fresh output directory. The two batches use the bundled solver only.
