"""Verify the speed degeneracy and a flux-unfolding prototype on conservative fronts."""
import argparse
import json
from pathlib import Path
import sys

from .contract import validate, write_json
from .jim_a5 import problem_entry


def triangular_front(sp):
    x, t = sp.symbols("x t", real=True)
    u,v = (sp.Function(name)(x,t) for name in ("u","v"))
    g = 1-u**2
    flux = (g,u*g+u*(v-u**2/2))
    equations = [sp.Eq(w.diff(t),w.diff(x,2)-f.diff(x)) for w,f in zip((u,v),flux,strict=True)]
    return dict(name="triangular_saddle_shock",x=x,t=t,equations=equations,unknowns=[u,v],
                wave=dict(kind="front",left={u:-1,v:sp.S.Half},
                          right={u:1,v:sp.S.Half},speed=0))


def configure_rho(entry, sp, target=0.0, guess=0.02):
    """Restore the A5-design unfolding to a rho=0 G2 or triangular specification."""
    rho = sp.Symbol("rho",real=True)
    x = entry["x"]
    u = entry["unknowns"][0]
    equations = list(entry["equations"])
    second = equations[1]
    equations[1] = sp.Eq(second.lhs,second.rhs-rho*(1-u*u).diff(x))
    wave = dict(entry["wave"],unfolding=dict(parameter="rho",target=target,guess=guess))
    wave.pop("guess",None)
    return dict(entry,equations=equations,wave=wave,parameters={**entry.get("parameters",{}),rho:target})


def native_probe(source, sp):
    from twstab.conservation import ConservativeSystem
    from twstab.problem import load_problem
    from twstab.symbolic import LocalSystem
    from twstab.wave import end_data, solve_wave, WaveError

    problem = load_problem(source)
    cons = ConservativeSystem(problem,LocalSystem(problem))
    system = cons.reduced
    spec = cons.reduced_problem.wave
    splitting = end_data(system,spec["left"],spec["right"],0,True)
    data = system.symbols
    substitutions = {p:sp.Rational(v) for p,v in zip(system.params,system.theta,strict=True)}
    G = sp.Matrix(data["G"]).xreplace(substitutions)
    jets = sp.Matrix(data["T"])
    viscosity = G.jacobian(jets).applyfunc(sp.simplify)
    scale = viscosity[0,0]
    scalar_viscosity = bool(scale.is_positive and not scale.free_symbols
                            and viscosity == scale*sp.eye(system.n))
    result = dict(physical_speed=cons.speed,rh_residual=cons.jump_residual,splitting=splitting,
                  scalar_constant_viscosity=scalar_viscosity,
                  free_speed_only_rescales=scalar_viscosity and splitting["free_c"])
    try:
        wave = solve_wave(cons.reduced_problem,system)
    except WaveError as exc:
        result.update(computed=False,reason=str(exc))
    else:
        result.update(computed=True,computed_reduced_speed=wave.c,nodes=wave.mesh.size,
                      residual_rms_max=wave.info["residual_rms_max"],tail_error=wave.info["tail_error"])
    return result,system,G,jets


def controlled_bvp(system, G, jets, mu, cubic, sp, np):
    from scipy.integrate import solve_bvp

    u,h = sp.symbols("u h",real=True)
    H = u**2/2+(5*u**3/24 if cubic else 0)
    g = 1-u**2
    k = mu*(1+u/2)
    # Verify that this 2D coordinate form is the native reduced field for rho=0.
    # In coordinates h=v-H(u), the derivative is h'=v'-H'(u)u'.
    physical = (-G.subs(dict.fromkeys(jets,0))).subs(
        dict(zip(system.symbols["low"],[u,H+h],strict=True)),simultaneous=True)
    hp = H.diff(u)
    transformed = sp.Matrix([physical[0],physical[1]-hp*physical[0]])
    expected = sp.Matrix([g+k*h,u*h])
    # Jim's reduction inserts the far-field flux constant as binary64.
    # Bound the resulting polynomial discrepancy on |u|, |h| <= 1.
    field_error = max(sum(abs(c) for c in sp.Poly(e,u,h).coeffs()) for e in (transformed-expected))
    if field_error > 1e-12:
        raise ValueError("controlled system does not match the native reduced system")

    L = 16
    mesh = np.linspace(-L,L,241)
    guess = np.array([np.tanh(mesh/1.2),.015/np.cosh(mesh)**2,np.zeros(mesh.size)])
    probes = np.linspace(-L,L,2001)
    physical_H = sp.lambdify(u,H,modules="numpy")
    physical_Hp = sp.lambdify(u,hp,modules="numpy")
    results = {}
    for mode in ("free_reduced_speed","flux_unfolding"):
        def rhs(z,y,p):
            u,h,_ = y
            g = 1-u*u
            rho = p[0] if mode=="flux_unfolding" else 0
            sigma = p[0] if mode=="free_reduced_speed" else 0
            ref = np.tanh(z)
            return np.array([(g+float(mu)*(1+u/2)*h)/(1+sigma),
                             (u*h+rho*g)/(1+sigma),(u-ref)*(1-ref*ref)])

        def boundary(left,right,p):
            rho = p[0] if mode=="flux_unfolding" else 0
            projected = []
            for side,y in ((-1,left),(1,right)):
                k = float(mu)*(1+side/2)
                jac = np.array([[-2*side,k],[-2*side*rho,side]])
                eigenvalues,vectors = np.linalg.eig(jac)
                selected = np.argmax(eigenvalues) if side==-1 else np.argmin(eigenvalues)
                direction = vectors[:,selected]
                projected.append(direction[0]*y[1]-direction[1]*(y[0]-side))
            return np.array([*projected,left[2],right[2]])

        solution = solve_bvp(rhs,boundary,mesh,guess,p=[0.0],tol=1e-8,max_nodes=40000)
        u,h,_ = solution.sol(probes)
        du,dh,_ = solution.sol(probes,1)
        Q = np.array([u,physical_H(u)+h])
        dQ = np.array([du,physical_Hp(u)*du+dh])
        target_u = np.tanh(probes)
        exact_Q = np.array([target_u,physical_H(target_u)])
        defect = dQ-system.rhs(Q,0)
        residual = float(np.max(abs(defect)))
        negative = defect.copy()
        negative[1] -= .05*(1-u*u)
        nonzero_target_residual = float(np.max(abs(negative)))
        results[mode] = dict(converged=bool(solution.success),message=solution.message,
                             fitted_parameter=float(solution.p[0]),nodes=solution.x.size,
                             exact_profile_max_error=float(np.max(abs(Q-exact_Q))),
                             original_equation_max_residual=residual,
                             residual_if_target_rho_is_point_05=nonzero_target_residual,
                             accepted_original=bool(solution.success and abs(solution.p[0])<1e-7
                                                    and residual<1e-7))
    return dict(native_field_error_bound=float(field_error),**results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver",type=Path,required=True)
    parser.add_argument("--examples",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    sys.path.insert(0,str(args.solver.resolve()))
    import numpy as np
    import scipy
    import sympy as sp
    import twstab

    manifest = json.loads((args.examples/"manifest.json").read_text())
    records = []
    for ident in manifest["instances"]:
        item = validate(json.loads((args.examples/"instances"/(ident+".json")).read_text()))
        entry = problem_entry(item,sp)
        entry["wave"].pop("guess")
        mu = sp.Rational(item["parameters"]["mu"])
        records.append((f"A5 mu={mu}",entry,mu,True,item["specification_hash"]))
    records.append(("triangular shock",triangular_front(sp),sp.S.Zero,False,None))
    results = []
    for name,entry,mu,cubic,source_hash in records:
        native,system,G,jets = native_probe(entry,sp)
        controlled = controlled_bvp(system,G,jets,mu,cubic,sp,np)
        results.append(dict(case=name,specification_hash=source_hash,native=native,controlled=controlled))
        print(name,native["computed"],controlled["flux_unfolding"]["accepted_original"],flush=True)
    normal,_,_,_ = native_probe(args.solver/"problems/p_system_shock.py",sp)
    results.append(dict(case="p-system shock",native=normal))
    write_json(args.out,dict(solver_version=twstab.__version__,scipy_version=scipy.__version__,
                            sympy_version=sp.__version__,cases=results))
    assert normal["computed"] and not normal["free_speed_only_rescales"]
    for row in results[:-1]:
        assert row["native"]["free_speed_only_rescales"]
        controlled = row["controlled"]
        assert not controlled["free_reduced_speed"]["accepted_original"]
        good = controlled["flux_unfolding"]
        assert good["accepted_original"] and good["residual_if_target_rho_is_point_05"]>.01


if __name__ == "__main__":
    main()
