"""Fixed-speed profile BVP with one caller-specified flux unfolding."""
import numpy as np
import sympy as sp
from scipy.integrate import solve_bvp

from .conservation import antiderivative
from .wave import Wave, WaveError, end_data, reference_from_guess


def unfold_wave(problem, cons, opts):
    spec = problem.wave
    config = spec["unfolding"]
    key, target, guess = (config[name] for name in ("parameter","target","guess"))
    if cons.K!=2 or cons.n!=2 or cons.depth!=1 or spec["kind"]!="front":
        raise WaveError("flux unfolding requires a two-component second-order conservative front")
    if spec["frequency"] is not None or spec["frequency_guess"] is not None:
        raise WaveError("flux unfolding does not support a rotating front")
    reduced = cons.reduced
    left,right = (np.asarray(spec[name],dtype=float) for name in ("left","right"))
    ends = end_data(reduced,left,right,0,True)
    if (ends["ku_m"],ends["ku_p"]) != (1,1):
        raise WaveError("flux unfolding requires a saddle-to-saddle profile")

    base = cons.base
    sym = base.symbols
    frozen = {p:sp.Rational(v) for p,v in problem.parameters.items() if p!=key}
    jets = [[*sym["low"][i:i+2],highest] for i,highest in zip(base.offset,sym["T"],strict=True)]
    flux = antiderivative([g.xreplace(frozen) for g in sym["G"]],jets,[2,2])
    if flux is None:
        raise WaveError("the symbolic flux could not be recovered with the unfolding parameter free")
    state = [component[0] for component in jets]
    first = [component[1] for component in jets]
    flux = sp.Matrix(flux)
    if (flux.jacobian(first)-sp.eye(2)).applyfunc(sp.simplify) != sp.zeros(2):
        raise WaveError("flux unfolding currently requires constant identity viscosity")
    advective = flux.subs(dict.fromkeys(first,0))
    end_flux = [advective.subs(dict(zip(state,map(sp.Rational,end),strict=True)),simultaneous=True)
                for end in (left,right)]
    jump = end_flux[1]-end_flux[0]
    if any(sp.simplify(v.diff(key))!=0 for v in jump):
        raise WaveError("unfolding parameter changes the jump conditions at the prescribed end states")
    velocity = (end_flux[0]-advective
                -sp.Rational(cons.speed)*sp.Matrix([u-sp.Rational(a) for u,a in zip(state,left,strict=True)]))
    if not velocity.has(key):
        raise WaveError("unfolding parameter does not change the reduced profile field")
    sensitivity = velocity.diff(key)
    if sp.simplify(velocity[0]*sensitivity[1]-velocity[1]*sensitivity[0]) == 0:
        raise WaveError("unfolding parameter only rescales the reduced profile field")
    field = sp.lambdify((*state,key),list(velocity),modules="numpy",cse=True)
    jacobian = sp.lambdify((*state,key),velocity.jacobian(state),modules="numpy",cse=True)
    variation = sp.lambdify((*state,key),list(sensitivity),modules="numpy",cse=True)

    def evaluate(fun, points, parameter):
        return np.stack([np.broadcast_to(v,points.shape[1:]) for v in fun(*points,parameter)])

    def normal(end, parameter, sign):
        eigenvalues,vectors = np.linalg.eig(jacobian(*end,parameter))
        if np.any(abs(eigenvalues.imag)>1e-10) or not min(eigenvalues.real)<0<max(eigenvalues.real):
            raise WaveError("endstate loses its saddle splitting during unfolding")
        direction = vectors[:,np.argmax(sign*eigenvalues.real)].real
        return np.array([-direction[1],direction[0]])

    orientation = (normal(left,target,1),normal(right,target,-1))
    ref = reference_from_guess(cons.reduced_problem,reduced)
    L = spec["L"]
    if L is None:
        L = float(np.clip(np.log(1/opts["tail_tol"])/ends["slow"],6,opts["L_max"]))
    if not np.isfinite(L) or not 0<L<=opts["L_max"]:
        raise WaveError("unfolding requires 0 < wave['L'] <= L_max")

    def rhs(z,y,p):
        phase = ((y[:2]-ref.U(z))*ref.dU(z)).sum(axis=0)
        return np.vstack((evaluate(field,y[:2],p[0]),phase))

    def boundary(a,b,p):
        residuals = []
        for end,y,sign,ref_normal in zip((left,right),(a,b),(1,-1),orientation,strict=True):
            row = normal(end,p[0],sign)
            if row@ref_normal<0:
                row = -row
            residuals.append(row@(y[:2]-end))
        return [*residuals,a[2],b[2]]

    mesh = np.linspace(-L,L,241)
    y0 = np.vstack((ref.Y(mesh),np.zeros(mesh.size)))
    first_sol = solve_bvp(rhs,boundary,mesh,y0,p=[guess],tol=opts["wave_tol"],max_nodes=opts["max_nodes"])
    if not first_sol.success:
        raise WaveError(f"unfolding BVP did not converge: {first_sol.message}")
    fine = solve_bvp(rhs,boundary,first_sol.x,first_sol.y,p=first_sol.p,
                     tol=opts["wave_check_tol"],max_nodes=opts["max_nodes"])
    if not fine.success:
        raise WaveError(f"refined unfolding BVP did not converge: {fine.message}")
    fitted = float(fine.p[0])
    tolerance = 1e-7*max(1,abs(target))
    if not np.isfinite(fitted) or abs(fitted-target)>tolerance:
        raise WaveError(f"unfolding fitted {key}={fitted:.9g} differs from target {target:.9g}")
    z = np.linspace(-L,L,2001)
    profile = fine.sol(z)[:2]
    derivative = fine.sol(z,1)[:2]
    target_velocity = reduced.rhs(profile,0)
    residual = float(np.max(abs(derivative-target_velocity)))
    original_jets = np.stack((profile,derivative),axis=1).reshape(base.N,z.size)
    original_top = base.rhs(original_jets,cons.speed)[base.top_rows]
    pde_residual = float(np.max(abs(fine.sol(z,2)[:2]-original_top)))
    tails = float(max(np.max(abs(profile[:,0]-left)),np.max(abs(profile[:,-1]-right))))
    if (not np.isfinite([residual,pde_residual,tails]).all() or residual>1e-7
            or pde_residual>1e-5 or tails>1e3*opts["tail_tol"]):
        raise WaveError(f"unfolding profile failed target checks (first integral {residual:.3g}, "
                        f"PDE {pde_residual:.3g}, tails {tails:.3g})")
    delta = evaluate(variation,profile,target)
    transverse = float(np.max(abs(target_velocity[0]*delta[1]-target_velocity[1]*delta[0])))
    if not np.isfinite(transverse) or transverse<=1e-10:
        raise WaveError("unfolding parameter only changes the speed along the computed orbit")
    difference = float(np.max(abs(profile-first_sol.sol(z)[:2])))
    return Wave(c=cons.speed,L=L,N=2,Y_minus=left,Y_plus=right,interpolant=fine.sol,
                mesh=fine.x,theta=reduced.theta,
                info=dict(started_from="parameter_unfolding",profile_error=difference,tail_error=tails,
                          target_equation_residual=residual,target_pde_residual=pde_residual,nodes=int(fine.x.size),
                          unfolding=dict(parameter=str(key),target=target,guess=guess,fitted=fitted,
                                         fit_tolerance=tolerance,normal_variation_max=transverse)))
