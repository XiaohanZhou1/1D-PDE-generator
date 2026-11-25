"""Exact profiles and continued origin counts for viscous shocks."""
import numpy as np
import sympy as sp

from .contour import box, circle, winding
from .evans import Evans
from .wave import Wave, WaveError


def exact_wave(problem, cons, opts):
    """Validate a supplied full-line front and expose it on the Evans mesh."""
    spec = problem.wave
    if (spec["kind"] != "front" or cons.K != 2 or cons.depth != 1
            or spec["speed"] is None or spec["frequency"] is not None or spec["frequency_guess"] is not None):
        raise WaveError("a second-order conservative front and an explicit speed are required")
    z = sp.Symbol("z", real=True)
    parameters = {key: sp.Rational(value) for key,value in problem.parameters.items()}
    profile = []
    for value in spec["exact_profile"]:
        value = value.xreplace(parameters)
        symbols = value.free_symbols
        if len(symbols) > 1 or any(str(s) != "z" for s in symbols):
            raise WaveError("the exact profile must depend only on z")
        profile.append(value.subs({s:z for s in symbols}))
    speed = sp.Rational(cons.speed)
    moving = {u:p.subs(z,problem.x-speed*problem.t) for u,p in zip(problem.unknowns,profile,strict=True)}
    for equation in problem.equations:
        residual = (equation.lhs-equation.rhs).xreplace(parameters).xreplace(moving).doit()
        residual = residual.subs({problem.t:0,problem.x:z}, simultaneous=True)
        if sp.simplify(residual) != 0:
            raise WaveError("the supplied expressions do not satisfy the travelling-wave equations")
    endpoints = [np.asarray(spec["left"]),np.asarray(spec["right"])]
    for sign, end in zip((-1,1),endpoints,strict=True):
        limits = [float(sp.limit(p,z,sign*sp.oo)) for p in profile]
        if not np.allclose(limits,end,atol=1e-12,rtol=1e-12):
            raise WaveError("the supplied profile limits differ from the prescribed end states")
    L = 20.0 if spec["L"] is None else spec["L"]
    if not np.isfinite(L) or not 0 < L <= opts["L_max"]:
        raise WaveError("exact profile truncation requires 0 < wave['L'] <= L_max")
    evaluators = [sp.lambdify(z,p,modules="numpy") for p in profile]

    def values(points):
        return np.stack([np.broadcast_to(f(points),np.shape(points)) for f in evaluators])

    mesh = np.linspace(-L,L,241)
    sample = values(mesh)
    tail = max(np.max(abs(sample[:,0]-endpoints[0])),np.max(abs(sample[:,-1]-endpoints[1])))
    if not np.isfinite(sample).all() or tail > opts["tail_tol"]:
        raise WaveError("the supplied profile is nonfinite or its tails need a larger domain")
    return Wave(c=cons.speed,L=L,N=cons.n,Y_minus=endpoints[0],Y_plus=endpoints[1],
                interpolant=values,mesh=mesh,theta=cons.reduced.theta,
                info=dict(started_from="exact_profile",profile_error=0.0,tail_error=tail,
                          symbolic_residual_zero=True))


class ContinuedExterior:
    """Spatial graphs for two-component identity-viscosity saddles."""
    N, k = 4, 2

    def __init__(self, system, left, right):
        if system.n != 2:
            raise WaveError("continued origin currently requires two components")
        self.bases = []
        for end in (left,right):
            coefficients = system.end_state(end)[2]
            if not np.allclose(coefficients[2],np.eye(2),rtol=0,atol=1e-12):
                raise WaveError("continued origin requires identity viscosity")
            speed,vectors = np.linalg.eig(-coefficients[1])
            if np.any(abs(speed.imag)>1e-12):
                raise WaveError("continued origin requires real nonzero endstate speeds")
            order = np.argsort(speed.real)
            speed,vectors = speed.real[order],vectors[:,order]
            if not speed[0] < 0 < speed[1]:
                raise WaveError("continued origin requires one characteristic of each sign at each end")
            self.bases.append((speed,vectors,np.linalg.inv(vectors)))
        outgoing = np.column_stack((self.bases[0][1][:,0],self.bases[1][1][:,1]))
        if abs(np.linalg.det(outgoing)) <= 1e-8:
            raise WaveError("outgoing characteristic directions do not span the state space")
        self.branch_distance = min(np.min(speed**2)/4 for speed,_,_ in self.bases)

    def at(self, lam):
        data = []
        for side,(speed,vectors,inverse) in enumerate(self.bases):
            root = np.sqrt(speed**2+4*complex(lam))
            plus,minus = (speed+root)/2,(speed-root)/2
            # Compute small roots without subtracting nearly equal numbers.
            plus[0],minus[1] = -lam/minus[0],-lam/plus[1]
            slopes = plus if side==0 else minus
            boundary = np.empty((2,4),dtype=complex)
            boundary[:,::2] = -(vectors*slopes) @ inverse
            boundary[:,1::2] = np.eye(2)
            data.append((boundary,np.r_[plus,minus]))
        return data[0][0],data[1][0],data[0][1],data[1][1],1+0j


class ContinuedEvans(Evans):
    def _log_norm(self, ev_minus, ev_plus):
        # Select the same normalization sheets throughout the detour.
        total = 0j
        for widths,exponents in ((self.h_left,ev_minus),(self.h_right,ev_plus)):
            v = np.outer(widths,exponents*np.array([1,1,-1,-1]))
            total += np.log(1+v/2+v*v/12).sum()
        return total


def continued_origin(system, wave, left, right, opts):
    coefficients = system.base.symbols
    values = {p:sp.Rational(v) for p,v in zip(system.base.params,system.base.theta,strict=True)}
    diffusion = sp.Matrix(coefficients["G"]).jacobian(coefficients["T"]).xreplace(values)
    if (diffusion-sp.eye(system.n)).applyfunc(sp.simplify) != sp.zeros(system.n):
        raise WaveError("continued origin requires constant identity viscosity")
    exterior = ContinuedExterior(system,left,right)
    radius = float(opts["conservation_origin_radius"])
    region = opts["region"]
    if not isinstance(region,(tuple,list)) or len(region)!=2:
        raise WaveError("continued origin requires a supplied high-frequency enclosure in options['region']")
    xmax,ymax = map(float,region)
    branch_limit = exterior.branch_distance*(1-1e-12)
    if not np.isfinite([radius,xmax,ymax]).all() or not 0 < radius < min(branch_limit,xmax,ymax):
        raise WaveError("origin radius must fit inside the search box and avoid the far-field branch points")
    whole = box(0,xmax,ymax,1.0)
    records = []
    for L,step in ((wave.L,0.08),(wave.L,0.04),(0.75*wave.L,0.04)):
        engine = ContinuedEvans.build(system,wave,exterior,L,L,step)
        for detour in (radius/2,radius):
            def outer(t):
                lam = whole(t)
                if t >= 0.75 and abs(lam.imag) < detour:
                    return complex(-np.sqrt(max(0,detour**2-lam.imag**2)),lam.imag)
                return lam
            outer_count = winding(engine,outer,initial=max(96,opts["initial_panels"]),
                                  limit=opts["phase_limit"],max_depth=21)
            origin_count = winding(engine,circle(0,detour),initial=48,limit=opts["phase_limit"])
            records.append(dict(L=L,step=step,radius=detour,outer=outer_count,origin=origin_count))
    # At lambda=0 the primitive Q+d has outgoing limits for a unique d:
    # the two outgoing directions span R^2. Each disk must contain only this zero.
    stable = all(r["outer"]["count"]==r["origin"]["count"]==1 for r in records)
    return dict(label="STABLE" if stable else "UNCERTAIN",
                qualifier="numerical stability modulo translation",
                reason=("one continued origin zero and no additional zeros inside the supplied enclosure"
                        if stable else "outer and origin windings did not isolate only the continued zero"),
                open_rhp_count=0 if stable else None,
                region=dict(x_min=0,x_max=xmax,y_max=ymax,source="supplied high-frequency enclosure"),
                windings=records,evaluations=sum(r[k]["evaluations"] for r in records for k in ("outer","origin")))
