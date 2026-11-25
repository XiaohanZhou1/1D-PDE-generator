"""Run the G2 A5 specifications through Jim's twstab 2.18.0."""
import argparse
import hashlib
import json
from math import ceil, exp, prod
from pathlib import Path
import sys
from unittest.mock import patch

from .algebra import evaluate
from .contract import validate, write_json


def polynomial(terms, variables, sp):
    return sum((sp.Rational(coefficient) *
                (1 if term == "1" else prod(variables[name] for name in term.split("*")))
                for term, coefficient in terms.items()), sp.S.Zero)


def problem_entry(item, sp):
    x, time, z = sp.symbols("x time z", real=True)
    fields = {name: sp.Function(name)(x, time) for name in item["variables"]}
    equations = item["equations"]
    entries = []
    for i, u in enumerate(fields.values()):
        D, A, F, G = (polynomial(equations[key][i], fields, sp) for key in ("D","A","F","G"))
        entries.append(sp.Eq(u.diff(time), D*u.diff(x,2)-A*u.diff(x)+F+G.diff(x)))
    target = item["target_wave"]
    basis = {"s": sp.sech(z), "t": sp.tanh(z)}
    guess = {u: polynomial(p, basis, sp)
             for u,p in zip(fields.values(), target["profile"], strict=True)}
    left,right = ({u:polynomial(p,{},sp) for u,p in zip(fields.values(), end, strict=True)}
                  for end in target["endstates"])
    return dict(name=item["instance_id"], x=x, t=time, equations=entries, unknowns=list(fields.values()),
                wave=dict(kind="front", left=left, right=right, speed=target["speed"], guess=guess))


def audit_coefficients(item, entry, native, np):
    from twstab.conservation import ConservativeSystem
    from twstab.symbolic import LocalSystem
    problem = native.load_problem(entry)
    local = LocalSystem(problem)
    cons = ConservativeSystem(problem,local)
    if cons.depth != 1 or cons.speed != 0 or cons.jump_residual > 1e-12:
        raise ValueError("native conservation reduction differs from the A5 specification")
    z = np.linspace(-18,18,401)
    env = dict(s=1/np.cosh(z),t=np.tanh(z))
    def vector(polys):
        return np.asarray([np.broadcast_to(evaluate(p,env),z.shape) for p in polys])
    target = item["target_wave"]
    U = vector(target["profile"])
    dU = vector(target["dz"])
    jets = np.stack((U,dU),axis=1).reshape(local.N,z.size)
    operator = item["linearised_operator"]
    B,C = (np.stack([vector(row).T for row in operator[key]],axis=1) for key in ("B","C"))
    D = np.broadcast_to(np.eye(2),B.shape)
    full = local.coefficients(jets,0)
    integrated = cons.coefficients(U)
    profile_defect = float(np.max(abs(cons.reduced.rhs(U,0)-dU)))
    full_defect = float(np.max(abs(full-np.stack((C,B,D)))))
    integrated_defect = float(np.max(abs(integrated-np.stack((np.zeros_like(C),B,D)))))
    if max(profile_defect,full_defect,integrated_defect) > 1e-11:
        raise ValueError("native profile or linearised coefficients differ from the A5 specification")
    return dict(profile_rhs_max_error=profile_defect, full_operator_max_error=full_defect,
                integrated_operator_max_error=integrated_defect)


def exact_search_region(item, sp):
    # Bound the Frobenius norm of B on -1 <= tanh(z) <= 1 using a grid
    # of exact rationals and the coefficient-sum bound on each derivative.
    t = sp.Symbol("t")
    s = sp.Symbol("s")
    grid = [sp.Rational(i,64) for i in range(-64,65)]
    squared = 0
    for row in item["linearised_operator"]["B"]:
        for terms in row:
            entry = sp.Poly(polynomial(terms,{"s":s,"t":t},sp).subs(s**2,1-t**2),t)
            lipschitz = sum(abs(c) for c in entry.diff().all_coeffs())
            bound = max(abs(entry.eval(point)) for point in grid) + lipschitz/128
            squared += bound**2
    # For integrated modes, Re(lambda) <= M²/4 and |Im(lambda)| <= M² when Re(lambda)>=0.
    return [int(ceil(squared/4))+1,int(ceil(squared))+1]


def continued_origin(entry, native, wave, region, np):
    """Use the native Evans engine on the A5-specific continued exterior sheets."""
    from twstab.conservation import ConservativeSystem
    from twstab.contour import box, circle, winding
    from twstab.evans import Evans
    from twstab.symbolic import LocalSystem
    problem = native.load_problem(entry)
    system = ConservativeSystem(problem,LocalSystem(problem))
    bases = []
    for side in ("left","right"):
        coefficients = system.end_state(problem.wave[side])[2]
        speed, vectors = np.linalg.eig(-coefficients[1])
        if not np.allclose(sorted(speed),[-1,2] if side=="left" else [-2,1]):
            raise ValueError("unexpected A5 endstate transport speeds")
        bases.append((speed,vectors,np.linalg.inv(vectors)))

    class ContinuedExterior:
        N, k = 4, 2

        def at(self, lam):
            data = []
            for side,(speed,vectors,inverse) in enumerate(bases):
                root = np.sqrt(speed**2+4*complex(lam))
                plus,minus = (speed+root)/2,(speed-root)/2
                slopes = plus if side==0 else minus
                boundary = np.empty((2,4),dtype=complex)
                boundary[:,::2] = -(vectors*slopes) @ inverse
                boundary[:,1::2] = np.eye(2)
                data.append((boundary,np.r_[plus,minus]))
            return data[0][0],data[1][0],data[0][1],data[1][1],1+0j

    class ContinuedEvans(Evans):
        def _log_norm(self, ev_minus, ev_plus):
            # The first two exponents stay on the growing sheets selected
            # in the open right half-plane, including on the left semicircle.
            total = 0j
            for widths,exponents in ((self.h_left,ev_minus),(self.h_right,ev_plus)):
                v = np.outer(widths,exponents*np.array([1,1,-1,-1]))
                total += np.log(1+v/2+v*v/12).sum()
            return total

    whole = box(0,*region,1.0)
    records = []
    for L, step in ((wave.L,0.08),(wave.L,0.04),(0.75*wave.L,0.04)):
        engine = ContinuedEvans.build(system,wave,ContinuedExterior(),L,L,step)
        for radius in (0.04,0.08):
            def outer(t):
                lam = whole(t)
                if t >= 0.75 and abs(lam.imag) < radius:
                    return complex(-np.sqrt(max(0,radius**2-lam.imag**2)),lam.imag)
                return lam
            outer_count = winding(engine,outer,initial=96,max_depth=21)
            origin_count = winding(engine,circle(0,radius),initial=48)
            records.append(dict(L=L,step=step,radius=radius,outer=outer_count,origin=origin_count))
    accounted = all(r["outer"]["count"]==r["origin"]["count"]==1 for r in records)
    return dict(windings=records,open_rhp_count=0 if accounted else None,
                outcome="NO_ADDITIONAL_ZEROS_NUMERICAL" if accounted else "UNCERTAIN")


def run_case(item, native, sp, np, mode, half_length):
    validate(item)
    if item["family_id"] != "A5":
        raise ValueError("this adapter requires an A5 instance")
    entry = problem_entry(item,sp)
    audit = audit_coefficients(item,entry,native,np)
    if mode in ("formal","endpoints"):
        target = entry["wave"].pop("guess")
        options = dict(conservation_method="axis",default_region=False)
        if mode == "formal":
            entry["wave"].update(exact_profile={u.func.__name__:p for u,p in target.items()},L=half_length)
            options.update(region=exact_search_region(item,sp),conservation_origin_radius=0.08)
        result = native.analyse(entry,**options)
        return dict(instance_id=item["instance_id"], specification_hash=item["specification_hash"],
                    parameters=item["parameters"], coefficient_audit=audit,
                    profile_mode=mode,native_options=options,native=result)
    target = item["target_wave"]["profile"]
    computed = {}
    solve_wave = native.solve_wave
    options = dict(conservation_method="axis",default_region=False)
    if mode != "native":
        options["region"] = exact_search_region(item,sp)
    waves = []

    def exact_values(z):
        env = dict(s=1/np.cosh(z),t=np.tanh(z))
        return np.asarray([evaluate(p,env) for p in target])

    def audited_solve(*args, **kwargs):
        if mode == "native":
            wave = solve_wave(*args, **kwargs)
        else:
            from twstab.wave import Wave
            problem, system = args
            if system.N != 2 or problem.wave["speed"] != 0:
                raise native.WaveError("unexpected A5 reduced profile system")
            L = half_length
            wave = Wave(c=0,L=L,N=2,Y_minus=np.asarray(problem.wave["left"]),
                        Y_plus=np.asarray(problem.wave["right"]),interpolant=exact_values,
                        mesh=np.linspace(-L,L,241),theta=system.theta,
                        info=dict(started_from="exact G2 profile",profile_error=0.0,
                                  analytic_tail_error_bound=3.25*exp(-2*L)))
        z = np.linspace(-wave.L,wave.L,2001)
        error = float(np.max(abs(wave.Y(z)-exact_values(z))))
        computed.update(L=wave.L,source=mode,exact_profile_max_error=error)
        if error > 1e-6:
            raise native.WaveError("computed profile does not match the designated G2 A5 shock")
        waves.append(wave)
        return wave

    with patch.object(native,"solve_wave",audited_solve):
        result = native.analyse(entry,**options)
    record = dict(instance_id=item["instance_id"], specification_hash=item["specification_hash"],
                  parameters=item["parameters"], coefficient_audit=audit, computed_profile=computed,
                  profile_mode=mode,native_options=options,native=result)
    if mode == "continued" and waves:
        try:
            record["continued_origin"] = continued_origin(entry,native,waves[0],options["region"],np)
        except ArithmeticError as error:
            record["continued_origin"] = dict(outcome="UNCERTAIN",open_rhp_count=None,reason=str(error))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver",type=Path,required=True,help="directory containing Jim's twstab package")
    parser.add_argument("--examples",type=Path,required=True,help="G2 batch directory")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--profile",choices=("native","exact","continued","formal","endpoints"),default="native")
    parser.add_argument("--half-length",type=float,default=20)
    args = parser.parse_args()
    if not 12 <= args.half_length <= 40:
        parser.error("--half-length must be between 12 and 40")
    solver = args.solver.resolve()
    sys.path.insert(0,str(solver))
    import numpy as np
    import sympy as sp
    from twstab import label as native
    from twstab import __version__ as solver_version
    sys.path.pop(0)
    if solver_version != "2.18.0" or not Path(native.__file__).is_relative_to(solver):
        parser.error("the adapter expects Jim's local twstab 2.18.0")
    if args.profile=="formal" and "conservation_origin_radius" not in native.DEFAULTS:
        parser.error("apply the twstab A5 interface patch before running --profile formal")
    manifest = json.loads((args.examples/"manifest.json").read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    source_hashes = {path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted((solver/"twstab").glob("*.py"))}
    write_json(args.out/"environment.json",dict(python=sys.version,numpy=np.__version__,
               sympy=sp.__version__,solver_version=solver_version,source_sha256=source_hashes))
    for ident in manifest["instances"]:
        item = json.loads((args.examples/"instances"/(ident+".json")).read_text())
        if item["instance_id"] != ident:
            raise ValueError("manifest ID does not match the instance")
        result = run_case(item,native,sp,np,args.profile,args.half_length)
        write_json(args.out/(ident+".json"),result)
        print(item["parameters"], result["native"]["label"],result["native"]["reason"],flush=True)
        if "continued_origin" in result:
            print("A5 continued origin:",result["continued_origin"]["outcome"],flush=True)


if __name__ == "__main__":
    main()
