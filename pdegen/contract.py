import hashlib
import json
from fractions import Fraction
from pathlib import Path
from jsonschema import Draft202012Validator
from . import VERSION
from .algebra import poly, encode, reduce_wave, dz, t
from .families import a1, a5, dispatch_limitation

ROOT = Path(__file__).resolve().parents[1]
A1_CAP = ["A1.design-v1", "expanded.second_order", "whole_line.unweighted",
          "exterior_graphs", "translation", "complete_open_rhp_count"]
A5_CAP = ["A5.design-v1", "conservative.flux", "integrated.crosscheck",
          "whole_line.unweighted", "continued.origin"]
MATH_KEYS = ("family_id","structure_id","template_version","variables","parameters","equations",
             "spatial_domain","boundary_conditions","target_wave","wave_id","branch_id",
             "function_space","operator_domain","weights","linearised_operator","far_field_data",
             "stability_question","task_type","stability_convention","neutral_mode_policy")


def digest(obj):
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode())


def schema_check(name, value):
    schema = json.loads((ROOT/"schemas"/(name+".json")).read_text())
    Draft202012Validator(schema).validate(value)


def require_zero(values, message):
    if any(reduce_wave(x).terms for x in values):
        raise ValueError(message)


def check_a5_shock(model, U1, B, C, wave_value):
    D, A, F, G = (model[key] for key in ("D","A","F","G"))
    mu, rho = (model["params"][key] for key in ("mu","rho"))
    if D != [1,1] or A != [0,0] or any(poly(f).terms for f in F):
        raise ValueError("A5 requires identity viscosity and a pure conservative flux")
    center = {"s":1,"t":0}
    if (model["wave"][0].at(center) != 0 or U1[0].at(center) <= 0 or model["c"] != 0):
        raise ValueError("shock phase or speed")
    left, right = (dict(zip(model["names"],end,strict=True)) for end in model["ends"])
    if any(poly(g).at(right)-poly(g).at(left) != model["c"]*(left[name]-right[name])
           for name,g in zip(model["names"],G,strict=True)):
        raise ValueError("A5 Rankine–Hugoniot identity")
    # The pinned native model uses zero far-field flux, fixing the constant in the first integral.
    if any(poly(g).at(left) for g in G):
        raise ValueError("A5 requires zero far-field flux")
    require_zero([derivative+wave_value(g) for derivative,g in zip(U1,G,strict=True)],
                 "integrated shock identity")

    # B = -J, where J is the on-wave physical flux Jacobian.
    trace = -B[0][0]-B[1][1]
    determinant = B[0][0]*B[1][1]-B[0][1]*B[1][0]
    discriminant = trace**2-4*determinant
    expected = 9*t*t+4*mu*(1+Fraction(1,2)*t)*(1+Fraction(5,4)*t)*(1-t*t)
    require_zero([discriminant-expected], "A5 on-wave hyperbolicity formula")
    if not (1 <= mu <= 6 and rho == 0):
        raise ValueError("A5 strict hyperbolicity bound requires the reviewed shock branch")
    # On [-1,-4/5], 9t² >= 144/25 and the negative term is bounded by 54/25.
    # On [-4/5,1], both terms are nonnegative and they cannot vanish together.
    for sign in (-1,1):
        tail = {"s":0,"t":sign}
        # Roots of the endpoint characteristic polynomial are -2*sign and sign.
        # One incoming characteristic at each end and a 2+2 spatial split in the open RHP.
        if (trace.at(tail) != -sign or determinant.at(tail) != -2
                or any(entry.at(tail) for row in C for entry in row)):
            raise ValueError("A5 endstate characteristic speeds or spectral limits")


def construct(model, seed):
    names, U = model["names"], model["wave"]
    n, c = len(names), model["c"]
    D, A, F, G = (model[k] for k in ("D","A","F","G"))
    if min(D) <= 0:
        raise ValueError("diffusion must be positive")
    on = dict(zip(names, U, strict=True))
    def wave_value(f):
        return reduce_wave(poly(f).at(on))
    U1, U2 = list(map(dz,U)), [dz(dz(u)) for u in U]
    require_zero([D[i]*U2[i]+(c-A[i])*U1[i]+wave_value(F[i])+dz(wave_value(G[i]))
                  for i in range(n)], "travelling-wave identity")
    B, C = [], []
    for i in range(n):
        row = [wave_value(poly(G[i]).diff(v)) for v in names]
        B.append([v + ((c-A[i]) if i == j else 0) for j,v in enumerate(row)])
        C.append([wave_value(poly(F[i]).diff(v)) + dz(row[j]) for j,v in enumerate(names)])
    require_zero([D[i]*dz(U2[i])+sum(B[i][j]*U2[j]+C[i][j]*U1[j] for j in range(n))
                  for i in range(n)], "translation identity")
    far = {}
    for side, sign, end in zip(("left","right"), (-1,1), model["ends"]):
        env = {"s":0, "t":sign}
        if [poly(v).at(env) for v in U] != list(end):
            raise ValueError("endstate mismatch")
        if any(poly(f).at(dict(zip(names,end))) for f in F):
            raise ValueError("endstate equilibrium")
        far[side] = encode({k:[[v.at(env) for v in row] for row in mat]
                            for k,mat in (("B",B),("C",C))})
    pulse = model["family"] == "A1"
    params = model["params"]
    feedback = pulse and bool(params["beta"])
    calibration = pulse and params["q"] == 0 and not feedback
    if pulse:
        center = {"s":1,"t":0}
        if U[0].at(center)!=1 or U1[0].at(center)!=0 or U2[0].at(center)!=-1 or c!=1:
            raise ValueError("pulse phase or speed")
        q, a = params["q"], params["a"]
        expected_B = [[(-q/2,0,q/2,-q)[i] if i==j else 0 for j in range(4)] for i in range(4)]
        expected_C = [[-1,q/2,0,0],[0,-3,0,0],[0,0,-a,-2],[0,0,2,-a]]
        if any(v != encode(dict(B=expected_B,C=expected_C)) for v in far.values()):
            raise ValueError("A1 analytic far-field bound does not apply")
    else:
        check_a5_shock(model,U1,B,C,wave_value)
    family = model["family"]
    wave_id = "sech-maximum-0" if pulse else "tanh-zero-0"
    task = "unstable_count" if pulse else "binary_stability"
    item = dict(schema_version="0.1-draft", generator_version=VERSION, seed=seed, family_id=family,
                template_version="1", structure_id=family+(".normal-y" if feedback else ".design-v1"),
                variables=names, parameters={k:str(v) for k,v in params.items()},
                equations=dict(form="D*X_xx-A*X_x+F+dx(G)", **encode(dict(D=D,A=A,F=F,G=G))),
                spatial_domain="real_line", boundary_conditions="specified limits at both infinities",
                target_wave=dict(basis="s=sech(z),t=tanh(z),z=x-c*time", speed=c, profile=encode(U),
                                 dz=encode(U1), dzz=encode(U2), endstates=encode(model["ends"]),
                                 phase="u(0)=1, u'(0)=0, u''(0)<0" if pulse else "u(0)=0, u'(0)>0",
                                 existence="EXACT_SUBSTITUTION"),
                wave_id=wave_id, branch_id=family+".explicit."+wave_id, function_space=f"L2(R)^{n}",
                operator_domain=f"H2(R)^{n}", weights="identity",
                linearised_operator=dict(form="D*p''+B*p'+C*p", **encode(dict(D=D,B=B,C=C))),
                far_field_data=dict(symbol="-k^2*diag(D)+i*k*B+C", coefficients=far,
                                    essential="Re<=-1" if pulse else
                                    "parabolas for b=-2,-1,1,2 and index region -Im^2<Re<-Im^2/4",
                                    rhp_splitting=[n,n]),
                stability_question="Determine N_+ at these parameters, counting algebraic multiplicity."
                                   if pulse else f"Is this shock spectrally stable modulo translation at mu={params['mu']}?",
                task_type=task, stability_convention="open_RHP_count" if pulse else "unweighted_shock_edge",
                neutral_mode_policy="Exclude all axis modes from N_+; record translation and extra modes."
                                    if pulse else "Permit essential edge at 0 and simple continued translation only.",
                required_solver_capabilities=(A1_CAP + (["normal_y.feedback"] if feedback else []))
                                             if pulse else A5_CAP,
                expected_proof_mechanisms=["scalar block count"] if calibration else
                    (["coupled point spectrum", "exclude missing unstable roots"] if pulse else
                     ["conservative origin matching","finite-frequency exclusion"]),
                diversity_level="L2" if feedback else "L1",
                novelty_assessment=dict(status="NOT_NOVEL" if calibration else "UNCERTAIN",
                    screen="reducible seed" if calibration else
                        ("new feedback requires reduction review" if feedback else
                         ("coupled operator; fixed-point count needs expert review" if pulse else
                          ("contract fixture; finite-frequency exclusion requires review" if params["mu"]==3 else
                           "A5 parameter variant; finite-frequency exclusion requires review"))),
                    grouping=family+".explicit-family", training_admission="HOLD"),
                validity_checks=dict(profile="PASS",endstates="PASS",phase="PASS",diffusion="PASS",
                    linearisation="PASS",translation="PASS", global_nonlinear_existence="UNVERIFIED",
                    proof_difficulty="UNVERIFIED" if not calibration else "NOT_APPLICABLE",
                    solver_dispatch="FAIL" if dispatch_limitation(model) else "PASS"),
                references=["5_1D_problems_v1.txt#"+family, "independent_audit_1.txt#"+family,
                            "docs/mathematics.md", "docs/interface.md"])
    item["specification_hash"] = digest({k:item[k] for k in MATH_KEYS})
    item["instance_id"] = family+"-"+item["specification_hash"][:20]
    schema_check("instance",item)
    return item


def validate(item):
    schema_check("instance", item)
    p = item["parameters"]
    if item["family_id"] == "A1":
        model = a1(p["q"],p["a"],p["beta"])
    else:
        model = a5(**p)
    if item != construct(model,item["seed"]):
        raise ValueError("record differs from the versioned construction or canonical hash")
    return item


def unique(records):
    seen = set()
    for record in records:
        key = record["specification_hash"]
        if key in seen:
            raise ValueError("duplicate mathematical specification")
        seen.add(key)
    return records
