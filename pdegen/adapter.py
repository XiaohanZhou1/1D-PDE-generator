"""Bridge for the attached solver release. Root scouting data stay on the answer side."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from fractions import Fraction
from .contract import ROOT, digest, validate, schema_check, write_json
from .algebra import evaluate
from .families import a1, a5, dispatch_limitation

def check_release(root):
    for name, expected in json.loads((ROOT/"schemas/solver-release-files.json").read_text()).items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"unreviewed solver file: {name}")


def parity(item, model):
    import numpy as np
    z = np.array([-8.,-2.,-.37,0.,.63,1.7,8.])
    wave_env = dict(s=1/np.cosh(z), t=np.tanh(z))
    def array(polys, env):
        return np.stack([np.broadcast_to(evaluate(p,env),z.shape) for p in polys],axis=-1)
    def same(actual, expected):
        np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)
    X = model.profile(z)
    same(array(item["target_wave"]["profile"],wave_env),X)
    # Independently exercise off-wave states so that vanishing feedback is visible.
    perturbed = X + np.sin(np.arange(X.size).reshape(X.shape)+1)/3
    off = dict(zip(item["variables"],perturbed.T))
    eq = item["equations"]
    if model.family=="A1":
        same(array(eq["F"],off),model.reaction(perturbed))
        same(array(eq["G"],off),np.stack([0*z,6*perturbed[:,0]**3,0*z,0*z],axis=1))
        same(array(eq["D"],off),np.tile(model.diffusion,(len(z),1)))
        same(model.speed-array(eq["A"],off),np.tile(model.drift,(len(z),1)))
        B,C = model.abc(X)
        for field,numeric in (("B",B),("C",C)):
            for i,row in enumerate(item["linearised_operator"][field]):
                same(array(row,wave_env),numeric[:,i,:])
    else:
        same(-array(eq["G"],off),model.flux(perturbed))
        operator = item["linearised_operator"]
        same(array(operator["D"],wave_env),np.ones_like(X))
        same(item["target_wave"]["speed"],model.speed)
        u = X[:,0]
        J = model.jac(u)
        # Independently differentiate the native Jacobian along the native profile.
        step = 1e-5
        dJ = (model.jac(model.profile(z+step)[:,0]) -
              model.jac(model.profile(z-step)[:,0]))/(2*step)
        for i,(b_row,c_row) in enumerate(zip(operator["B"],operator["C"],strict=True)):
            same(array(b_row,wave_env),-J[:,i,:])
            np.testing.assert_allclose(array(c_row,wave_env),-dJ[:,i,:],rtol=2e-8,atol=2e-8)
        for sign,side in ((-1,"left"),(1,"right")):
            env = dict(s=0,t=sign)
            end = array(item["target_wave"]["endstates"][(sign+1)//2],env)
            same(end,model.profile(np.full(z.shape,30.0*sign)))
            tail = item["far_field_data"]["coefficients"][side]
            for i,b_row in enumerate(tail["B"]):
                same(array(b_row,env),-model.jac(end[:,0])[:,i,:])


def prepare(item, root):
    import numpy as np
    from spectral.models import A1,A5
    from spectral.checks import eigenpairs
    params = {k:float(Fraction(v)) for k,v in item["parameters"].items()}
    model = A1(params["q"],params["a"]) if item["family_id"]=="A1" else A5(params["mu"])
    parity(item,model)
    cfg = json.loads((root/"benchmark.json").read_text())
    if model.family=="A5":
        case = ["A5",model.mu]
    else:
        case = next((c for c in cfg["cases"] if c[:3]==["A1",model.q,model.a]),None)
        if case is None:
            # Scouting branch implemented only for the pilot neighbourhood.
            if not (31/32 <= model.q <= 1 and 2 <= model.a <= 2+1/32):
                raise ValueError("no reviewed root-scouting policy for this parameter point")
            scout = eigenpairs(model,14,.08,[4.47+1.11j])[0]["value"]
            if not np.isfinite(scout).all() or scout[1]<=0:
                raise ValueError("scout returned no finite upper-half-plane centre")
            case = ["A1",model.q,model.a,[scout,[scout[0],-scout[1]],[0,0]]]
    cfg["cases"] = [case]
    return cfg


def execute(item, solver, dest):
    validate(item)
    start = time.perf_counter()
    dest.mkdir(parents=True,exist_ok=False)
    result = dict(schema_version="0.1-draft",instance_id=item["instance_id"],
        specification_hash=item["specification_hash"],solver_status="NOT_RUN",solver_version=None,
        settings_hash=None,label=None,evidence_grade=None,unstable_count=None,complete_numerical_count=False,
        spectral_search=None,verification=None,artifact_refs=[],runtime_seconds=0,failure_reason=None,
        post_solver_review=dict(disposition="HOLD",finding="Awaiting compatible solver."))
    model = a1(**item["parameters"]) if item["family_id"]=="A1" else a5(**item["parameters"])
    limitation = dispatch_limitation(model)
    if limitation:
        result["failure_reason"]=limitation
    else:
        try:
            root = solver.resolve()
            check_release(root)
            for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
                os.environ[key]="1"
            sys.path.insert(0,str(root))
            try:
                cfg = prepare(item,root)
            finally:
                sys.path.pop(0)
            write_json(dest/"settings.json",cfg)
            result["settings_hash"]=digest(cfg)
            result["artifact_refs"].append("settings.json")
            proc = subprocess.run([sys.executable,str(root/"run.py"),"--config",str((dest/"settings.json").resolve()),
                                   "--output",str((dest/"raw").resolve())],capture_output=True,text=True)
            (dest/"solver.log").write_text(proc.stdout+proc.stderr,encoding="utf-8")
            result["artifact_refs"].append("solver.log")
            proc.check_returncode()
            records = list((dest/"raw").glob("*/case.json"))
            if len(records)!=1:
                raise ValueError("expected exactly one fresh native result")
            raw = json.loads(records[0].read_text())
            expected = {k:float(Fraction(v)) for k,v in item["parameters"].items() if k!="beta"}
            if raw["problem"]!=item["family_id"] or raw["parameters"]!=expected or raw["settings"]!=cfg:
                raise ValueError("solver returned different input")
            paths = [records[0],dest/"raw/environment.json"] + [dest/"raw"/p for p in raw["artifacts"]]
            if not all(p.is_file() for p in paths):
                raise ValueError("incomplete evidence files")
            if raw["label"]=="UNCERTAIN" or not raw["complete_numerical_count"]:
                finding = "Complete count remains unresolved. Retain available witness diagnostics."
            elif item["parameters"].get("q")=="0":
                finding = "Analytically reducible calibration point. Exclude from proof training."
            elif raw["problem"]=="A5":
                finding = "Fixed-parameter shock result. Interval-wide finite-frequency exclusion remains open."
            else:
                witnesses = [raw["roots"][i]["lambda_value"] for i in raw["positive_witness_indices"]]
                nonreal = sum(abs(im)>raw["settings"]["local_radius"] for _,im in witnesses)
                finding = (f"{nonreal} positive witnesses have disks disjoint from the real axis. "
                           "Nearby samples share a proof family. Collision and completeness proofs require review.")
            result.update(solver_status="COMPLETED",solver_version=raw["solver_version"],label=raw["label"],
                evidence_grade=raw["evidence_grade"],unstable_count=raw["numerical_open_rhp_count"],
                complete_numerical_count=raw["complete_numerical_count"],spectral_search=raw.get("search"),
                verification=dict(roots=raw.get("roots",[]),positive_witness_indices=raw.get("positive_witness_indices",[]),
                                  native_elapsed_seconds=raw["elapsed_seconds"]),
                failure_reason=raw["failure_reason"],post_solver_review=dict(disposition="HOLD",finding=finding))
            result["artifact_refs"].extend(p.relative_to(dest).as_posix() for p in paths)
        except Exception as exc:
            result.update(solver_status="FAILED",failure_reason=f"{type(exc).__name__}: {exc}",
                          post_solver_review=dict(disposition="HOLD",finding="Bridge or solver execution failed."))
    result["runtime_seconds"]=time.perf_counter()-start
    schema_check("result",result)
    write_json(dest/"result.json",result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--solver",type=Path,required=True)
    parser.add_argument("--examples",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.examples/"manifest.json").read_text())
    paths=[args.examples/"instances"/(k+".json") for k in manifest["instances"]]
    if "fixture" in manifest:
        paths.append(args.examples/manifest["fixture"])
    failed=False
    for path in paths:
        item=json.loads(path.read_text())
        result=execute(item,args.solver,args.out/item["instance_id"])
        print(item["parameters"],result["solver_status"],result["label"],result["unstable_count"],flush=True)
        failed |= result["solver_status"]=="FAILED"
    raise SystemExit(int(failed))


if __name__=="__main__":
    main()
