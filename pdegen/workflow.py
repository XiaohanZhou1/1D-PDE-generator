"""Generate, audit and solve a batch with its family-specific backend."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from .contract import ROOT, validate, write_json
from .audit import a5_evidence, audit_batch
from .bloch import bloch_batch, solve_roll, validate_roll
from .cf_calibration import (CALIBRATION_FAMILIES, calibration_batch, certify_calibration,
                             native_reference, validate_calibration)
from .generate import a5_batch, batch
from .floquet import solve_spatial_orbit, spatial_orbit, validate_spatial_orbit
from .jim_a5_unfolding import analyse_a5
from .periodic import periodic_batch, solve_periodic_roll, validate_periodic_roll


def bundled_solver():
    solver = ROOT/"solver"
    sys.path.insert(0,str(solver))
    import sympy as sp
    import twstab
    from twstab import label as native

    if not Path(twstab.__file__).resolve().is_relative_to(solver.resolve()):
        raise RuntimeError("expected the solver bundled with the generator")
    return sp, twstab, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=20261124)
    parser.add_argument("--family",choices=("A1","A5","B1","B2",*CALIBRATION_FAMILIES,"F2"),default="A5")
    parser.add_argument("--audit-only",action="store_true")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--roll", nargs=2, action="append", metavar=("Q", "AMPLITUDE"),
                        help="exact B2 wavenumber and first cosine amplitude")
    parser.add_argument("--native-check", action="store_true",
                        help="also run the bundled solver against a C1–F1 calibration")
    parser.add_argument("--instance", type=Path, help="read a generated F2 instance JSON")
    args = parser.parse_args()
    if args.roll and args.family != "B2":
        parser.error("--roll requires --family B2")
    if args.native_check and (args.family not in CALIBRATION_FAMILIES or args.audit_only):
        parser.error("--native-check requires a C1–F1 calibration solve")
    if args.instance is not None and args.family != "F2":
        parser.error("--instance requires --family F2")
    import numpy as np
    import scipy
    environment = dict(seed=args.seed,family=args.family,python=sys.version,
                       numpy=np.__version__,scipy=scipy.__version__,
                       solver_enabled=not (args.audit_only or args.family=="A1"))
    if args.family=="A1":
        items = batch(args.seed)
        validator,backend = validate,None
        environment["backend"] = "audit only"
    elif args.family=="B1":
        items = bloch_batch(args.seed)
        validator,backend = validate_roll,solve_roll
        environment["backend"] = "B1 analytical dispersion and physical finite differences"
    elif args.family=="B2":
        try:
            items = periodic_batch(args.seed, args.roll)
        except (ValueError, ZeroDivisionError) as exc:
            parser.error(str(exc))
        validator,backend = validate_periodic_roll,solve_periodic_roll
        environment["backend"] = "B2 Fourier continuation, Bloch Galerkin and physical finite differences"
        source = ROOT / "pdegen" / "periodic.py"
        environment["solver_source_sha256"] = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
    elif args.family=="F2":
        item = spatial_orbit(args.seed)
        if args.instance is not None:
            try:
                raw = args.instance.read_bytes()
                item = json.loads(raw)
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
            environment["input_sha256"] = hashlib.sha256(raw).hexdigest()
            if isinstance(item,dict) and type(item.get("seed")) is int:
                environment["seed"] = item["seed"]
        items = [item]
        validator,backend = validate_spatial_orbit,solve_spatial_orbit
        environment["backend"] = "F2 spatial Fourier–Bloch discretisation and temporal monodromy"
        environment["solver_source_sha256"] = {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in ("pdegen/floquet.py", "schemas/floquet-instance.json", "docs/f2.md")}
    elif args.family in CALIBRATION_FAMILIES:
        items = calibration_batch(args.family,args.seed)
        validator,backend = validate_calibration,certify_calibration
        source = ROOT / "pdegen" / "cf_calibration.py"
        evidence_sources = [source, ROOT / "docs" / "cf-calibration.md"]
        environment["backend"] = "C–F exact calibration proofs and F1 numerical monodromy regression"
        if args.native_check:
            analyse = None
            if args.family != "F1":
                sp,twstab,native = bundled_solver()
                analyse = native.analyse
                environment.update(sympy=sp.__version__,twstab=twstab.__version__)
                evidence_sources.extend((ROOT/"solver"/"twstab").glob("*.py"))
                for name in ("gkdv_soliton.py", "delay_kink_uniform.py", "stefan_linear.py",
                             "stefan_linear_moving.py"):
                    evidence_sources.append(ROOT/"solver"/"problems"/name)

            def backend(item):
                result = certify_calibration(item)
                try:
                    result["native_check"] = native_reference(item, analyse)
                except (ArithmeticError, RuntimeError, ValueError, OSError) as error:
                    result["native_check"] = dict(status="ERROR", reason=str(error))
                return result
        environment["solver_source_sha256"] = {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(evidence_sources)}
    elif args.audit_only:
        items = a5_batch(args.seed)
        validator,backend = validate,None
        environment["backend"] = "audit only"
    else:
        solver = ROOT/"solver"
        sp,twstab,native = bundled_solver()
        items = a5_batch(args.seed)
        validator = validate
        def backend(item):
            return a5_evidence(analyse_a5(item,native,sp))
        environment.update(sympy=sp.__version__,twstab=twstab.__version__,
                           solver_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                                 for p in sorted((solver/"twstab").glob("*.py"))})
    args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/"environment.json",environment)
    def solve(item):
        result = backend(item)
        print(item["parameters"],result["label"],result["open_rhp_count"],flush=True)
        return result

    entries = audit_batch(items,args.out,validator,solve if backend is not None and not args.audit_only else None)
    write_json(args.out/"manifest.json",dict(seed=environment["seed"],entries=entries))


if __name__ == "__main__":
    main()
