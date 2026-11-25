"""Exercise Jim's fixed-speed unfolding API on the G2 A5 batch and controls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from .a5_bvp_audit import configure_rho, triangular_front
from .contract import validate, write_json
from .jim_a5 import exact_search_region, problem_entry


def analyse_a5(item, native, sp):
    """Solve the original A5 flux at the supplied exact target parameters."""
    entry = configure_rho(problem_entry(item,sp),sp)
    return native.analyse(entry,conservation_method="axis",
                          conservation_origin_radius=.08,region=exact_search_region(item,sp))


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
    from scipy.optimize import brentq
    from twstab import label as native
    from twstab.conservation import ConservativeSystem
    from twstab.conservative_unfolding import unfold_wave
    from twstab.symbolic import LocalSystem

    def profile_error(entry, cubic):
        problem = native.load_problem(entry)
        cons = ConservativeSystem(problem,LocalSystem(problem))
        wave = unfold_wave(problem,cons,native.DEFAULTS)
        shift = brentq(lambda z:wave.Y([z])[0,0],-wave.L, wave.L)
        z = np.linspace(-8,8,1601)
        u = np.tanh(z)
        v = u*u/2+(5*u**3/24 if cubic else 0)
        return float(np.max(abs(wave.Y(z+shift)-np.vstack((u,v)))))

    manifest = json.loads((args.examples/"manifest.json").read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    source = args.solver/"twstab"
    write_json(args.out/"environment.json",dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
               sympy=sp.__version__,twstab=twstab.__version__,source_sha256={
                   p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))}))
    a5_wrong = None
    for ident in manifest["instances"]:
        item = validate(json.loads((args.examples/"instances"/(ident+".json")).read_text()))
        entry = configure_rho(problem_entry(item,sp),sp)
        spectrum = analyse_a5(item,native,sp)
        error = profile_error(entry,True)
        write_json(args.out/(ident+".json"),dict(specification_hash=item["specification_hash"],
                   parameters=item["parameters"],phase_aligned_max_error=error,native=spectrum))
        assert spectrum["wave"]["started_from"]=="parameter_unfolding" and error<1e-7
        assert spectrum["label"]=="STABLE" and spectrum["closed_half_plane"]["open_rhp_count"]==0
        if item["parameters"]["mu"]=="3":
            a5_wrong = native.analyse(configure_rho(problem_entry(item,sp),sp,target=.05))
        print(item["parameters"],spectrum["label"],error,flush=True)

    triangular = configure_rho(triangular_front(sp),sp)
    second = native.analyse(triangular,conservation_method="axis",default_region=False)
    error = profile_error(triangular,False)
    wrong = native.analyse(configure_rho(triangular_front(sp),sp,target=.05))
    p_system = native.analyse(args.solver/"problems/p_system_shock.py")
    write_json(args.out/"controls.json",dict(triangular=second,triangular_phase_aligned_max_error=error,
                                            wrong_target=wrong,a5_wrong_target=a5_wrong,p_system=p_system))
    assert second["wave"]["computed"] and second["wave"]["started_from"]=="parameter_unfolding" and error<1e-7
    for rejected in (wrong,a5_wrong):
        assert rejected["label"]=="UNCERTAIN" and "differs from target" in rejected["reason"]
        assert "closed_half_plane" not in rejected and not rejected["wave"].get("computed")
    assert p_system["wave"]["computed"] and p_system["wave"]["started_from"]=="guess"
    print("triangular profile",error,"wrong target",wrong["reason"],"p-system",p_system["label"])


if __name__ == "__main__":
    main()
