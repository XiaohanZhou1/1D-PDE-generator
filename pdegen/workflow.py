"""Generate a seed-controlled G2 batch and run the bundled A5 solver."""
import argparse
import hashlib
import sys
from pathlib import Path

from .contract import ROOT, validate, write_json
from .generate import a5_batch
from .jim_a5_unfolding import analyse_a5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=20261124)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    solver = ROOT/"solver"
    sys.path.insert(0,str(solver))
    import numpy as np
    import scipy
    import sympy as sp
    import twstab
    from twstab import label as native

    if not Path(twstab.__file__).resolve().is_relative_to(solver.resolve()):
        raise RuntimeError("expected the solver bundled with the generator")
    args.out.mkdir(parents=True,exist_ok=False)
    items = a5_batch(args.seed)
    write_json(args.out/"environment.json",dict(seed=args.seed,python=sys.version,
               numpy=np.__version__,scipy=scipy.__version__,sympy=sp.__version__,twstab=twstab.__version__,
               solver_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted((solver/"twstab").glob("*.py"))}))
    entries = []
    for item in items:
        validate(item)
        ident = item["instance_id"]
        instance_path = f"instances/{ident}.json"
        result_path = f"results/{ident}.json"
        write_json(args.out/instance_path,item)
        result = analyse_a5(item,native,sp)
        count = result.get("closed_half_plane",{}).get("open_rhp_count")
        write_json(args.out/result_path,dict(instance_id=ident,specification_hash=item["specification_hash"],
                   parameters=item["parameters"],native=result))
        entries.append(dict(instance=instance_path,result=result_path,label=result["label"],
                            open_rhp_count=count))
        print(item["parameters"],result["label"],count,flush=True)
    write_json(args.out/"manifest.json",dict(seed=args.seed,entries=entries))


if __name__ == "__main__":
    main()
