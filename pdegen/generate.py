import argparse
from fractions import Fraction
import hashlib
from pathlib import Path
from .contract import construct, unique, write_json
from .families import a1, a5, a5_fixture


def batch(seed):
    # A small dyadic neighbourhood of (1,2) for reproducible local root scouting.
    bits = hashlib.sha256(f"A1-pilot-v1/{seed}".encode()).digest()
    q = 1 - Fraction(1 + bits[0] % 4, 128)
    a = 2 + Fraction(1 + bits[1] % 4, 128)
    beta = Fraction(1 if bits[2] % 2 else -1, 4)
    models = [a1(0,4), a1(1,2), a1(q,a), a1(q,a,beta)]
    return unique([construct(m,seed) for m in models])


def a5_batch(seed, mu_values=None):
    if mu_values is None:
        bits = hashlib.sha256(f"A5-pilot-v1/{seed}".encode()).digest()
        # One new point on each side of mu=3, strictly inside the reviewed interval.
        lower = 1 + Fraction(1 + bits[0] % 15, 8)
        upper = 3 + Fraction(1 + bits[1] % 23, 8)
        mu_values = [1,3,6,lower,upper]
    return unique([construct(a5(mu),seed) for mu in mu_values])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20261124)
    p.add_argument("--family", choices=("A1","A5"), default="A1")
    p.add_argument("--mu", action="append", help="exact A5 parameter, repeat for multiple values")
    args = p.parse_args()
    if args.mu and args.family != "A5":
        p.error("--mu requires --family A5")
    try:
        records = a5_batch(args.seed,args.mu) if args.family=="A5" else batch(args.seed)
    except (ValueError,ZeroDivisionError) as exc:
        p.error(str(exc))
    for record in records:
        write_json(args.out/"instances"/(record["instance_id"]+".json"),record)
    manifest = dict(seed=args.seed,instances=[r["instance_id"] for r in records])
    if args.family=="A1":
        manifest["fixture"] = "fixtures/a5-mu3.json"
        write_json(args.out/manifest["fixture"],construct(a5_fixture(),args.seed))
    write_json(args.out/"manifest.json",manifest)
    print(f"{len(records)} {args.family} candidates")


if __name__ == "__main__":
    main()
