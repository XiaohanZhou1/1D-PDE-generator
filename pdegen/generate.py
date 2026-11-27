import argparse
from fractions import Fraction
import hashlib
from pathlib import Path
from .contract import construct, unique, write_json
from .bloch import bloch_batch
from .cf_calibration import CALIBRATION_FAMILIES, calibration_batch
from .families import a1, a5, a5_fixture
from .floquet import spatial_orbit
from .periodic import periodic_batch


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
    p.add_argument("--family", choices=("A1","A5","B1","B2",*CALIBRATION_FAMILIES,"F2"), default="A1")
    p.add_argument("--mu", action="append", help="exact A5 parameter, repeat for multiple values")
    p.add_argument("--q2", action="append", help="exact B1 squared wavenumber, repeat for multiple values")
    p.add_argument("--roll", nargs=2, action="append", metavar=("Q", "AMPLITUDE"),
                   help="exact B2 wavenumber and first cosine amplitude, repeat for multiple rolls")
    args = p.parse_args()
    if args.mu and args.family != "A5":
        p.error("--mu requires --family A5")
    if args.q2 and args.family != "B1":
        p.error("--q2 requires --family B1")
    if args.roll and args.family != "B2":
        p.error("--roll requires --family B2")
    try:
        if args.family=="B1":
            records = bloch_batch(args.seed,args.q2)
        elif args.family=="B2":
            records = periodic_batch(args.seed,args.roll)
        elif args.family=="F2":
            records = [spatial_orbit(args.seed)]
        elif args.family in CALIBRATION_FAMILIES:
            records = calibration_batch(args.family,args.seed)
        else:
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
