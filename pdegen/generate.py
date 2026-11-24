import argparse
from fractions import Fraction
import hashlib
from pathlib import Path
from .contract import construct, unique, write_json
from .families import a1, a5_fixture


def batch(seed):
    # A small dyadic neighbourhood of (1,2) for reproducible local root scouting.
    bits = hashlib.sha256(f"A1-pilot-v1/{seed}".encode()).digest()
    q = 1 - Fraction(1 + bits[0] % 4, 128)
    a = 2 + Fraction(1 + bits[1] % 4, 128)
    beta = Fraction(1 if bits[2] % 2 else -1, 4)
    models = [a1(0,4), a1(1,2), a1(q,a), a1(q,a,beta)]
    return unique([construct(m,seed) for m in models])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20261124)
    args = p.parse_args()
    records = batch(args.seed)
    for record in records:
        write_json(args.out/"instances"/(record["instance_id"]+".json"),record)
    write_json(args.out/"fixtures"/"a5-mu3.json",construct(a5_fixture(),args.seed))
    write_json(args.out/"manifest.json",dict(seed=args.seed, instances=[r["instance_id"] for r in records],
                                            fixture="fixtures/a5-mu3.json"))
    print(f"{len(records)} A1 candidates and one A5 contract fixture")


if __name__ == "__main__":
    main()
