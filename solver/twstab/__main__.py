"""Command line:  python -m twstab list.py [more.py ...] [--out DIR] [--single-pass] [--verbose]

A file that defines the list `problems` (batch.py): one line for each wave of each equation, and the
output list in <out>/<file>_labels.json.

A file with one problem.  The output is one line for each wave: its label.  The evidence is in the JSON
record.  A file that names a wave gives the label of this wave.  A file without `wave` starts
the search of survey.py: equilibria, candidates, and one line for each wave found.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

from .batch import lines, solve
from .label import analyse
from .problem import ProblemError, load_problem
from .survey import summary, survey


def _list_of(path):
    """The list `problems` of the file; None: the file holds one problem."""
    if not Path(path).is_file():
        return None                                       # analyse() reports the error
    ns = runpy.run_path(str(Path(path).resolve()))
    return list(ns["problems"]) if "problems" in ns else None


def main():
    if sys.argv[1:2] == ["floquet"]:
        from .temporal_floquet import main as floquet_main
        floquet_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(prog="twstab", description="Spectral-stability label of a travelling wave.",
                                     epilog="For sampled F2 orbits, use %(prog)s floquet --help.")
    parser.add_argument("files", nargs="+", type=Path, help="problem files written with SymPy")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "results",
                        help="directory for the JSON records")
    parser.add_argument("--single-pass", action="store_true", help="skip the audit pass")
    parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                        help="replace parameter values of the problem file")
    parser.add_argument("--verbose", action="store_true", help="also print speed, time, reason and roots")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    values = {item.split("=", 1)[0]: float(item.split("=", 1)[1]) for item in args.set}
    tag = "".join(f"_{k}{v:g}" for k, v in values.items())
    for path in args.files:
        problems = _list_of(path)
        if problems is not None:
            if values:
                parser.error("--set is not available for a file with the list `problems`")
            results = solve(problems, on_result=lambda r: print("\n".join(lines(r)), flush=True),
                            two_pass=not args.single_pass)
            target = args.out / (Path(path).stem + "_labels.json")
            target.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"{len(results)} equations, {sum(r['seconds'] for r in results):.1f} s; output list: {target}")
            continue
        target = args.out / (Path(path).stem + tag + ".json")
        try:
            searching = load_problem(path, values).wave is None
        except ProblemError:
            searching = False                             # analyse() reports the error
        if searching:
            result = survey(path, two_pass=not args.single_pass, parameters=values)
            target = target.with_name("search_" + target.name)       # a problem file can have the same name
            target.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print("\n".join(summary(result)) + f"\n  ({result['seconds']:.1f} s)")
            continue
        record = analyse(path, two_pass=not args.single_pass, parameters=values)
        target.write_text(json.dumps(record, indent=2), encoding="utf-8")
        # A weaker sense of STABLE: the qualifier is always printed with the label.
        qualified = f" ({record['qualifier']})" if record.get("qualifier") and record["label"] == "STABLE" else ""
        if not args.verbose:
            note = f"   ({record['reason']})" if record["label"] == "UNCERTAIN" else ""
            if record.get("weighted"):                    # unstable in L2 through the essential spectrum
                note = f" in L2; in an exponentially weighted space: {record['weighted']['label']}"
            print(f"{Path(path).stem + tag}: {record['label']}{qualified or note}")
            continue
        speed = record.get("wave", {}).get("speed")
        print(f"{Path(path).stem}: {record['label']}{qualified} (pass {record['decided_in_pass']}), "
              f"c = {speed if speed is None else format(speed, '.8g')}, "
              + (f"omega = {record['symmetry']['frequency']:.8g}, " if record.get("symmetry", {}).get("frequency") is not None else "")
              +
              f"{record['timings']['total']:.2f} s")
        print(f"  reason: {record['reason']}")
        branch = record.get("wave", {}).get("continuation")
        if branch:
            folds = ", ".join(f"{f['value']:.6g}" for f in branch["folds"]) or "none"
            print(f"  continuation in {branch['parameter']}: {branch['start']:g} -> {branch['end']:g}, "
                  f"{branch['steps']} steps, folds at: {folds}")
        for root in record.get("roots", []):
            re, im = root["value"]
            print(f"  root {re:+.8f} {im:+.8f}i  error {root['error']:.2e}  {root['status']}")
        print(f"  record: {target}")


if __name__ == "__main__":
    main()
