"""Freeze and measure one prospective A–F pilot through the shared generator."""
import argparse
from collections import Counter
import hashlib
from itertools import product
from pathlib import Path
import shutil
from time import perf_counter

from .batch_workflow import candidates, load_reviews, read_json, review_status, run
from .contract import ROOT, digest, write_json

SEED = 64312026
PER_STRATUM = 4
# Values and boundaries are fixed before the prospective solver runs.
STRATA = {
    "A1": {"keys": ("q", "a"), "low": (("31/32", "125/128"), ("257/128", "129/64", "259/128", "65/32")),
           "high": (("63/64", "127/128"), ("257/128", "129/64", "259/128", "65/32"))},
    "B3": {"keys": ("damping", "skew"), "low": (("5/2", 3, "7/2", 4), (1, 2, 3, 4)),
           "high": (("9/2", 5, "11/2", 6, 7), (1, 2, 3, 4))},
    "C2": {"keys": ("alpha", "beta"), "low": (("1/4", "3/8", "1/2", "3/4"), ("1/16", "1/8", "1/4")),
           "high": (("7/4", 2, "9/4", "5/2", "11/4", 3, 4, 5), ("1/4", "1/2", "3/4", 1))},
    "D2": {"keys": ("sigma", "tau", "epsilon"),
           "low": ((2, "5/2", 3), (1, "5/4", "3/2", "7/4", 2), ("1/4", "1/2", "3/4")),
           "high": (("7/2", 4, "9/2", 5), (1, "5/4", "3/2", "7/4", 2), ("1/4", "1/2", "3/4"))},
    "E3": {"keys": ("gain",), "low": ((3, "7/2", "9/2", "19/4", "21/4", "11/2", "23/4"),),
           "high": (("25/4", "13/2", "27/4", 7, "29/4", "15/2", "17/2", 9, 10, 11, 13),)},
    "F3": {"keys": ("gain",), "low": (("1/5", "1/3", "2/5", "1/2", "3/5", "7/10", "4/5"),),
           "high": (("7/8", "9/10", "19/20", "11/10", "6/5", "7/5", "3/2"),)},
}


def rank(key, context):
    return hashlib.sha256(f"{SEED}/{context}/{key}".encode()).hexdigest()


def prior_specifications():
    known = set()
    for path in (ROOT / "evidence").glob("**/instances/*.json"):
        item = read_json(path)
        if item.get("family_id") in STRATA:
            known.add(item["specification_hash"])
    return known


def freeze(known):
    selected = []
    exclusions = set()
    for family, rule in STRATA.items():
        for stratum in ("low", "high"):
            pool = []
            for values in product(*rule[stratum]):
                params = dict(zip(rule["keys"], values, strict=True))
                row = dict(family=family, parameters=params)
                key = candidates(SEED, [row])[0]["specification_hash"]
                if key in known:
                    exclusions.add(key)
                else:
                    pool.append(dict(**row, stratum=stratum, specification_hash=key))
            if len(pool) < PER_STRATUM:
                raise ValueError(f"too few fresh points in {family}/{stratum}")
            selected.extend(sorted(pool, key=lambda row: rank(row["specification_hash"], "sample"))[:PER_STRATUM])
    body = dict(version="af-pilot-0.1", seed=SEED, per_stratum=PER_STRATUM,
                sampling="SHA-256 rank within fixed Cartesian strata after excluding prior specification hashes",
                strata=STRATA, excluded_known_hashes=sorted(exclusions),
                review_sampling="At most two reliable non-calibration tasks per family; cover both labels when available; "
                                "choose SHA-256-ranked cases, then reorder across families using a separate rank.",
                mechanisms="Repeat count is sample count minus number of represented sampling groups.",
                cases=selected)
    return dict(**body, plan_sha256=digest(body))


def validate_plan(frozen):
    body = {key: value for key, value in frozen.items() if key != "plan_sha256"}
    if digest(body) != frozen["plan_sha256"] or frozen["version"] != "af-pilot-0.1":
        raise ValueError("frozen plan hash or version mismatch")
    items = candidates(frozen["seed"], frozen["cases"])
    hashes = [item["specification_hash"] for item in items]
    if (hashes != [row["specification_hash"] for row in frozen["cases"]]
            or len(set(hashes)) != len(hashes) or set(hashes) & set(frozen["excluded_known_hashes"])):
        raise ValueError("plan no longer matches its fresh, unique specifications")
    return items


def eligible_for_review(row):
    return (row["label_gate"] == "PASS"
            and row["preliminary_difficulty"] not in ("CALIBRATION", "SHORTCUT_FOUND"))


def review_selection(rows):
    selected = []
    for family in STRATA:
        pool = sorted((r for r in rows if r["family"] == family and eligible_for_review(r)),
                      key=lambda row: rank(row["specification_hash"], "review"))
        if not pool:
            continue
        first = pool[0]
        second = next((row for row in pool if row["label"] != first["label"]), None)
        selected.append(first)
        if second is not None:
            selected.append(second)
        elif len(pool) > 1:
            selected.append(pool[1])
    return sorted(selected, key=lambda row: rank(row["specification_hash"], "share"))


def ratio(count, total):
    return count/total if total else None


def group_metrics(rows, selected, observations):
    reliable = [row for row in rows if row["label_gate"] == "PASS"]
    statuses = [review_status(observations.get(row["specification_hash"], [])) for row in selected]
    reviewed = [status for status in statuses if status != "PENDING_SINGLE_CASE_REVIEW"]
    blind_passes = reviewed.count("NOT_SOLVED_IN_BLIND_REVIEW")

    def repetition(items):
        groups = Counter(row["sampling_group"] for row in items if row["sampling_group"])
        count = sum(groups.values())
        repeats = count-len(groups)
        return dict(count=count, groups=dict(groups), repeats=repeats, rate=ratio(repeats, count))

    return dict(attempted=len(rows), reliable=len(reliable), label_rate=ratio(len(reliable), len(rows)),
                labels=dict(Counter(row["label"] for row in reliable)),
                gate_counts=dict(Counter(row["label_gate"] for row in rows)),
                exact_duplicates=sum(row["exact_duplicate_of"] is not None for row in rows),
                repetition_all=repetition(rows), repetition_reliable=repetition(reliable),
                blind_selected=len(selected), blind_reviewed=len(reviewed),
                blind_passes=blind_passes, blind_pass_rate=ratio(blind_passes, len(reviewed)),
                blind_coverage=ratio(len(reviewed), len(selected)))


def export_packet(out, rows):
    selected = review_selection(rows)
    destination = out / "blind-sample"
    destination.mkdir()
    index = []
    for n, row in enumerate(selected, 1):
        filename = f"review-{n:02}.md"
        shutil.copyfile(out / row["prompt"], destination / filename)
        index.append(dict(file=filename, family=row["family"], label=row["label"],
                          specification_hash=row["specification_hash"], prompt_sha256=row["prompt_sha256"]))
    write_json(out / "internal" / "blind-sample-index.json", index)


def score(out, observations):
    manifest = read_json(out / "internal" / "manifest.json")
    index = read_json(out / "internal" / "blind-sample-index.json")
    rows = manifest["rows"]
    report = dict(overall=group_metrics(rows, index, observations),
                  by_family={family: group_metrics([row for row in rows if row["family"] == family],
                                                   [row for row in index if row["family"] == family], observations)
                             for family in STRATA},
                  reviews=[dict(**row, status=review_status(observations.get(row["specification_hash"], [])),
                                observations=observations.get(row["specification_hash"], [])) for row in index])
    write_json(out / "internal" / "pilot-metrics.json", report)
    total = report["overall"]
    print(f"{total['reliable']}/{total['attempted']} reliable; "
          f"{total['blind_reviewed']}/{total['blind_selected']} blind reviews completed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    prepare.add_argument("--plan", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--out", type=Path, required=True)
    execute.add_argument("--legacy-solver", type=Path, required=True)
    execute.add_argument("--reviews", type=Path)
    metrics = commands.add_parser("score")
    metrics.add_argument("--out", type=Path, required=True)
    metrics.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        if args.plan.exists():
            parser.error("the frozen plan already exists; choose a new path")
        frozen = freeze(prior_specifications())
        write_json(args.plan, frozen)
        print(f"Frozen {len(frozen['cases'])} specifications, SHA-256 {frozen['plan_sha256']}")
        return
    if args.command == "run":
        frozen = read_json(args.plan)
        start = perf_counter()
        observations = load_reviews(args.reviews)
        run(args.out, validate_plan(frozen), observations, args.legacy_solver)
        write_json(args.out / "internal" / "frozen-plan.json", frozen)
        write_json(args.out / "internal" / "runtime.json", dict(wall_seconds=perf_counter()-start))
        export_packet(args.out, read_json(args.out / "internal" / "manifest.json")["rows"])
    else:
        observations = load_reviews(args.reviews)
    score(args.out, observations)


if __name__ == "__main__":
    main()
