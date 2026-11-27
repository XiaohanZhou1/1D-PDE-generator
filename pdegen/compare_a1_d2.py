"""Compare A1 and D2 using one label gate and separate shortcut screens."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from .audit import audit_batch
from .contract import ROOT, validate, write_json
from .delay_feedback import shifted_delay_kink, solve_shifted_delay, validate_shifted_delay
from .generate import batch
from .label_pilot import SEED, a1_result, readiness, summarise
from .workflow import bundled_solver


def d2_batch(seed):
    grid = [(sigma, tau, "3/4") for sigma in (3, 5) for tau in ("1", "3/2")]
    controls = [("1/2", "1", "3/4"), (5, "1", 0)]
    return [shifted_delay_kink(sigma, tau, epsilon, seed) for sigma, tau, epsilon in grid + controls]


def screen(item, result, audit):
    label, difficulty = readiness(item, result, audit)
    if item["family_id"] == "A1":
        notes = ("Decoupled calibration or an unsupported variant." if difficulty != "PRELIMINARY_PASS" else
                 "The coupled regime does not follow directly from the known decoupled calculation.")
    elif item["calibration"]:
        notes = "Small-gain control or symmetric Pöschl–Teller control."
    elif label == "PASS" and result.get("independent_shooting"):
        difficulty = "PRELIMINARY_PASS"
        notes = ("Positive real roots are excluded by the delayed-block energy identity. "
                 "The shifted negative feedback does not inherit D1's nonreal exclusion; "
                 "the asymmetric spatial factor removes the exact symmetric reduction. "
                 "Independent blind review remains pending.")
    else:
        notes = "The general checks did not yield a reliable label; difficulty remains unreviewed."
    return label, difficulty, notes


def run(out, legacy, seed):
    _, twstab, native = bundled_solver()
    out.mkdir(parents=True, exist_ok=False)
    configurations = (
        ("A1", batch(seed), validate,
         lambda item: a1_result(item, legacy, out / "A1" / "native" / item["instance_id"])),
        ("D2", d2_batch(seed), validate_shifted_delay,
         lambda item: solve_shifted_delay(item, native.analyse)),
    )
    sources = ("pdegen/compare_a1_d2.py", "pdegen/delay_feedback.py",
               "solver/problems/shifted_delay_kink.py")
    write_json(out / "environment.json", dict(seed=seed, twstab_version=twstab.__version__,
               source_hashes={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in sources}))
    cases, by_family = [], {}
    for family, items, validate_item, solve in configurations:
        start = perf_counter()
        folder = out / family
        entries = audit_batch(items, folder, validate_item, solve)
        rows = []
        for item, entry in zip(items, entries, strict=True):
            audit = json.loads((folder / entry["audit"]).read_bytes())
            result = json.loads((folder / entry["result"]).read_bytes()) if entry["result"] else {}
            label, difficulty, notes = screen(item, result, audit)
            rows.append(dict(family=family, instance_id=item["instance_id"], parameters=item["parameters"],
                             solver_label=result.get("label"), evidence_grade=result.get("evidence_grade"),
                             label_gate=label, difficulty_gate=difficulty, difficulty_notes=notes,
                             sampling_group=audit["sampling_group"],
                             reason=result.get("reason", audit["reason"]),
                             audit=f"{family}/{entry['audit']}",
                             result=f"{family}/{entry['result']}" if entry["result"] else None))
        write_json(folder / "manifest.json", dict(seed=seed, entries=entries))
        cases.extend(rows)
        by_family[family] = dict(**summarise(rows), runtime_seconds=perf_counter() - start)
        print(f"{family}: {by_family[family]['labels_passed']}/{len(rows)} label passes; "
              f"{by_family[family]['preliminary_joint_passed']}/{len(rows)} joint passes", flush=True)
    write_json(out / "summary.json", dict(seed=seed, overall=summarise(cases), by_family=by_family,
                                          blind_review="PENDING", rows=cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-solver", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(args.out, args.legacy_solver, args.seed)


if __name__ == "__main__":
    main()
