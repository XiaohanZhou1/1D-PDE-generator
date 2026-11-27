"""Measure reproducible label yield on a fixed 30-case G4 pilot."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import scipy

from .adapter import execute
from .audit import a5_evidence, audit_batch
from .bloch import bloch_batch, solve_roll, validate_roll
from .contract import ROOT, validate, write_json
from .generate import a5_batch, batch
from .jim_a5_unfolding import analyse_a5
from .periodic import periodic_batch, solve_periodic_roll, validate_periodic_roll
from .workflow import bundled_solver


SEED = 20261126


def a1_result(item, solver, directory):
    result = execute(item, solver, directory)
    verification = result["verification"] or {}
    roots = verification.get("roots", [])
    positive = any(roots[index]["unit_local_count"]
                   and roots[index]["lambda_value"][0] > 10 * roots[index]["observed_error_scale"] > 0
                   for index in verification.get("positive_witness_indices", []))
    passed = (result["solver_status"] == "COMPLETED" and result["label"] == "UNSTABLE"
              and result["evidence_grade"] == "CONVERGED_NUMERICAL" and positive)
    return dict(label="UNSTABLE" if passed else "UNCERTAIN", open_rhp_count=result["unstable_count"],
                evidence_grade="CONVERGED_NUMERICAL" if passed else None,
                reason=("Evans and physical finite-difference checks corroborate a positive witness."
                        if passed else result["failure_reason"] or "Positive witness failed the pilot gate."),
                solver_status=result["solver_status"], adapter_record=result)


def readiness(item, result, audit):
    label = result.get("label")
    grade = result.get("evidence_grade")
    if label == "STABLE":
        if grade in ("EXACT_LINEAR_SPECTRUM", "RIGOROUSLY_CERTIFIED"):
            label_status = "PASS"
        else:
            label_status = "NEEDS_FULL_DOMAIN_CERTIFICATE"
    elif label == "UNSTABLE" and grade in ("CONVERGED_NUMERICAL", "RIGOROUSLY_CERTIFIED",
                                           "EXACT_LINEAR_SPECTRUM"):
        label_status = "PASS"
    elif result.get("solver_status") == "FAILED":
        label_status = "SOLVER_FAILED"
    else:
        label_status = audit["status"]
    family = item["family_id"]
    if item.get("calibration") or audit["proof_screen"].get("elementary_decision"):
        difficulty = "CALIBRATION"
    elif family == "A1" and item["parameters"]["beta"] == "0":
        difficulty = "PRELIMINARY_PASS"
    else:
        difficulty = "REVIEW_PENDING"
    return label_status, difficulty


def summarise(rows):
    passed = [row for row in rows if row["label_gate"] == "PASS"]
    joint = [row for row in passed if row["difficulty_gate"] == "PRELIMINARY_PASS"]
    return dict(attempted=len(rows), labels_passed=len(passed),
                label_pass_rate=len(passed) / len(rows), preliminary_joint_passed=len(joint),
                preliminary_joint_pass_rate=len(joint) / len(rows), joint_mechanism_groups=sorted({
                    row["sampling_group"] for row in joint}),
                label_gate_counts=dict(Counter(row["label_gate"] for row in rows)),
                accepted_label_counts=dict(Counter(row["solver_label"] for row in passed)))


def run(out, legacy, seed):
    sp, twstab, native = bundled_solver()
    out.mkdir(parents=True, exist_ok=False)
    configurations = (
        ("A1", batch(seed), validate,
         lambda item: a1_result(item, legacy, out / "A1" / "native" / item["instance_id"])),
        ("A5", a5_batch(seed), validate, lambda item: a5_evidence(analyse_a5(item, native, sp))),
        ("B1", bloch_batch(seed, ["1/4"]), validate_roll, solve_roll),
        ("B2", periodic_batch(seed, [(q, a) for q in ("9/10", "19/20", "1", "21/20", "11/10")
                                     for a in ("1/10", "1/5", "3/10", "2/5")]),
         validate_periodic_roll, solve_periodic_roll),
    )
    sources = (ROOT / "pdegen" / name for name in
               ("label_pilot.py", "adapter.py", "audit.py", "periodic.py", "bloch.py", "jim_a5_unfolding.py"))
    write_json(out / "environment.json", dict(seed=seed, python=sys.version, numpy=np.__version__,
               scipy=scipy.__version__, sympy=sp.__version__, twstab=twstab.__version__,
               source_sha256={str(path.relative_to(ROOT).as_posix()): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sources}))
    rows = []
    by_family = {}
    for family, items, check, solve in configurations:
        start = perf_counter()
        destination = out / family
        entries = audit_batch(items, destination, check, solve)
        family_rows = []
        for item, entry in zip(items, entries, strict=True):
            audit = json.loads((destination / entry["audit"]).read_bytes())
            result = json.loads((destination / entry["result"]).read_bytes()) if entry["result"] else {}
            label, difficulty = readiness(item, result, audit)
            family_rows.append(dict(family=family, instance_id=item["instance_id"],
                                    parameters=item["parameters"], solver_label=result.get("label"),
                                    evidence_grade=result.get("evidence_grade"), label_gate=label,
                                    difficulty_gate=difficulty, sampling_group=audit["sampling_group"],
                                    reason=result.get("reason", audit["reason"]),
                                    audit=f"{family}/{entry['audit']}",
                                    result=f"{family}/{entry['result']}" if entry["result"] else None))
        write_json(destination / "manifest.json", dict(seed=seed, entries=entries))
        by_family[family] = dict(**summarise(family_rows), runtime_seconds=perf_counter() - start)
        rows.extend(family_rows)
        print(f"{family}: {by_family[family]['labels_passed']}/{len(items)} label passes", flush=True)
    write_json(out / "summary.json", dict(seed=seed, overall=summarise(rows), by_family=by_family, rows=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-solver", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(args.out, args.legacy_solver, args.seed)


if __name__ == "__main__":
    main()
