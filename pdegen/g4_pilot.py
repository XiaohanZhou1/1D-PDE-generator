"""G4 proof-readiness pre-pilot over frozen A–F manifests and one new D task."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

from .contract import ROOT, write_json
from .workflow import bundled_solver


RUNS = ("g3-a1", "g3-a5", "b1", "b2", "cf/C1", "cf/D1", "cf/E1", "cf/F1", "f2")
TASK = "G4-D-count-01"


def load_json(path):
    return json.loads(path.read_bytes())


def review_existing():
    rows = []
    for name in RUNS:
        run = ROOT / "evidence" / name
        manifest = load_json(run / "manifest.json")
        for entry in manifest["entries"]:
            audit = load_json(run / entry["audit"])
            item = load_json(run / entry["instance"])
            result = load_json(run / entry["result"]) if entry["result"] else {}
            key = item["specification_hash"]
            if audit["specification_hash"] != key or result.get("specification_hash", key) != key:
                raise ValueError(f"unlinked evidence for {item['instance_id']}")
            grade = result.get("evidence_grade")
            label = result.get("label")
            if audit["status"] == "HOLD":
                finding = "SOLVER_DISPATCH_HOLD"
            elif audit["status"] == "VALIDATED":
                finding = "SPECIFICATION_ONLY"
            elif label == "UNCERTAIN":
                finding = "UNRESOLVED"
            elif grade == "CONVERGED_NUMERICAL":
                finding = "NUMERICAL_EVIDENCE_ONLY"
            elif item.get("calibration") and grade in {"RIGOROUSLY_CERTIFIED", "EXACT_LINEAR_SPECTRUM"}:
                finding = "ANALYTIC_CALIBRATION"
            else:
                finding = "MANUAL_EVIDENCE_REVIEW"
            rows.append(dict(family=item["family_id"], instance_id=item["instance_id"],
                             specification_hash=key, source=f"evidence/{name}/{entry['audit']}",
                             label=label, evidence_grade=grade, finding=finding,
                             proof_admission=audit["dataset_admission"],
                             proof_screen=audit["proof_screen"]))
    hashes = Counter(row["specification_hash"] for row in rows)
    return dict(rows=rows, cases=len(rows), distinct_specifications=len(hashes),
                findings=dict(Counter(row["finding"] for row in rows)),
                labels=dict(Counter(row["label"] or "NOT_RUN" for row in rows)),
                proof_training_admitted=sum(row["proof_admission"] == "ADMIT" for row in rows))


def eigenvalue_probe(epsilon, halfwidth, intervals):
    """Independent Dirichlet finite differences on the delayed perturbation block."""
    x = np.linspace(-halfwidth, halfwidth, intervals + 1)[1:-1]
    h = 2 * halfwidth / intervals
    u = np.tanh(x / np.sqrt(2))
    weight = (1 - u**2) * (1 + epsilon * u)
    off_diagonal = np.full(len(x) - 1, 1 / h**2)

    def top_three(lam):
        kernel = 1.0 if lam == 0 else (-np.expm1(-lam) / (2 * lam) - np.expm1(-2 * lam) / (4 * lam))
        diagonal = -2 / h**2 - 1 - lam + 6 * kernel * weight
        return eigh_tridiagonal(diagonal, off_diagonal, select="i",
                                select_range=(len(x) - 3, len(x) - 1), eigvals_only=True)[::-1]

    roots = [float(brentq(lambda lam: top_three(lam)[index], 0, 5, xtol=1e-11)) for index in (0, 1)]
    return dict(halfwidth=halfwidth, intervals=intervals,
                top_three_at_zero=top_three(0).tolist(), positive_roots=roots)


def check_count_task(analyse):
    samples = []
    for exact in ("0", "1/16", "1/8"):
        start = perf_counter()
        epsilon = float(Fraction(exact))
        estimates = [eigenvalue_probe(epsilon, width, n) for width, n in ((12, 1200), (20, 4000))]
        record = analyse(ROOT / "solver" / "problems" / "g4_delay_count.py",
                         parameters={"epsilon": epsilon})
        record["file"] = "solver/problems/g4_delay_count.py"
        native = sorted((complex(*root["value"]) for root in record.get("roots", [])
                         if root["status"] == "unstable"), key=lambda z: z.real, reverse=True)
        fine = np.asarray(estimates[-1]["positive_roots"])
        refine_error = float(np.max(np.abs(fine - estimates[0]["positive_roots"])))
        matched = (record["label"] == "UNSTABLE" and len(native) == len(fine)
                   and np.max(np.abs(native - fine)) < 1e-3)
        samples.append(dict(epsilon=exact, finite_difference=estimates, native=record,
                            refinement_error=refine_error, runtime_seconds=perf_counter() - start,
                            diagnostic_status="PASS" if matched and refine_error < 1e-3 else "UNVERIFIED"))
    return dict(task_id=TASK, prompt="docs/g4-review/delay-count-task.md",
                proof="docs/g4-review/delay-count-reference.md", proof_training_status="HOLD",
                proof_novelty="UNCERTAIN", blind_review="PENDING",
                analytic_open_rhp_count=2, analytic_scope="all 0 <= epsilon <= 1/8",
                numerical_probe_status="PASS" if all(sample["diagnostic_status"] == "PASS"
                                                     for sample in samples) else "UNVERIFIED",
                samples=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    review = review_existing()
    _, twstab, native = bundled_solver()
    task = check_count_task(native.analyse)
    write_json(args.out / "baseline.json", review)
    write_json(args.out / "delay-count.json", dict(twstab_version=twstab.__version__, **task))
    print(f"{review['cases']} baseline cases; candidate {task['numerical_probe_status']}; blind review pending")


if __name__ == "__main__":
    main()
