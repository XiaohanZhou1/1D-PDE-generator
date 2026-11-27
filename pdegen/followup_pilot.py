"""Run fixed follow-up D2 samples and the first E2 moving-interface pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy
import scipy

from .audit import audit_batch
from .cf_calibration import calibration_batch, certify_calibration, native_reference, validate_calibration
from .compare_a1_d2 import screen
from .contract import ROOT, write_json
from .delay_feedback import shifted_delay_kink, solve_shifted_delay, validate_shifted_delay
from .interface_feedback import interface_wave, solve_interface_wave, validate_interface_wave
from .label_pilot import readiness, summarise
from .workflow import bundled_solver

SEED = 20261127
D2_POINTS = ((3, "5/4", "1/2"), (4, "1", "1/2"), (4, "7/4", "3/4"), (2, "3/2", "3/4"))
E2_POINTS = ((1, -4), (1, -8), (2, -4), (2, -8))


def control_evidence(item, analyse):
    result = certify_calibration(item)
    result["native_check"] = native_reference(item, analyse)
    return result


def assess(item, result, audit):
    if item["family_id"] == "D2":
        return screen(item, result, audit)
    label, difficulty = readiness(item, result, audit)
    if item["family_id"] == "E1":
        return label, difficulty, "Exact scalar boundary determinant control."
    if item["parameters"]["beta"] == "-8":
        return label, "SHORTCUT_FOUND", "A comparison supersolution gives the sign change of the real boundary response."
    return label, difficulty, "No full-domain certificate or independent difficulty decision is assigned."


def run(out):
    sp, solver, native = bundled_solver()
    out.mkdir(parents=True, exist_ok=False)
    groups = (
        ("D2", [shifted_delay_kink(*point, SEED) for point in D2_POINTS],
         validate_shifted_delay, lambda item: solve_shifted_delay(item, native.analyse)),
        ("E1", calibration_batch("E1", SEED)[:1],
         validate_calibration, lambda item: control_evidence(item, native.analyse)),
        ("E2", [interface_wave(*point, SEED) for point in E2_POINTS],
         validate_interface_wave, lambda item: solve_interface_wave(item, native.analyse)),
    )
    files = ("pdegen/followup_pilot.py", "pdegen/delay_feedback.py", "pdegen/interface_feedback.py",
             "solver/problems/interface_feedback_wave.py", "solver/problems/shifted_delay_kink.py")
    write_json(out / "environment.json", dict(python=sys.version, numpy=numpy.__version__,
               scipy=scipy.__version__, sympy=sp.__version__, twstab=solver.__version__,
               source_sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}))
    rows, counts = [], {}
    for name, items, check, solve in groups:
        start = perf_counter()
        directory = out / name
        entries = audit_batch(items, directory, check, solve)
        group_rows = []
        for item, entry in zip(items, entries, strict=True):
            audit = json.loads((directory / entry["audit"]).read_bytes())
            result = json.loads((directory / entry["result"]).read_bytes()) if entry["result"] else {}
            label, difficulty, notes = assess(item, result, audit)
            group_rows.append(dict(family=name, instance_id=item["instance_id"], parameters=item["parameters"],
                                   label_gate=label, difficulty_gate=difficulty, difficulty_notes=notes,
                                   solver_label=result.get("label"), evidence_grade=result.get("evidence_grade"),
                                   sampling_group=audit["sampling_group"], reason=result.get("reason", audit["reason"]),
                                   audit=f"{name}/{entry['audit']}",
                                   result=f"{name}/{entry['result']}" if entry["result"] else None))
        write_json(directory / "manifest.json", dict(seed=SEED, entries=entries))
        counts[name] = dict(**summarise(group_rows), runtime_seconds=perf_counter() - start)
        rows.extend(group_rows)
        print(f"{name}: {counts[name]['labels_passed']}/{len(items)} reliable labels; "
              f"{counts[name]['preliminary_joint_passed']}/{len(items)} joint passes", flush=True)
    write_json(out / "summary.json", dict(seed=SEED, by_family=counts, overall=summarise(rows), rows=rows,
                                          blind_review="PENDING"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out)


if __name__ == "__main__":
    main()
