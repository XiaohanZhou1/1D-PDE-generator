"""Run the E3 dynamic-interface label and difficulty pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy
import scipy

from .audit import audit_batch
from .contract import ROOT, write_json
from .dynamic_interface import dynamic_interface, solve_dynamic_interface, validate_dynamic_interface
from .label_pilot import readiness, summarise

GAINS = ("1/2", "1", "4", "5", "6", "8", "12")
SEED = 20261126


def run(out):
    items = [dynamic_interface(gain, SEED) for gain in GAINS]
    out.mkdir(parents=True, exist_ok=False)
    sources = ("pdegen/dynamic_interface.py", "pdegen/e3_pilot.py")
    write_json(out / "environment.json", dict(seed=SEED, python=sys.version, numpy=numpy.__version__,
                                              scipy=scipy.__version__, sha256={
                                                  name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                                  for name in sources}))
    entries = audit_batch(items, out, validate_dynamic_interface, solve_dynamic_interface)
    rows = []
    for item, entry in zip(items, entries, strict=True):
        result = json.loads((out / entry["result"]).read_bytes())
        audit = json.loads((out / entry["audit"]).read_bytes())
        label, difficulty = readiness(item, result, audit)
        if label == "PASS" and result["label"] == "STABLE" and not item["calibration"]:
            difficulty = "SHORTCUT_FOUND"
        elif label == "PASS" and result["label"] == "UNSTABLE":
            difficulty = "PRELIMINARY_PASS"
        rows.append(dict(family="E3", parameters=item["parameters"], instance_id=item["instance_id"],
                         solver_label=result["label"], evidence_grade=result["evidence_grade"],
                         label_gate=label, difficulty_gate=difficulty, sampling_group=audit["sampling_group"],
                         result=entry["result"], audit=entry["audit"]))
        print(f"{item['parameters']['gain']}: {result['label']} ({difficulty})", flush=True)
    write_json(out / "manifest.json", dict(seed=SEED, entries=entries))
    write_json(out / "summary.json", dict(overall=summarise(rows), rows=rows, blind_review="PENDING",
                                          notes="Exploratory gain grid; the existing native solver was not used."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out)


if __name__ == "__main__":
    main()
