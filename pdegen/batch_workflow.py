"""Unified generation, label verification and blind export for the supported PDE families."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

from .anisotropic_bloch import anisotropic_roll, solve_roll, validate_roll
from .audit import audit_batch
from .blind_prompt import render_problem
from .compare_a1_d2 import d2_batch, screen
from .contract import ROOT, construct, validate, write_json
from .delay_feedback import shifted_delay_kink, solve_shifted_delay, validate_shifted_delay
from .dynamic_interface import dynamic_interface, solve_dynamic_interface, validate_dynamic_interface
from .e3_certificate import certify_gain
from .families import a1
from .generate import batch
from .hamiltonian_polarization import polarization_wave, solve_polarization, validate_polarization
from .label_pilot import SEED, a1_result, readiness
from .parametric_floquet import parametric_orbit, solve_parametric, validate_parametric_orbit
from .switched_floquet import solve_switch, switched_orbit, validate_switch
from .workflow import bundled_solver

REVIEW_FILE = ROOT / "evidence" / "blind-review" / "observations.json"
VALIDATORS = {"A1": validate, "D2": validate_shifted_delay, "E3": validate_dynamic_interface,
              "B3": validate_roll, "C2": validate_polarization, "F3": validate_switch,
              "F4": validate_parametric_orbit}

BCF_PLAN = [
    *[dict(family="B3", parameters=dict(damping=damping, skew=skew))
      for damping, skew in ((3, 1), (4, 2), (6, 2))],
    *[dict(family="C2", parameters=dict(alpha=alpha, beta=beta))
      for alpha, beta in (("1/4", "1/8"), (2, "1/2"), ("5/2", 1))],
    *[dict(family="F3", parameters=dict(gain=gain)) for gain in ("1/4", "3/4", 1, "5/4")],
]


def read_json(path):
    return json.loads(path.read_bytes())


def candidates(seed, plan=None):
    if plan is None:
        return [*batch(seed), *d2_batch(seed), *(dynamic_interface(gain, seed) for gain in (5, 6, 8, 12))]
    if not isinstance(plan, list) or not plan:
        raise ValueError("the plan must be a nonempty JSON list")
    constructors = {"A1": lambda q, a, seed, beta=0: construct(a1(q, a, beta), seed),
                    "D2": shifted_delay_kink,
                    "E3": dynamic_interface, "B3": anisotropic_roll, "C2": polarization_wave,
                    "F3": switched_orbit, "F4": parametric_orbit}
    return [constructors[row["family"]](**row["parameters"], seed=seed) for row in plan]


def review_index(records):
    if not isinstance(records, list):
        raise ValueError("blind-review records must be a JSON list")
    by_hash = {}
    for observation in records:
        key = observation["specification_hash"]
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("blind-review records require the full specification hash")
        if (observation["session"] not in ("single", "group")
                or observation["outcome"] not in ("complete_proof", "incomplete_proof")
                or not isinstance(observation["no_hints"], bool)
                or not isinstance(observation["no_numerics"], bool)):
            raise ValueError("invalid blind-review protocol or outcome")
        by_hash.setdefault(key, []).append(observation)
    return by_hash


def review_status(observations):
    qualifying = [row for row in observations if row["session"] == "single"
                  and row["no_hints"] and row["no_numerics"]]
    if any(row["outcome"] == "complete_proof" for row in qualifying):
        return "PROOF_COMPLETED"
    if qualifying:
        return "NOT_SOLVED_IN_BLIND_REVIEW"
    return "PENDING_SINGLE_CASE_REVIEW"


def load_reviews(path=None):
    return {} if path is None else review_index(read_json(path))


def solve_e3(item):
    gain = Fraction(item["parameters"]["gain"])
    try:
        certificate = certify_gain(gain)
    except (ArithmeticError, ValueError) as error:
        result = solve_dynamic_interface(item)
        result["interval_certificate_unavailable"] = str(error)
        return result
    return dict(label=certificate["label"], open_rhp_count=certificate["open_rhp_count"],
                evidence_grade=certificate["evidence_grade"],
                reason="Verified count on the full open right half-plane.",
                certificate=certificate)


def run(out, items, reviews, legacy=None):
    need_a1 = any(item["family_id"] == "A1" and item["validity_checks"]["solver_dispatch"] == "PASS" for item in items)
    if need_a1 and legacy is None:
        raise ValueError("dispatchable A1 cases require --legacy-solver")
    solver_version = None
    if any(item["family_id"] == "D2" for item in items):
        _, twstab, native = bundled_solver()
        solver_version = twstab.__version__
    out.mkdir(parents=True, exist_ok=False)
    private = out / "internal"
    blind = out / "blind"
    blind.mkdir()
    solvers = {"A1": lambda item: a1_result(item, legacy, private / "native" / item["instance_id"]),
               "D2": lambda item: solve_shifted_delay(item, native.analyse),
               "E3": solve_e3, "B3": solve_roll, "C2": solve_polarization,
               "F3": solve_switch, "F4": solve_parametric}

    def validate_item(item):
        return VALIDATORS[item["family_id"]](item)

    def solve_item(item):
        return solvers[item["family_id"]](item)

    entries = audit_batch(items, private, validate_item, solve_item)
    rows = []
    prompt_count = 0
    for item, entry in zip(items, entries, strict=True):
        audit = read_json(private / entry["audit"])
        result = read_json(private / entry["result"]) if entry["result"] else {}
        family = item["family_id"]
        if entry["result"]:
            if family in ("A1", "D2"):
                gate, screen_status, difficulty_note = screen(item, result, audit)
            else:
                gate, screen_status = readiness(item, result, audit)
                difficulty_note = "The spectral evidence concerns label reliability; difficulty is reviewed separately."
        else:
            gate, screen_status = audit["status"], "NOT_SCREENED"
            difficulty_note = None
        observations = reviews.get(item["specification_hash"], [])
        blind_status = review_status(observations)
        prompt = None
        prompt_hash = None
        if gate == "PASS":
            prompt_count += 1
            prompt = f"blind/problem-{prompt_count:04d}.md"
            encoded = render_problem(item).encode("utf-8")
            (out / prompt).write_bytes(encoded)
            prompt_hash = hashlib.sha256(encoded).hexdigest()
        rows.append(dict(family=family, parameters=item["parameters"], instance_id=audit["instance_id"],
                         specification_hash=audit["specification_hash"], sampling_group=audit["sampling_group"],
                         exact_duplicate_of=audit.get("duplicate_of"), label=entry["label"],
                         evidence_grade=audit["evidence_grade"], label_gate=gate,
                         preliminary_difficulty=screen_status, difficulty_note=difficulty_note, blind_status=blind_status,
                         blind_observations=observations, prompt=prompt, prompt_sha256=prompt_hash,
                         instance=f"internal/{entry['instance']}" if entry["instance"] else None,
                         evidence=f"internal/{entry['result']}" if entry["result"] else None,
                         audit=f"internal/{entry['audit']}"))
    passed = [row for row in rows if row["label_gate"] == "PASS"]
    sources = ("pdegen/batch_workflow.py", "pdegen/blind_prompt.py", "pdegen/e3_certificate.py",
               "pdegen/dynamic_interface.py", "pdegen/adapter.py", "pdegen/delay_feedback.py", "pdegen/label_pilot.py",
               "solver/problems/shifted_delay_kink.py", "pdegen/extension_contract.py",
               "pdegen/anisotropic_bloch.py", "pdegen/hamiltonian_polarization.py", "pdegen/switched_floquet.py",
               "pdegen/f3_certificate.py", "pdegen/parametric_floquet.py")
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    summary = dict(attempted=len(rows), reliable_labels=len(passed), prompts_exported=prompt_count,
                   labels=dict(Counter(row["label"] for row in passed)),
                   outcomes=dict(Counter(row["label_gate"] for row in rows)),
                   reliable_with_blind_threshold=sum(row["blind_status"] == "NOT_SOLVED_IN_BLIND_REVIEW"
                                                     for row in passed),
                   groups=sorted({row["sampling_group"] for row in passed}))
    write_json(private / "manifest.json", dict(version="batch-0.1", seed=items[0]["seed"],
               twstab_version=solver_version, source_sha256=source_hashes, summary=summary, rows=rows))
    print(f"{len(rows)} specifications; {len(passed)} reliable labels; {prompt_count} blind prompts", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--legacy-solver", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--plan", type=Path)
    selection.add_argument("--preset", choices=("bcf",), help="run the B3, C2 and F3 extension pilot")
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    try:
        plan = read_json(args.plan) if args.plan is not None else BCF_PLAN if args.preset == "bcf" else None
        items = candidates(args.seed, plan)
        reviews = load_reviews(args.reviews)
        run(args.out, items, reviews, args.legacy_solver)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
