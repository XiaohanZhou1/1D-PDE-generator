"""Canonical record helpers for the B3, C2 and F3 constructions."""
from .contract import digest


def specification(problem, seed, calibration=False, references=()):
    family = problem["family_id"]
    key = digest(problem)
    return dict(**problem, seed=seed, schema_version=f"{family.lower()}-0.1",
                instance_id=f"{family}-{key[:20]}", specification_hash=key, calibration=calibration,
                validity_checks=dict(profile="PASS", linearisation="PASS", solver_dispatch="PASS",
                                     proof_difficulty="NOT_APPLICABLE" if calibration else "UNVERIFIED"),
                novelty_assessment=dict(status="NOT_NOVEL" if calibration else "UNCERTAIN",
                                        grouping=problem["structure_id"], training_admission="HOLD"),
                references=list(references))


def validate_instance(item, constructor):
    if type(item["seed"]) is not int or item != constructor(**item["parameters"], seed=item["seed"]):
        raise ValueError(f"{item['family_id']} differs from the canonical specification")
    return item


def outcome(label="UNCERTAIN", grade=None, count=None, reason="No resolved positive witness or full-domain bound."):
    return dict(label=label, evidence_grade=grade, open_rhp_count=count, reason=reason)
