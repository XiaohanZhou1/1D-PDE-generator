"""G3 decision records for generated specifications and their evidence."""
from fractions import Fraction
import math

from jsonschema import ValidationError

from .contract import write_json


def sampling_group(item):
    # Use the family-wide generator group to keep controls and nearby parameter
    # variants together when a benchmark is split.
    return item["novelty_assessment"]["grouping"]


def proof_screen(item):
    family = item["family_id"]
    if family=="B1":
        return dict(known_reduction="periodic rotation reduces the operator to Fourier blocks",
                    elementary_decision="Eckhaus dispersion relation",review_required=False)
    if family=="B2":
        return dict(known_reduction="classical Swift–Hohenberg roll and selfadjoint Bloch fibers",
                    elementary_decision=None,review_required=False)
    if family=="F2":
        return dict(known_reduction="ground-state transform and energy estimate",
                    elementary_decision="full-zone contraction",review_required=False)
    if item.get("schema_version")=="cf-calibration-0.1":
        return dict(known_reduction="reviewed exact calibration", elementary_decision=None,
                    review_required=False)
    if family=="A1" and item["parameters"]["q"]=="0" and item["parameters"]["beta"]=="0":
        return dict(known_reduction="decoupled scalar blocks",elementary_decision="analytic calibration",
                    review_required=False)
    if family=="A5":
        return dict(known_reduction="conservative integration and an origin calculation",
                    elementary_decision=None,review_required=True)
    return dict(known_reduction=None,elementary_decision=None,review_required=True)


def near_match(item, earlier):
    """Identify nearby A5 parameters without claiming mathematical equivalence."""
    if item["family_id"] != "A5":
        return None
    mu = Fraction(item["parameters"]["mu"])
    for other in earlier:
        if (other["structure_id"] == item["structure_id"]
                and other["parameters"]["rho"] == item["parameters"]["rho"]
                and abs(mu-Fraction(other["parameters"]["mu"])) <= Fraction(1,8)):
            return other["instance_id"]
    return None


def audit_batch(items, destination, validate, solve=None):
    """Record invalid, duplicate, unsupported and unresolved cases as well as successes.

    The backend returns a dict containing label, open_rhp_count, evidence_grade
    and reason, plus any backend-specific diagnostics.
    """
    seen = {}
    entries = []
    for ordinal,item in enumerate(items,1):
        audit_path = f"audit/{ordinal:04d}.json"
        entry = dict(audit=audit_path,instance=None,result=None,label=None,open_rhp_count=None)
        audit = dict(audit_version="1",candidate_index=ordinal,instance_id=None,
                     specification_hash=None,checks={},sampling_group=None,near_duplicate_of=None,
                     solver_dispatched=False,dataset_admission="HOLD",proof_review="UNVERIFIED",
                     evidence_grade=None)
        try:
            validate(item)
        except (ValidationError,ValueError,KeyError,TypeError,ZeroDivisionError) as error:
            candidate_path = f"rejected/{ordinal:04d}.json"
            write_json(destination/candidate_path,item)
            reason = (str(error) or type(error).__name__).splitlines()[0]
            audit.update(status="REJECTED",rejected_candidate=candidate_path,reason=reason)
            audit["checks"]["contract"] = "FAIL"
        else:
            ident = item["instance_id"]
            key = item["specification_hash"]
            novelty = item["novelty_assessment"]
            audit.update(instance_id=ident,specification_hash=key,sampling_group=sampling_group(item),
                         checks={**item["validity_checks"],"contract":"PASS"},
                         novelty_assessment=novelty,proof_screen=proof_screen(item),references=item["references"],
                         proof_review=item["validity_checks"]["proof_difficulty"])
            if key in seen:
                audit.update(status="DUPLICATE",reason="Exact mathematical specification already present.",
                             duplicate_of=seen[key]["instance_id"])
                entry["instance"] = f"instances/{ident}.json"
            else:
                audit["near_duplicate_of"] = near_match(item,seen.values())
                seen[key] = item
                entry["instance"] = f"instances/{ident}.json"
                write_json(destination/entry["instance"],item)
                if audit["checks"]["solver_dispatch"] != "PASS":
                    audit.update(status="HOLD",reason="The specification is outside the enabled solver capability.")
                elif solve is None:
                    audit.update(status="VALIDATED",reason="Mathematical contract checked without a solver run.")
                else:
                    audit["solver_dispatched"] = True
                    try:
                        result = solve(item)
                    except (ArithmeticError,RuntimeError,ValueError) as error:
                        result = dict(label="UNCERTAIN",open_rhp_count=None,evidence_grade=None,
                                      reason=f"{type(error).__name__}: {error}")
                    entry.update(result=f"results/{ident}.json",label=result["label"],
                                 open_rhp_count=result["open_rhp_count"])
                    audit.update(status="UNCERTAIN" if result["label"]=="UNCERTAIN" else "EVALUATED",
                                 evidence_grade=result["evidence_grade"],reason=result["reason"])
                    write_json(destination/entry["result"],dict(instance_id=ident,specification_hash=key,
                               parameters=item["parameters"],**result))
        write_json(destination/audit_path,audit)
        entries.append(entry)
    return entries


def a5_evidence(native):
    """Grade the existing numerical workflow without changing its native record."""
    spectrum = native.get("closed_half_plane",{})
    contours = spectrum.get("windings",[])
    residual = native.get("wave",{}).get("target_pde_residual")
    accepted = (native["label"]=="STABLE" and spectrum.get("open_rhp_count")==0 and len(contours)>=6
                and all(trace[key]["resolved"] and trace[key]["count"]==1
                        for trace in contours for key in ("outer","origin"))
                and isinstance(residual,(int,float)) and math.isfinite(residual) and 0<=residual<=1e-5)
    return dict(label="STABLE" if accepted else "UNCERTAIN",open_rhp_count=0 if accepted else None,
                evidence_grade="CONVERGED_NUMERICAL" if accepted else None,
                reason=native["reason"] if accepted or native["label"]=="UNCERTAIN" else
                "Native label lacks the fixed-speed profile and resolved origin evidence required by this workflow.",
                native=native)
