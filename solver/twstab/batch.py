"""A list of equations in, a list of labels out (DESIGN.md, sections 4.1 and 4.3).

Input: a list.  Entry k describes one equation; its name is eqn<k> unless the entry gives `name`.
    dict(equations=..., parameters={symbol: value}, wave=dict(...))
The names are those of a problem file (problem.py): `equations` (one sympy.Eq or a list), `parameters`,
`wave` (the known conditions on the wave: kind, end states, speed), and, where needed, `unknowns`,
`free_boundary`, `options`, `x`, `t`.  An entry without parameters and without `wave` may be the
sympy.Eq itself (or the list of sympy.Eq of a system).

Output: a list with one entry for each equation, in the order of the input.
    dict(name, waves=[dict(wave, speed, label, qualifier, reason), ...], note, seconds)
An entry with `wave` gives one wave.  An entry without `wave` gives every wave that the search of
survey.py finds, and one statement for each family of waves that has one label.
`label` is STABLE, UNSTABLE or UNCERTAIN.  `qualifier` (or None) narrows the label and belongs to it.
`note` (or None) says why `waves` is empty.
"""
from __future__ import annotations

import runpy
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import sympy as sp

from .conservation import ConservativeSystem
from .label import analyse
from .problem import ProblemError, load_problem
from .survey import DEFAULT_RANGE, find_equilibria, survey
from .symbolic import LocalSystem, StructureError
from .wave import WaveError, end_data

SEARCH_ONLY = ("equilibria_range", "partners", "double_pulses")      # options of survey, not of analyse


def read_list(path):
    """The list `problems` of a file written with SymPy."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ProblemError(f"no such file: {path}")
    ns = runpy.run_path(str(path))
    if "problems" not in ns:
        raise ProblemError("the file must define the list `problems`")
    return list(ns["problems"])


def _entry(item, k):
    """Entry k of the list as a dictionary with a name."""
    if isinstance(item, sp.Equality) or (isinstance(item, (list, tuple)) and item
                                         and all(isinstance(e, sp.Equality) for e in item)):
        item = dict(equations=item)
    if not isinstance(item, Mapping):
        raise ProblemError("an entry must be a dict with the key 'equations', or a sympy.Eq")
    entry = dict(item)
    entry["name"] = str(entry.get("name") or f"eqn{k}")
    return entry


def _show(state):
    return "(" + ", ".join(f"{k} = {v:.6g}" for k, v in state.items()) + ")"


def _given(problem):
    """Description of the wave that the entry names."""
    spec, names = problem.wave, [u.func.__name__ for u in problem.unknowns]
    left, right = (_show(dict(zip(names, spec[side]))) for side in ("left", "right"))
    if spec["kind"] == "front":
        return f"front {left} -> {right}"
    if spec["kind"] == "semi":
        return f"semi-wave behind the free boundary, far state {left}"
    return f"pulse on {left}" if spec["copies"] == 1 else f"{spec['copies']} pulses on {left}"


def _found(w):
    """Description of a wave of the search (survey.py)."""
    if w.get("copies"):
        text = f"two pulses on {_show(w['left'])}, distance {w['distance']:.6g}"
    elif w["kind"] == "front":
        text = f"front {_show(w['left'])} -> {_show(w['right'])}"
    elif w["kind"] == "pulse":
        text = f"pulse on {_show(w['left'])}, extremum {_show(w['extremum'] or {})}"
    else:
        text = f"semi-wave behind the free boundary, far state {_show(w['left'])}"
    return text + (f" [{w['note']}]" if w.get("note") else "")


def _item(wave, speed, record):
    """One wave of the output list, from the record of label.analyse."""
    label = record["label"]
    qualifier = record.get("qualifier") if label == "STABLE" else None
    if label == "UNSTABLE" and record.get("weighted"):      # unstable through the essential spectrum only
        qualifier = f"in L2; in an exponentially weighted space: {record['weighted']['label']}"
    if speed is not None:
        speed = 0.0 if abs(speed) < 1e-9 else float(speed)
    return dict(wave=wave, speed=speed, label=label, qualifier=qualifier, reason=record["reason"])


def _pulse_guesses(problem):
    """Guesses for a pulse of a conservation law that the input names without a guess (DESIGN 2.11).

    The reduced profile equation (conservation.py) has the rest state and further constant solutions.
    A pulse surrounds one of them: its amplitude is of the order of the distance to it.  The width is
    the inverse of the slowest decay rate at the rest state.  Nearest constant solution first.
    """
    try:
        cons = ConservativeSystem(problem, LocalSystem(problem))
        rest = np.asarray(problem.wave["left"], dtype=float)
        width = 1.0 / end_data(cons.reduced, rest, rest, 0.0, True)["slow"]
        limit = float(problem.options.get("equilibria_range", DEFAULT_RANGE))
        others = [u - rest for u in find_equilibria(cons.reduced, -limit, limit)]
    except (StructureError, WaveError, np.linalg.LinAlgError, ArithmeticError):
        return []
    others = sorted((d for d in others if np.abs(d).max() > 1e-9), key=lambda d: np.abs(d).max())
    names = [u.func.__name__ for u in problem.unknowns]
    return [dict(amplitude=dict(zip(names, (alpha * d).tolist())), width=w * width)
            for d in others for alpha in (1.5, 1.0, 2.0) for w in (1.0, 2.0)]


def _named(entry, problem, options):
    """The waves of an entry that names a wave: list of output items.

    With a guess in the entry, or when the default guess of wave.py gives the wave: this one wave.
    Otherwise the wave is found with trial guesses.  Conservation law: the guesses of _pulse_guesses, first
    success.  Other equations: the search of survey.py; its waves with the kind, the end states and the
    speed of the entry are listed (a wave and its mirror image are one wave).
    """
    opts = {key: v for key, v in options.items() if key not in SEARCH_ONLY}
    spec = problem.wave

    def computed(record):
        return record["label"] != "UNCERTAIN" or bool((record.get("wave") or {}).get("computed"))

    def item(record):
        return _item(_given(problem), (record.get("wave") or {}).get("speed"), record)

    first = analyse(entry, **opts)
    hinted = (spec.get("exact_profile") is not None or spec.get("unfolding") is not None or spec["guess"] is not None
              or spec["amplitude"] is not None or spec["copies"] > 1
              or spec["continuation"] is not None
              or spec["frequency"] is not None or spec["frequency_guess"] is not None)   # the search has frequency 0
    if computed(first) or hinted:
        return [item(first)]
    if "conservation" in first:
        for trial in (_pulse_guesses(problem) if spec["kind"] == "pulse" else []):
            record = analyse(entry, wave={**entry["wave"], **trial}, **opts)
            if computed(record):
                return [item(record)]
        return [item(first)]
    found = survey({key: v for key, v in entry.items() if key != "wave"}, **{**options, "double_pulses": 0})
    names = [u.func.__name__ for u in problem.unknowns]
    ends = [spec["left"], spec["right"]]
    symmetric = bool(found.get("search", {}).get("reflection_symmetric"))    # then the search lists one of two mirror images

    def close(a, b):
        return np.allclose(a, b, rtol=1e-6, atol=1e-9)

    def as_named(w):
        """The wave w of the search in the orientation of the entry; None: it is not the wave of the entry."""
        states = [[w[side][name] for name in names] for side in ("left", "right")]
        mirrored = symmetric and not close(states, ends) and close(states, ends[::-1])
        if w["kind"] != spec["kind"] or w.get("copies") or not (mirrored or close(states, ends)):
            return None
        speed = -w["speed"] if mirrored else w["speed"]
        if spec["speed"] is not None and not close(speed, spec["speed"]):
            if not (symmetric and spec["kind"] == "pulse" and close(-speed, spec["speed"])):
                return None
            speed = -speed                              # the mirror image of the pulse of the search
        return dict(w, left=w["right"], right=w["left"], speed=speed) if mirrored else dict(w, speed=speed)

    waves = [w for w in map(as_named, found["waves"]) if w is not None]
    return [_item(_found(w), w["speed"], w["record"]) for w in waves] or [item(first)]


def solve(problems, on_result=None, **options):
    """Label the travelling waves of every equation of the list `problems`.  Returns the output list.

    on_result : called with the result of each equation as soon as it is ready
    options   : options of label.analyse and survey.survey, for all entries (for example two_pass=False)
    An entry that cannot be read, or that fails, gives `waves = []` and a `note`; the list goes on.
    """
    results = []
    for k, item in enumerate(problems, start=1):
        start = time.perf_counter()
        name, waves, note = f"eqn{k}", [], None
        try:
            entry = _entry(item, k)
            name = entry["name"]
            problem = load_problem(entry)
            if problem.wave is not None:
                waves += _named(entry, problem, options)
            else:
                found = survey(entry, **options)
                note = found.get("note")
                waves += [_item(_found(w), w["speed"], w["record"]) for w in found["waves"]]
                waves += [dict(wave=f.get("text") or "every wave that ends at " + _show(f["state"]), speed=None,
                               label=f["label"], qualifier=None, reason=f["statement"])
                          for f in found["families"]]
                if not waves and note is None:
                    note = f"no travelling wave found ({len(found['not_found'])} candidates tried)"
        except ProblemError as error:
            note = f"input error: {error}"
        except Exception as error:            # one entry must not stop the list
            note = f"solver error: {type(error).__name__}: {error}"
        results.append(dict(name=name, waves=waves, note=note, seconds=time.perf_counter() - start))
        if on_result is not None:
            on_result(results[-1])
    return results


def lines(result):
    """One line for each wave of one equation: name, wave, label."""
    if not result["waves"]:
        return [f"{result['name']}: no label ({result['note']})"]
    out = []
    for w in result["waves"]:
        speed = "" if w["speed"] is None else f", c = {w['speed']:.6g}"
        tail = (f" ({w['qualifier']})" if w["qualifier"]
                else f"   ({w['reason']})" if w["label"] == "UNCERTAIN" else "")
        out.append(f"{result['name']}: {w['wave']}{speed}: {w['label']}{tail}")
    return out
