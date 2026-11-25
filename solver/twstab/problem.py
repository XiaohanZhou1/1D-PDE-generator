"""Read and validate a problem written with SymPy: a file, or one entry of a list (DESIGN.md, section 4.1)."""
from __future__ import annotations

import math
import runpy
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import sympy as sp
from sympy.core.function import AppliedUndef


class ProblemError(ValueError):
    """The input file does not follow the input schema."""


@dataclass
class Problem:
    path: Path | None         # None: the problem is an entry of a list
    name: str
    x: sp.Symbol
    t: sp.Symbol
    equations: list
    unknowns: list            # applied functions u_j(x, t), in component order
    parameters: dict          # Symbol -> float
    wave: dict | None         # normalised wave specification; None: the file names no wave (survey.py)
    options: dict = field(default_factory=dict)
    boundary: dict | None = None   # free boundary: conditions, Stefan condition, values of u_j = value


def _number(value, parameters, what):
    try:
        return float(sp.sympify(value).subs(parameters))
    except (TypeError, ValueError) as error:
        raise ProblemError(f"{what} is not a number once the parameters are inserted") from error


def _state(spec, unknowns, parameters, what):
    """Turn {u: value} or {'u': value} into a list ordered like `unknowns`."""
    if spec is None:
        raise ProblemError(f"wave['{what}'] is missing")
    by_name = {}
    for key, value in spec.items():
        name = key.func.__name__ if isinstance(key, AppliedUndef) else str(key)
        by_name[name] = value
    values = []
    for u in unknowns:
        name = u.func.__name__
        if name not in by_name:
            raise ProblemError(f"wave['{what}'] has no value for {name}")
        values.append(_number(by_name[name], parameters, f"wave['{what}'][{name}]"))
    return values


def _continuation(spec, parameters):
    """Normalise wave['continuation']: the guess of the file is valid at `start`, not at the file value."""
    if spec is None:
        return None
    name = str(spec.get("parameter"))
    if name not in {str(k) for k in parameters}:
        raise ProblemError(f"wave['continuation']: the file has no parameter named {name}")
    if "start" not in spec:
        raise ProblemError("wave['continuation'] needs the entry 'start'")
    start = float(spec["start"])
    target = {str(k): v for k, v in parameters.items()}[name]
    direction = spec.get("direction")
    if direction is None and start == target:
        raise ProblemError("wave['continuation']: 'direction' (+1 or -1) is needed when start equals the file value")
    return dict(parameter=name, start=start,
                direction=None if direction is None else (1.0 if float(direction) > 0 else -1.0),
                step=None if spec.get("step") is None else abs(float(spec["step"])),
                max_steps=int(spec.get("max_steps", 200)))


def _boundary(spec, unknowns, parameters, x):
    """Normalise `free_boundary`: conditions at x = s(t) and the Stefan condition ds/dt = speed."""
    spec = dict(spec)
    conditions = list(spec.get("conditions", []))
    if not 1 <= len(conditions) <= len(unknowns) or not all(isinstance(e, sp.Equality) for e in conditions):
        raise ProblemError("free_boundary['conditions'] must hold sympy.Eq entries, at most one for every unknown")
    if "speed" not in spec:
        raise ProblemError("free_boundary needs the entry 'speed' (Stefan condition ds/dt = speed)")
    speed = sp.sympify(spec["speed"])
    symbol = spec.get("speed_symbol")           # stands for ds/dt inside the conditions
    if symbol is not None and speed.has(symbol):
        raise ProblemError("free_boundary['speed'] must not contain free_boundary['speed_symbol']")
    # Conditions of the form u_j = value give the boundary values of the default guess.
    values = [None] * len(unknowns)
    for eq in conditions:
        for j, u in enumerate(unknowns):
            other = eq.rhs if eq.lhs == u else eq.lhs if eq.rhs == u else None
            if other is None or values[j] is not None:
                continue
            other = sp.sympify(other)
            if not other.has(AppliedUndef) and (symbol is None or not other.has(symbol)):
                values[j] = _number(other, parameters, "a boundary value")
    if symbol is not None:
        conditions = [sp.Eq((eq.lhs - eq.rhs).subs(symbol, speed), 0) for eq in conditions]
    return dict(conditions=conditions, speed=speed, values=values)


def source_name(source):
    """Name of an input for the records: the path of a file, or the name of an entry of a list."""
    return str(source.get("name", "entry")) if isinstance(source, Mapping) else str(source)


def _variables(equations):
    """(x, t) of an entry that does not give them: every unknown is written as a function of (x, t)."""
    pairs = set()
    for eq in equations:
        for d in eq.atoms(sp.Derivative):
            f = d.expr
            if (isinstance(f, AppliedUndef) and len(f.args) == 2 and all(isinstance(a, sp.Symbol) for a in f.args)
                    and f.args[1] in d.variables):
                pairs.add(f.args)
    if len(pairs) != 1:
        raise ProblemError("x and t could not be read from the equations: write every unknown as a function "
                           "of (x, t), with a time derivative in the equations, or give `x` and `t`")
    return pairs.pop()


def load_problem(path, parameters=None, wave=None) -> Problem:
    """Read a problem: a file, or a dictionary with the same names (one entry of a list, batch.py).

    parameters : {name: value}, replaces values given in the file
    wave       : a wave specification in the syntax of the file; it replaces the one of the file
    A file without `wave` (and no `wave` argument) gives a problem with wave = None.
    An entry may leave out `x` and `t`: they are read from the equations.  `equations` may be one sympy.Eq.
    """
    if isinstance(path, Mapping):
        ns, name, path = dict(path), source_name(path), None
        if "equations" not in ns:
            raise ProblemError("the entry must define `equations`")
    else:
        path = Path(path).resolve()
        if not path.is_file():
            raise ProblemError(f"no such file: {path}")
        ns, name = runpy.run_path(str(path)), path.stem
        for required in ("x", "t", "equations"):
            if required not in ns:
                raise ProblemError(f"the file must define `{required}`")
    equations = ns["equations"]
    equations = [equations] if isinstance(equations, sp.Equality) else list(equations)
    if not equations or not all(isinstance(e, sp.Equality) for e in equations):
        raise ProblemError("`equations` must be a sympy.Eq or a non-empty list of sympy.Eq")
    x, t = (ns["x"], ns["t"]) if "x" in ns and "t" in ns else _variables(equations)
    replace = {str(k): float(v) for k, v in (parameters or {}).items()}
    parameters = {sp.sympify(k): float(v) for k, v in dict(ns.get("parameters", {})).items()}
    unknown_names = set(replace) - {str(k) for k in parameters}
    if unknown_names:
        raise ProblemError(f"the file has no parameter named {sorted(unknown_names)}")
    parameters = {k: replace.get(str(k), v) for k, v in parameters.items()}

    if "unknowns" in ns:
        unknowns = list(ns["unknowns"])
    else:
        # Order of first appearance as a first-order time derivative, then by name.
        seen = []
        for eq in equations:
            for d in sorted(eq.atoms(sp.Derivative), key=str):
                if d.variables == (t,) and isinstance(d.expr, AppliedUndef) and d.expr not in seen:
                    seen.append(d.expr)
        # Unknowns of constraint equations have no time derivative: add them, by name.
        others = {f for eq in equations for f in eq.atoms(AppliedUndef) if f.args == (x, t)} - set(seen)
        unknowns = seen + sorted(others, key=str)
    if len(unknowns) != len(equations):
        raise ProblemError(
            f"{len(equations)} equations but {len(unknowns)} unknowns; define `unknowns` explicitly")
    for u in unknowns:
        if not (isinstance(u, AppliedUndef) and u.args == (x, t)):
            raise ProblemError(f"unknown {u} must be written as a function of (x, t)")

    # Free boundary: the domain is x < s(t).  A wave is a semi-wave: `far` is its state at x -> -infinity.
    boundary = _boundary(ns["free_boundary"], unknowns, parameters, x) if "free_boundary" in ns else None
    if wave is None and ns.get("wave") is None:
        return Problem(path=path, name=name, x=x, t=t, equations=equations, unknowns=unknowns,
                       parameters=parameters, wave=None, options=dict(ns.get("options", {})), boundary=boundary)
    wave = dict(wave if wave is not None else ns["wave"])
    kind = wave.get("kind")
    if boundary is not None:
        if kind != "semi" or "far" not in wave:
            raise ProblemError("with a free boundary, wave['kind'] must be 'semi' and wave['far'] must be given")
        wave.setdefault("left", wave["far"])
        at_boundary = _state(wave["far"], unknowns, parameters, "far")
        at_boundary = [v if v is not None else f for v, f in zip(boundary["values"], at_boundary)]
        wave.setdefault("right", {u.func.__name__: v for u, v in zip(unknowns, at_boundary)})
    elif kind not in ("front", "pulse"):
        raise ProblemError("wave['kind'] must be 'front' or 'pulse'")
    if kind == "pulse" and "rest" in wave:
        wave.setdefault("left", wave["rest"])
        wave.setdefault("right", wave["rest"])
    spec = dict(kind=kind,
                left=_state(wave.get("left"), unknowns, parameters, "left"),
                right=_state(wave.get("right"), unknowns, parameters, "right"),
                speed=None if wave.get("speed") is None else _number(wave["speed"], parameters, "wave['speed']"),
                speed_guess=_number(wave.get("speed_guess", 0.0), parameters, "wave['speed_guess']"),
                width=_number(wave.get("width", 1.0), parameters, "wave['width']"),
                guess=wave.get("guess"), amplitude=wave.get("amplitude"),
                L=None if wave.get("L") is None else float(wave["L"]))
    if wave.get("exact_profile") is not None:
        raw = wave["exact_profile"]
        names = [u.func.__name__ for u in unknowns]
        if not isinstance(raw, Mapping) or set(raw) != set(names):
            raise ProblemError("wave['exact_profile'] must map every component name to a SymPy expression")
        if not all(isinstance(raw[name], (sp.Expr, int)) for name in names):
            raise ProblemError("wave['exact_profile'] expressions must be constructed with SymPy")
        spec["exact_profile"] = [sp.sympify(raw[name]) for name in names]
    if wave.get("unfolding") is not None:
        raw = wave["unfolding"]
        if (not isinstance(raw, Mapping) or set(raw) != {"parameter","target","guess"}
                or not isinstance(raw["parameter"], (str,sp.Symbol))):
            raise ProblemError("wave['unfolding'] needs parameter, target and guess")
        matches = [key for key in parameters if str(key)==str(raw["parameter"])]
        if len(matches)!=1:
            raise ProblemError("unfolding parameter must name one PDE parameter")
        target = _number(raw["target"],parameters,"unfolding target")
        guess = _number(raw["guess"],parameters,"unfolding guess")
        if not all(map(math.isfinite,(target,guess))) or parameters[matches[0]]!=target:
            raise ProblemError("unfolding target must equal the finite PDE parameter value and guess must be finite")
        if spec.get("exact_profile") is not None or wave.get("continuation") is not None:
            raise ProblemError("unfolding cannot be combined with an exact profile or continuation")
        spec["unfolding"] = dict(parameter=matches[0],target=target,guess=guess)
    # A wave with a rotation symmetry, u = exp(omega t J) U(x - c t): omega is given, or it is an unknown.
    for key in ("frequency", "frequency_guess"):
        spec[key] = None if wave.get(key) is None else _number(wave[key], parameters, f"wave['{key}']")
    if spec["frequency"] is not None and spec["frequency_guess"] is not None:
        raise ProblemError("give wave['frequency'] or wave['frequency_guess'], not both")
    spec["continuation"] = _continuation(wave.get("continuation"), parameters)
    spec["copies"] = int(wave.get("copies", 1))
    if spec["copies"] > 1:
        # Multi-pulse: `copies` pulses, each gap near `separation` (one number or one number for each gap).
        if kind != "pulse" or "separation" not in wave:
            raise ProblemError("wave['copies'] > 1 needs kind 'pulse' and wave['separation']")
        gaps = wave["separation"] if isinstance(wave["separation"], (list, tuple)) else [wave["separation"]]
        if len(gaps) not in (1, spec["copies"] - 1):
            raise ProblemError("wave['separation'] must be one number or one number for each gap")
        spec["separation"] = [_number(g, parameters, "wave['separation']") for g in gaps]
    if kind == "pulse" and spec["left"] != spec["right"]:
        raise ProblemError("a pulse must have identical end states")
    if kind == "front" and spec["left"] == spec["right"]:
        raise ProblemError("a front must have different end states")
    return Problem(path=path, name=name, x=x, t=t, equations=equations, unknowns=unknowns,
                   parameters=parameters, wave=spec, options=dict(ns.get("options", {})), boundary=boundary)
