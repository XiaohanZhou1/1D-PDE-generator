"""Structural classification of the input PDE (DESIGN.md, sections 2.1 and 2.5).

The classifier only reads the SymPy expressions.  It decides which sub-solver
applies: L (local), F (free boundary), N (nonlocal), D (delay), G (degenerate),
or OUT (outside the scope).  Milestone M1 implements L only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp
from sympy.core.function import AppliedUndef

from .delay import find_delay
from .kernel import find_kernels
from .symbolic import StructureError

IMPLEMENTED = {"L", "F", "D", "N"}


@dataclass
class Classification:
    sub_solver: str
    supported: bool
    reasons: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def as_dict(self):
        return dict(sub_solver=self.sub_solver, supported=self.supported,
                    reasons=self.reasons, details=self.details)


def classify(problem) -> Classification:
    x, t = problem.x, problem.t
    funcs = {u.func for u in problem.unknowns}
    exprs = [sp.sympify(eq.lhs - eq.rhs) for eq in problem.equations]

    def result(sub, *reasons, **details):
        return Classification(sub, sub in IMPLEMENTED and not details.get("blocked", False),
                              list(reasons), details)

    integrals = any(e.has(sp.Integral) for e in exprs)
    distributed = []                                  # integrals over a delay (delay.py), not over space
    if integrals:
        try:
            found = find_delay(problem)
        except StructureError as error:
            return result("D", str(error), blocked=True)
        distributed = [entry["atom"] for entry in found["integrals"]] if found else []
        integrals = any(i not in distributed for e in exprs for i in e.atoms(sp.Integral))
        if integrals and distributed:
            return result("N", "a convolution in space together with a distributed delay is not supported",
                          blocked=True)
    if integrals:
        try:
            find_kernels(problem)
        except StructureError as error:
            return result("N", f"{error}; a kernel with a rational Fourier transform can also be written "
                          "with an auxiliary unknown and a constraint equation (DESIGN.md 2.7)", blocked=True)
        if getattr(problem, "boundary", None) is not None:
            return result("N", "a convolution together with a free boundary is not supported", blocked=True)
    if integrals or distributed:
        # The remaining checks look at the local terms only.
        exprs = [e.xreplace({i: sp.Dummy() for i in e.atoms(sp.Integral)}) for e in exprs]

    delayed = set()
    for e in exprs:
        for f in e.atoms(AppliedUndef):
            if f.func not in funcs:
                return result("OUT", f"{f.func} is an undefined function that is not an unknown")
            if f.args != (x, t):
                if len(f.args) == 2 and f.args[0] == x:
                    delayed.add(f)                    # shifted time argument: delay
                else:
                    return result("N", f"{f} has a shifted space argument (nonlocal)")
    if delayed and integrals:
        return result("N", "a convolution together with a delay is not supported", blocked=True)
    if delayed or distributed:
        try:
            find_delay(problem)
        except StructureError as error:
            return result("D", str(error), blocked=True)
        if getattr(problem, "boundary", None) is not None:
            return result("D", "a delay together with a free boundary is not supported", blocked=True)

    # Explicit dependence on x or t breaks translation invariance: no travelling wave.
    hidden = {u: sp.Dummy() for u in list(problem.unknowns) + list(delayed)}
    for e in exprs:
        bare = e.doit().xreplace({d: sp.Dummy() for d in e.doit().atoms(sp.Derivative)}).xreplace(hidden)
        if bare.has(x) or bare.has(t):
            return result("OUT", "the equations depend explicitly on x or t")

    time_order, space_order, mixed = 0, 0, False
    for e in exprs:
        for d in e.doit().atoms(sp.Derivative):
            if d.expr in delayed:
                return result("D", "derivatives of the delayed terms are not supported", blocked=True)
            counts = dict(d.variable_count)
            nt, nx = int(counts.get(t, 0)), int(counts.get(x, 0))
            if set(counts) - {x, t}:
                return result("OUT", f"derivative {d} is not with respect to x or t")
            if nt and nx:
                mixed = True
            time_order, space_order = max(time_order, nt), max(space_order, nx)
    details = dict(components=len(problem.unknowns), time_order=time_order, space_order=space_order)
    if mixed:
        return result("L", "mixed x-t derivatives are not implemented in M1", blocked=True, **details)
    if time_order not in (1, 2):
        return result("L", f"time order {time_order} is not supported (first and second order only)",
                      blocked=True, **details)
    if integrals:
        if mixed or time_order != 1:
            return result("N", "with a convolution, only first-order time derivatives are supported",
                          blocked=True, **details)
        return result("N", "local terms and convolution integrals", **details)
    if delayed or distributed:
        if mixed or time_order != 1:
            return result("D", "with a delay, only first-order time derivatives are supported",
                          blocked=True, **details)
        return result("D", "local terms and time delays (discrete or distributed)", **details)
    if getattr(problem, "boundary", None) is None:
        return result("L", "local system, first order in time" if time_order == 1
                      else "local system with second-order time derivatives", **details)
    if time_order != 1:
        return result("F", "with a free boundary, only first-order time derivatives are supported",
                      blocked=True, **details)
    return result("F", "local system on x < s(t) with a Stefan condition", **details)
