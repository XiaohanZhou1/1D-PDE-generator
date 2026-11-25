"""Continuation of a travelling wave in one parameter (DESIGN.md, section 3.2).

The first step is a natural-parameter step: the parameter is changed and the
profile problem is solved again from the previous wave.  All further steps are
pseudo-arclength steps: the parameter is an unknown and the step length along
the branch is prescribed.  Such a step passes a fold of the branch, where the
parameter reaches an extremum and a natural step fails.

Distance on the branch:  ds^2 = int |dU|^2 dz + dc^2 + (dp / scale)^2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline

from .symbolic import StructureError
from .wave import Wave, WaveError, bvp, end_data, merged_grid, padded, reference_from_wave


MAX_NODES = 8000              # a step from a good predictor needs few new nodes; more means failure


@dataclass
class Point:
    value: float              # parameter value
    c: float
    wave: Wave
    left: np.ndarray          # end states at this parameter value
    right: np.ndarray


@dataclass
class Branch:
    parameter: str
    points: list
    status: str               # 'reached', 'max_steps', or the reason of the stop
    folds: list = field(default_factory=list)
    final: Wave | None = None  # the wave at the target value, when it was reached

    def summary(self):
        return dict(parameter=self.parameter, status=self.status, steps=len(self.points) - 1,
                    start=self.points[0].value, end=self.points[-1].value,
                    folds=[dict(value=f["value"], speed=f["speed"]) for f in self.folds],
                    values=[p.value for p in self.points], speeds=[p.c for p in self.points])


def _thin(mesh, limit=1500):
    """The collocation mesh only grows inside one solve.  Keep it bounded along the branch."""
    if mesh.size <= limit:
        return mesh
    k = int(np.ceil(mesh.size / limit))
    return np.unique(np.concatenate((mesh[::k], mesh[-1:])))


def _point(system, sol, left, right, L, free_c, c_fixed, pulse, index):
    N = system.N
    Ym, Yp = system.state(left), system.state(right)
    if pulse and np.abs(sol.y[system.offset] - Ym[system.offset][:, None]).max() < 1e-6 * max(1.0, np.abs(Ym).max()):
        raise WaveError("the solution collapsed to the constant state")
    c = float(sol.p[0]) if free_c else float(c_fixed)
    wave = Wave(c=c, L=float(L), N=N, Y_minus=Ym, Y_plus=Yp, interpolant=sol.sol, mesh=sol.x,
                info=dict(nodes=int(sol.x.size), residual_rms_max=float(sol.rms_residuals.max()),
                          speed_is_unknown=free_c), theta=system.theta)
    return Point(value=float(system.theta[index]), c=c, wave=wave, left=np.array(left, float),
                 right=np.array(right, float))


def _fixed(system, point, index, free_c, tol, pulse, L_new=None, blend=None):
    """Solve at the current parameter values of the system, starting from `point`.

    blend = (other point, weight): the initial guess is the linear interpolation
    between `point` (weight 0) and the other point (weight 1).
    """
    w, N = point.wave, system.N
    left, right = system.equilibrium(point.left), system.equilibrium(point.right)
    mesh = _thin(w.mesh)
    Y0, c_start = w.Y(mesh), w.c
    if blend is not None:
        other, weight = blend
        Y0 = (1 - weight) * Y0 + weight * other.wave.Y(mesh)
        c_start = (1 - weight) * w.c + weight * other.c if free_c else w.c
    Y0 = np.vstack((Y0, np.zeros(mesh.size)))
    L = w.L
    if L_new is not None and L_new > L:
        mesh, Y0 = padded(mesh, Y0, L, L_new, system.state(left), system.state(right), N)
        L = L_new
    sol = bvp(system, left, right, c_start, free_c, mesh, Y0, reference_from_wave(system, w), tol,
              max_nodes=MAX_NODES)
    if not sol.success:
        raise WaveError(f"natural step: {sol.message}")
    return _point(system, sol, left, right, L, free_c, w.c, pulse, index)


def _difference(system, a, b, zs, scale):
    """b - a as (dU on zs, dc, dpi)."""
    off = system.offset
    return b.wave.Y(zs)[off] - a.wave.Y(zs)[off], b.c - a.c, (b.value - a.value) / scale


def _inner(zs, x, y):
    return float(np.trapezoid((x[0] * y[0]).sum(axis=0), zs) + x[1] * y[1] + x[2] * y[2])


def _grid(*points):
    L = max(p.wave.L for p in points)
    return merged_grid(*[p.wave.mesh for p in points], np.linspace(-L, L, 2001))


def _arclength(system, prev, cur, index, ds, free_c, scale, tol, pulse):
    """One pseudo-arclength step from `cur` along the secant through `prev` and `cur`."""
    system.theta = cur.wave.theta
    off = system.offset
    zs = _grid(prev, cur)
    secant = _difference(system, prev, cur, zs, scale)
    norm = np.sqrt(_inner(zs, secant, secant))
    dUd, dc, dpi = secant[0] / norm, secant[1] / norm, secant[2] / norm
    tangent = CubicSpline(zs, dUd, axis=1)            # smooth: the error control of the solve needs it
    arc = dict(index=index, scale=scale, U=lambda zz: cur.wave.Y(zz)[off], dU=tangent,
               c=cur.c, dc=dc, pi=cur.value / scale, dpi=dpi, ds=ds,
               pi_guess=cur.value / scale + ds * dpi)
    mesh = _thin(cur.wave.mesh)
    Ycur = cur.wave.Y(mesh)
    Y0 = np.vstack((Ycur + ds * (Ycur - prev.wave.Y(mesh)) / norm, np.zeros((2, mesh.size))))
    c_start = cur.c + ds * dc if free_c else cur.c
    sol = bvp(system, cur.left, cur.right, c_start, free_c, mesh, Y0,
              reference_from_wave(system, cur.wave), tol, max_nodes=MAX_NODES, arc=arc)
    if not sol.success:
        raise WaveError(f"arclength step: {sol.message}")
    left, right = system.equilibrium(cur.left), system.equilibrium(cur.right)
    return _point(system, sol, left, right, cur.wave.L, free_c, cur.c, pulse, index)


def _fold(system, a, b, c, scale):
    """Parameter extremum on the parabola through three consecutive points."""
    zs = _grid(a, b, c)
    d1 = np.sqrt(_inner(zs, *(2 * [_difference(system, a, b, zs, scale)])))
    d2 = np.sqrt(_inner(zs, *(2 * [_difference(system, b, c, zs, scale)])))
    s = np.array([0.0, d1, d1 + d2])
    pv = np.polyfit(s, [a.value, b.value, c.value], 2)
    s_star = -pv[1] / (2 * pv[0])
    cv = np.polyfit(s, [a.c, b.c, c.c], 2)
    return dict(value=float(np.polyval(pv, s_star)), speed=float(np.polyval(cv, s_star)),
                between=[b.value, c.value])


def trace(system, left, right, wave, parameter, direction=None, step=None, until=None,
          max_steps=200, tol=1e-6, tail_tol=1e-9, L_min=6.0, L_max=2000.0, fixed_L=False,
          growth=1.3, value_range=None, min_step=1e-4):
    """Follow the branch of the wave in `parameter`.

    system    : LocalSystem at the parameter values of `wave`
    left/right: end states at these values
    direction : +1 or -1, the sign of the first parameter step (default: towards `until`)
    until     : stop when the parameter crosses this value; the wave there is `Branch.final`
    min_step  : the trace stops when the step length falls below min_step times the first step
    The system is left at the parameter value of the last point.
    """
    i = system.parameter_index(parameter)
    system.theta = wave.theta or system.theta
    p0 = float(system.theta[i])
    if direction is None:
        if until is None or until == p0:
            raise WaveError("continuation needs a direction")
        direction = np.sign(until - p0)
    scale = max(abs(p0), abs(until) if until is not None else 0.0)
    scale = scale if scale > 0 else 1.0
    step = abs(step) if step else 0.02 * scale
    pulse = bool(np.allclose(left, right))
    free_c = end_data(system, left, right, wave.c, True)["free_c"]
    base = system.theta

    def at(value):
        system.theta = base[:i] + (float(value),) + base[i + 1:]

    start = Point(p0, wave.c, wave, np.array(left, float), np.array(right, float))
    points = [start]
    if until is not None and 0 < (until - p0) * direction <= step:
        at(until)
        end = _fixed(system, start, i, free_c, tol, pulse)
        return Branch(str(parameter), points + [end], "reached", final=end.wave)

    # --- first step: natural parameter ---------------------------------------
    first = None
    for _ in range(8):
        try:
            at(p0 + direction * step)
            first = _fixed(system, start, i, free_c, tol, pulse)
            break
        except (WaveError, StructureError, np.linalg.LinAlgError):
            step /= 2
    if first is None:
        at(p0)
        return Branch(str(parameter), points, "the first step failed")
    points.append(first)
    zs = _grid(start, first)
    ds0 = ds = np.sqrt(_inner(zs, *(2 * [_difference(system, start, first, zs, scale)])))
    folds, status = [], "max_steps"

    # --- pseudo-arclength steps -----------------------------------------------
    while len(points) - 1 < max_steps:
        prev, cur = points[-2], points[-1]
        try:
            new = _arclength(system, prev, cur, i, ds, free_c, scale, tol, pulse)
        except (WaveError, StructureError, np.linalg.LinAlgError) as error:
            ds /= 2
            if ds < min_step * ds0:
                status = f"the step length became too small ({error})"
                break
            continue
        zs = _grid(prev, cur, new)
        old, now = _difference(system, prev, cur, zs, scale), _difference(system, cur, new, zs, scale)
        cosine = _inner(zs, old, now) / np.sqrt(_inner(zs, old, old) * _inner(zs, now, now))
        if cosine < 0.5 and ds > 1e-3 * ds0:
            ds /= 2                               # the branch turns too fast for this step
            continue
        turned = (new.value - cur.value) * (cur.value - prev.value) < 0
        fold = [_fold(system, prev, cur, new, scale)] if turned else []
        if not fixed_L:
            slow = end_data(system, new.left, new.right, new.c, True)["slow"]
            need = float(np.clip(np.log(1.0 / tail_tol) / slow, L_min, L_max))
            if need > 1.2 * new.wave.L:
                try:                                  # the tails decay more slowly: longer domain
                    new = _fixed(system, new, i, free_c, tol, pulse, L_new=need)
                except (WaveError, StructureError, np.linalg.LinAlgError):
                    system.theta = new.wave.theta
        if until is not None and (cur.value - until) * (new.value - until) <= 0:
            # The target lies between the last two points: interpolate, then solve at the target.
            weight = (until - new.value) / (cur.value - new.value) if cur.value != new.value else 0.0
            at(until)
            try:
                end = _fixed(system, new, i, free_c, tol, pulse, blend=(cur, weight))
            except (WaveError, StructureError, np.linalg.LinAlgError) as error:
                ds /= 2                               # bracket the target with a shorter step
                if ds < min_step * ds0:
                    status = f"the last step to the target value failed ({error})"
                    break
                continue
            points.extend((new, end))
            return Branch(str(parameter), points, "reached", folds + fold, end.wave)
        points.append(new)
        folds += fold
        if value_range is not None and not value_range[0] <= new.value <= value_range[1]:
            status = "the parameter left the given range"
            break
        ds = min(ds * growth, 8 * ds0)
    system.theta = points[-1].wave.theta
    return Branch(str(parameter), points, status, folds)
