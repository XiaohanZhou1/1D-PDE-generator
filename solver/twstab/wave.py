"""Computation of the travelling-wave profile (DESIGN.md, sections 2.2 and 3.2).

The profile solves  Y' = f(Y; c)  on [-L, L] with projection boundary conditions
and an integral phase condition.  The speed c is an unknown when the index
iota = k_u(left) + k_s(right) - N is 0, and it is prescribed when iota is 1.

bvp() is the single collocation solve.  solve_wave() adds the choice of the
domain and the checks of the wave.  The continuation engine (continuation.py)
calls bvp() with one more unknown: a parameter of the equations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.integrate import solve_bvp
from scipy.interpolate import CubicSpline

from .asymptotic import split, reference_rows
from .symbolic import StructureError


class WaveError(RuntimeError):
    """The wave could not be computed or failed a check."""


@dataclass
class Wave:
    c: float
    L: float
    N: int
    Y_minus: np.ndarray
    Y_plus: np.ndarray
    interpolant: object
    mesh: np.ndarray
    info: dict = field(default_factory=dict)
    theta: tuple = ()             # parameter values of the system for this wave
    right: float | None = None    # upper end of the domain; None means L (free boundary: 0)

    @property
    def hi(self):
        return self.L if self.right is None else self.right

    def Y(self, z):
        """Profile state at the points z, shape (N, len(z)).  End states outside [-L, hi]."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        out = self.interpolant(np.clip(z, -self.L, self.hi))[:self.N]
        out[:, z < -self.L] = self.Y_minus[:, None]
        out[:, z > self.hi] = self.Y_plus[:, None]
        return out


@dataclass
class Reference:
    """Initial guess and phase reference.  U, dU: z -> (n, m).  Y: z -> (N, m)."""
    U: object
    dU: object
    Y: object


def _boundary_values(spec, system, width):
    """Boundary values b of the default semi-wave guess  far + (b - far) exp(z / width).

    Newton iteration on the boundary conditions, from the values in spec['right'].
    A condition u_j = value holds at the start.  Without convergence the start values stay.
    """
    far, start = np.array(spec["left"], dtype=float), np.array(spec["right"], dtype=float)
    slope = np.zeros((system.N, system.n))             # derivative of the state with respect to b
    for j in range(system.n):
        for k in range(system.mu[j]):
            slope[system.offset[j] + k, j] = width ** -k

    def state(b):
        return system.state(b) + slope @ (b - far) - system.state(b - far)

    b = start.copy()
    for _ in range(30):
        residual = system.boundary_residual(state(b))
        if np.abs(residual).max() <= 1e-12 * max(1.0, np.abs(b).max()):
            return b
        b = b + np.linalg.lstsq(system.boundary_jacobian(state(b)) @ slope, -residual, rcond=None)[0]
        if not np.all(np.isfinite(b)):
            break
    return start


def _guess_expressions(problem, system):
    """SymPy expressions U_j(z) for the initial guess and the phase reference."""
    spec = problem.wave
    z = sp.Symbol("z", real=True)
    w = spec["width"]
    user = {}
    for key, value in (spec.get("guess") or {}).items():
        name = key.func.__name__ if hasattr(key, "func") and not isinstance(key, str) else str(key)
        expr = sp.sympify(value).subs(problem.parameters)
        free = expr.free_symbols
        if len(free) > 1:
            raise WaveError(f"the guess for {name} must depend on z only")
        if expr.has(sp.Min, sp.Max, sp.Abs, sp.Piecewise, sp.Heaviside, sp.sign):
            # A kink makes the weight U_ref' of the phase condition discontinuous;
            # the collocation error control then never reaches its tolerance.
            raise WaveError(f"the guess for {name} must be continuously differentiable "
                            "(no Min, Max, Abs, Piecewise)")
        user[name] = expr.subs({s: z for s in free})
    amplitude = {}
    for key, value in (spec.get("amplitude") or {}).items():
        name = key.func.__name__ if hasattr(key, "func") and not isinstance(key, str) else str(key)
        amplitude[name] = float(sp.sympify(value).subs(problem.parameters))
    at_boundary = _boundary_values(spec, system, w) if spec["kind"] == "semi" else None
    exprs = []
    for j, u in enumerate(problem.unknowns):
        name = u.func.__name__
        if name in user:
            exprs.append(user[name])
        elif spec["kind"] == "front":
            a, b = spec["left"][j], spec["right"][j]
            exprs.append(a + (b - a) * (1 + sp.tanh(z / w)) / 2)
        elif spec["kind"] == "semi":
            a, b = spec["left"][j], at_boundary[j]            # far state and boundary value
            exprs.append(a + (b - a) * sp.exp(z / w))
        elif name in amplitude:
            exprs.append(spec["left"][j] + amplitude[name] / sp.cosh(z / w) ** 2)
        else:
            raise WaveError(f"a pulse needs wave['guess'] or wave['amplitude'] for {name}")
    return z, exprs


def reference_from_guess(problem, system):
    """Reference built from the guess expressions of the problem file."""
    z, exprs = _guess_expressions(problem, system)
    n, N = system.n, system.N
    modules = ["scipy", "numpy"]                       # scipy: special functions (erf) on arrays
    ref = [sp.lambdify(z, e, modules) for e in exprs]
    dref = [sp.lambdify(z, sp.diff(e, z), modules) for e in exprs]
    derivs = [[sp.lambdify(z, sp.diff(e, z, k), modules) for k in range(system.mu[j])]
              for j, e in enumerate(exprs)]

    def on(f, zz):
        return np.broadcast_to(np.asarray(f(zz), dtype=float), np.shape(zz))

    def state(zz):
        Y0 = np.zeros((N, zz.size))
        for j in range(n):
            for k in range(system.mu[j]):
                Y0[system.offset[j] + k] = on(derivs[j][k], zz)
        return Y0

    Y = state
    if hasattr(system, "from_jets"):
        # A system in flux variables (taxis.py): the guess is turned into its state.
        spec = problem.wave
        c0 = spec["speed"] if spec["speed"] is not None else spec["speed_guess"]
        Y = lambda zz: system.from_jets(state(zz), c0)
    return Reference(U=lambda zz: np.array([on(f, zz) for f in ref]),
                     dU=lambda zz: np.array([on(f, zz) for f in dref]), Y=Y)


def merged_grid(*grids):
    """Sorted union of grids without nearly coincident points."""
    zs = np.unique(np.concatenate(grids))
    keep = np.concatenate(([True], np.diff(zs) > 1e-9 * max(1.0, zs[-1] - zs[0])))
    keep[-1] = True
    return zs[keep]


def reference_from_wave(system, wave):
    """Reference built from a computed wave."""
    off = system.offset
    zs = merged_grid(wave.mesh, np.linspace(-wave.L, wave.L, 2001))
    Yd = wave.Y(zs)
    # U' from the profile equation at the parameter values of the wave.  Difference
    # quotients on this irregular grid would amplify rounding errors.
    current, system.theta = system.theta, wave.theta or system.theta
    try:
        dUd = system.rhs(Yd, wave.c)[off]
    finally:
        system.theta = current
    smooth = CubicSpline(zs, dUd, axis=1)             # a smooth weight keeps the error control effective

    def weight(zz):
        zz = np.asarray(zz, dtype=float)
        return np.where(np.abs(zz) <= wave.L, smooth(np.clip(zz, -wave.L, wave.L)), 0.0)

    return Reference(U=lambda zz: wave.Y(zz)[off], dU=weight, Y=wave.Y)


def end_data(system, left, right, c, speed_given):
    """Splitting of the two end states: dimensions, index iota and the slowest decay rate."""
    N = system.N
    ku_m, _, evu_m, evs_m = split(system.end_state(left, c)[0])
    ku_p, _, evu_p, evs_p = split(system.end_state(right, c)[0])
    rates = np.concatenate((evu_m.real, -evs_p.real, -evs_m.real, evu_p.real))
    if rates.size == 0 or rates.min() < 1e-9 * max(1.0, abs(rates).max()):
        raise WaveError("an end state is not hyperbolic in the profile equation at the given speed")
    iota = ku_m + (N - ku_p) - N
    if iota == 0:
        free_c = True
    elif iota == 1:
        free_c = False
        if not speed_given:
            raise WaveError("the waves form a one-parameter family: wave['speed'] must be given")
    else:
        raise WaveError(f"index iota = {iota}: the wave is not determined by its speed alone (not in M1)")
    slow = min(evu_m.real.min() if ku_m else np.inf, (-evs_p.real).min() if ku_p < N else np.inf)
    fast = max(np.abs(ev).max() for ev in (evu_m, evs_m, evu_p, evs_p) if ev.size)
    return dict(ku_m=int(ku_m), ku_p=int(ku_p), iota=int(iota), free_c=free_c, slow=float(slow),
                fast=float(fast))


def bvp(system, left, right, c0, free_c, mesh, Y0, ref, tol, max_nodes=40000, arc=None):
    """One collocation solve of the profile problem.  Returns the SciPy solution object.

    ref : Reference; its U and dU fix the translation by  int (U - U_ref).U_ref' dz = 0.
    arc : None, or the data of a pseudo-arclength step.  The parameter with index
          arc['index'] (value = arc['scale'] * pi) is then an unknown, the end states
          follow it, and one more integral condition is added:
              int (U - U_k).dU ds + (c - c_k) dc + (pi - pi_k) dpi = ds.
          On success system.theta holds the new parameter value.
    Rows of the state: N profile rows, the phase integral, and with `arc` the arclength integral.
    """
    N, off, eye = system.N, system.offset, np.eye(system.N)
    theta0 = system.theta

    def unpack(p):
        c = p[0] if free_c else c0
        if arc is not None:
            i = arc["index"]
            system.theta = theta0[:i] + (float(p[-1]) * arc["scale"],) + theta0[i + 1:]
        return c

    def ends():
        if arc is None:
            return left, right
        return system.equilibrium(left), system.equilibrium(right)

    def splits(c):
        uL, uR = ends()
        return uL, uR, split(system.end_state(uL, c)[0]), split(system.end_state(uR, c)[0])

    def fun(zz, Ya, p=None):
        c = unpack(p)
        rows = [system.rhs(Ya[:N], c), ((Ya[off] - ref.U(zz)) * ref.dU(zz)).sum(axis=0)]
        if arc is not None:
            rows.append(((Ya[off] - arc["U"](zz)) * arc["dU"](zz)).sum(axis=0))
        return np.vstack(rows)

    def bc(Ya, Yb, p=None):
        c = unpack(p)
        uL, uR, (km, Pm, *_), (kp, Pp, *_) = splits(c)
        if km != ku_m or kp != ku_p:
            raise WaveError("the end-state splitting changed during the iteration")
        out = [C_left @ ((eye - Pm) @ (Ya[:N] - system.state(uL))),
               C_right @ (Pp @ (Yb[:N] - system.state(uR))), [Ya[N], Yb[N]]]
        if arc is not None:
            out.append([Ya[N + 1], Yb[N + 1] + (c - arc["c"]) * arc["dc"]
                        + (p[-1] - arc["pi"]) * arc["dpi"] - arc["ds"]])
        return np.concatenate(out)

    start = ([c0] if free_c else []) + ([arc["pi_guess"]] if arc is not None else [])
    try:
        unpack(start)
        _, _, (ku_m, Pu_m, *_), (ku_p, Pu_p, *_) = splits(c0)
        C_left = reference_rows(eye - Pu_m, N - ku_m).real
        C_right = reference_rows(Pu_p, ku_p).real
        # The projection conditions must be N (speed unknown) or N - 1 (speed given).  The number
        # changes when a parameter moves an end state through a loss of hyperbolicity.
        if (N - ku_m) + ku_p != N - (0 if free_c else 1):
            raise WaveError("the end-state splitting does not match the number of unknowns")
        sol = solve_bvp(fun, bc, mesh, Y0, p=start or None, tol=tol, max_nodes=max_nodes)
        if arc is not None and sol.success:
            unpack(sol.p)
        else:
            system.theta = theta0
        return sol
    except (np.linalg.LinAlgError, StructureError) as error:
        system.theta = theta0
        raise WaveError(f"the profile problem broke down: {error}") from error
    except WaveError:
        system.theta = theta0
        raise


def padded(mesh, Y, L, L_new, Ym, Yp, N):
    """Extend a solution from [-L, L] to [-L_new, L_new] with the end states."""
    grid = np.concatenate((np.linspace(-L_new, -L, 12)[:-1], mesh, np.linspace(L, L_new, 12)[1:]))
    Y0 = np.zeros((Y.shape[0], grid.size))
    inside = np.abs(grid) <= L
    Y0[:, inside] = Y
    Y0[:N, grid < -L] = Ym[:, None]
    Y0[:N, grid > L] = Yp[:, None]
    return grid, Y0


def _second_phase(system, ref, extra, c):
    """Data of the second phase condition  int (U - U_ref).(J U_ref) dz = 0  (rotation symmetry).

    One parameter of the system (index extra['index']) is the additional unknown.  The data have
    the form of a pseudo-arclength step of `bvp` without its two increments.
    """
    value = system.theta[extra["index"]]
    scale = max(1.0, abs(value))
    return dict(index=extra["index"], scale=scale, pi_guess=value / scale, pi=value / scale, c=c,
                U=ref.U, dU=lambda zz: extra["J"] @ ref.U(zz), dc=0.0, dpi=0.0, ds=0.0)


def solve_wave(problem, system, tol=1e-8, check_tol=5e-10, tail_tol=1e-9,
               L_min=6.0, L_max=2000.0, max_nodes=40000, start=None, extra=None):
    """Compute and check the wave of a problem.

    start : None, or a Wave.  It then replaces the guess of the problem file as the
            initial guess and as the phase reference.
    extra : None, or dict(index, J): a wave with a rotation symmetry (symmetry.py).  The parameter
            with this index is one more unknown, and a second phase condition is added.  On return
            system.theta holds the value found.
    """
    spec = problem.wave
    N = system.N
    left, right = spec["left"], spec["right"]
    Ym, Yp = system.state(left), system.state(right)
    if spec["speed"] is not None:
        c0 = spec["speed"]
    else:
        c0 = start.c if start is not None else spec["speed_guess"]
    ends = end_data(system, left, right, c0, spec["speed"] is not None)
    free_c, slow = ends["free_c"], ends["slow"]

    # Truncation length from the slowest relevant decay rate.
    L = spec["L"] if spec["L"] else float(np.clip(np.log(1.0 / tail_tol) / slow, L_min, L_max))

    def run(zz, Y0, c_start, tolerance):
        arc = None if extra is None else _second_phase(system, ref, extra, c_start)
        if arc is not None and Y0.shape[0] == N + 1:
            Y0 = np.vstack((Y0, np.zeros(Y0.shape[1])))      # the integral of the second phase condition
        return bvp(system, left, right, c_start, free_c, zz, Y0, ref, tolerance, max_nodes, arc)

    if extra is not None and (start is not None or not free_c):
        raise WaveError("a wave with a frequency needs an isolated speed and a guess from the problem file")
    if start is None:
        ref = reference_from_guess(problem, system)
        if extra is not None and np.abs(extra["J"] @ ref.U(np.linspace(-L, L, 201))).max() == 0:
            raise WaveError("the guess has no rotating component: a wave with a frequency needs one")
        # Initial mesh: at least 241 nodes, and a spacing below 1/(2 nu) for the fastest end-state rate nu.
        grid = np.linspace(-L, L, int(np.clip(4 * L * ends["fast"], 240, 4000)) + 1)
        Y0 = np.vstack((ref.Y(grid), np.zeros(grid.size)))
    else:
        ref = reference_from_wave(system, start)
        grid, Y0 = start.mesh, np.vstack((start.Y(start.mesh), np.zeros(start.mesh.size)))
        if L > start.L:
            grid, Y0 = padded(grid, Y0, start.L, L, Ym, Yp, N)
        else:
            L = start.L
    sol = run(grid, Y0, c0, tol)
    scale = max(1.0, float(np.abs(sol.y[:N]).max())) if sol.success else 1.0
    extensions = 0
    while sol.success and spec["L"] is None and L < L_max and extensions < 3:
        tails = max(np.abs(sol.y[:N, 0] - Ym).max(), np.abs(sol.y[:N, -1] - Yp).max()) / scale
        if tails < 1e3 * tail_tol:
            break
        L_new = min(1.5 * L, L_max)
        grid, Y0 = padded(sol.x, sol.y, L, L_new, Ym, Yp, N)
        L = L_new
        sol = run(grid, Y0, sol.p[0] if free_c else c0, tol)
        extensions += 1
    if not sol.success:
        raise WaveError(f"the boundary value problem did not converge: {sol.message}")

    # Check: solve again with a tighter tolerance, starting from the first solution.
    fine = run(sol.x, sol.y, sol.p[0] if free_c else c0, check_tol)
    if not fine.success:
        raise WaveError(f"the tighter solve failed: {fine.message}")
    c_first = float(sol.p[0]) if free_c else float(c0)
    c = float(fine.p[0]) if free_c else float(c0)
    if free_c and spec["speed"] is not None and abs(c - spec["speed"]) > 1e-6 * max(1.0, abs(spec["speed"])):
        # The index says that the speed is isolated, so it was an unknown.  A different result means
        # that the wave found is not the wave named in the file.
        raise WaveError(f"the wave found has the speed {c:.6g}, not the given speed {spec['speed']:g}: the given "
                        "speed is inaccurate, or the speed is not isolated (a family of waves from a symmetry)")
    probe = np.linspace(-L, L, 2001)
    Yf = fine.sol(probe)[:N]
    profile_error = float(np.abs(Yf - sol.sol(probe)[:N]).max())
    tails = float(max(np.abs(Yf[:, 0] - Ym).max(), np.abs(Yf[:, -1] - Yp).max()))
    # The projection boundary conditions are exact to first order only: the profile
    # must be close to its end states at both ends of the domain.
    if tails > 1e3 * tail_tol * scale:
        raise WaveError(f"the profile has not reached its end states at the ends of the domain "
                        f"(distance {tails:.2g}, L = {L:.4g}); a longer domain is needed (options['L_max'])")

    # A pulse must not collapse to the constant state.
    base = Ym[:, None] if spec["kind"] == "pulse" else np.linspace(Ym, Yp, probe.size).T
    departure = float(np.abs(Yf[system.offset] - base[system.offset]).max())
    if spec["kind"] == "pulse" and departure < 1e-6 * max(1.0, scale):
        raise WaveError("the solution collapsed to the constant state")

    info = dict(speed_is_unknown=free_c, index_iota=ends["iota"], unstable_dim_left=ends["ku_m"],
                stable_dim_right=int(N - ends["ku_p"]), nodes=int(fine.x.size), domain_extensions=extensions,
                residual_rms_max=float(fine.rms_residuals.max()), speed_first_solve=c_first,
                speed_error=abs(c - c_first), profile_error=profile_error, tail_error=tails,
                amplitude=departure, slowest_rate=slow, started_from="wave" if start is not None else "guess")
    return Wave(c=c, L=float(L), N=N, Y_minus=Ym, Y_plus=Yp, interpolant=fine.sol, mesh=fine.x,
                info=info, theta=system.theta)


def solve_wave_at_speed(problem, system, **kwargs):
    """Wave with the speed given in the problem file, for a system with second-order time derivatives.

    Without damping the speed is a symmetry parameter: every speed below the characteristic one
    has a wave, and the problem with c as an unknown is degenerate.  The profile equation at the
    given speed c is solved as a wave of an auxiliary problem of first order in time,
        v_t = E(v, v_x, v_xx; c),
    whose own speed d unfolds the problem.  d must come out as zero.
    """
    from .problem import Problem
    from .symbolic import LocalSystem

    sym, n = system.symbols, system.n
    speed = float(problem.wave["speed"])
    if not end_data(system, problem.wave["left"], problem.wave["right"], speed, True)["free_c"]:
        return solve_wave(problem, system, **kwargs)       # a one-parameter family in the speed: nothing to unfold
    U = [[sym["low"][system.offset[j] + k] for k in range(system.mu[j])] + [sym["T"][j]] for j in range(n)]
    E = [(g - sym["c"] ** 2 * U[i][2] if system.inertia[i] else g + sym["c"] * U[i][1]).xreplace({sym["c"]: speed})
         for i, g in enumerate(sym["G"])]
    us, x = problem.unknowns, problem.x
    back = {U[j][k]: (us[j] if k == 0 else sp.Derivative(us[j], (x, k))) for j in range(n) for k in range(len(U[j]))}
    auxiliary = Problem(path=problem.path, name=problem.name, x=x, t=problem.t, unknowns=us,
                        equations=[sp.Eq(sp.Derivative(us[i], problem.t), e.xreplace(back)) for i, e in enumerate(E)],
                        parameters=problem.parameters, wave=dict(problem.wave, speed=None, speed_guess=0.0),
                        options=problem.options)
    wave = solve_wave(auxiliary, LocalSystem(auxiliary), **kwargs)
    if abs(wave.c) > 1e-6 * max(1.0, abs(speed)):
        raise WaveError(f"there is no wave with the given speed {speed:g} (the unfolding parameter is {wave.c:.3g})")
    return Wave(c=speed, L=wave.L, N=wave.N, Y_minus=wave.Y_minus, Y_plus=wave.Y_plus, interpolant=wave.interpolant,
                mesh=wave.mesh, info=dict(wave.info, speed_is_unknown=False, unfolding_parameter=wave.c),
                theta=system.theta, right=wave.right)
