"""Sub-solver F: waves behind a free boundary (DESIGN.md, sections 2.6 and 3.5).

The equations hold on x < s(t).  At x = s(t) there are boundary conditions
b(u, u_x) = 0 and the Stefan condition gives ds/dt.  In the frame z = x - s(t)
the domain is z < 0 and the boundary is fixed at z = 0.  A wave is a steady
state U(z) with ds/dt = c.  A component with diffusion has one condition.  A
component without diffusion has one condition if and only if its
characteristics enter the domain at the boundary.

Linearisation with  ds/dt = c + eta exp(lambda t)  (P: first-order state of p):
    lambda p = L p + eta U',     J P(0) = 0,     eta = grad S . P(0).
The scalar eta becomes one more state with eta' = 0.  The first-order system
has dimension N + 1 and the matching determinant of evans.py applies.
There is no translation eigenvalue: the boundary fixes the position.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.integrate import solve_bvp
from scipy.optimize import brentq

from .asymptotic import reference_rows, split
from .evans import Evans, SplittingError, graded_nodes
from .scout import _difference_matrices
from .symbolic import StructureError
from .wave import Wave, WaveError, reference_from_guess


def solve_semi_wave(problem, system, tol=1e-8, check_tol=5e-10, tail_tol=1e-9,
                    L_min=6.0, L_max=2000.0, max_nodes=40000):
    """Wave on [-L, 0]: far state at -L, boundary conditions and Stefan condition at 0."""
    spec = problem.wave
    N, eye = system.N, np.eye(system.N)
    far = spec["left"]
    Ym = system.state(far)
    c0 = spec["speed"] if spec["speed"] is not None else spec["speed_guess"]

    def far_split(c):
        return split(system.end_state(far, c)[0])

    ku, Pu, evu, evs = far_split(c0)
    rates = np.concatenate((evu.real, -evs.real))
    if rates.size == 0 or rates.min() < 1e-9 * max(1.0, abs(rates).max()):
        raise WaveError("the far state is not hyperbolic in the profile equation at the given speed")
    # Unknowns: ku coordinates on the unstable manifold and c.  Conditions at z = 0: n_boundary + 1.
    if ku != system.n_boundary:
        raise WaveError(f"the unstable manifold of the far state has dimension {ku}, but there are "
                        f"{system.n_boundary} boundary conditions: the speed is not isolated")
    slow, fast = float(evu.real.min()), float(max(abs(evu).max(), abs(evs).max() if evs.size else 0.0))
    L = spec["L"] if spec["L"] else float(np.clip(np.log(1.0 / tail_tol) / slow, L_min, L_max))
    C_left = reference_rows(eye - Pu, N - ku).real
    ref = reference_from_guess(problem, system)

    def fun(zz, Y, p):
        return system.rhs(Y, p[0])

    def bc(Ya, Yb, p):
        k, P, *_ = far_split(p[0])
        if k != ku:
            raise WaveError("the splitting at the far state changed during the iteration")
        return np.concatenate((C_left @ ((eye - P) @ (Ya - Ym)), system.boundary_residual(Yb),
                               [p[0] - system.stefan(Yb)[0]]))

    def run(zz, Y0, c_start, tolerance):
        try:
            return solve_bvp(fun, bc, zz, Y0, p=[c_start], tol=tolerance, max_nodes=max_nodes)
        except (np.linalg.LinAlgError, StructureError) as error:
            raise WaveError(f"the profile problem broke down: {error}") from error

    grid = np.linspace(-L, 0.0, int(np.clip(2 * L * fast, 240, 4000)) + 1)
    sol = run(grid, ref.Y(grid), c0, tol)
    scale = max(1.0, float(np.abs(sol.y).max())) if sol.success else 1.0
    extensions = 0
    while sol.success and spec["L"] is None and L < L_max and extensions < 3:
        if np.abs(sol.y[:, 0] - Ym).max() / scale < 1e3 * tail_tol:
            break
        L_new = min(1.5 * L, L_max)
        grid = np.concatenate((np.linspace(-L_new, -L, 12)[:-1], sol.x))
        Y0 = np.hstack((np.repeat(Ym[:, None], 11, axis=1), sol.y))
        L = L_new
        sol = run(grid, Y0, sol.p[0], tol)
        extensions += 1
    if not sol.success:
        raise WaveError(f"the boundary value problem did not converge: {sol.message}")
    fine = run(sol.x, sol.y, sol.p[0], check_tol)
    if not fine.success:
        raise WaveError(f"the tighter solve failed: {fine.message}")
    c = float(fine.p[0])
    probe = np.linspace(-L, 0.0, 2001)
    Yf = fine.sol(probe)
    profile_error = float(np.abs(Yf - sol.sol(probe)).max())
    tail = float(np.abs(Yf[:, 0] - Ym).max())
    if tail > 1e3 * tail_tol * scale:
        raise WaveError(f"the profile has not reached its far state at the end of the domain "
                        f"(distance {tail:.2g}, L = {L:.4g}); a longer domain is needed (options['L_max'])")
    departure = float(np.abs(Yf[system.offset] - Ym[system.offset][:, None]).max())
    if departure < 1e-6 * scale:
        raise WaveError("the solution collapsed to the constant state")
    Yb = fine.sol(0.0)
    info = dict(speed_is_unknown=True, unstable_dim_far=int(ku), nodes=int(fine.x.size),
                domain_extensions=extensions, residual_rms_max=float(fine.rms_residuals.max()),
                speed_first_solve=float(sol.p[0]), speed_error=abs(c - float(sol.p[0])),
                profile_error=profile_error, tail_error=tail, amplitude=departure, slowest_rate=slow,
                stefan_residual=abs(c - system.stefan(Yb)[0]),
                boundary_residual=float(np.abs(system.boundary_residual(Yb)).max()))
    return Wave(c=c, L=float(L), N=N, Y_minus=Ym, Y_plus=Yb, interpolant=fine.sol, mesh=fine.x,
                info=info, theta=system.theta, right=0.0)


class BoundaryAsymptotics:
    """Far-field data and boundary rows of the extended system (state P and eta)."""

    def __init__(self, system, far, wave, lam_ref):
        self.M0m, self.M1m, _ = system.end_state(far, wave.c)
        N0 = self.N0 = system.N
        ref = complex(lam_ref).real if complex(lam_ref).imag == 0 else complex(lam_ref)
        k0, Pu, *_ = split(self.M0m + ref * self.M1m)
        if k0 != system.n_boundary:
            raise SplittingError("the number of boundary conditions does not match the splitting")
        self.k0, self.N, self.k = k0, N0 + 1, k0 + 1
        self.lam_ref = complex(lam_ref)
        self.CL = reference_rows(np.eye(N0) - Pu, N0 - k0)
        # Rows at z = 0:  linearised conditions J P = 0  and  eta - grad S . P = 0.
        self.BR = np.zeros((k0 + 1, N0 + 1))
        self.BR[:k0, :N0] = system.boundary_jacobian(wave.Y_plus)
        self.BR[k0, :N0] = -system.stefan(wave.Y_plus)[1]
        self.BR[k0, N0] = 1.0
        self.cache = {}

    def at(self, lam):
        lam = complex(lam)
        hit = self.cache.get(lam)
        if hit is not None:
            return hit
        N0 = self.N0
        k, Pu, eu, es = split(self.M0m + lam * self.M1m)
        if k != self.k0:
            raise SplittingError(f"lambda = {lam:.6g} is outside the region of consistent splitting")
        rows = self.CL @ (np.eye(N0) - Pu)
        BL = np.hstack((rows, np.zeros((N0 - k, 1))))
        gamma = complex(np.linalg.det(rows @ self.CL.conj().T)) if rows.shape[0] else 1.0 + 0j
        value = (BL, self.BR, np.concatenate((eu, es, [0.0])), np.zeros(0, complex), gamma)
        self.cache[lam] = value
        return value


def build_engine(system, wave, asym, La, h, cap=1.0, variation=None):
    """Matching determinant on [-La, 0] for the extended first-order system."""
    if cap > 1.0 and variation is not None:
        nodes = graded_nodes(variation, La, 0.0, h, cap)
    else:
        nodes = np.linspace(-La, 0.0, max(4, int(np.ceil(La / h))) + 1)
    points = np.empty(2 * nodes.size - 1)
    points[0::2] = nodes
    points[1::2] = (nodes[1:] + nodes[:-1]) / 2
    Y = wave.Y(points)
    M0, M1 = system.first_order(Y, wave.c)
    N0, m = system.N, points.size
    M0e, M1e = np.zeros((m, N0 + 1, N0 + 1)), np.zeros((m, N0 + 1, N0 + 1))
    M0e[:, :N0, :N0], M1e[:, :N0, :N0] = M0, M1
    H = system._solve_top(system._points(Y), wave.c)[0]
    dU = system.rhs(Y, wave.c)[system.offset].T * system.mass         # U'(z), shape (m, n)
    M0e[:, system.top_rows, N0] = -np.linalg.solve(H, dU[..., None])[..., 0]   # forcing eta U'
    return Evans(asym, nodes, M0e, M1e)


def fd_operator_boundary(system, wave, La, nz):
    """Finite differences of  lambda B p = L p + U' (grad S . P(0))  on [-La, 0].  Returns (A, B).

    p = 0 at z = -La.  The node z = 0 is an unknown.  There, the rows of the components
    that carry a boundary condition hold the linearised conditions (entry 0 in B).  The
    other components (no diffusion, characteristics leaving the domain) keep their
    equation, with one-sided differences.
    """
    if system.K != 2:
        raise StructureError("the difference operator with a free boundary needs a system of second order")
    n, m, off = system.n, nz + 1, system.offset
    z = np.linspace(-La, 0.0, nz + 2)[1:]
    h = z[1] - z[0]
    Y = wave.Y(z)
    a = system.coefficients(Y, wave.c)                                # (3, m, n, n)
    D = [d.tolil() for d in _difference_matrices(m, h, 2)]
    one_sided = np.array([1.0, -4.0, 3.0]) / (2 * h)                  # p'(0), second order
    D[1][m - 1, m - 3:] = one_sided
    D[2][m - 1, m - 4:] = np.array([-1.0, 4.0, -5.0, 2.0]) / h ** 2
    D = [d.tocsr() for d in D]
    blocks = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            block = sparse.csr_matrix((m, m))
            for k in range(3):
                if np.any(a[k, :, i, j]):
                    block = block + sparse.diags(a[k, :, i, j]) @ D[k]
            blocks[i][j] = block
    A = sparse.bmat(blocks, format="csr")

    # Components that carry a condition: all with diffusion, then those without diffusion
    # whose characteristics enter the domain most strongly (largest coefficient of p_j').
    second = [j for j in range(n) if system.mu[j] == 2]
    first = sorted((j for j in range(n) if system.mu[j] == 1), key=lambda j: -a[1, -1, j, j])
    extra = system.n_boundary - len(second)
    if not 0 <= extra <= len(first):
        raise StructureError("the number of boundary conditions does not fit the orders of the equations")
    carriers = second + first[:extra]

    def functional(gradient):
        """Row vector on the grid of  gradient . P(0)."""
        out = np.zeros(n * m)
        for j in range(n):
            last = j * m + m - 1
            out[last] += gradient[off[j]]
            if system.mu[j] == 2:
                out[last - 2:last + 1] += gradient[off[j] + 1] * one_sided
        return sparse.csr_matrix(out)

    keep = np.ones(n * m)
    keep[[j * m + m - 1 for j in carriers]] = 0.0
    forcing = (system.rhs(Y, wave.c)[off] * system.mass[:, None]).reshape(-1)        # mass U'(z)
    eta = functional(system.stefan(wave.Y_plus)[1])
    conditions = sparse.lil_matrix(A.shape)
    for row, j in zip(system.boundary_jacobian(wave.Y_plus), carriers):
        conditions[j * m + m - 1] = functional(row)
    A = sparse.diags(keep) @ (A + sparse.csr_matrix(forcing[:, None]) @ eta) + conditions.tocsr()
    B = sparse.diags(keep * np.repeat(system.mass, m), format="csc")
    return A.tocsc(), B


def boundary_bounds(system, wave, bounds, samples=4001):
    """Extra suprema of the energy estimate with a free boundary (DESIGN 2.6)."""
    def no(reason):
        return dict(bounds, applies=False, reason=bounds["reason"] or reason)

    if not bounds["applies"]:
        return no(None)
    if "general" in bounds:
        return no("the free-boundary bound needs equations of second order")
    if np.any(system.mass == 0):
        return no("the free-boundary bound does not cover constraint equations")
    n, off = system.n, system.offset
    Hs = list(system.first_order_components)
    Ps = [j for j in range(n) if j not in Hs]
    if Hs and bounds["gamma1"] > 0:
        return no("with a free boundary, the equations of the components with diffusion may contain "
                  "the diffusion-free components without derivatives only")

    # Type of every component at the boundary, from the linearised conditions J P(0) = 0:
    # D (p_j(0) = 0) or R (p_j'(0) = -r_j p_j(0)).
    kind, r = {}, np.zeros(n)
    for row in system.boundary_jacobian(wave.Y_plus):
        value = row[off]
        slope = np.array([row[off[j] + 1] if j in Ps else 0.0 for j in range(n)])
        owners = np.nonzero((value != 0) | (slope != 0))[0]
        if owners.size != 1 or int(owners[0]) in kind:
            return no("the free-boundary bound needs boundary conditions that contain one component each")
        j = int(owners[0])
        kind[j] = "D" if slope[j] == 0 else "R"
        r[j] = 0.0 if slope[j] == 0 else value[j] / slope[j]
    if any(j not in kind for j in Ps):
        return no("a component with diffusion has no boundary condition")
    Ds = [j for j in Ps if kind[j] == "D"]
    Rs = [j for j in Ps if kind[j] == "R"]
    Fs = [j for j in Hs if j not in kind]              # no diffusion and no condition: must leave the domain

    z = np.linspace(-wave.L, 0.0, samples)
    Y = wave.Y(z)
    a = system.coefficients(Y, wave.c)
    dU = system.rhs(Y, wave.c)[off]
    gradient = system.stefan(wave.Y_plus)[1]
    g0 = gradient[off]
    g1 = np.array([gradient[off[j] + 1] if j in Ps else 0.0 for j in range(n)])
    end = lambda k, rows, cols: a[k][-1][np.ix_(rows, cols)]         # coefficient block at z = 0
    sym = lambda M: (M + M.T) / 2
    if Fs:
        if np.linalg.eigvalsh(sym(end(1, Fs, Fs))).max() > 0:
            return no("a diffusion-free component enters the domain at the boundary without a boundary value")
        if np.abs(g0[Fs]).max() > 0:
            return no("the Stefan condition contains a diffusion-free component that has no boundary value")
    if Ds and Rs and np.abs(end(2, Rs, Ds)).max() > 0:
        return no("cross-diffusion between components with different types of boundary condition")

    norm_l2 = lambda rows: float(np.sqrt(np.trapezoid((dU[rows] ** 2).sum(axis=0), z))) if rows else 0.0
    rho_b = sigma_b = 0.0
    if Rs:
        flux = end(2, Rs, Rs) @ np.diag(r[Rs])                       # p_R* A2 p_R'(0) = -p_R* flux p_R
        rho_b = float(np.linalg.norm(flux, 2))
        sigma_b = max(0.0, float(np.linalg.eigvalsh(sym(end(1, Rs, Rs)) / 2 - sym(flux)).max()))
    A2 = a[2][:, Ps][:, :, Ps]
    return dict(bounds, m2=float(np.linalg.norm(np.linalg.inv(A2), 2, axis=(1, 2)).max()),
                psi2=norm_l2(Ps), psi_h=norm_l2(Hs),
                kappa=float(np.linalg.norm(g1[Ds])), kappa_r=float(np.linalg.norm((g0 - g1 * r)[Rs])),
                rho_b=rho_b, sigma_b=sigma_b, boundary_types=[kind.get(j, "none") for j in range(n)])


def boundary_estimate(bounds, delta):
    """The inequalities of DESIGN 2.6 for an eigenfunction with Re(lambda) >= -delta.

    p1: components with diffusion, ||p1|| = 1, t = ||p1'||, e = |eta| <= e_D + e_R.
        Re lambda   <= -d t^2 + q t + top + R1 e,
        |Im lambda| <=  a t^2 + lin t + const + R1 e,
        e_R <= kappa_r sqrt(2 t),
        e_D^2 <= K t (|lambda| + b t + lam0 + coupling + R1 e),   K = 2 kappa^2 m2.
    Robin conditions enter q (2 sigma_b) and lin (2 rho_b).  Diffusion-free components p2
    enter through ||p2|| <= (alpha1 t + alpha0 + psi_h e) / g,  g = -delta - beta.
    Returns functions re(t, e), im(t, e), e_max(t), rho(t, e); None if g <= 0.
    """
    d, a2 = bounds["d"], bounds["a"]
    q, top = bounds["bK"] + bounds["d1"] + 2 * bounds["sigma_b"], bounds["c0"] + bounds["s1"] / 2
    lin, const = bounds["b"] + bounds["d1"] + 2 * bounds["rho_b"], bounds["c2"]
    R1, slope, level, rho = bounds["psi2"], 0.0, 0.0, lambda t, e: 0.0
    if "beta" in bounds:
        g = -delta - bounds["beta"]
        if g <= 0:
            return None
        rho = lambda t, e: (bounds["alpha1"] * t + bounds["alpha0"] + bounds["psi_h"] * e) / g
        R1 += bounds["gamma0"] * bounds["psi_h"] / g
        slope, level = bounds["gamma0"] * bounds["alpha1"] / g, bounds["gamma0"] * bounds["alpha0"] / g
    q, top, lin, const = q + slope, top + level, lin + slope, const + level
    K = 2 * bounds["kappa"] ** 2 * bounds["m2"]

    def e_max(t):
        e_r = bounds["kappa_r"] * np.sqrt(2 * t)
        alpha = (delta + q * t + abs(top) + a2 * t * t + lin * t + const
                 + bounds["b"] * t + bounds["lam0"] + slope * t + level)
        beta = 3 * R1
        return e_r + (K * t * beta + np.sqrt((K * t * beta) ** 2 + 4 * K * t * (alpha + beta * e_r))) / 2

    return dict(re=lambda t, e: -d * t * t + q * t + top + R1 * e,
                im=lambda t, e: a2 * t * t + lin * t + const + R1 * e, e_max=e_max, rho=rho)


def boundary_region(bounds, delta, safety=1.1):
    """Rectangle that contains all point spectrum with Re(lambda) >= -delta; None without a bound."""
    estimate = boundary_estimate(bounds, delta)
    if estimate is None:
        return None
    re_max = lambda t: estimate["re"](t, estimate["e_max"](t))
    grid = np.concatenate(([0.0], np.geomspace(1e-6, 1e8, 2801)))
    admissible = np.array([re_max(t) + delta >= 0 for t in grid])
    if not admissible.any():
        t_max = 0.0
    else:
        last = int(np.nonzero(admissible)[0].max())
        if last == grid.size - 1:
            return None
        t_max = brentq(lambda t: re_max(t) + delta, grid[last], grid[last + 1])
    ts = np.concatenate((grid[grid <= t_max], [t_max]))
    x_max = max(re_max(t) for t in ts)
    y_max = estimate["im"](t_max, estimate["e_max"](t_max))
    floor = 4 * delta
    return float(safety * max(x_max, floor)), float(safety * max(y_max, floor))
