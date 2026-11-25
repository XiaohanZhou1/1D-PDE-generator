"""Independent discretisation of L and the bound on the search region.

* fd_operator(), fd_probe(): centred finite differences of the operator in
  expanded (non-conservative) form.  They share no code with the Evans discretisation.
* coefficient_bounds(), search_region(): the energy estimate of DESIGN.md 2.4.
"""
from __future__ import annotations

from math import comb

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigs


def _difference_matrices(nz, h, order):
    """Centred differences of second-order accuracy for the derivatives 0, ..., order (zero values outside)."""
    D = []
    for k in range(order + 1):
        stencil = np.array([1.0])
        for _ in range(k // 2):
            stencil = np.convolve(stencil, [1.0, -2.0, 1.0])
        if k % 2:
            stencil = np.convolve(stencil, [-0.5, 0.0, 0.5])
        reach = (stencil.size - 1) // 2                                # weights at the offsets -reach, ..., reach
        D.append(sum(w * sparse.eye(nz, k=d, format="csc")
                     for d, w in zip(range(-reach, reach + 1), stencil) if w) / h ** k)
    return D


def fd_operator(system, wave, La, Lb, nz):
    """Sparse matrix of L on nz interior points of [-La, Lb], zero values outside."""
    z = np.linspace(-La, Lb, nz + 2)[1:-1]
    h = z[1] - z[0]
    a = system.coefficients(wave.Y(z), wave.c)          # (K+1, nz, n, n)
    D = _difference_matrices(nz, h, system.K)
    n = system.n
    blocks = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            block = sparse.csc_matrix((nz, nz))
            for k in range(system.K + 1):
                coefficient = a[k, :, i, j]
                if np.any(coefficient):
                    block = block + sparse.diags(coefficient) @ D[k]
            blocks[i][j] = block
    return sparse.bmat(blocks, format="csc")


def fd_companion(system, wave, La, Lb, nz):
    """Second order in time: (A, B) with  A (p, q) = lambda B (p, q),  q = lambda p.

    L p = lambda (N1 p - 2 c N2 p') + lambda^2 N2 p  becomes an ordinary (generalised) eigenvalue
    problem of twice the size.  The blocks use the centred differences of fd_operator().
    """
    L = fd_operator(system, wave, La, Lb, nz)
    z = np.linspace(-La, Lb, nz + 2)[1:-1]
    n = system.n
    N1 = system.damping(wave.Y(z), wave.c)
    D1 = _difference_matrices(nz, z[1] - z[0], 1)[1]
    empty = sparse.csc_matrix((nz, nz))
    C = sparse.bmat([[sparse.diags(N1[:, i, j]) - (2 * wave.c * system.inertia[i] * D1 if i == j else empty)
                      for j in range(n)] for i in range(n)], format="csc")
    eye = sparse.identity(n * nz, format="csc")
    A = sparse.bmat([[None, eye], [L, -C]], format="csc")
    B = sparse.block_diag((eye, sparse.diags(np.repeat(system.inertia, nz))), format="csc")
    return A, B


def fd_probe(A, shifts, k=8, restarts=60, mass=None):
    """Eigenvalues of A nearest to each shift (shift-and-invert Arnoldi).

    This is a probe near the given shifts, not the complete spectrum.  No dense
    eigenvalue solver of large order is used: in the `premesh` environment the
    dense nonsymmetric LAPACK solver fails above order of about 400.

    The number of restarts is limited.  A shift far from every isolated eigenvalue
    sees only the discretised essential spectrum, at nearly equal distances; the
    iteration then does not converge and would run for a long time.  The
    eigenvalues that did converge are kept.

    mass: None, or the diagonal matrix B of the generalised problem  A p = lambda B p
    (constraint components have the entry 0).
    """
    rng = np.random.default_rng(20261123)
    v0 = rng.normal(size=A.shape[0])
    k = min(k, A.shape[0] - 2)
    found = []
    for sigma in shifts:
        matrix = A.astype(complex) if np.iscomplex(sigma) else A
        try:
            values = eigs(matrix, k=k, M=mass, sigma=sigma, v0=v0.astype(matrix.dtype), tol=1e-9,
                          return_eigenvectors=False, maxiter=restarts)
        except ArpackNoConvergence as error:
            values = error.eigenvalues
        except (ArpackError, RuntimeError):
            continue
        found.extend(np.atleast_1d(values))
    return np.array(found, dtype=complex)


def coefficient_bounds(system, wave, samples=4001, scaling=None):
    """Suprema along the wave that enter the energy estimate.  None if it does not apply.

    scaling: positive numbers d_j.  The estimate is then made for the variables d_j p_j,
    that is for the coefficients D A_k D^{-1}.  The eigenvalues do not change.
    """
    if getattr(system, "quadratic", False):
        return _quadratic_bounds(system, wave, samples)
    z = np.linspace(-wave.L, wave.hi, samples)
    a = system.coefficients(wave.Y(z), wave.c)
    if scaling is not None:
        scaling = np.asarray(scaling, dtype=float)
        a = a * (scaling[:, None] / scaling[None, :])
    norm = lambda A: float(np.linalg.norm(A, 2, axis=(1, 2)).max())
    sym = lambda A: (A + A.transpose(0, 2, 1)) / 2
    skew = lambda A: (A - A.transpose(0, 2, 1)) / 2
    out = dict(lam0=norm(a[0]), applies=False, reason=None)
    Hs = list(system.first_order_components)
    Cs = [j for j in range(system.n) if system.mass[j] == 0]         # constraint components
    Ps = [j for j in range(system.n) if j not in Hs and j not in Cs]
    if system.K > 2:
        return _higher_order_bounds(system, a, z, out)
    if system.K != 2 or not Ps:
        out["reason"] = "the energy bound needs second-order components"
        return out
    if Cs and Hs:
        out["reason"] = "the energy bound does not combine constraint and first-order components"
        return out
    block = lambda A, rows, cols: A[:, rows][:, :, cols]
    A0, A1, A2 = block(a[0], Ps, Ps), block(a[1], Ps, Ps), block(a[2], Ps, Ps)
    d = float(np.linalg.eigvalsh(sym(A2)).min())
    if d <= 0:
        out.update(reason="the symmetric part of the second-order coefficient matrix is not positive definite",
                   indefinite=True, d=d)
        return out
    data = dict(d=d, a=norm(skew(A2)), d1=norm(np.gradient(A2, z, axis=0)),
                bK=norm(skew(A1)), b=norm(A1), s1=norm(np.gradient(sym(A1), z, axis=0)),
                c0=float(np.linalg.eigvalsh(sym(A0)).max()), c2=norm(skew(A0)))
    if Hs:
        # Partly parabolic system (DESIGN 2.4): set H = components without a second derivative.
        if np.abs(a[2][:, Hs, :]).max() > 0:
            out["reason"] = "a second derivative appears in the equation of a first-order component"
            return out
        if np.abs(block(a[2], Ps, Hs)).max() > 0:
            out["reason"] = ("the second derivative of a component without diffusion appears in another "
                             "equation (taxis): the energy estimate does not cover this term")
            return out
        S = block(a[1], Hs, Hs)
        if np.abs(skew(S)).max() > 1e-12 * max(1.0, np.abs(S).max()):
            out["reason"] = "the first-order coefficient matrix of the first-order components is not symmetric"
            return out
        cross = block(a[1], Ps, Hs)
        data.update(beta=float(np.linalg.eigvalsh(sym(block(a[0], Hs, Hs))
                                                  - np.gradient(S, z, axis=0) / 2).max()),
                    alpha0=norm(block(a[0], Hs, Ps)), alpha1=norm(block(a[1], Hs, Ps)),
                    gamma0=norm(block(a[0], Ps, Hs)), gamma1=norm(cross),
                    gamma1p=norm(np.gradient(cross, z, axis=0)))
    if Cs:
        # Constraint 0 = L_CC q + A0_CD p  (DESIGN 2.7): elliptic estimate for q, then a
        # bounded term in the estimate for p.
        if max(np.abs(a[2][:, Cs][:, :, Ps]).max(), np.abs(a[1][:, Cs][:, :, Ps]).max(),
               np.abs(a[2][:, Ps][:, :, Cs]).max()) > 0:
            out["reason"] = ("the constraint may contain the other unknowns without derivatives only, "
                             "and the other equations no second derivative of the constraint unknown")
            return out
        C2, C1, C0 = block(a[2], Cs, Cs), block(a[1], Cs, Cs), block(a[0], Cs, Cs)
        if np.linalg.eigvalsh(sym(C2)).max() < 0:                   # sign convention of the constraint
            C2, C1, C0 = -C2, -C1, -C0
        dc = float(np.linalg.eigvalsh(sym(C2)).min())
        r = norm(np.gradient(C2, z, axis=0)) + norm(skew(C1))
        omega = -float(np.linalg.eigvalsh(sym(C0)).max()) - norm(np.gradient(sym(C1), z, axis=0)) / 2
        if dc <= 0 or omega - r * r / (2 * dc) <= 0:
            out["reason"] = "the constraint operator is not coercive"
            return out
        omega -= r * r / (2 * dc)
        alpha = norm(block(a[0], Cs, Ps))
        K0, K1 = alpha / omega, alpha * np.sqrt(2 / (dc * omega))   # ||q|| <= K0 ||p||, ||q'|| <= K1 ||p||
        coupling = norm(block(a[1], Ps, Cs)) * K1 + norm(block(a[0], Ps, Cs)) * K0
        data.update(c0=data["c0"] + coupling, c2=data["c2"] + coupling, constraint_coupling=coupling)
    out.update(applies=True, **data)
    return out


def _quadratic_bounds(system, wave, samples):
    """Suprema for the energy estimate with second-order time derivatives (DESIGN 2.12).

    In the variables of the resting frame,  Lambda^2 p + gamma Lambda p = D p'' + b p' + a p  with
    Lambda = lambda - c d/dz:  D = a2 + c^2,  b = a1 - c gamma,  a = a0,  gamma = N1.
    """
    z = np.linspace(-wave.L, wave.hi, samples)
    Y, c, n = wave.Y(z), wave.c, system.n
    a = system.coefficients(Y, c)
    norm = lambda A: float(np.linalg.norm(A, 2, axis=(1, 2)).max())
    out = dict(lam0=norm(a[0]), applies=False, reason=None, quadratic=True)
    if system.K != 2 or np.any(system.inertia == 0) or any(k != 2 for k in system.orders):
        out["reason"] = "the bound for second-order time derivatives needs u_tt and u_xx in every component"
        return out
    gamma = system.damping(Y, c)
    D = a[2] + c * c * np.eye(n)[None]
    scale = max(norm(D), 1e-300)
    if np.abs(D - D[:1]).max() > 1e-9 * scale or np.abs(D[0] - D[0].T).max() > 1e-9 * scale:
        out["reason"] = ("the bound for second-order time derivatives needs a constant symmetric "
                         "coefficient matrix of u_xx")
        return out
    d = np.linalg.eigvalsh((D[0] + D[0].T) / 2)
    if d.min() <= 0:
        out["reason"] = "the coefficient matrix of u_xx is not positive definite"
        return out
    out.update(applies=True, d_min=float(d.min()), d_max=float(d.max()), speed=abs(float(c)),
               gamma_min=float(np.linalg.eigvalsh((gamma + gamma.transpose(0, 2, 1)) / 2).min()),
               gamma_sup=norm(gamma), b_sup=norm(a[1] - c * gamma), a_sup=norm(a[0]))
    return out


def _quadratic_region(bounds, delta, safety):
    """Rectangle for second-order time derivatives; None if the estimate gives no bound.

    ||p|| = 1, w = lambda p - c p', X = ||w||, t = ||p'||, x = Re(lambda).  Two identities:
        x (X^2 + <D p', p'>) + Re <gamma w, w> = Re <b p' + a p, w>,
        X^2 - <D p', p'> = 2 x Re <w, p> + Re <gamma w, p> - Re <b p' + a p, p>.
    The second one bounds t by X; the first one then bounds X.  |lambda| <= X + |c| t.
    """
    dmin, dmax, g, G = bounds["d_min"], bounds["d_max"], bounds["gamma_min"], bounds["gamma_sup"]
    b, a, c = bounds["b_sup"], bounds["a_sup"], bounds["speed"]
    scale = max(a, G, 1e-300)
    X = scale * np.geomspace(1e-8, 1e8, 3201)

    def reach(x):
        """Largest |lambda| with Re(lambda) = x: None if there is none, inf if the estimate does not close."""
        t_up = (b + np.sqrt(b * b + 4 * dmin * (X * X + (2 * abs(x) + G) * X + a))) / (2 * dmin)
        if x < 0:
            feasible = (g + x) * X * X + x * dmax * t_up ** 2 <= (b * t_up + a) * X
        else:
            t = np.minimum(t_up, b * X / (2 * x * dmin)) if x > 0 else t_up
            feasible = x * dmin * t * t - b * X * t + (x + g) * X * X - a * X <= 0
        if not feasible.any():
            return None
        return np.inf if feasible[-1] else float((X + c * t_up)[feasible].max())

    floor = 4 * delta
    edge = reach(-delta)
    if edge is None:                                   # no eigenvalue with Re(lambda) >= -delta
        return float(safety * floor), float(safety * floor)
    if not np.isfinite(edge):
        return None
    x_max = 0.0
    for x in scale * np.geomspace(1e-8, 1e8, 1601):
        r = reach(x)
        if r is None or r < x:
            break
        x_max = x
    else:
        return None
    return float(safety * max(x_max, floor)), float(safety * max(edge, floor))


def _higher_order_bounds(system, a, z, out):
    """Energy estimate for components of even order 2 m_j (DESIGN 2.4, orders above 2).

    ||p|| = 1, tau_j = ||p_j^(m_j)||, T_g^2 = sum of tau_j^2 over the components of half-order g.
    <lambda p, p> is expanded term by term; derivatives are moved by parts until every factor
    carries at most m derivatives, and ||p^(s)|| <= tau^(s/m) (interpolation).  The result is
        Re lambda   <= c0 + sum_g ( -d_g T_g^2 + sum of w T_g^gamma ),     gamma < 2,
        |Im lambda| <= c2 + sum_g ( skew_g T_g^2 + sum of w T_g^gamma ).
    Terms that couple two groups are split by Young's inequality.
    """
    orders, n = list(system.orders), system.n
    if np.any(system.mass == 0) or any(k < 2 or k % 2 for k in orders):
        out["reason"] = ("the energy bound for orders above 2 needs components of even order "
                         "(no constraint and no component without diffusion)")
        return out
    half = [k // 2 for k in orders]
    groups = sorted(set(half))
    tiny = 1e-10 * max(float(np.abs(a).max()), 1e-300)

    def sup(values, s=0):
        """sup |d^s values / dz^s|; coefficients below the rounding level count as zero."""
        for _ in range(s):
            values = np.gradient(values, z, edge_order=2)
        value = float(np.abs(values).max())
        return value if value > tiny else 0.0

    info = {}
    for m in groups:
        g = [j for j in range(n) if half[j] == m]
        P = (-1) ** (m + 1) * a[2 * m][:, g][:, :, g]
        d = float(np.linalg.eigvalsh((P + P.transpose(0, 2, 1)) / 2).min())
        if d <= 0:
            out.update(reason=f"the leading coefficient matrix of order {2 * m} is not positive definite",
                       indefinite=True, d=d)
            return out
        info[m] = dict(order=2 * m, components=g, d=d, re={}, im={},
                       skew=float(np.linalg.norm((P - P.transpose(0, 2, 1)) / 2, 2, axis=(1, 2)).max()))

    def add(m, power, re, im):
        for key, weight in (("re", re), ("im", im)):
            if weight > 0:
                info[m][key][power] = info[m][key].get(power, 0.0) + weight

    for i in range(n):
        for j in range(n):
            mi, mj = half[i], half[j]
            for k in range(1, min(a.shape[0], orders[j] + 1)):
                c = a[k][:, i, j]
                if sup(c) == 0.0:
                    continue
                if i == j:
                    q = k // 2
                    if k % 2 == 0:
                        # (-1)^q <c p^(q), p^(q)> is real; for k = 2 m it is the leading term (in d).
                        if k < 2 * mj:
                            add(mj, 2 * q / mj, max(0.0, float(((-1) ** q * c).max())), 0.0)
                        lower, top = range(q), q
                    else:
                        # Re <c p^(q+1), p^(q)> = -<c' p^(q), p^(q)> / 2.
                        add(mj, 2 * q / mj, sup(c, 1) / 2, 0.0)
                        add(mj, (2 * q + 1) / mj, 0.0, sup(c))
                        lower, top = range(q), q + 1
                    for r in lower:                    # derivatives that fall on the coefficient
                        weight = comb(q, r) * sup(c, q - r)
                        add(mj, (top + r) / mj, weight, weight)
                    continue
                moved = max(0, k - mj)                 # derivatives moved from p_j to the other factor
                if moved > mi:
                    out["reason"] = (f"the equation of component {i} contains a derivative of component {j} "
                                     f"of order {k}, above the sum {mi + mj} of their half-orders")
                    return out
                for r in range(moved + 1):
                    weight = comb(moved, r) * sup(c, moved - r)
                    alpha, beta = (k - moved) / mj, r / mi         # powers of tau_j and of tau_i
                    if weight == 0.0 or (mi == mj and k == 2 * mj and r == moved):
                        continue                       # nothing, or a leading term (in d and skew)
                    if alpha + beta >= 2 - 1e-12:
                        out["reason"] = (f"components {i} and {j} have different orders and are coupled "
                                         "at the critical order")
                        return out
                    if mi == mj or alpha == 0 or beta == 0:
                        add(mj if beta == 0 else mi if alpha == 0 else mi, alpha + beta, weight, weight)
                    else:                              # Young: x^alpha y^beta <= (alpha x^s + beta y^s) / s
                        add(mj, alpha + beta, weight * alpha / (alpha + beta), weight * alpha / (alpha + beta))
                        add(mi, alpha + beta, weight * beta / (alpha + beta), weight * beta / (alpha + beta))
    sym = lambda A: (A + A.transpose(0, 2, 1)) / 2
    out.update(applies=True, c0=float(np.linalg.eigvalsh(sym(a[0])).max()),
               c2=float(np.linalg.norm(a[0] - sym(a[0]), 2, axis=(1, 2)).max()),
               general=[dict(info[m], re=sorted(info[m]["re"].items()), im=sorted(info[m]["im"].items()))
                        for m in groups])
    return out


def _general_region(bounds, delta, safety):
    """Rectangle from the bounds of _higher_order_bounds; None if a group has no bound."""
    grid = np.concatenate(([0.0], np.geomspace(1e-8, 1e12, 4001)))
    value = lambda terms, leading, T: leading * T * T + sum(w * T ** p for p, w in terms)
    curves = [value(g["re"], -g["d"], grid) for g in bounds["general"]]
    peaks = [float(curve.max()) for curve in curves]
    x_max, y_max = bounds["c0"] + sum(peaks), bounds["c2"]
    for g, curve, peak in zip(bounds["general"], curves, peaks):
        admissible = np.nonzero(curve >= -delta - bounds["c0"] - (sum(peaks) - peak))[0]
        if admissible.size and admissible[-1] == grid.size - 1:
            return None
        T = float(grid[admissible[-1] + 1]) if admissible.size else 0.0
        y_max += float(value(g["im"], g["skew"], T))
    floor = 4 * delta
    return float(safety * max(x_max, floor)), float(safety * max(y_max, floor))


def scaled_bounds(system, wave, delta, samples=1001):
    """Bounds after a diagonal change of variables that makes sym(D A2 D^{-1}) positive definite.

    Cross-diffusion makes A2 non-symmetric.  Its symmetric part can be indefinite
    although the system is parabolic.  The search minimises the area of the
    search rectangle over the scalings; None if no admissible scaling is found.
    """
    if system.n < 2 or system.first_order_components:
        return None

    def scaling(log_d):
        return np.exp(np.clip(np.concatenate(([0.0], log_d)), -14.0, 14.0))

    def cost(log_d):
        b = coefficient_bounds(system, wave, samples, scaling(log_d))
        if not b["applies"]:
            return 1e3 - b.get("d", -1e3)             # slope towards a positive definite matrix
        region = search_region(b, delta)
        return np.log(region[0]) + np.log(region[1]) if region else 1e3

    start = np.zeros(system.n - 1)
    simplex = np.vstack((start, start + 3.0 * np.eye(system.n - 1)))
    best = minimize(cost, start, method="Nelder-Mead",
                    options=dict(initial_simplex=simplex, xatol=0.05, fatol=0.01, maxiter=60 * system.n))
    out = coefficient_bounds(system, wave, scaling=scaling(best.x))
    if not out["applies"]:
        return None
    out["scaling"] = scaling(best.x)
    return out


def search_region(bounds, delta, safety=1.1):
    """Rectangle that contains all point spectrum with Re(lambda) >= -delta.

    Returns None when the estimate gives no bound for this delta (partly parabolic
    systems need -delta > beta and a positive effective diffusion).
    """
    if "general" in bounds:
        return _general_region(bounds, delta, safety)
    if bounds.get("quadratic"):
        return _quadratic_region(bounds, delta, safety)
    q = bounds["bK"] + bounds["d1"]
    top = bounds["c0"] + bounds["s1"] / 2
    d, a2 = bounds["d"], bounds["a"]
    lin, const = bounds["b"] + bounds["d1"], bounds["c2"]
    if "beta" in bounds:
        g = -delta - bounds["beta"]
        if g <= 0:
            return None
        e = bounds["gamma1p"] + bounds["gamma0"]
        t2 = bounds["gamma1"] * bounds["alpha1"] / g
        t1 = (bounds["gamma1"] * bounds["alpha0"] + e * bounds["alpha1"]) / g
        t0 = e * bounds["alpha0"] / g
        d -= t2
        if d <= 0:
            return None
        q, top, a2, lin, const = q + t1, top + t0, a2 + t2, lin + t1, const + t0
    x_max = top + q * q / (4 * d)
    t_max = (q + np.sqrt(max(q * q + 4 * d * (top + delta), 0.0))) / (2 * d)
    y_max = a2 * t_max ** 2 + lin * t_max + const
    floor = 4 * delta
    return float(safety * max(x_max, floor)), float(safety * max(y_max, floor))
