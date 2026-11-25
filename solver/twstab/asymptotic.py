"""Constant-coefficient analysis at the end states (DESIGN.md, sections 2.4 and 3.2).

* split():  spectral projector onto the unstable subspace of a matrix.
  The projector does not depend on a choice of basis and is analytic in any
  parameter as long as no eigenvalue crosses the imaginary axis.
* essential_bound():  supremum of Re(lambda) on the dispersion curves.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eig, schur, solve_sylvester
from scipy.optimize import minimize_scalar


def split(M):
    """Return (k, Pu, ev_u, ev_s) for the splitting of M by the sign of Re(eigenvalue).

    k   : dimension of the unstable subspace (Re > 0)
    Pu  : projector onto the unstable subspace along the stable one
    ev_u, ev_s : the unstable and the stable eigenvalues
    """
    M = np.asarray(M)
    N = M.shape[0]
    if np.iscomplexobj(M):
        T, Z, k = schur(M, output="complex", sort="rhp")
        ev = np.diag(T)
        ev_u, ev_s = ev[:k], ev[k:]
    else:
        T, Z, k = schur(M, output="real", sort="rhp")
        ev_u = np.linalg.eigvals(T[:k, :k]) if k else np.zeros(0, complex)
        ev_s = np.linalg.eigvals(T[k:, k:]) if k < N else np.zeros(0, complex)
    if 0 < k < N:
        X = solve_sylvester(T[:k, :k], -T[k:, k:], -T[:k, k:])
        Pu = Z[:, :k] @ (Z[:, :k].conj().T - X @ Z[:, k:].conj().T)
    elif k == N:
        Pu = np.eye(N, dtype=T.dtype)
    else:
        Pu = np.zeros((N, N), dtype=T.dtype)
    return int(k), Pu, ev_u, ev_s


class SlowModeError(ValueError):
    """The slow eigenvalues of an end state cannot be assigned to the two subspaces."""


def split_touching(M, M1, tol=1e-7):
    """Splitting of M = M(lambda) that stays continuous on the closed half-plane Re(lambda) >= 0.

    For the integrated operator of a conservation law the end-state matrix has slow eigenvalues
    mu_j = lambda / alpha_j + ...  (alpha_j: minus the characteristic speeds relative to the wave).
    They reach the imaginary axis when lambda does, and they all vanish at lambda = 0; the sign of
    their real part is then no criterion.  They belong to the unstable subspace for Re(lambda) > 0
    if and only if they move to the right when Re(lambda) grows: Re(d mu / d lambda) > 0.
    The derivative comes from M1 = dM/dlambda, reduced to the invariant subspace of these eigenvalues.

    Eigenvalues away from the imaginary axis are assigned by the sign of their real part, as in split().
    Returns (k, Pu, ev_u, ev_s) like split().
    """
    M = np.asarray(M, dtype=complex)
    N = M.shape[0]
    scale = max(np.linalg.norm(M, 2), 1e-300)
    re = np.linalg.eigvals(M).real
    near = np.abs(re) <= tol * scale
    if not near.any():
        return split(M)
    far = np.abs(re[~near])
    theta = 0.5 * (np.abs(re[near]).max() + (far.min() if far.size else 2 * tol * scale))
    eye = np.eye(N)
    k_hi, P_hi, ev_hi, _ = split(M - theta * eye)          # Re > theta
    k_lo, P_lo, _, ev_lo = split(M + theta * eye)          # Re > -theta; ev_lo: Re < -theta
    m = k_lo - k_hi
    if m != int(near.sum()):
        raise SlowModeError("the eigenvalues next to the imaginary axis are not separated from the others")
    Pc = P_lo - P_hi                                        # projector onto the slow subspace
    R = np.linalg.svd(Pc)[0][:, :m]
    C = R.conj().T @ Pc @ M @ R                             # M on the slow subspace
    A = R.conj().T @ Pc @ np.asarray(M1, dtype=complex) @ R  # dM/dlambda on the slow subspace
    mu, X = np.linalg.eig(C)
    gaps = np.abs(mu[:, None] - mu[None, :])[~np.eye(m, dtype=bool)]
    if m > 1 and gaps.min() <= 1e-10 * scale:
        # The slow eigenvalues coincide (lambda = 0): the eigenvectors of the derivative separate them.
        rate, X = np.linalg.eig(A)
        mu = np.zeros(m, dtype=complex)
    else:
        rate = np.diag(np.linalg.solve(X, A @ X))
    size = max(np.abs(rate).max(), 1e-300)
    if np.abs(rate.real).min() <= 1e-6 * size:
        raise SlowModeError("a characteristic speed of an end state equals the speed of the wave, "
                            "or the characteristic speeds are not real")
    up = rate.real > 0
    Xi = np.linalg.inv(X)
    P_sel = R @ X[:, up] @ Xi[up, :] @ R.conj().T @ Pc
    return (int(k_hi + up.sum()), P_hi + P_sel, np.concatenate((ev_hi + theta, mu[up])),
            np.concatenate((mu[~up], ev_lo - theta)))


def reference_rows(P, rank):
    """Constant rows C (rank x N) such that C @ P has full row rank."""
    U = np.linalg.svd(P)[0]
    return U[:, :rank].conj().T


ROUNDING = 64 * np.finfo(float).eps


def symbol_norm(a_k, kappa):
    """2-norm of the symbol sum_k a_k (i kappa)^k at one wavenumber; 0 at kappa = infinity (a limit, no eigenvalues)."""
    if not np.isfinite(kappa):
        return 0.0
    powers = (1j * float(kappa)) ** np.arange(a_k.shape[0])
    return float(np.linalg.norm(np.einsum("k,kij->ij", powers, a_k), 2))


def axis_tolerance(sides, scale0):
    """Tolerance for the comparison of sup Re(lambda) with zero.

    A relative part, and the rounding error of the eigenvalues of the symbol at the wavenumber of
    the supremum.  The second part matters if a dispersion curve lies on the imaginary axis for all
    wavenumbers (no dissipation): the computed real parts are then rounding errors of the order
    eps |symbol|, and the largest one is found at a large wavenumber.
    """
    return 1e-9 * scale0 + max(v.get("rounding", 0.0) for v in sides.values())


def dispersion(a_k, kappa, mass=None):
    """Eigenvalues of the symbol sum_k a_k (i kappa)^k, shape (len(kappa), number of dynamic components).

    With constraint components (mass 0) the eigenvalue problem is  symbol p = lambda diag(mass) p.
    The constraint block is eliminated; it must be invertible for every real kappa.
    """
    kappa = np.atleast_1d(np.asarray(kappa, dtype=float))
    powers = (1j * kappa[:, None]) ** np.arange(a_k.shape[0])[None, :]
    symbol = np.einsum("lk,kij->lij", powers, a_k)
    if mass is None or np.all(np.asarray(mass) != 0):
        return np.linalg.eigvals(symbol)
    D, C = np.nonzero(mass)[0], np.nonzero(np.asarray(mass) == 0)[0]
    block = lambda rows, cols: symbol[:, rows][:, :, cols]
    constraint = block(C, C)
    if np.linalg.cond(constraint).max() > 1e12:
        raise np.linalg.LinAlgError("the constraint is not invertible for some real wavenumber")
    return np.linalg.eigvals(block(D, D) - block(D, C) @ np.linalg.solve(constraint, block(C, D)))


def quadratic_dispersion(a_k, N1, inertia, c, kappa):
    """Roots lambda of  det(lambda^2 N2 + lambda (N1 - 2 i c kappa N2) - sum_k a_k (i kappa)^k) = 0.

    N2 = diag(inertia).  Companion form in (p, lambda p).  A component without inertia gives
    infinite eigenvalues of the pencil; they are returned as nan.  Shape (len(kappa), 2 n).
    """
    kappa = np.atleast_1d(np.asarray(kappa, dtype=float))
    n = N1.shape[0]
    powers = (1j * kappa[:, None]) ** np.arange(a_k.shape[0])[None, :]
    N2 = np.diag(np.asarray(inertia, dtype=float))
    A = np.zeros((kappa.size, 2 * n, 2 * n), dtype=complex)
    A[:, :n, n:] = np.eye(n)
    A[:, n:, :n] = np.einsum("lk,kij->lij", powers, a_k)
    A[:, n:, n:] = -(N1[None] - 2j * c * kappa[:, None, None] * N2[None])
    if np.all(np.asarray(inertia) != 0):
        A[:, n:, :] = np.linalg.solve(N2, A[:, n:, :])
        return np.linalg.eigvals(A)
    B = np.eye(2 * n)
    B[n:, n:] = N2
    out = np.full((kappa.size, 2 * n), np.nan, dtype=complex)
    with np.errstate(all="ignore"):
        for i in range(kappa.size):
            values = eig(A[i], B, right=False)
            values = values[np.isfinite(values)]
            out[i, :values.size] = values
    return out


def quadratic_essential_bound(end_minus, end_plus, c, kappa_max=1.0e4, samples=3001):
    """Supremum of Re(lambda) over the dispersion curves of the two end states (second order in time).

    end = (a_k, N1, inertia).  A hyperbolic branch tends to a vertical line as kappa -> infinity:
    the curve counts as rising only if it still gains more than 1e-7 over the last part of the range.
    """
    kappa = np.sinh(np.linspace(0.0, np.arcsinh(kappa_max), samples))
    top = lambda end, q: np.nanmax(quadratic_dispersion(*end, c, q).real, axis=1)
    sides = {}
    for name, end in (("left", end_minus), ("right", end_plus)):
        re = top(end, kappa)
        i = int(np.argmax(re))
        lo, hi = kappa[max(i - 1, 0)], kappa[min(i + 1, samples - 1)]
        s, where = float(re[i]), float(kappa[i])
        if hi > lo:
            fit = minimize_scalar(lambda q: -top(end, q)[0], bounds=(lo, hi),
                                  method="bounded", options=dict(xatol=1e-10 * max(1.0, hi)))
            if -fit.fun > s:
                s, where = float(-fit.fun), float(fit.x)
        growing = bool(i >= samples - 3 and re[-1] - re[-200] > 1e-7 * max(1.0, abs(re[-1])))
        a_k, N1, inertia = end
        weight = np.abs(np.asarray(inertia, dtype=float))
        companion = 1.0 + (symbol_norm(a_k, where) + np.linalg.norm(N1, 2)
                           + 2 * abs(c) * where * weight.max()) / weight[weight > 0].min()
        sides[name] = dict(s=s, kappa=where, growing_at_large_kappa=growing, limit_at_infinity=float(re[-1]),
                           rounding=ROUNDING * companion)
    return max(v["s"] for v in sides.values()), sides


def high_wavenumber_limit(a_k, first_order):
    """Limit of Re(lambda) on the bounded dispersion branches as kappa -> infinity.

    `first_order` lists the components without a second derivative (set H; the
    others form the set P).  With B = A0_HH - A1_HP A2_PP^{-1} A1_PH and S = A1_HH,
    the bounded branches behave like  i kappa sigma + eig(B restricted to the
    eigenspace of S for sigma).  Returns None when this structure is absent:
    second derivatives in the H rows, or S without a real diagonal form.
    """
    Hs = list(first_order)
    if not Hs or a_k.shape[0] != 3:
        return None
    A0, A1, A2 = a_k
    Ps = [j for j in range(A0.shape[0]) if j not in Hs]
    if np.abs(A2[Hs, :]).max() > 0:
        return None
    taxis = bool(Ps) and np.abs(A2[np.ix_(Ps, Hs)]).max() > 0
    if taxis and np.abs(A1[np.ix_(Hs, Ps)]).max() > 0:
        return None
    B = A0[np.ix_(Hs, Hs)]
    if Ps:
        B = B - A1[np.ix_(Hs, Ps)] @ np.linalg.solve(A2[np.ix_(Ps, Ps)], A1[np.ix_(Ps, Hs)])
    if taxis:
        B = B - A0[np.ix_(Hs, Ps)] @ np.linalg.solve(A2[np.ix_(Ps, Ps)], A2[np.ix_(Ps, Hs)])
    sigma, V = np.linalg.eig(A1[np.ix_(Hs, Hs)])
    scale = max(1.0, np.abs(sigma).max())
    if np.abs(sigma.imag).max() > 1e-9 * scale or np.linalg.cond(V) > 1e8:
        return None
    Bt = np.linalg.solve(V, B @ V)
    order = np.argsort(sigma.real)
    groups, current = [], [order[0]]
    for j in order[1:]:
        if sigma.real[j] - sigma.real[current[-1]] < 1e-9 * scale:
            current.append(j)
        else:
            groups.append(current)
            current = [j]
    groups.append(current)
    return float(max(np.linalg.eigvals(Bt[np.ix_(g, g)]).real.max() for g in groups))


def essential_bound(a_minus, a_plus, kappa_max=1.0e4, samples=3001, first_order=(), mass=None):
    """Supremum of Re(lambda) over the dispersion curves of the two end states.

    `first_order`: components without a second derivative.  Their branches stay
    bounded as kappa -> infinity; the limit is included in the supremum.
    """
    kappa = np.sinh(np.linspace(0.0, np.arcsinh(kappa_max), samples))
    sides = {}
    for name, a_k in (("left", a_minus), ("right", a_plus)):
        limit = high_wavenumber_limit(a_k, first_order)
        re = dispersion(a_k, kappa, mass).real.max(axis=1)
        i = int(np.argmax(re))
        lo, hi = kappa[max(i - 1, 0)], kappa[min(i + 1, samples - 1)]
        s, where = float(re[i]), float(kappa[i])
        if hi > lo:
            fit = minimize_scalar(lambda q: -dispersion(a_k, q, mass).real.max(), bounds=(lo, hi),
                                  method="bounded", options=dict(xatol=1e-10 * max(1.0, hi)))
            if -fit.fun > s:
                s, where = float(-fit.fun), float(fit.x)
        # The supremum must not sit at the end of the sampled range.
        growing = bool(i >= samples - 3 and re[-1] > re[-2] > re[-3])
        if limit is not None:
            # A bounded branch that approaches its limit from below is not a growing one.
            if re[-1] <= limit + 1e-6 * max(1.0, abs(limit)):
                growing = False
            if limit > s:
                s, where = limit, np.inf
        sides[name] = dict(s=s, kappa=where, growing_at_large_kappa=growing, limit_at_infinity=limit,
                           rounding=ROUNDING * symbol_norm(a_k, where))
    return max(v["s"] for v in sides.values()), sides
