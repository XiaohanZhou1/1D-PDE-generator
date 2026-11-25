"""Exponentially weighted spaces (DESIGN.md, section 2.13).

A wave whose essential spectrum reaches Re(lambda) > 0 is unstable in L2.  The instability can
be convective: a perturbation grows, but it is carried away from the wave.  In the space with
the norm  ||exp(phi) p||,  phi'(z) = a(z),  a -> a_- / a_+  at the two ends, such a perturbation
decays.  With  p = exp(-phi) q  the eigenvalue problem keeps its form:

    d/dz -> d/dz - a(z)        in the operator L,
    M    -> M + a(z) I         in the first-order system.

* find_rate         the rate a at one end state that moves its essential spectrum furthest left
* WeightedSystem    a LocalSystem seen through the weight; the wave carries a(z) and a'(z)
* weighted_wave     the wave with the two extra rows
* translation_mode  is exp(phi) U' square integrable?  (then lambda = 0 stays an eigenvalue)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .asymptotic import essential_bound, split
from .wave import Wave


def weighted_coefficients(a, rate, slope=0.0):
    """Coefficients of  exp(phi) L exp(-phi)  for an operator of order two: (d - a)^2 = d^2 - 2 a d + a^2 - a'.

    a: (3, ..., n, n);  rate, slope: numbers, or arrays over the middle axes.
    """
    rate, slope = np.asarray(rate, dtype=float)[..., None, None], np.asarray(slope, dtype=float)[..., None, None]
    return np.stack((a[0] - rate * a[1] + (rate * rate - slope) * a[2], a[1] - 2 * rate * a[2], a[2]))


def end_supremum(a_k, rate, first_order, mass, samples=3001):
    """Supremum of Re(lambda) on the dispersion curves of one end state in the weight exp(rate z)."""
    w = weighted_coefficients(a_k, rate)
    return essential_bound(w, w, samples=samples, first_order=first_order, mass=mass)[0]


def find_rate(a_k, first_order, mass):
    """Rate of the weight at one end state: dict(rate, supremum, interval).

    rate 0 if the end state is stable without a weight.  Otherwise the rate with the smallest
    supremum; `interval` is the range of rates around it with a negative supremum (None if there is none).
    """
    plain = end_supremum(a_k, 0.0, first_order, mass)
    scale0 = max(float(np.linalg.norm(a_k[0], 2)), 1e-300)
    if plain < -1e-9 * scale0:
        return dict(rate=0.0, supremum=plain, interval=None, without_weight=plain)
    coarse = lambda rate: end_supremum(a_k, rate, first_order, mass, samples=601)
    diffusion = np.linalg.eigvalsh((a_k[2] + a_k[2].T) / 2)
    d = float(diffusion[diffusion > 0].min()) if np.any(diffusion > 0) else 1.0
    reach = 4 * max(float(np.linalg.norm(a_k[1], 2)) / (2 * d), np.sqrt(scale0 / d))
    grid = reach * np.linspace(-1.0, 1.0, 81)
    values = np.array([coarse(rate) for rate in grid])
    i = int(np.argmin(values))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
    fit = minimize_scalar(coarse, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-8 * reach))
    rate = float(fit.x) if fit.fun < values[i] else float(grid[i])
    supremum = end_supremum(a_k, rate, first_order, mass)
    interval = None
    if supremum < 0:
        ends = []
        for step in (-1, 1):                           # walk outwards to the first rate with a supremum >= 0
            j = i
            while 0 <= j + step < grid.size and values[j + step] < 0:
                j += step
            if 0 <= j + step < grid.size:
                inside = rate if j == i else grid[j]
                ends.append(float(brentq(coarse, *sorted((inside, grid[j + step])))) if coarse(inside) < 0
                            else float(grid[j]))
            else:
                ends.append(float(grid[j]))
        interval = [min(ends), max(ends)]
    return dict(rate=rate, supremum=float(supremum), interval=interval, without_weight=float(plain))


class WeightedSystem:
    """A LocalSystem of order two, first order in time, in the weight exp(phi).

    The state has two more rows: a(z) = phi'(z) and a'(z).  Everything that the spectral
    part of the pipeline needs is provided; the other attributes are those of the system.
    """

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def state(self, v):
        """v = (u_1, ..., u_n, rate): extended state of the constant solution u in the weight exp(rate z)."""
        v = np.asarray(v, dtype=float)
        return np.concatenate((self.base.state(v[:self.base.n]), [v[self.base.n], 0.0]))

    def _split(self, Y):
        rows = getattr(self.base, "state_rows", self.base.N)           # conservation.py: fewer rows than N
        Y = np.asarray(Y, dtype=float).reshape(rows + 2, -1)
        return Y[:-2], Y[-2], Y[-1]

    def coefficients(self, Y, c):
        Y0, rate, slope = self._split(Y)
        return weighted_coefficients(self.base.coefficients(Y0, c), rate, slope)

    def first_order(self, Y, c):
        Y0, rate, _ = self._split(Y)
        M0, M1 = self.base.first_order(Y0, c)
        return M0 + rate[:, None, None] * np.eye(self.base.N)[None], M1

    pencil = first_order

    def end_state(self, v, c):
        Y = self.state(v)
        M0, M1 = self.first_order(Y, c)
        return M0[0], M1[0], self.coefficients(Y, c)[:, 0]


def weighted_wave(wave, rate_minus, rate_plus):
    """The wave with the rows a(z), a'(z):  a = a_- + (a_+ - a_-)(1 + tanh(z / l)) / 2,  l = L / 10."""
    width = wave.L / 10
    step = rate_plus - rate_minus

    def interpolant(z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        th = np.tanh(z / width)
        return np.vstack((wave.interpolant(z)[:wave.N], rate_minus + step * (1 + th) / 2,
                          step * (1 - th * th) / (2 * width)))

    return Wave(c=wave.c, L=wave.L, N=wave.N + 2, Y_minus=np.concatenate((wave.Y_minus, [rate_minus, 0.0])),
                Y_plus=np.concatenate((wave.Y_plus, [rate_plus, 0.0])), interpolant=interpolant, mesh=wave.mesh,
                info=wave.info, theta=wave.theta, right=wave.right)


def translation_mode(system, wave, rate_minus, rate_plus):
    """True if exp(phi) U' is square integrable, False if not, None if the rates are too close to tell.

    U' decays like exp(-r z) at +infinity and like exp(r z) at -infinity; r is the modulus of the real
    part of an eigenvalue of the profile equation at the end state.  The rate that the wave uses is
    measured on the profile and matched to these eigenvalues.
    """
    answers = []
    for sign, Y_end, rate in ((1, wave.Y_plus, rate_plus), (-1, wave.Y_minus, rate_minus)):
        if rate == 0.0:
            answers.append(True)
            continue
        z = sign * np.linspace(0.5 * wave.L, 0.8 * wave.L, 61)
        distance = np.linalg.norm(wave.Y(z) - Y_end[:, None], axis=0)
        ok = distance > 1e-300
        measured = -np.polyfit(np.abs(z[ok]), np.log(distance[ok]), 1)[0]
        u_end = Y_end[system.offset]
        eigenvalues = np.concatenate(split(system.end_state(u_end, wave.c)[0])[2:])
        candidates = np.unique(np.round(np.abs(eigenvalues.real[sign * eigenvalues.real < 0]), 12))
        if candidates.size == 0:
            return None
        decay = float(candidates[np.argmin(np.abs(candidates - measured))])
        margin = decay - sign * rate                   # exp(sign rate |z|) exp(-decay |z|) must decay
        if abs(margin) < 1e-6 * max(1.0, decay):
            return None
        answers.append(margin > 0)
    return all(answers)
