"""Contours, adaptive winding numbers and root polishing (DESIGN.md, section 3.2)."""
from __future__ import annotations

import numpy as np


class ContourError(ArithmeticError):
    """A contour passes through a zero, or an iteration failed."""


def circle(center, radius):
    return lambda s: center + radius * np.exp(2j * np.pi * s)


def box(x_min, x_max, y_max, y_scale):
    """Counter-clockwise rectangle [x_min, x_max] x [-y_max, y_max].

    The vertical edges use asinh spacing with scale y_scale: fine near the real axis.
    """
    T = np.arcsinh(y_max / y_scale)

    def curve(s):
        q = (s % 1.0) * 4.0
        i = min(3, int(q))
        u = q - i
        if i == 0:
            return complex(x_min + (x_max - x_min) * u, -y_max)
        if i == 1:
            return complex(x_max, y_scale * np.sinh((2 * u - 1) * T))
        if i == 2:
            return complex(x_max + (x_min - x_max) * u, y_max)
        return complex(x_min, y_scale * np.sinh((1 - 2 * u) * T))
    return curve


def rectangle(x0, x1, y0, y1):
    """Counter-clockwise rectangle [x0, x1] x [y0, y1] with uniform speed on each edge."""
    corners = [complex(x0, y0), complex(x1, y0), complex(x1, y1), complex(x0, y1)]

    def curve(s):
        q = (s % 1.0) * 4.0
        i = min(3, int(q))
        return corners[i] + (corners[(i + 1) % 4] - corners[i]) * (q - i)
    return curve


def isolate(engine, rect, known=(), min_size=1e-3, max_boxes=200, initial=16, limit=0.55):
    """Seeds for the zeros in rect = (x0, x1, y0, y1) that are not listed in `known`.

    Recursive bisection with winding numbers (argument principle).  A box with one
    unknown zero is handed to the secant iteration; otherwise it is cut again.
    The cut avoids the middle, so that it does not run along the real axis.
    A box smaller than min_size that still holds m zeros gives m equal seeds.
    The entries of `known` are approximations: a zero closer than min_size to one
    of them is that known zero.
    """
    def inside(r, z):
        return r[0] < z.real < r[1] and r[2] < z.imag < r[3]

    def is_known(z):
        return any(abs(z - complex(q)) < min_size for q in known)

    def unknown(r):
        out = winding(engine, rectangle(*r), initial, limit)
        if not out["resolved"]:
            raise ContourError("a contour of the subdivision is not resolved")
        return out["count"] - sum(inside(r, complex(z)) for z in known)

    seeds, boxes = [], 1
    stack = [(tuple(map(float, rect)), unknown(rect))]
    while stack:
        r, count = stack.pop()
        if count <= 0:
            continue
        width, height = r[1] - r[0], r[3] - r[2]
        centre = complex((r[0] + r[1]) / 2, (r[2] + r[3]) / 2)
        if count == 1:
            try:
                z = locate(engine, centre, 0.1 * min(width, height))
                if inside(r, z) and not is_known(z):
                    seeds.append(z)
                    continue
            except ArithmeticError:
                pass
        if max(width, height) < min_size:
            if not is_known(centre):
                seeds.extend([centre] * count)
            continue
        for fraction in (0.382, 0.618, 0.5, 0.27, 0.73):
            if width >= height:
                cut = r[0] + fraction * width
                children = [(r[0], cut, r[2], r[3]), (cut, r[1], r[2], r[3])]
            else:
                cut = r[2] + fraction * height
                children = [(r[0], r[1], r[2], cut), (r[0], r[1], cut, r[3])]
            try:
                counts = [unknown(child) for child in children]
            except ArithmeticError:
                continue
            if sum(counts) == count:
                break
        else:
            raise ContourError("the subdivision found no admissible cut")
        boxes += 2
        if boxes > max_boxes:
            raise ContourError("the subdivision exceeded its budget of boxes")
        stack.extend(zip(children, counts))
    return seeds


def winding(engine, curve, initial=64, limit=0.55, max_depth=14, log_limit=1.0):
    """Winding number of the engine's determinant along a closed curve.

    A panel is accepted when both half-panel phase increments are below `limit`,
    they add up to the full-panel increment, and log|D| changes by less than
    `log_limit`.  Otherwise the panel is bisected.  Every panel is bisected once.
    The same samples give the winding number of the certificate gamma.
    """
    def sample(s):
        lam = complex(curve(s))
        log_abs, phase, gamma = engine(lam)
        if not np.isfinite(log_abs) or gamma == 0:
            raise ContourError("the contour passes through a zero")
        return log_abs, phase, gamma / abs(gamma)

    grid = np.linspace(0.0, 1.0, initial + 1)
    values = [sample(s) for s in grid[:-1]]
    values.append(values[0])
    panels = [(grid[i], grid[i + 1], values[i], values[i + 1]) for i in range(initial)]
    evaluations = initial
    total = gamma_total = 0.0
    max_step = gamma_step = 0.0
    depth = 0
    while panels and depth <= max_depth:
        pending = []
        for sa, sb, va, vb in panels:
            sm = 0.5 * (sa + sb)
            vm = sample(sm)
            evaluations += 1
            first = np.angle(vm[1] * np.conj(va[1]))
            last = np.angle(vb[1] * np.conj(vm[1]))
            coarse = np.angle(vb[1] * np.conj(va[1]))
            bad = (max(abs(first), abs(last)) > limit or abs(first + last - coarse) > 1e-2
                   or max(abs(vm[0] - va[0]), abs(vb[0] - vm[0])) > log_limit)
            if bad or depth == 0:
                pending.append((sa, sm, va, vm))
                pending.append((sm, sb, vm, vb))
            else:
                total += first + last
                max_step = max(max_step, abs(first), abs(last))
                g1 = np.angle(vm[2] * np.conj(va[2]))
                g2 = np.angle(vb[2] * np.conj(vm[2]))
                gamma_total += g1 + g2
                gamma_step = max(gamma_step, abs(g1), abs(g2))
        panels = pending
        depth += 1
    resolved = not panels
    return dict(count=int(round(total / (2 * np.pi))) if resolved else None, resolved=resolved,
                evaluations=evaluations, max_phase_step=float(max_step),
                gamma_count=int(round(gamma_total / (2 * np.pi))) if resolved else None,
                gamma_max_step=float(gamma_step))


def locate(engine, seed, step, tol=1e-11, max_iter=40):
    """Complex secant iteration on the determinant, started at seed and seed + step."""
    z0, z1 = complex(seed) + step, complex(seed)
    for _ in range(max_iter):
        l0, p0, _ = engine(z0)
        l1, p1, _ = engine(z1)
        if not np.isfinite(l1):
            return z1
        if not np.isfinite(l0):
            return z0
        shift = max(l0, l1)
        f0, f1 = p0 * np.exp(l0 - shift), p1 * np.exp(l1 - shift)
        if abs(f1 - f0) < 1e-300:
            raise ContourError("the secant iteration stalled")
        dz = f1 * (z1 - z0) / (f1 - f0)
        z0, z1 = z1, z1 - dz
        if abs(dz) < tol * max(1.0, abs(z1)):
            return z1
    raise ContourError("the secant iteration did not converge")
