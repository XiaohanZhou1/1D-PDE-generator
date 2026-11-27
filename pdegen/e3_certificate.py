"""Verified E3 winding count using fixed-point interval arithmetic."""
import argparse
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from math import factorial, isqrt
from pathlib import Path

from .contract import write_json
from .dynamic_interface import dynamic_interface

SCALE = 1 << 96
ORDER = 20
WALL = 3
MAX_DEPTH = 18
TAIL = Fraction(8**(ORDER + 1), factorial(ORDER + 1)**2) / (1 - Fraction(8, (ORDER + 2)**2))


def ceildiv(a, b):
    return -((-a) // b)


@dataclass(frozen=True)
class Interval:
    lo: int
    hi: int

    @classmethod
    def point(cls, value):
        scaled = Fraction(value) * SCALE
        return cls(scaled.numerator // scaled.denominator, ceildiv(scaled.numerator, scaled.denominator))

    def __add__(self, other):
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        products = [a*b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return Interval(min(products) // SCALE, ceildiv(max(products), SCALE))

    def reciprocal(self):
        if self.lo <= 0 <= self.hi:
            raise ArithmeticError("interval divisor contains zero")
        return Interval(SCALE**2 // self.hi, ceildiv(SCALE**2, self.lo))

    def square(self):
        lower = 0 if self.lo <= 0 <= self.hi else min(self.lo**2, self.hi**2)
        return Interval(lower // SCALE, ceildiv(max(self.lo**2, self.hi**2), SCALE))

    def sqrt(self):
        if self.lo < 0:
            raise ArithmeticError("square-root interval extends below zero")
        bottom, top = self.lo*SCALE, self.hi*SCALE
        upper = isqrt(top)
        return Interval(isqrt(bottom), upper + (upper**2 != top))

    def midpoint(self):
        return (self.lo + self.hi) // 2

    def zero_distance(self):
        return max(0, self.lo, -self.hi)


@dataclass(frozen=True)
class Box:
    re: Interval
    im: Interval

    @classmethod
    def real(cls, value):
        return cls(Interval.point(value), Interval(0, 0))

    def __add__(self, other):
        return Box(self.re + other.re, self.im + other.im)

    def __mul__(self, other):
        return Box(self.re*other.re - self.im*other.im, self.re*other.im + self.im*other.re)

    def __truediv__(self, other):
        inverse = (other.re.square() + other.im.square()).reciprocal()
        return self * Box(other.re*inverse, -other.im*inverse)

    def sqrt_positive_real(self):
        magnitude = (self.re.square() + self.im.square()).sqrt()
        re = ((magnitude + self.re) * Interval.point(Fraction(1, 2))).sqrt()
        return Box(re, self.im * (re * Interval.point(2)).reciprocal())


ZERO, ONE, TWO, FOUR = map(Box.real, (0, 1, 2, 4))
HALF = Box.real(Fraction(1, 2))
QUARTER = Box.real(Fraction(1, 4))
SPEED = Box.real(Fraction(3, 2))
MASS_V = Box.real(Fraction(5, 2))
EPSILON = Interval.point(2 * TAIL).hi


def characteristic(lam, gain):
    """Enclose the exact infinite-domain characteristic on a right-half-plane box."""
    sigma_u = (SPEED*SPEED + FOUR*(ONE + lam)).sqrt_positive_real()
    sigma_v = (SPEED*SPEED + FOUR*(MASS_V + lam)).sqrt_positive_real()
    kappa = (SPEED + sigma_u) * HALF
    nu = (Box.real(Fraction(-3, 2)) + sigma_v) * HALF
    qden = nu + ONE + SPEED
    integral = ZERO
    normalisation = ZERO
    coefficient = ONE
    for n in range(ORDER + 1):
        offset = Box.real(Fraction(n, 2))
        if n:
            coefficient = TWO * coefficient / (offset * (sigma_u + offset))
        k = kappa + offset
        term = HALF / (k + HALF) + ONE / (qden * (k + ONE) * (k + nu))
        integral = integral + coefficient * term
        normalisation = normalisation + coefficient
    error = Box(Interval(-EPSILON, EPSILON), Interval(-EPSILON, EPSILON))
    response = (integral + error) / (normalisation + error) + ONE / qden
    polynomial = lam*lam + QUARTER*lam + ONE
    return polynomial + Box.real(gain) * response


def parameter_box(left, right):
    return Box(*(Interval(min(a, b), max(a, b)) for a, b in zip(left, right, strict=True)))


def polygon_winding(vertices):
    count = 0
    for (x, y), (s, t) in zip(vertices, vertices[1:]):
        cross = x*t - y*s
        if y <= 0 < t and cross > 0:
            count += 1
        elif t <= 0 < y and cross < 0:
            count -= 1
    return count


def certify_gain(gain):
    gain = Fraction(gain)
    outside_margin = WALL**2 - Fraction(WALL, 4) - 1 - gain * (Fraction(1, 5) + Fraction(2, 7))
    if gain <= 0 or outside_margin <= 0:
        raise ValueError("gain is outside this contour's analytic tail bound")

    @cache
    def vertex(point):
        image = characteristic(parameter_box(point, point), gain)
        return image.re.midpoint(), image.im.midpoint()

    segments = []

    def verify(left, right, depth):
        try:
            image = characteristic(parameter_box(left, right), gain)
            ends = vertex(left), vertex(right)
            bounds = [Interval(min(interval.lo, ends[0][axis], ends[1][axis]),
                               max(interval.hi, ends[0][axis], ends[1][axis]))
                      for axis, interval in enumerate((image.re, image.im))]
            margin = max(bound.zero_distance() for bound in bounds)
        except ArithmeticError:
            margin = 0
        if margin:
            segments.append(dict(start=left, end=right, polygon_start=ends[0], polygon_end=ends[1],
                                 image_box=[[b.lo, b.hi] for b in bounds], zero_margin=margin))
            return
        if depth == MAX_DEPTH:
            raise ArithmeticError("contour segment could not be separated from zero")
        middle = tuple((a+b)//2 for a, b in zip(left, right, strict=True))
        verify(left, middle, depth + 1)
        verify(middle, right, depth + 1)

    side = WALL*SCALE
    corners = [(0, -side), (side, -side), (side, side), (0, side), (0, -side)]
    for left, right in zip(corners, corners[1:]):
        verify(left, right, 0)
    vertices = [segments[0]["polygon_start"]] + [row["polygon_end"] for row in segments]
    count = polygon_winding(vertices)
    if count < 0:
        raise ArithmeticError("negative count for the analytic characteristic")
    return dict(gain=str(gain), label="STABLE" if count == 0 else "UNSTABLE",
                evidence_grade="RIGOROUSLY_CERTIFIED", open_rhp_count=count,
                neutral_mode="simple constant interface displacement",
                fixed_point_scale=SCALE, series_order=ORDER, series_tail_bound=str(TAIL),
                contour_coordinates="counterclockwise [0,3] x [-3,3]",
                outside_margin_lower_bound=str(outside_margin),
                contour_min_zero_margin=str(Fraction(min(row["zero_margin"] for row in segments), SCALE)),
                segment_count=len(segments), segments=segments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for gain in (5, 6, 8, 12):
        item = dynamic_interface(gain, 20261126)
        ident = item["instance_id"]
        try:
            evidence = certify_gain(gain)
        except ArithmeticError as error:
            evidence = dict(gain=str(gain), label="UNCERTAIN", open_rhp_count=None,
                            evidence_grade=None, reason=str(error))
        evidence.update(instance_id=ident, specification_hash=item["specification_hash"],
                        proof="docs/e3-verified-stability.md")
        instance = f"instances/{ident}.json"
        result = f"certificates/{ident}.json"
        write_json(args.out / instance, item)
        write_json(args.out / result, evidence)
        rows.append({key: evidence[key] for key in ("gain", "label", "open_rhp_count", "evidence_grade",
                                                    "instance_id", "specification_hash")})
        rows[-1].update(instance=instance, certificate=result, difficulty_review="PENDING")
        print(gain, evidence["label"], evidence["open_rhp_count"], flush=True)
    write_json(args.out / "manifest.json", dict(rows=rows))


if __name__ == "__main__":
    main()
