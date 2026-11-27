"""Verified whole-right-half-plane count for the D2 delayed kink."""
from fractions import Fraction
from functools import cache
from math import factorial, floor

from .e3_certificate import (Box, Interval, ONE, SCALE, TWO, parameter_box,
                             polygon_winding)

ORDER = 40
EXP_ORDER = 24
MAX_DEPTH = 18
SQRT_TWO = TWO.sqrt_positive_real()
HALF = Box.real(Fraction(1, 2))
QUARTER = Box.real(Fraction(1, 4))
real = cache(Box.real)


def disk_bound(radius):
    limit = Interval.point(radius).hi
    interval = Interval(-limit, limit)
    return Box(interval, interval)


def series_remainder(bound, averaged=False):
    first = EXP_ORDER + 1
    advance = int(averaged)
    return (bound**first / factorial(first+advance) /
            (1 - bound/Fraction(first+advance+1)))


def exponential(z, bound):
    # Reduce the pointwise argument to absolute value at most one.
    halvings = 0
    while bound > 1:
        z = HALF*z
        bound /= 2
        halvings += 1
    value = real(Fraction(1, factorial(EXP_ORDER)))
    for n in range(EXP_ORDER-1, -1, -1):
        value = real(Fraction(1, factorial(n))) + z*value
    value = value + disk_bound(series_remainder(bound))
    for _ in range(halvings):
        value = value*value
    return value


def delay_average(lam, tau, wall):
    # On [0,wall] + i[-wall,wall], |lambda| <= 3*wall/2.
    bound = Fraction(3*wall, 2)
    z = real(Fraction(-1, 4))*lam
    average = real(Fraction(1, factorial(EXP_ORDER+1)))
    for n in range(EXP_ORDER-1, -1, -1):
        average = real(Fraction(1, factorial(n+1))) + z*average
    average = average + disk_bound(series_remainder(bound/4, averaged=True))
    return exponential(real(-tau)*lam, tau*bound)*average


def spatial_tail_bounds(sigma, epsilon):
    # For Re lambda >= 0, Re(2*p) >= sqrt(2) > 1 and |g| <= sigma.
    coupling = 2*sigma*(1+epsilon)
    skew = 4*sigma*epsilon
    if coupling+skew > 2*(ORDER+1):
        raise ValueError("D2 series order is insufficient for this parameter range")
    previous, current = Fraction(0), Fraction(1)
    for n in range(ORDER):
        previous, current = current, ((n*(n+1)+coupling)*current+skew*previous)/((n+1)*(n+2))
    majorant = max(current, previous)
    return majorant/2**ORDER, majorant*(ORDER+2)/2**(ORDER-1)


def half_line(p, g, signed_epsilon, errors):
    first = real(2*(1-signed_epsilon))*g
    second = real(4*signed_epsilon)*g
    previous, current = real(0), ONE
    value, derivative = ONE, real(0)
    for n in range(ORDER):
        n1 = n+1
        coefficient = ((real(n*n1)+first)*current + second*previous) / (real(n1)*(real(n1)+TWO*p))
        previous, current = current, coefficient
        value = value + real(Fraction(1, 2**n1))*coefficient
        derivative = derivative + real(Fraction(n1, 2**n))*coefficient
    return value+errors[0], derivative+errors[1]


def characteristic(lam, sigma, tau, epsilon, wall, errors):
    kappa = (ONE+lam).sqrt_positive_real()
    p = kappa/SQRT_TWO
    g = real(sigma)*delay_average(lam, tau, wall)
    left, dl = half_line(p, g, epsilon, errors)
    right, dr = half_line(p, g, -epsilon, errors)
    return TWO*kappa*left*right + SQRT_TWO*QUARTER*(dl*right+left*dr)


def certify_delay(sigma, tau, epsilon):
    sigma, tau, epsilon = map(Fraction, (sigma, tau, epsilon))
    if not 0 < sigma <= 8 or not 1 <= tau <= 2 or not 0 <= epsilon <= Fraction(3, 4):
        raise ValueError("outside the reviewed D2 parameter range")
    # On U >= 0, W <= 1+epsilon*U-U**2; on U <= 0, W <= 1.
    gain_bound = sigma*(1+epsilon**2/4)
    wall = floor(gain_bound)+1
    tails = spatial_tail_bounds(sigma, epsilon)
    errors = tuple(map(disk_bound, tails))

    def image(left, right):
        return characteristic(parameter_box(left, right), sigma, tau, epsilon, wall, errors)

    @cache
    def vertex(point):
        box = image(point, point)
        return box.re.midpoint(), box.im.midpoint()

    segments = []

    def verify(left, right, depth):
        try:
            ends = vertex(left), vertex(right)
            box = image(left, right)
            bounds = [Interval(min(interval.lo, ends[0][axis], ends[1][axis]),
                               max(interval.hi, ends[0][axis], ends[1][axis]))
                      for axis, interval in enumerate((box.re, box.im))]
            margin = max(bound.zero_distance() for bound in bounds)
        except ArithmeticError:
            margin = 0
        if margin:
            segments.append(dict(start=left, end=right, polygon_start=ends[0], polygon_end=ends[1],
                                 image_box=[[b.lo, b.hi] for b in bounds], zero_margin=margin))
            return
        if depth == MAX_DEPTH:
            raise ArithmeticError("D2 contour image could not be separated from zero")
        middle = tuple((a+b)//2 for a, b in zip(left, right, strict=True))
        verify(left, middle, depth+1)
        verify(middle, right, depth+1)

    side = wall*SCALE
    corners = [(0, -side), (side, -side), (side, side), (0, side), (0, -side)]
    for left, right in zip(corners, corners[1:]):
        verify(left, right, 0)
    vertices = [segments[0]["polygon_start"]] + [entry["polygon_end"] for entry in segments]
    count = polygon_winding(vertices)
    if count < 0:
        raise ArithmeticError("negative winding for the analytic D2 Wronskian")
    return dict(method="INTERVAL_WRONSKIAN_WINDING", label="STABLE" if count == 0 else "UNSTABLE",
                open_rhp_count=count, evidence_grade="RIGOROUSLY_CERTIFIED",
                scope="entire open right half-plane", fixed_point_scale=SCALE,
                spatial_series_order=ORDER, exponential_series_order=EXP_ORDER,
                spatial_tail_bounds=[str(value) for value in tails],
                contour_wall=wall, outside_resolvent_margin=str(wall-gain_bound),
                contour_min_zero_margin=str(Fraction(min(s["zero_margin"] for s in segments), SCALE)),
                segment_count=len(segments), segments=segments,
                proof="docs/a1-d2-comparison.md#d2-full-right-half-plane-certificate")
