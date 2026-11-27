"""Outward-rounded power bounds for the F3 monodromy on all Bloch fibers."""
from fractions import Fraction
from functools import cache
from math import factorial, isqrt

import numpy as np

from .e3_certificate import Interval, SCALE as INTERVAL_SCALE


HALF_BITS = 22
SCALE = 1 << (2*HALF_BITS)
CUTOFF = 32
DEGREE = 13
MAX_DEPTH = 10
MAX_POWER = 5
TIME_UPPER = Fraction(22, 7)
DAMPING_FACTOR = Fraction(23, 50)


def ceildiv(a, b):
    return -(-a//b)


def round_up(value):
    return Fraction(ceildiv(value.numerator*SCALE, value.denominator), SCALE)


def atan_bounds(inverse):
    terms = [Fraction((-1)**n, (2*n+1)*inverse**(2*n+1)) for n in range(34)]
    return sum(terms), sum(terms[:-1])


def pi_interval():
    lo5, hi5 = atan_bounds(5)
    lo239, hi239 = atan_bounds(239)
    # Machin's identity and the alternating-series bounds.
    return Interval(Interval.point(16*lo5-4*hi239).lo, Interval.point(16*hi5-4*lo239).hi)


PI = pi_interval()


@cache
def bulk_coefficients(cutoff):
    root = Interval.point(3).sqrt()
    leading = -Interval.point(2)*root.reciprocal()
    ratio = root-Interval.point(2)
    values = [Interval.point(1)+leading]
    term = leading
    for _ in range(2*cutoff):
        term = term*ratio
        values.append(term)
    return values


def fixed_value(interval):
    value = round(Fraction(interval.midpoint()*SCALE, INTERVAL_SCALE))
    if max(abs(value*INTERVAL_SCALE-edge*SCALE) for edge in (interval.lo, interval.hi)) > INTERVAL_SCALE:
        raise ArithmeticError("F3 time step coefficient enclosure exceeds the rounding allowance")
    return value


def product(a, b):
    """Compute floor(a @ b / SCALE) exactly with split integer products."""
    limits = [max(abs(int(matrix.min())), abs(int(matrix.max()))) for matrix in (a, b)]
    if max(limits) >= 8*SCALE or a.shape[1] > 256:
        raise ArithmeticError("F3 fixed-point product exceeded the checked accumulation bound")
    ah, bh = a >> HALF_BITS, b >> HALF_BITS
    al, bl = a-(ah << HALF_BITS), b-(bh << HALF_BITS)
    middle = ah @ bl + al @ bh + ((al @ bl) >> HALF_BITS)
    return ah @ bh + (middle >> HALF_BITS)


def triangular_product(a, b):
    diagonal = product(a[0], b[0])
    return (diagonal, *(product(a[0], right)+product(left, b[0])
                        for left, right in zip(a[1:], b[1:], strict=True)))


def frobenius(*matrices):
    squared = sum(int((matrix.astype(object)**2).sum()) for matrix in matrices)
    lower = isqrt(squared)
    return Fraction(lower+(lower**2 < squared), SCALE)


def time_step(gain, xi, cutoff, halvings):
    """Propose rounded blocks of t times the two Galerkin generators."""
    n = 2*cutoff+1
    step = PI*Interval.point(Fraction(1, 2**halvings))
    potential = bulk_coefficients(cutoff)
    coeffs = [fixed_value(step*value) for value in potential]
    modes = np.arange(-cutoff, cutoff+1)
    differences = modes[:, None]-modes
    bulk = np.array(coeffs, dtype=np.int64)[abs(differences)]
    for i, mode in enumerate(range(-cutoff, cutoff+1)):
        value = potential[0]-Interval.point((mode+xi)**2+Fraction(1, 4))
        bulk[i, i] = fixed_value(step*value)
    shear = fixed_value(step*Interval.point(gain/2))
    identity = np.eye(n, dtype=np.int64)
    cos = shear*(identity+(abs(differences)==1))
    sin_re = shear*identity
    sin_im = shear*((differences==-1).astype(np.int64)-(differences==1))
    return (bulk, cos, sin_re, sin_im), identity


def finite_segments(gain, xi, cutoff):
    n = 2*cutoff+1
    shear = Fraction(3, 2)*gain
    generator_bound = (cutoff+Fraction(1, 2))**2+Fraction(5, 4)+shear
    halvings = 0
    while TIME_UPPER*generator_bound/2**halvings > Fraction(1, 2):
        halvings += 1
    step, identity = time_step(gain, xi, cutoff, halvings)
    zero = np.zeros_like(identity)
    polynomial = (SCALE*identity, zero, zero, zero)
    for k in range(DEGREE, 0, -1):
        diagonal, *off = triangular_product(step, polynomial)
        polynomial = (SCALE*identity+diagonal//k, *(entry//k for entry in off))

    # Each rounded generator differs by at most 3*n/SCALE in operator norm.
    # Compare the Horner recursion directly with the exact Taylor recursion.
    assembly = Fraction(3*n, SCALE)
    if assembly > Fraction(1, 100):
        raise ArithmeticError("F3 assembly error exceeds the Taylor recursion allowance")
    t = TIME_UPPER/2**halvings
    rounding = Fraction(8*n, SCALE)
    error = round_up(5*assembly+3*rounding+Fraction(2, 2**(DEGREE+1)*factorial(DEGREE+1)))
    for _ in range(halvings):
        polynomial = triangular_product(polynomial, polynomial)
        error = round_up((2*(1+shear*t)+error)*error+Fraction(5*n, SCALE))
        t *= 2
    return polynomial, error


def monodromy_data(gain, xi, cutoff):
    (e, b, c_real, c_imag), error = finite_segments(gain, xi, cutoff)
    n = len(e)
    f1 = frobenius(e, e, b)
    f2 = frobenius(e, e, c_real, c_imag)
    ee = product(e, e)
    real = np.block([[ee, product(e, b)], [product(c_real, e), ee+product(c_real, b)]])
    zero = np.zeros_like(e)
    imag = np.block([[zero, zero], [product(c_imag, e), product(c_imag, b)]])
    matrix = real, imag
    matrix_error = round_up(error*(f1+f2+error)+Fraction(8*n, SCALE))
    norm = frobenius(*matrix)
    powers = [norm]
    current = matrix
    power_error = Fraction(0)
    for _ in range(1, MAX_POWER):
        a_real, a_imag = current
        current = (product(a_real, real)-product(a_imag, imag),
                   product(a_real, imag)+product(a_imag, real))
        power_error = round_up(power_error*norm+Fraction(8*n, SCALE))
        powers.append(frobenius(*current)+power_error)
    return matrix, matrix_error, (f1+error, f2+error), powers


def tail_estimate(gain, cutoff):
    """Bounds for the core-core error, cross blocks and tail-tail blocks."""
    shear = Fraction(3, 2)*gain
    coupling = (1+gain)/2
    tail_decay = (cutoff+Fraction(1, 2))**2-Fraction(1, 12)-shear/2
    gap = tail_decay-Fraction(1, 4)
    t = TIME_UPPER
    propagation = DAMPING_FACTOR*(1+shear*t)
    core = coupling**2/gap*DAMPING_FACTOR*(t+shear*t**2+shear**2*t**3/6)
    cross = coupling/gap*propagation
    # Since pi > 3, e^(-pi*a) <= 2/(3*a)^2.
    tail = Fraction(2)/(3*tail_decay)**2+(coupling/gap)**2*propagation
    return core, cross, tail, propagation


def interval_bound(left, right, width, tail_bounds):
    core, cross, tail, propagation = tail_bounds
    # Cauchy's estimate on radius 3/8 with e^(2*pi*(3/8)^2) < 3.
    interpolation = Fraction(128, 3)*width**2/8
    f1, f2 = (max(left[2][i], right[2][i])+interpolation*propagation for i in range(2))
    offdiagonal = cross*(max(f1, f2)+core+tail)
    tail_error = offdiagonal + max(core*(f1+f2)+core**2+cross**2, cross**2+tail**2)
    difference = frobenius(*(a-b for a, b in zip(left[0], right[0], strict=True)))
    error = round_up(tail_error+interpolation*propagation**2+max(left[1], right[1])+difference/2)
    norm = max(left[3][0], right[3][0])
    for p, (bound_left, bound_right) in enumerate(zip(left[3], right[3], strict=True), 1):
        bound = round_up(max(bound_left, bound_right)+(norm+error)**p-norm**p)
        if bound < 1:
            return p, bound, error
    return None


def certify_switch(gain):
    """Return a strict all-fiber bound, or None if the finite certificate fails."""
    gain = Fraction(gain)
    if not 0 < gain <= 2:
        raise ValueError("outside the reviewed F3 gain range")
    if PI.lo <= Fraction(157, 50)*INTERVAL_SCALE or PI.hi >= TIME_UPPER*INTERVAL_SCALE:
        raise ArithmeticError("F3 checked pi interval falls outside the analytic bounds")
    # e^(pi/4) > 50/23 by the first six terms at pi/4 > 157/200.
    x = Fraction(157, 200)
    if sum(x**k/factorial(k) for k in range(6)) <= 1/DAMPING_FACTOR:
        raise ArithmeticError("F3 damping estimate did not verify")

    @cache
    def evaluate(xi):
        return monodromy_data(gain, xi, CUTOFF)

    tail_bounds = tail_estimate(gain, CUTOFF)
    remaining = [(Fraction(0), Fraction(1, 2), 0)]
    intervals = []
    while remaining:
        lo, hi, depth = remaining.pop()
        bound = interval_bound(evaluate(lo), evaluate(hi), hi-lo, tail_bounds)
        if bound is not None:
            power, upper, error = bound
            intervals.append(dict(interval=[str(lo), str(hi)], power=power,
                                  power_norm_upper=str(upper), approximation_error=str(error)))
            continue
        if depth >= MAX_DEPTH:
            return None
        mid = (lo+hi)/2
        remaining.extend(((mid, hi, depth+1), (lo, mid, depth+1)))

    return dict(method="FULL_BLOCH_MONODROMY_POWER", cutoff=CUTOFF, fixed_point_scale=SCALE,
                reflection="complex conjugation", scope="all Bloch fibers in [-1/2,1/2]",
                segment_tail_block_bounds=[str(bound) for bound in tail_bounds[:3]],
                intervals=intervals,
                proof="docs/bcf-extension.md#f3-full-bloch-floquet-certificate")
