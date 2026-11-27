"""Exact-checked Fourier energy certificates for the B3 Bloch fibers."""
from fractions import Fraction
from math import comb, lcm

import numpy as np
from scipy.linalg import solve, solve_continuous_lyapunov


SCALE = 1 << 36
CUTOFF = 6
MIN_NEAR_ZERO = Fraction(1, 512)
MAX_DEPTH = 12


def real_matrix(matrix):
    result = np.empty((2*matrix.shape[0], 2*matrix.shape[1]), dtype=float)
    result[0::2, 0::2] = result[1::2, 1::2] = matrix.real
    result[1::2, 0::2] = matrix.imag
    result[0::2, 1::2] = -matrix.imag
    return result


def integer_matrix(matrix):
    rounded = np.rint(matrix)
    if not np.all(np.isfinite(rounded)) or np.max(abs(rounded)) >= 2**62:
        raise OverflowError("B3 rounded matrix is outside the integer conversion range")
    return rounded.astype(np.int64).astype(object)


def exact_positive(numerator, denominator):
    """A rounded whitening matrix is a witness, checked with Python integers."""
    try:
        lower = np.linalg.cholesky(np.array(numerator, dtype=float)/denominator)
        factor = solve(lower.T, np.eye(len(lower)), check_finite=False)
        whitening = integer_matrix(SCALE*factor)
    except (ValueError, OverflowError, np.linalg.LinAlgError):
        return False
    transformed = whitening.T @ numerator @ whitening
    return all(2*transformed[i, i] > sum(abs(entry) for entry in transformed[i])
               for i in range(len(transformed)))


def polynomial_positive(coefficients, denominator, left, right):
    """Use the convex hull of the exact Bernstein coefficient matrices."""
    degree = len(coefficients)-1
    width = right-left
    for i in range(degree+1):
        weights = [sum(Fraction(comb(i, j)*comb(k, j), comb(degree, j))
                       * left**(k-j)*width**j for j in range(min(i, k)+1))
                   for k in range(degree+1)]
        common = lcm(*(weight.denominator for weight in weights))
        numerator = sum(int(weight*common)*matrix for weight, matrix in zip(weights, coefficients, strict=True))
        if not exact_positive(numerator, denominator*common):
            return False
    return True


def fourier_coefficients(damping, skew, cutoff):
    """Return exact integer matrices with a common denominator."""
    scale = 2*lcm(damping.denominator, skew.denominator)
    # All entries and partial sums of the integer assembly fit in binary64.
    if scale * (4*(cutoff+2)**2+32) >= 1 << 48:
        raise ValueError("B3 exact Fourier assembly denominator exceeds its limit")
    b, c = int(scale*damping/2), int(scale*skew/2)
    mixing = np.array([[b, -c], [c, b]])
    modes = [*range(-cutoff, cutoff+1), -cutoff-2, -cutoff-1, cutoff+1, cutoff+2]
    n = 2*len(modes)
    coeffs = [np.zeros((n, n), dtype=complex) for _ in range(3)]
    diffusion = np.diag([scale, 2*scale])
    potential = {0: -2*mixing, 2: -mixing @ np.array([[1, -1j], [-1j, -1]]),
                 -2: -mixing @ np.array([[1, 1j], [1j, -1]])}
    for i, mode in enumerate(modes):
        block = slice(2*i, 2*i+2)
        coeffs[0][block, block] = (1-mode**2)*diffusion
        coeffs[1][block, block] = -2*mode*diffusion
        coeffs[2][block, block] = -diffusion
        for j, source in enumerate(modes):
            if mode-source in potential:
                coeffs[0][block, 2*j:2*j+2] += potential[mode-source]
    return coeffs, scale, mixing


def dissipation(coeffs, metric, metric_scale, a_scale, core, tail_bound):
    """Compress the core and its interacting tail, bounding the entire tail below."""
    result = [np.zeros_like(coeffs[0]) for _ in range(len(coeffs)+len(metric)-1)]
    for i, a in enumerate(coeffs):
        for j, p in enumerate(metric):
            pa = p @ a
            result[i+j] -= pa+pa.T
    for matrix in result:
        matrix[core:, core:] = 0
    bound = tail_bound * metric_scale * a_scale
    if bound.denominator != 1:
        raise ValueError("B3 tail bound is not an integer at the proof scale")
    result[0][core:, core:] = int(bound)*np.eye(len(result[0])-core, dtype=int).astype(object)
    return result


def certify_roll(damping, skew, cutoff=CUTOFF):
    coeffs, a_scale, mixing = fourier_coefficients(damping, skew, cutoff)
    size = 2*(2*cutoff+1)
    core = 2*size
    total = 2*len(coeffs[0])
    exact_a = [integer_matrix(real_matrix(a)) for a in coeffs]
    a0, a1, a2 = [a[:size, :size]/a_scale for a in coeffs]
    k = np.zeros(size, dtype=complex)
    k[2*(cutoff-1):2*(cutoff-1)+2] = (-1j, 1)
    k[2*(cutoff+1):2*(cutoff+1)+2] = (1j, 1)
    w_integer = (k.reshape(-1, 2) @ mixing.T).ravel()
    b = int(mixing[0, 0])
    w = w_integer/b
    projection = np.outer(k, w.conj())/4
    complement = np.eye(size)-projection
    regular = a0-projection
    weights = np.repeat(2*(1+np.arange(-cutoff, cutoff+1)**2), 2)
    stable_metric = solve_continuous_lyapunov(regular.conj().T, -np.diag(weights))
    # Derivatives of the right and left vectors of the simple finite-core
    # translation eigenvalue. They only propose the metric.
    speed = np.vdot(w, a1 @ k)/4
    dk = solve(regular, -(a1 @ k-speed*k))
    dw = solve(regular.conj().T, -(a1.conj().T @ w-speed.conjugate()*w))
    dr = -(np.outer(dk, w.conj())+np.outer(k, dw.conj()))/4
    p1 = dr.conj().T @ stable_metric @ complement + complement.conj().T @ stable_metric @ dr
    p1 += (np.outer(dw, w.conj())+np.outer(w, dw.conj()))/4

    def sym_round(p):
        return integer_matrix(SCALE*real_matrix((p+p.conj().T)/2))

    w_full = np.pad(w_integer, (0, len(coeffs[0])-size))
    k_full = np.pad(k, (0, len(coeffs[0])-size))
    wk = integer_matrix(real_matrix(w_full[:, None]))
    kk = integer_matrix(real_matrix(k_full[:, None]))
    if any(np.any(product) for product in (exact_a[0]@kk, exact_a[0].T@wk)):
        raise ArithmeticError("B3 translation or adjoint kernel does not match the exact coefficients")
    scale_r = 4*b
    identity = np.eye(total, dtype=int).astype(object)
    r = scale_r*identity-kk @ wk.T
    p0_scale = SCALE*scale_r**2
    candidate = SCALE*identity.copy()
    candidate[:core, :core] = sym_round(stable_metric)
    p0_int = r.T @ candidate @ r + 4*SCALE*(wk @ wk.T)
    p1_int = np.zeros_like(identity)
    p1_int[:core, :core] = scale_r**2*sym_round(p1)
    min_tail = 2*((Fraction(cutoff)+Fraction(1, 2))**2-1-2*(damping+skew))
    if min_tail <= 0:
        return None

    h = dissipation(exact_a, (p0_int, p1_int), p0_scale, a_scale, core, min_tail)
    retained = np.ones(total, dtype=bool)
    # Remove the two real coordinates for the second component of mode +1.
    # The two real kernel columns form the identity in these coordinates.
    retained[4*(cutoff+1)+2:4*(cutoff+1)+4] = False
    change = np.column_stack((kk, identity[:, retained]))
    transformed = [change.T @ matrix @ change for matrix in h]
    if (np.any(transformed[0][:2]) or np.any(transformed[0][:, :2])
            or np.any(transformed[1][:2, :2])):
        raise ArithmeticError("B3 neutral-mode divisions do not cancel exactly")
    divided = [matrix.copy() for matrix in transformed]
    zero = np.zeros_like(identity)
    for degree, matrix in enumerate(divided):
        next_matrix = transformed[degree+1] if degree+1 < len(transformed) else zero
        next_two = transformed[degree+2] if degree+2 < len(transformed) else zero
        matrix[:2, 2:] = next_matrix[:2, 2:]
        matrix[2:, :2] = next_matrix[2:, :2]
        matrix[:2, :2] = next_two[:2, :2]

    edge = Fraction(1, 8)
    if not exact_positive(p0_int, p0_scale):
        return None
    while edge >= MIN_NEAR_ZERO:
        edge_d = edge.denominator
        if (exact_positive(edge_d*p0_int+edge.numerator*p1_int, edge_d*p0_scale)
                and polynomial_positive(divided, p0_scale*a_scale, Fraction(0), edge)):
            break
        edge /= 2
    else:
        return None

    remaining = [(edge, Fraction(1, 2), 0)]
    intervals = []
    weight_matrix = complement.conj().T @ (weights[:, None]*complement)
    while remaining:
        left, right, depth = remaining.pop()
        mid = float((left+right)/2)
        a = a0+mid*a1+mid**2*a2
        target = weight_matrix+mid**2*np.outer(w, w.conj())/4
        proposed = solve_continuous_lyapunov(a.conj().T, -target)
        p = SCALE*identity.copy()
        p[:core, :core] = sym_round(proposed)
        if exact_positive(p, SCALE):
            h = dissipation(exact_a, (p,), SCALE, a_scale, core, min_tail)
            if polynomial_positive(h, SCALE*a_scale, left, right):
                intervals.append([str(left), str(right)])
                continue
        if depth >= MAX_DEPTH:
            return None
        middle = (left+right)/2
        remaining.extend(((middle, right, depth+1), (left, middle, depth+1)))

    return dict(method="EXACT_FOURIER_ENERGY_BERNSTEIN", cutoff=cutoff, near_zero=["0", str(edge)],
                intervals=intervals, tail_bound=str(min_tail), metric_rounding_scale=SCALE,
                scope="all Bloch fibers in [-1/2,1/2]", reflection="complex conjugation",
                proof="docs/bcf-extension.md#b3-full-bloch-certificate")
