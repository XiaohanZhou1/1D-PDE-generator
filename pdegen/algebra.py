"""Small exact polynomial algebra. Monomials are sorted tuples of variable names."""
from fractions import Fraction
from math import prod


class Poly:
    def __init__(self, terms):
        self.terms = {m: Fraction(c) for m, c in terms.items() if c}

    def __add__(self, other):
        terms = self.terms.copy()
        for monomial, coefficient in poly(other).terms.items():
            terms[monomial] = terms.get(monomial, 0) + coefficient
        return Poly(terms)

    __radd__ = __add__

    def __neg__(self):
        return Poly({m: -c for m, c in self.terms.items()})

    def __sub__(self, other):
        return self + (-poly(other))

    def __rsub__(self, other):
        return poly(other) + (-self)

    def __mul__(self, other):
        terms = {}
        for m, c in self.terms.items():
            for n, d in poly(other).terms.items():
                key = tuple(sorted(m + n))
                terms[key] = terms.get(key, 0) + c*d
        return Poly(terms)

    __rmul__ = __mul__

    def __pow__(self, degree):
        if type(degree) is not int or degree < 0:
            raise ValueError("expected a nonnegative integer power")
        return prod([self] * degree, start=poly(1))

    def diff(self, var):
        terms = {}
        for m, c in self.terms.items():
            if var in m:
                i = m.index(var)
                terms[m[:i] + m[i+1:]] = c*m.count(var)
        return Poly(terms)

    def at(self, env):
        return sum((c * prod(env[v] for v in m) for m, c in self.terms.items()), 0)

    def json(self):
        return {"*".join(m) or "1": str(c) for m, c in sorted(self.terms.items())}


def poly(value):
    return value if isinstance(value, Poly) else Poly({(): value})


def symbol(name):
    return Poly({(name,): 1})


s, t = symbol("s"), symbol("t")


def reduce_wave(value):
    """Normal form modulo t^2 = 1-s^2 for s=sech(z), t=tanh(z)."""
    result = poly(0)
    for m, c in poly(value).terms.items():
        if set(m) - {"s", "t"}:
            raise ValueError("unsubstituted wave variable")
        pairs, odd = divmod(m.count("t"), 2)
        result += c * s**m.count("s") * t**odd * (1-s*s)**pairs
    return result


def dz(value):
    f = poly(value)
    return reduce_wave(-s*t*f.diff("s") + s*s*f.diff("t"))


def encode(tree):
    if isinstance(tree, dict):
        return {k: encode(v) for k, v in tree.items()}
    if isinstance(tree, (tuple, list)):
        return [encode(v) for v in tree]
    return poly(tree).json()


def evaluate(encoded, env):
    """Evaluate wire polynomials on scalars or arrays, without parsing Python."""
    return sum(float(Fraction(c)) * (1 if m == "1" else prod(env[x] for x in m.split("*")))
               for m, c in encoded.items())
