"""F4 smoothly forced periodic orbit with Hill-type parametric resonance."""
from fractions import Fraction
from math import factorial

from .e3_certificate import Interval, SCALE
from .extension_contract import outcome, specification, validate_instance
from .f3_certificate import PI
from .families import rational


STEPS = 32
ORDER = 18
DAMPING = Fraction(1, 8)
ZERO = Interval.point(0)
ONE = Interval.point(1)


def parametric_orbit(stiffness, modulation, seed):
    stiffness, modulation = map(rational, (stiffness, modulation))
    if not Fraction(3, 4) <= stiffness <= Fraction(5, 4) or not Fraction(1, 4) <= modulation <= Fraction(3, 4):
        raise ValueError("outside the reviewed F4 stiffness or modulation range")
    problem = dict(
        family_id="F4", structure_id="F4.smooth-damped-hill-resonance", template_version="1",
        parameters=dict(stiffness=str(stiffness), modulation=str(modulation)),
        equations=["X_t=X_xx-(f_xx/f)*X+f*r'(t)+B(t)*(X-f*r(t))",
                   "f=2+cos(x)+sin(2*x)/4", "r=(cos(2*t),sin(2*t))",
                   "B=[[-1/8,1],[-k(t),-1/8]]",
                   "k(t)=stiffness+modulation*(cos(2*t)+sin(4*t)/4)"],
        target_wave=dict(profile="f(x)*r(t)", phase="prescribed time phase",
                         space_period="2*pi", time_period="pi"),
        eigenproblem=dict(space="unweighted L2(R;C^2)", domain="H2 for the linear evolution",
                          bloch_zone="[-1/2,1/2]",
                          question="Does the full-line time monodromy have spectrum outside the unit disk?"))
    return specification(problem, seed)


def validate_parametric_orbit(item):
    return validate_instance(item, parametric_orbit)


def symmetric_error(radius):
    upper = Interval.point(radius).hi
    return Interval(-upper, upper)


def trig_at(argument):
    """Reduce to a small angle before outward-rounded Taylor evaluation."""
    halvings = 0
    half = Interval.point(Fraction(1, 2))
    while max(abs(argument.lo), abs(argument.hi)) > SCALE//2:
        argument = half*argument
        halvings += 1
    order = 16
    square = argument.square()
    cosine = ZERO
    sine = ZERO
    for n in range(order, -1, -1):
        cosine = cosine*square + Interval.point(Fraction((-1)**n, factorial(2*n)))
        sine = sine*square + Interval.point(Fraction((-1)**n, factorial(2*n+1)))
    first = 2*order+1
    cosine = cosine+symmetric_error(Fraction(1, 2**first*factorial(first)))
    sine = argument*sine+symmetric_error(Fraction(1, 2**(first+1)*factorial(first+1)))
    for _ in range(halvings):
        cosine, sine = cosine.square()-sine.square(), Interval.point(2)*cosine*sine
    return cosine, sine


def coefficient_series(start, stiffness, modulation):
    cos2, sin2 = trig_at(Interval.point(2)*start)
    cos4, sin4 = trig_at(Interval.point(4)*start)
    cos_cycle = (cos2, -sin2, -cos2, sin2)
    sin_cycle = (sin4, cos4, -sin4, -cos4)
    coefficients = [
        Interval.point(modulation*Fraction(2**n, factorial(n)))*cos_cycle[n % 4]
        + Interval.point(modulation*Fraction(4**n, 4*factorial(n)))*sin_cycle[n % 4]
        for n in range(ORDER)
    ]
    coefficients[0] = coefficients[0]+Interval.point(stiffness)
    return coefficients


def local_column(coefficients, step, initial):
    x, y = [initial[0]], [initial[1]]
    for n in range(ORDER):
        inverse = Interval.point(Fraction(1, n+1))
        x.append(inverse*y[n])
        y.append(-inverse*sum((coefficients[j]*x[n-j] for j in range(n+1)), ZERO))
    # On the complex radius-1/2 disk the generator has infinity norm <= 6,
    # so the normalized fundamental solution is bounded by e^3 < 27.
    remainder = symmetric_error(Fraction(135, 4*5**(ORDER+1)))

    def evaluate(series):
        value = ZERO
        for coefficient in reversed(series):
            value = value*step + coefficient
        return value+remainder

    return evaluate(x), evaluate(y)


def damping_threshold():
    """Enclose exp(pi/8)+exp(-pi/8) with a positive even series."""
    argument = PI*Interval.point(DAMPING)
    if argument.hi >= SCALE//2:
        raise ArithmeticError("F4 damping argument outside the even-series bound")
    square = argument.square()
    value = ZERO
    for n in range(ORDER, -1, -1):
        value = value*square+Interval.point(Fraction(2, factorial(2*n)))
    first = 2*ORDER+2
    tail = Fraction(4, 2**first*factorial(first))
    return value+Interval(0, Interval.point(tail).hi)


def certify_parametric(stiffness, modulation):
    stiffness, modulation = map(rational, (stiffness, modulation))
    if not Fraction(3, 4) <= stiffness <= Fraction(5, 4) or not Fraction(1, 4) <= modulation <= Fraction(3, 4):
        raise ValueError("outside the reviewed F4 stiffness or modulation range")
    step = PI*Interval.point(Fraction(1, STEPS))
    if step.hi*10 >= SCALE:
        raise ArithmeticError("F4 time step outside the Taylor remainder bound")
    a, b, c, d = ONE, ZERO, ZERO, ONE
    for i in range(STEPS):
        start = PI*Interval.point(Fraction(i, STEPS))
        coefficients = coefficient_series(start, stiffness, modulation)
        e, g = local_column(coefficients, step, (ONE, ZERO))
        f, h = local_column(coefficients, step, (ZERO, ONE))
        a, b, c, d = e*a+f*c, e*b+f*d, g*a+h*c, g*b+h*d

    trace = a+d
    threshold = damping_threshold()
    trace_upper = max(abs(trace.lo), abs(trace.hi))
    trace_lower = trace.zero_distance()
    if trace_upper < threshold.lo:
        label, margin = "STABLE", threshold.lo-trace_upper
    elif trace_lower > threshold.hi:
        label, margin = "UNSTABLE", trace_lower-threshold.hi
    else:
        label, margin = "UNCERTAIN", 0

    def endpoints(interval):
        return [str(Fraction(edge, SCALE)) for edge in (interval.lo, interval.hi)]

    return dict(method="HILL_TRACE_INTERVAL_TAYLOR", label=label, fixed_point_scale=SCALE,
                time_steps=STEPS, taylor_order=ORDER, undamped_trace_interval=endpoints(trace),
                damped_threshold_interval=endpoints(threshold),
                trace_threshold_margin_lower=str(Fraction(margin, SCALE)),
                scope="all Bloch fibers and full-line monodromy via the spatial semigroup factor",
                proof="docs/f4-parametric-resonance.md")


def solve_parametric(item):
    p = validate_parametric_orbit(item)["parameters"]
    cert = certify_parametric(**p)
    label = cert["label"]
    if label == "UNCERTAIN":
        return dict(outcome(reason="The verified Hill trace interval overlaps the damping threshold."),
                    certificate=cert)
    return dict(outcome(label, "RIGOROUSLY_CERTIFIED",
                        reason="The validated Hill trace lies strictly on one side of the damping threshold."),
                certificate=cert)
