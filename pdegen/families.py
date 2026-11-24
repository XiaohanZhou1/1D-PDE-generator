"""Physical PDEs in laboratory coordinates, independently transcribed from design v1."""
from fractions import Fraction as Q
from .algebra import symbol, poly, s, t


def P(j, u):
    return j*j*u**j - j*(j+1)*u**(j+2)


def rational(x):
    if isinstance(x, bool) or not isinstance(x, (str, int, Q)):
        raise ValueError("use an integer or exact rational string")
    return Q(x)


def a1(q, a, beta=0):
    q, a, beta = map(rational, (q, a, beta))
    if not (0 <= q <= 1 and 2 <= a <= 6 and beta in (0, Q(1,4), Q(-1,4))):
        raise ValueError("outside reviewed A1 rectangle or feedback menu")
    names = ["u", "y", "v", "w"]
    u, y, v, w = map(symbol, names)
    R, S = v-u**2, w-u**3
    diffusion = [1, 3, 1, 1+q/2]
    transport = [1+q/2, 1, 1-q/2, 1+q]
    reaction = [-P(1,u)+(q/2)*y+q*u*(3*R+2*S),
                -3*y+beta*u*u*R,
                -P(2,u)-q*u*y+(6*u*u-a)*R-2*S,
                -(1+q/2)*P(3,u)+3*q*u*u*y+(6*u*u-a+q*u)*S+2*R]
    return dict(family="A1", names=names, params=dict(q=q, a=a, beta=beta), c=1,
                D=diffusion, A=transport, F=reaction, G=[0,6*u**3,0,0],
                wave=[s,-s*t,s*s,s**3], ends=[[0]*4,[0]*4])


def a5_fixture():
    # One contract fixture. The rho unfolding belongs to the design, fixed at zero here.
    mu, rho = Q(3), Q(0)
    u, v = map(symbol, ["u","v"])
    def H(x):
        return Q(1,2)*x*x + Q(5,24)*x**3
    hp, g, h, k = u+Q(5,8)*u*u, 1-u*u, v-H(u), mu*(1+Q(1,2)*u)
    flux = [g+k*h, hp*g+(hp*k+u)*h+rho*g]
    return dict(family="A5", names=["u","v"], params=dict(mu=mu,rho=rho), c=0,
                D=[1,1], A=[0,0], F=[poly(0),poly(0)], G=[-f for f in flux],
                wave=[t,H(t)], ends=[[-1,H(-1)],[1,H(1)]])


def dispatch_limitation(model):
    if model["family"] == "A5":
        return None
    q, a, beta = (model["params"][k] for k in ("q","a","beta"))
    if beta:
        return "The built-in A1 model omits beta*u^2*(v-u^2) in the y reaction."
    fixed = {(0,4),(1,2),(1,4),(1,6),(Q(1,2),4)}
    if (q,a) not in fixed and not (Q(31,32)<=q<=1 and 2<=a<=2+Q(1,32)):
        return "No reviewed root-scouting policy at this parameter point."
    if any(Q.from_float(float(v)) != v for v in (q,a)):
        return "Native binary64 conversion would change an exact parameter."
    return None
