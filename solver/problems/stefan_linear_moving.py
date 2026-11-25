"""The second wave of stefan_linear.py (same equations and parameters; it exists for kappa < -1).

U = exp(m z) with m = -1/(1 + kappa) and speed c = (1 - m^2)/m.
Known result: one eigenvalue lambda = kappa (1 - m) if c < 1, none otherwise.
For kappa = -3:   c = 1.5,  no eigenvalue: stable.
For kappa = -1.5: c = -1.5, lambda = 1.5: unstable.
"""
import runpy
from pathlib import Path

globals().update({key: value for key, value in
                  runpy.run_path(str(Path(__file__).with_name("stefan_linear.py"))).items()
                  if not key.startswith("__")})

m = -1 / (1 + kappa)
# The guess is 10 percent away from the exact speed and from the exact decay length.
wave = dict(wave, speed_guess=0.9 * (1 - m ** 2) / m, width=1.1 / m)
