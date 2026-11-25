"""The second wave of stefan_robin.py for r = 0 (Neumann condition); it exists for kappa > 1.

U = exp(m z)/m with m = kappa - 1 and speed c = (1 - m^2)/m.
Known result: mu^2 + mu/m - kappa = 0, lambda = (mu - m)(mu + 1/m).
For kappa = 6:   c = -4.8, no eigenvalue to the right of the essential spectrum: stable.
For kappa = 1.5: c = 1.5,  lambda = 0.209431: unstable.
The two waves exchange stability at kappa = 2.
"""
import runpy
from pathlib import Path

globals().update({key: value for key, value in
                  runpy.run_path(str(Path(__file__).with_name("stefan_robin.py"))).items()
                  if not key.startswith("__")})

m = kappa - 1
# The guess is 10 percent away from the exact speed and from the exact decay length.
wave = dict(wave, speed_guess=0.9 * (1 - m ** 2) / m, width=1.1 / m)
