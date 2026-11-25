"""FitzHugh-Nagumo equations: the slow pulse (same equations and parameters as fhn_fast_pulse.py).

The fast and the slow pulse lie on one branch.  The branch has a fold in eps:
both pulses exist below the fold value, none above.  The slow pulse is reached
by continuation: start at the fast pulse, increase eps, pass the fold, and
return to the eps of this file.
Known result: the slow pulse is unstable (Flores 1991).
"""
import runpy
from pathlib import Path

globals().update({key: value for key, value in
                  runpy.run_path(str(Path(__file__).with_name("fhn_fast_pulse.py"))).items()
                  if not key.startswith("__")})

# The guess of the fast-pulse file is valid at `start`; `direction` is the sign of the first step in eps.
wave = dict(wave, continuation=dict(parameter=eps, start=parameters[eps], direction=+1))
