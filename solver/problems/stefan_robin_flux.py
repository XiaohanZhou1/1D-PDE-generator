"""stefan_robin.py with the speed of the boundary inside the boundary condition.

With ds/dt = kappa (u - 1), the condition u_x + r u = 1 + r is the same as

    u_x + (r / kappa) ds/dt = 1          at x = s(t).

`speed_symbol` names the symbol that stands for ds/dt in the conditions.
The results must equal those of stefan_robin.py (r = 1, kappa = 9: lambda = 3).
"""
import runpy
from pathlib import Path

globals().update({key: value for key, value in
                  runpy.run_path(str(Path(__file__).with_name("stefan_robin.py"))).items()
                  if not key.startswith("__")})

sdot = sp.Symbol("sdot", real=True)
free_boundary = dict(conditions=[sp.Eq(u.diff(x) + r / kappa * sdot, 1)], speed=kappa * (u - 1),
                     speed_symbol=sdot)
parameters = {kappa: 9.0, r: 1.0}
