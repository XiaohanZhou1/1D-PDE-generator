import json
import unittest

import numpy as np
import sympy as sp

from pdegen.floquet import fourier_generator, solve_spatial_orbit, spatial_orbit, validate_spatial_orbit


class SpatialFloquetTests(unittest.TestCase):
    def test_exact_manufactured_orbit(self):
        x, t = sp.symbols("x t", real=True)
        f = 2 + sp.cos(x)
        u = f * sp.Matrix([sp.cos(t), sp.sin(t)])
        diffusion = sp.diag(1, 2)
        rotation = sp.Matrix([[0, -1], [1, 0]])
        pde = diffusion * (u.diff(x, 2) - f.diff(x, 2) * u / f) + rotation * u + (f*f - u.dot(u)) * u
        self.assertEqual((u.diff(t) - pde).applyfunc(sp.simplify), sp.zeros(2, 1))

    def test_wire_format_and_coupled_monodromy(self):
        item = json.loads(json.dumps(spatial_orbit(4)))
        self.assertEqual(validate_spatial_orbit(item), item)
        altered = {**item, "target_wave": {**item["target_wave"], "temporal_period": "pi"}}
        with self.assertRaises(ValueError):
            validate_spatial_orbit(altered)

        generator, dimension = fourier_generator(0.25, 4)
        first, second = generator(0), generator(np.pi / 4)
        self.assertGreater(np.linalg.norm(first[:2, 2:4]), 0.1)
        self.assertGreater(np.linalg.norm(first @ second - second @ first), 0.1)
        self.assertEqual(first.shape, (dimension, dimension))

        result = solve_spatial_orbit(item)
        self.assertEqual(result["label"], "STABLE")
        self.assertEqual(result["evidence_grade"], "RIGOROUSLY_CERTIFIED")
        self.assertEqual(result["numerical_check"]["status"], "PASS")
        rows = {row["xi"]: row for row in result["numerical_check"]["fibers"]}
        self.assertLess(rows[0]["phase_residual"], 1e-9)
        self.assertAlmostEqual(rows[0]["fine_spectral_radius"], 1.0, places=9)
        self.assertAlmostEqual(rows[-0.25]["fine_spectral_radius"], rows[0.25]["fine_spectral_radius"], places=9)
        self.assertAlmostEqual(rows[-0.5]["fine_spectral_radius"], rows[0.5]["fine_spectral_radius"], places=9)
        self.assertLess(rows[1 / 32]["fine_spectral_radius"], 1)
