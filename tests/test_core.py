"""Tests for adaptive_quadrature_py.core.

Every test uses a deterministic integrand and compares against a closed-form
value with an explicit tolerance. No wall-clock time is involved.
"""

import math
import unittest

from adaptive_quadrature_py import QuadratureError, integrate


class TestIntegrate(unittest.TestCase):
    def assertClose(self, got, want, tol):
        self.assertLessEqual(abs(got - want), tol,
                             msg=f"got {got!r}, want {want!r} (tol {tol!r})")

    def test_constant(self):
        # Integral of 3 over [0, 2] is 6. A constant is exactly representable
        # by Simpson's rule, so the tolerance can be tight.
        self.assertClose(integrate(lambda x: 3.0, 0.0, 2.0), 6.0, 1e-12)

    def test_linear(self):
        # Integral of x over [0, 1] is 0.5. Simpson is exact for cubics.
        self.assertClose(integrate(lambda x: x, 0.0, 1.0), 0.5, 1e-12)

    def test_quadratic(self):
        # Integral of x**2 over [0, 3] is 9.
        self.assertClose(integrate(lambda x: x * x, 0.0, 3.0), 9.0, 1e-10)

    def test_sine(self):
        # Integral of sin over [0, pi] is 2.
        self.assertClose(integrate(math.sin, 0.0, math.pi), 2.0, 1e-10)

    def test_gaussian_like_bump(self):
        # A smooth bump with a known integral: int_0^1 exp(x) dx = e - 1.
        self.assertClose(integrate(math.exp, 0.0, 1.0), math.e - 1.0, 1e-10)

    def test_reversed_limits(self):
        # Swapping limits must negate the result.
        forward = integrate(math.sin, 0.0, math.pi)
        backward = integrate(math.sin, math.pi, 0.0)
        self.assertClose(backward, -forward, 1e-12)

    def test_empty_interval(self):
        # a == b gives exactly zero, regardless of the integrand.
        self.assertEqual(integrate(lambda x: 1.0 / (x - 5.0), 2.0, 2.0), 0.0)

    def test_sharp_peak_requires_refinement(self):
        # A narrow peak at x = 0.5 with width 1e-3. The adaptive scheme must
        # subdivide to resolve it. The exact integral of exp(-((x - 0.5)/w)^2)
        # over the whole real line is sqrt(pi) * w, so over [0, 1] it is
        # sqrt(pi) * w to within the tails.
        width = 1e-3

        def peak(x):
            return math.exp(-((x - 0.5) / width) ** 2)

        expected = math.sqrt(math.pi) * width
        self.assertClose(integrate(peak, 0.0, 1.0, tol=1e-8), expected, 1e-8)

    def test_oscillatory(self):
        # sin(50x) over [0, pi]. The exact integral is (1 - cos(50 pi)) / 50.
        expected = (1.0 - math.cos(50.0 * math.pi)) / 50.0
        self.assertClose(integrate(lambda x: math.sin(50.0 * x), 0.0, math.pi,
                                   tol=1e-9), expected, 1e-8)

    def test_tolerance_is_respected(self):
        # A loose tolerance should still give a usable answer.
        self.assertClose(integrate(math.sin, 0.0, math.pi, tol=1e-3), 2.0, 1e-3)

    def test_raises_when_depth_exhausted(self):
        # A discontinuous integrand forces refinement past any reasonable
        # depth, so the routine must report failure rather than lie.
        def step(x):
            return 0.0 if x < 0.5 else 1.0

        with self.assertRaises(QuadratureError):
            integrate(step, 0.0, 1.0, tol=1e-14, max_depth=4)

    def test_zero_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            integrate(math.sin, 0.0, 1.0, tol=0.0)

    def test_negative_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            integrate(math.sin, 0.0, 1.0, tol=-1e-6)

    def test_negative_max_depth_rejected(self):
        with self.assertRaises(ValueError):
            integrate(math.sin, 0.0, 1.0, max_depth=-1)

    def test_zero_max_depth_still_returns_when_accurate(self):
        # With no refinement allowed, a function Simpson integrates exactly
        # should still succeed.
        self.assertClose(integrate(lambda x: x * x, 0.0, 1.0, max_depth=0),
                         1.0 / 3.0, 1e-12)


if __name__ == "__main__":
    unittest.main()
