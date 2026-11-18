# adaptive_quadrature_py

Integrates a scalar function on an interval using adaptive Simpson's rule to a specified tolerance.

## Usage

```python
import math

from adaptive_quadrature_py import integrate, QuadratureError

value = integrate(math.sin, 0.0, math.pi)
print(value)  # ~2.0

# Tighter tolerance and a reversed interval.
value = integrate(math.exp, 1.0, 0.0, tol=1e-12)
print(value)  # ~-(e - 1)

try:
    # A discontinuous integrand cannot meet a very tight tolerance.
    integrate(lambda x: 0.0 if x < 0.5 else 1.0, 0.0, 1.0,
              tol=1e-14, max_depth=4)
except QuadratureError as exc:
    print(exc)
```

The public API is exactly two names: `integrate(f, a, b, tol=1e-10, max_depth=30)`
and `QuadratureError`.

## Why this exists

Fixed-step Simpson's rule needs the caller to guess a step size before knowing
anything about the integrand. Too coarse and the answer is wrong; too fine and
work is wasted on regions that were already flat. Adaptive refinement starts
coarse and only subdivides where the local error estimate is large, so smooth
integrals cost a handful of evaluations while a narrow peak still gets resolved.

The trade-off is that the error estimate is heuristic. The routine compares the
coarse Simpson estimate on an interval with the sum of the two half-interval
estimates and uses Richardson extrapolation (dividing by 15) to bound the error.
That bound is not a proof; it is the standard practical criterion. For an
integrand with a discontinuity the estimate never settles, so the routine stops
at `max_depth` and raises `QuadratureError` instead of returning a number it
cannot vouch for.

## The edge you will hit

If `f` is discontinuous or has a singularity inside `[a, b]`, the error estimate
will not converge and you will get a `QuadratureError` once `max_depth` is
exhausted. That is deliberate: the library will not silently hand back a value
that violates the tolerance you asked for. Either loosen `tol`, raise
`max_depth`, or split the integral at the discontinuity and integrate the smooth
pieces separately.

Reversed limits are supported: `integrate(f, b, a)` returns the negation of
`integrate(f, a, b)`. An empty interval (`a == b`) returns `0.0` without calling
`f`.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

