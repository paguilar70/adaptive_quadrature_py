"""Adaptive Simpson's rule integration.

The implementation subdivides the interval recursively: an interval is only
accepted once the Simpson estimate has converged to the requested tolerance.
This keeps the number of function evaluations low for smooth functions while
still resolving sharp features.
"""

from __future__ import annotations

from typing import Callable, Tuple

__all__ = ["integrate", "QuadratureError"]


class QuadratureError(Exception):
    """Raised when the adaptive refinement cannot reach the tolerance.

    This happens when ``max_depth`` is exhausted before the local error
    estimate drops below the requested tolerance. It is a real failure, not a
    warning: silently returning a less accurate value would hide the problem
    from the caller.
    """


# A depth of 30 allows up to 2**30 subdivisions in principle. In practice the
# recursion stops far earlier because each level halves the error estimate for
# smooth integrands. The cap exists only to bound pathological functions.
_DEFAULT_MAX_DEPTH = 30


def _simpson(f: Callable[[float], float], a: float, b: float,
             fa: float, fm: float, fb: float) -> float:
    """Composite Simpson estimate on [a, b] given the endpoint and midpoint values."""
    return (b - a) * (fa + 4.0 * fm + fb) / 6.0


def _adaptive(f: Callable[[float], float], a: float, b: float,
              fa: float, fm: float, fb: float, whole: float,
              tol: float, depth: int, max_depth: int) -> Tuple[float, float]:
    """Recursively refine [a, b].

    Returns ``(value, error_estimate)``. The error estimate is the absolute
    difference between the coarse and refined Simpson estimates, which is the
    standard practical bound used by adaptive Simpson implementations.
    """
    mid = 0.5 * (a + b)
    left_mid = 0.5 * (a + mid)
    right_mid = 0.5 * (mid + b)

    flm = f(left_mid)
    frm = f(right_mid)

    left = _simpson(f, a, mid, fa, flm, fm)
    right = _simpson(f, mid, b, fm, frm, fb)
    refined = left + right

    # The classic Richardson extrapolation factor for Simpson's rule is 15.
    err = abs(refined - whole) / 15.0

    if depth >= max_depth:
        # Even at the depth limit we return the best estimate we have; the
        # caller decides whether the accumulated error is acceptable.
        return refined, err

    if err <= tol:
        return refined, err

    half_tol = 0.5 * tol
    left_val, left_err = _adaptive(
        f, a, mid, fa, flm, fm, left, half_tol, depth + 1, max_depth
    )
    right_val, right_err = _adaptive(
        f, mid, b, fm, frm, fb, right, half_tol, depth + 1, max_depth
    )
    return left_val + right_val, left_err + right_err


def integrate(f: Callable[[float], float], a: float, b: float,
              tol: float = 1e-10, max_depth: int = _DEFAULT_MAX_DEPTH) -> float:
    """Integrate a scalar function on ``[a, b]`` with adaptive Simpson's rule.

    Parameters
    ----------
    f:
        A callable taking a single float and returning a float. It must be
        defined on the closed interval; no smoothing or extrapolation is done.
    a, b:
        Integration limits. ``a`` may be greater than ``b``; the result is
        negated to preserve the orientation of the integral.
    tol:
        Target absolute error. The routine refines until the estimated error
        is at or below this value, or until ``max_depth`` is reached.
    max_depth:
        Maximum recursion depth. Raising this permits more refinement for
        difficult integrands at the cost of more function evaluations.

    Returns
    -------
    float
        The estimated value of the integral.

    Raises
    ------
    QuadratureError
        If the estimated error still exceeds ``tol`` after ``max_depth``
        levels of refinement. The message reports the achieved error so the
        caller can choose a looser tolerance or a larger depth.
    ValueError
        If ``tol`` is not positive or ``max_depth`` is negative.
    """
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")

    if a == b:
        return 0.0

    # Swap and negate so the recursion always works on an increasing interval.
    # This keeps the sign bookkeeping in one place instead of inside every
    # recursive call.
    sign = 1.0
    if a > b:
        a, b = b, a
        sign = -1.0

    fa = f(a)
    fb = f(b)
    mid = 0.5 * (a + b)
    fm = f(mid)

    whole = _simpson(f, a, b, fa, fm, fb)
    value, err = _adaptive(f, a, b, fa, fm, fb, whole, tol, 0, max_depth)

    if err > tol:
        raise QuadratureError(
            "adaptive refinement did not reach tolerance: "
            f"estimated error {err:.3e} > tol {tol:.3e} at max_depth {max_depth}"
        )

    return sign * value
