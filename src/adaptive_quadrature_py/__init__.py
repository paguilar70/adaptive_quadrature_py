"""Adaptive Simpson's rule quadrature for scalar functions.

This package exposes a single public entry point, :func:`integrate`, plus the
:class:`QuadratureError` raised when the requested tolerance cannot be met.
"""

from .core import QuadratureError, integrate

__all__ = ["integrate", "QuadratureError"]
