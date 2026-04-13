"""
Shared numerical utilities (nonlinear solvers, iteration guards).
"""

from __future__ import annotations

from .solvers import newton_raphson_1d, solve_clothing_surface_temperature_coupled

__all__ = ["newton_raphson_1d", "solve_clothing_surface_temperature_coupled"]
