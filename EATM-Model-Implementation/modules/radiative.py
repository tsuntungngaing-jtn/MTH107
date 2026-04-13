"""
Radiative exchange corrections for high-density transit microclimates.

The module implements asymptotic interpersonal view factors, wall occlusion,
and the Stefan-Boltzmann mean radiant temperature T_mr' used in the EATM
radiative network. When clothing temperature is unknown a priori, the coupled
balance should be solved with ``utils.solvers.solve_clothing_surface_temperature_coupled``.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    MAX_OCCUPANT_DENSITY_PER_M2,
    VIEW_FACTOR_INTERPERSONAL_RATE_KP,
    VIEW_FACTOR_MAX_PP,
    VIEW_FACTOR_WALL_BASELINE,
    VIEW_FACTOR_WALL_DECAY_KW,
)


def _density_clipped(density_per_m2: float) -> float:
    """GB 50157 / report crowding bound [0, 9] pass/m^2."""

    return min(MAX_OCCUPANT_DENSITY_PER_M2, max(0.0, float(density_per_m2)))


@dataclass(frozen=True)
class RadiativeOutputs:
    """Density-dependent radiative descriptors for coupling to PMV solvers."""

    f_pp: float
    f_p_wall: float
    t_mr_prime_celsius: float


def interpersonal_view_factor(density_per_m2: float) -> float:
    """
    Human-to-human view factor F_p-p(D) with asymptotic saturation.

    The report specifies a saturating interpersonal field; the implemented
    smooth form is::

        F_p-p = F_max * (1 - exp(-k_p * D)),

    which is zero at zero density and approaches F_max for large crowding.
    """

    d = _density_clipped(density_per_m2)
    return VIEW_FACTOR_MAX_PP * (1.0 - math.exp(-VIEW_FACTOR_INTERPERSONAL_RATE_KP * d))


def wall_view_factor(density_per_m2: float) -> float:
    """
    Passenger-to-wall view factor with exponential occlusion by neighbors.

        F_p-wall = F_wall,0 * exp(-k_omega * D).
    """

    d = _density_clipped(density_per_m2)
    return VIEW_FACTOR_WALL_BASELINE * math.exp(-VIEW_FACTOR_WALL_DECAY_KW * d)


def clip_view_factors(f_pp: float, f_pwall: float, *, air_channel_floor: float = 1e-3) -> Tuple[float, float]:
    """
    Ensure F_pp + F_pwall <= 1 - eps so the air radiance channel stays positive.

    If the sum violates the inequality, both factors are scaled uniformly.
    """

    if f_pp < 0.0 or f_pwall < 0.0:
        raise ValueError("View factors must be non-negative.")
    total = f_pp + f_pwall
    limit = 1.0 - air_channel_floor
    if total <= limit:
        return f_pp, f_pwall
    scale = limit / total
    return f_pp * scale, f_pwall * scale


def mean_radiant_temperature_corrected_celsius(
    t_air_celsius: float,
    t_wall_celsius: float,
    t_neighbor_clothing_celsius: float,
    f_pp: float,
    f_pwall: float,
) -> float:
    """
    Stefan-Boltzmann mean radiant temperature (deg C) from radiance weighting.

        T_mr' = (F_pp * T_cl,n^4 + F_pwall * T_wall^4 + (1 - F_pp - F_pwall) * T_a^4)^(1/4) - 273.15,

    where all temperatures on the right-hand side are converted to kelvin before
    raising to the fourth power.
    """

    f_pp_c, f_pw_c = clip_view_factors(f_pp, f_pwall)
    ta_k = t_air_celsius + 273.15
    tw_k = t_wall_celsius + 273.15
    tcl_n_k = t_neighbor_clothing_celsius + 273.15
    tmr_k4 = (
        f_pp_c * tcl_n_k**4
        + f_pw_c * tw_k**4
        + (1.0 - f_pp_c - f_pw_c) * ta_k**4
    )
    return tmr_k4**0.25 - 273.15


def radiative_bundle_from_density(
    density_per_m2: float,
    t_air_celsius: float,
    t_wall_celsius: float,
    t_neighbor_clothing_celsius: float,
) -> RadiativeOutputs:
    """
    Convenience wrapper returning view factors and T_mr' for a symmetric crowd.

    Neighboring passengers are assumed to share the target clothing temperature
    proxy ``t_neighbor_clothing_celsius`` (often lagged from the previous Newton
    iterate or an explicit symmetry assumption).
    """

    f_pp = interpersonal_view_factor(density_per_m2)
    f_pw = wall_view_factor(density_per_m2)
    f_pp_c, f_pw_c = clip_view_factors(f_pp, f_pw)
    t_mr_p = mean_radiant_temperature_corrected_celsius(
        t_air_celsius, t_wall_celsius, t_neighbor_clothing_celsius, f_pp_c, f_pw_c
    )
    return RadiativeOutputs(f_pp=f_pp_c, f_p_wall=f_pw_c, t_mr_prime_celsius=t_mr_p)


__all__ = [
    "RadiativeOutputs",
    "clip_view_factors",
    "interpersonal_view_factor",
    "wall_view_factor",
    "mean_radiant_temperature_corrected_celsius",
    "radiative_bundle_from_density",
]
