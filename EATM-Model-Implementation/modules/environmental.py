"""
Environmental mapping from nominal HVAC fields to local microclimate quantities.

The module implements airflow attenuation with occupant density, ISO-style
convective coefficients tied to local velocity, Magnus-type vapor pressures,
crowding-induced humidity spikes coupled to the transient metabolic trajectory,
and clothing insulation corrections for compression and sweat uptake.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    CLOTHING_COMPRESSION_K,
    CLOTHING_WETNESS_DECAY_ALPHA,
    CRITICAL_SKIN_WETTEDNESS,
    FLOW_ATTENUATION_GAMMA,
    METABOLIC_TRAIN_REFERENCE_MET,
    METABOLIC_VAPOR_COUPLING_ETA,
    MIN_LOCAL_AIR_VELOCITY_MS,
)


@dataclass(frozen=True)
class EnvironmentalOutputs:
    """Local microclimate scalars passed to heat-balance and PMV routines."""

    local_air_velocity_ms: float
    flow_attenuation_factor: float
    convective_coefficient_w_m2_k: float
    vapor_pressure_env_pa: float
    vapor_pressure_local_pa: float
    clothing_compression_factor: float
    clothing_wetness_factor: float
    corrected_clothing_insulation_clo: float


def flow_ventilation_factor(
    nominal_velocity_ms: float,
    density_per_m2: float,
    gamma: float = FLOW_ATTENUATION_GAMMA,
    epsilon_floor_ms: float = MIN_LOCAL_AIR_VELOCITY_MS,
) -> float:
    """
    Ventilation attenuation factor linking nominal and local velocities.

    The report gives both a rational form and an exponential approximation; the
    exponential mapping is adopted here for numerical robustness::

        V_local ≈ epsilon + V * exp(-gamma * D),

    and the returned factor is f_vent = V_local / V when V > 0.
    """

    if nominal_velocity_ms < 0.0:
        raise ValueError("nominal_velocity_ms must be non-negative.")
    if density_per_m2 < 0.0:
        raise ValueError("density_per_m2 must be non-negative.")
    v_local = epsilon_floor_ms + nominal_velocity_ms * math.exp(
        -gamma * density_per_m2
    )
    if nominal_velocity_ms == 0.0:
        return 1.0
    return max(0.0, min(1.0, v_local / nominal_velocity_ms))


def local_air_velocity_ms(
    nominal_velocity_ms: float,
    density_per_m2: float,
    gamma: float = FLOW_ATTENUATION_GAMMA,
    epsilon_floor_ms: float = MIN_LOCAL_AIR_VELOCITY_MS,
) -> float:
    """Local air speed (m/s) after crowding attenuation."""

    if nominal_velocity_ms < 0.0:
        raise ValueError("nominal_velocity_ms must be non-negative.")
    if density_per_m2 < 0.0:
        raise ValueError("density_per_m2 must be non-negative.")
    return epsilon_floor_ms + nominal_velocity_ms * math.exp(
        -gamma * density_per_m2
    )


def convective_heat_transfer_coefficient_w_m2_k(local_velocity_ms: float) -> float:
    """
    Forced-convection coefficient on the nude/clothing outer surface (W/(m^2 K)).

        h_c,v = 8.3 * V_local^0.6  (V in m/s),

    consistent with the ISO 7730 correlation family used in the report.
    """

    if local_velocity_ms < 0.0:
        raise ValueError("local_velocity_ms must be non-negative.")
    return 8.3 * (local_velocity_ms**0.6)


def saturation_vapor_pressure_magnus_pa(t_celsius: float) -> float:
    """
    Saturation vapor pressure (Pa) via the Tetens/Magnus form used in ASHRAE references.

    The manuscript cites an exponential curve fit; for SI pascals this implementation
    adopts the conventional Tetens expression::

        p_sat = 610.78 * exp(17.27 * T / (T + 237.3)),

    which remains aligned with ISO 7730 humidity conversions while avoiding unit
    ambiguity in the typeset coefficients.
    """

    denom = t_celsius + 237.3
    if abs(denom) < 1e-6:
        raise ValueError("Invalid temperature for Magnus denominator.")
    return 610.78 * math.exp(17.27 * t_celsius / denom)


def ambient_vapor_pressure_pa(t_celsius: float, relative_humidity_0_1: float) -> float:
    """Partial pressure of water vapor (Pa) from relative humidity (0-1 scale)."""

    if not 0.0 <= relative_humidity_0_1 <= 1.0:
        raise ValueError("relative_humidity_0_1 must lie in [0, 1].")
    return relative_humidity_0_1 * saturation_vapor_pressure_magnus_pa(t_celsius)


def local_vapor_pressure_pa(
    p_vap_env_pa: float,
    density_per_m2: float,
    metabolic_rate_met: float,
    m_train_reference_met: float = METABOLIC_TRAIN_REFERENCE_MET,
    eta: float = METABOLIC_VAPOR_COUPLING_ETA,
) -> float:
    """
    Local vapor pressure (Pa) with metabolic-density coupling.

        p_vap,local = p_vap,env * (1 + eta * D * M(t) / M_train).

    The coupling factor eta is calibrated in the report for peak-density humidity
    excursions (~20% at nine persons per square meter).
    """

    if p_vap_env_pa < 0.0:
        raise ValueError("p_vap_env_pa must be non-negative.")
    if density_per_m2 < 0.0:
        raise ValueError("density_per_m2 must be non-negative.")
    if m_train_reference_met <= 0.0:
        raise ValueError("m_train_reference_met must be positive.")
    factor = 1.0 + eta * density_per_m2 * (metabolic_rate_met / m_train_reference_met)
    return p_vap_env_pa * max(0.0, factor)


def clothing_compression_factor(
    density_per_m2: float,
    k_compression: float = CLOTHING_COMPRESSION_K,
) -> float:
    """
    Crowding compression factor f_clo(D) reducing effective insulation.

    A simple saturating rational form keeps the factor bounded::

        f_clo = 1 / (1 + k * D).
    """

    if density_per_m2 < 0.0:
        raise ValueError("density_per_m2 must be non-negative.")
    return 1.0 / (1.0 + k_compression * density_per_m2)


def clothing_wetness_insulation_factor(
    skin_wettedness: float,
    omega_crit: float = CRITICAL_SKIN_WETTEDNESS,
    alpha_de: float = CLOTHING_WETNESS_DECAY_ALPHA,
) -> float:
    """
    Moisture-dependent clothing resistance modifier f_wet(omega).

        f_wet = min(1.0, max(0.6, exp(-alpha_de * (omega - omega_crit)))).
    """

    if not 0.0 <= skin_wettedness <= 1.0:
        raise ValueError("skin_wettedness must lie in [0, 1].")
    raw = math.exp(-alpha_de * (skin_wettedness - omega_crit))
    return max(0.6, min(1.0, raw))


def corrected_clothing_insulation_clo(
    icl_base_clo: float,
    density_per_m2: float,
    skin_wettedness: float,
) -> float:
    """
    Effective clothing insulation (clo) after compression and sweat uptake.

        I_cl,final = I_cl,0 * f_clo(D) * f_wet(omega).
    """

    if icl_base_clo <= 0.0:
        raise ValueError("icl_base_clo must be positive.")
    return (
        icl_base_clo
        * clothing_compression_factor(density_per_m2)
        * clothing_wetness_insulation_factor(skin_wettedness)
    )


def environmental_bundle(
    nominal_velocity_ms: float,
    density_per_m2: float,
    t_air_celsius: float,
    relative_humidity_0_1: float,
    metabolic_rate_met: float,
    icl_base_clo: float,
    skin_wettedness: float,
) -> EnvironmentalOutputs:
    """Aggregate microclimate mapping for a single state vector."""

    v_loc = local_air_velocity_ms(nominal_velocity_ms, density_per_m2)
    f_vent = flow_ventilation_factor(nominal_velocity_ms, density_per_m2)
    h_cv = convective_heat_transfer_coefficient_w_m2_k(v_loc)
    p_env = ambient_vapor_pressure_pa(t_air_celsius, relative_humidity_0_1)
    p_loc = local_vapor_pressure_pa(p_env, density_per_m2, metabolic_rate_met)
    f_clo = clothing_compression_factor(density_per_m2)
    f_wet = clothing_wetness_insulation_factor(skin_wettedness)
    icl_corr = icl_base_clo * f_clo * f_wet
    return EnvironmentalOutputs(
        local_air_velocity_ms=v_loc,
        flow_attenuation_factor=f_vent,
        convective_coefficient_w_m2_k=h_cv,
        vapor_pressure_env_pa=p_env,
        vapor_pressure_local_pa=p_loc,
        clothing_compression_factor=f_clo,
        clothing_wetness_factor=f_wet,
        corrected_clothing_insulation_clo=icl_corr,
    )


__all__ = [
    "EnvironmentalOutputs",
    "ambient_vapor_pressure_pa",
    "clothing_compression_factor",
    "clothing_wetness_insulation_factor",
    "convective_heat_transfer_coefficient_w_m2_k",
    "corrected_clothing_insulation_clo",
    "environmental_bundle",
    "flow_ventilation_factor",
    "local_air_velocity_ms",
    "local_vapor_pressure_pa",
    "saturation_vapor_pressure_magnus_pa",
]
