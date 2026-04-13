"""
Physiological pathway models for the EATM framework.

This module implements the transient metabolic trajectory M(t), the heat
storage relaxation S(t), skin wettedness omega, and the psychological offset
lambda(t) that couples Thermal History Intensity (THI) with a physiological
boundary-defense filter on skin moisture.

The functional forms follow the analytical expressions typeset in the EATM
report (first-order metabolic decay, exponential heat-storage relaxation,
ratio-based skin wettedness, logistic boundary defense, and multiplicative
expectancy gain).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from pythermalcomfort.models import set_tmp

# Allow imports when the project root is not on PYTHONPATH (e.g., IDE test runs).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    ACTIVE_SKIN_WETTEDNESS_WEIGHT,
    ALPHA_RETENTION_COOLING_STEP,
    ALPHA_RETENTION_WARMING_STEP,
    BASELINE_SKIN_WETTEDNESS,
    BOUNDARY_DEFENSE_STEEPNESS_K,
    CRITICAL_SKIN_WETTEDNESS,
    DEFAULT_HEAT_STORAGE_INITIAL_WM2,
    LINEAR_EXPECTANCY_GAIN_G,
    METABOLIC_PLATFORM_MET,
    METABOLIC_TRAIN_MET,
    PARTICIPANT_HEIGHT_M,
    PARTICIPANT_MASS_KG,
    TAU_EXPECTANCY_S,
    TAU_PHYSIOLOGICAL_S,
)


StepKind = Literal["auto", "cooling", "warming"]


@dataclass(frozen=True)
class PhysiologicalOutputs:
    """Primary time-dependent physiological scalars for downstream PMV coupling."""

    metabolic_rate_met: float
    heat_storage_rate_wm2: float
    skin_wettedness: float
    boundary_defense_factor: float
    thermal_history_intensity: float
    psychological_offset: float


@dataclass(frozen=True)
class SetTmpInputs:
    """
    Input bundle for Standard Effective Temperature (SET) via pythermalcomfort.

    Relative humidity ``rh`` follows pythermalcomfort conventions (0-100 %).
    """

    tdb: float
    tr: float
    v: float
    rh: float
    met: float
    clo: float
    wme: float = 0.0


def standard_effective_temperature_celsius(inputs: SetTmpInputs) -> float:
    """Return SET (degrees Celsius) using ``pythermalcomfort.models.set_tmp``."""

    res = set_tmp(
        tdb=inputs.tdb,
        tr=inputs.tr,
        v=inputs.v,
        rh=inputs.rh,
        met=inputs.met,
        clo=inputs.clo,
        wme=inputs.wme,
    )
    return float(res.set)


def delta_set_kelvin(previous: SetTmpInputs, current: SetTmpInputs) -> float:
    """
    Spatial SET increment Delta SET_{n-1} = SET_n - SET_{n-1} (Kelvin / Celsius delta).

    Both SET values are evaluated through ``set_tmp`` to remain aligned with the
    CBE / ISO SET implementation bundled in pythermalcomfort.
    """

    return standard_effective_temperature_celsius(current) - standard_effective_temperature_celsius(previous)


def dubois_body_surface_area_m2(mass_kg: float, height_m: float) -> float:
    """
    DuBois body surface area (m^2) used to convert whole-body powers to fluxes.

    A_D = 0.202 * m^0.425 * l^0.725 (mass in kg, stature in m).
    """

    return 0.202 * (mass_kg**0.425) * (height_m**0.725)


def metabolic_rate_transient_met(
    time_s: float,
    m_platform_met: float = METABOLIC_PLATFORM_MET,
    m_train_met: float = METABOLIC_TRAIN_MET,
    tau_physiological_s: float = TAU_PHYSIOLOGICAL_S,
) -> float:
    """
    MAPS-style metabolic relaxation between platform activity and train sedentary levels.

    M(t) = M_train + (M_platform - M_train) * exp(-t / T_phys)

    Parameters
    ----------
    time_s:
        Exposure time after the transition into the carriage environment (s).
    m_platform_met, m_train_met:
        Initial high-intensity metabolic rate and asymptotic sedentary rate (met).
    tau_physiological_s:
        Physiological time constant controlling the decay (s).
    """

    if tau_physiological_s <= 0.0:
        raise ValueError("tau_physiological_s must be positive.")
    delta_m = m_platform_met - m_train_met
    return m_train_met + delta_m * math.exp(-time_s / tau_physiological_s)


def metabolic_decay_gradient_met_per_s(
    time_s: float,
    m_platform_met: float = METABOLIC_PLATFORM_MET,
    m_train_met: float = METABOLIC_TRAIN_MET,
    tau_physiological_s: float = TAU_PHYSIOLOGICAL_S,
) -> float:
    """
    Time derivative dM/dt for sensitivity studies (met / s).

    dM/dt = -(M_platform - M_train) / tau * exp(-t / tau).
    """

    if tau_physiological_s <= 0.0:
        raise ValueError("tau_physiological_s must be positive.")
    delta_m = m_platform_met - m_train_met
    return -(delta_m / tau_physiological_s) * math.exp(-time_s / tau_physiological_s)


def heat_storage_transient_wm2(
    time_s: float,
    s_initial_wm2: float = DEFAULT_HEAT_STORAGE_INITIAL_WM2,
    tau_physiological_s: float = TAU_PHYSIOLOGICAL_S,
) -> float:
    """
    Exponential relaxation of the body heat-storage term S(t) (W/m^2).

    S(t) = S_initial * exp(-t / tau)

    The initial storage rate should be calibrated with the scenario enthalpy
    budget; the bundled default is a finite placeholder for cold-start runs.
    """

    if tau_physiological_s <= 0.0:
        raise ValueError("tau_physiological_s must be positive.")
    return s_initial_wm2 * math.exp(-time_s / tau_physiological_s)


def skin_wettedness_from_fluxes(e_sk_wm2: float, e_max_wm2: float) -> float:
    """
    ISO-style skin wettedness defined as the ratio of actual to maximum latent loss.

    omega_sk = E_sk / E_max, clipped to [0, 1].
    """

    if e_max_wm2 <= 0.0:
        raise ValueError("e_max_wm2 must be positive for skin wettedness.")
    ratio = e_sk_wm2 / e_max_wm2
    return max(0.0, min(1.0, ratio))


def skin_wettedness_refined(
    e_rsw_wm2: float,
    e_max_wm2: float,
    omega_basal: float = BASELINE_SKIN_WETTEDNESS,
    active_weight: float = ACTIVE_SKIN_WETTEDNESS_WEIGHT,
) -> float:
    """
    Refined skin wettedness isolating basal diffusion from regulatory sweat.

    omega = omega_basal + active_weight * (E_rsw / E_max)

    E_rsw is the active regulatory evaporative heat loss from the skin (W/m^2).
    """

    if e_max_wm2 <= 0.0:
        raise ValueError("e_max_wm2 must be positive for skin wettedness.")
    if not 0.0 <= omega_basal <= 1.0:
        raise ValueError("omega_basal must lie in [0, 1].")
    if not 0.0 <= active_weight <= 1.0:
        raise ValueError("active_weight must lie in [0, 1].")
    ratio = e_rsw_wm2 / e_max_wm2
    omega = omega_basal + active_weight * ratio
    return max(0.0, min(1.0, omega))


def boundary_defense_logistic(
    skin_wettedness: float,
    omega_crit: float = CRITICAL_SKIN_WETTEDNESS,
    steepness_k: float = BOUNDARY_DEFENSE_STEEPNESS_K,
) -> float:
    """
    Physiological veto / boundary-defense multiplier on expectancy credit.

    f_def(omega) = 1 / (1 + exp(k * (omega - omega_crit)))

    For omega << omega_crit the multiplier approaches unity; beyond the ISO
    critical skin wettedness it collapses toward zero, removing psychological offset.
    """

    if steepness_k <= 0.0:
        raise ValueError("steepness_k must be positive.")
    exponent = steepness_k * (skin_wettedness - omega_crit)
    # Clip exponent to avoid overflow in exp for extreme arguments.
    exponent = max(-60.0, min(60.0, exponent))
    return 1.0 / (1.0 + math.exp(exponent))


def retention_alpha_for_set_step(
    delta_set_k: float,
    step_kind: StepKind = "auto",
) -> float:
    """
    Asymmetric expectancy-retention factor keyed to the sign of the SET step-change.

    Cooling / relief transitions (Delta SET < 0) retain more credit than warming
    stress (Delta SET > 0), matching the habituation logic in the EATM narrative.
    """

    if step_kind == "cooling":
        return ALPHA_RETENTION_COOLING_STEP
    if step_kind == "warming":
        return ALPHA_RETENTION_WARMING_STEP
    if step_kind != "auto":
        raise ValueError("step_kind must be 'auto', 'cooling', or 'warming'.")
    if delta_set_k < 0.0:
        return ALPHA_RETENTION_COOLING_STEP
    if delta_set_k > 0.0:
        return ALPHA_RETENTION_WARMING_STEP
    return 0.5 * (ALPHA_RETENTION_COOLING_STEP + ALPHA_RETENTION_WARMING_STEP)


def thermal_history_intensity(
    delta_set_k: float,
    time_s: float,
    space_index: int,
    tau_expectancy_s: float = TAU_EXPECTANCY_S,
    step_kind: StepKind = "auto",
    alpha_override: Optional[float] = None,
) -> float:
    """
    Thermal History Intensity (THI) entering the psychological offset.

    THI = Delta SET * alpha^(n - 1) * exp(-t / tau_exp)

    Parameters
    ----------
    delta_set_k:
        Standard-effective-temperature step between the previous and current space (K).
    time_s:
        Local dwell time after the transition (s).
    space_index:
        1-based spatial index n along the itinerary (first entry uses alpha^0 = 1).
    tau_expectancy_s:
        Expectancy window controlling temporal habituation (72 s baseline).
    step_kind / alpha_override:
        Optional asymmetric retention selector or explicit alpha value.
    """

    if tau_expectancy_s <= 0.0:
        raise ValueError("tau_expectancy_s must be positive.")
    if space_index < 1:
        raise ValueError("space_index must be >= 1 (1-based itinerary).")
    alpha = alpha_override if alpha_override is not None else retention_alpha_for_set_step(delta_set_k, step_kind)
    retention_chain = alpha ** (space_index - 1)
    habituation = math.exp(-time_s / tau_expectancy_s)
    return delta_set_k * retention_chain * habituation


def psychological_offset_lambda(
    thermal_history_intensity_value: float,
    skin_wettedness: float,
    expectancy_gain: float = LINEAR_EXPECTANCY_GAIN_G,
    omega_crit: float = CRITICAL_SKIN_WETTEDNESS,
    steepness_k: float = BOUNDARY_DEFENSE_STEEPNESS_K,
) -> float:
    """
    Psychological offset lambda coupling THI with the boundary-defense filter.

    lambda = G * THI * f_def(omega)
    """

    f_def = boundary_defense_logistic(skin_wettedness, omega_crit, steepness_k)
    return expectancy_gain * thermal_history_intensity_value * f_def


def evaluate_physiological_state(
    time_s: float,
    delta_set_k: float,
    space_index: int,
    e_sk_wm2: float,
    e_max_wm2: float,
    *,
    e_rsw_wm2: Optional[float] = None,
    use_refined_skin_wettedness: bool = False,
    s_initial_wm2: float = DEFAULT_HEAT_STORAGE_INITIAL_WM2,
    step_kind: StepKind = "auto",
    m_platform_met: float = METABOLIC_PLATFORM_MET,
    m_train_met: float = METABOLIC_TRAIN_MET,
    tau_physiological_s: float = TAU_PHYSIOLOGICAL_S,
    tau_expectancy_s: float = TAU_EXPECTANCY_S,
) -> PhysiologicalOutputs:
    """
    Convenience bundle for a single instant in the time-marching loop.

    External heat-balance solvers supply evaporative fluxes; this routine only
    maps them into omega, THI, lambda, S(t), and M(t).
    """

    m_t = metabolic_rate_transient_met(time_s, m_platform_met, m_train_met, tau_physiological_s)
    s_t = heat_storage_transient_wm2(time_s, s_initial_wm2, tau_physiological_s)
    if use_refined_skin_wettedness:
        if e_rsw_wm2 is None:
            raise ValueError("e_rsw_wm2 is required when use_refined_skin_wettedness is True.")
        omega = skin_wettedness_refined(e_rsw_wm2, e_max_wm2)
    else:
        omega = skin_wettedness_from_fluxes(e_sk_wm2, e_max_wm2)

    f_def = boundary_defense_logistic(omega)
    thi = thermal_history_intensity(delta_set_k, time_s, space_index, tau_expectancy_s, step_kind)
    lam = psychological_offset_lambda(thi, omega)

    return PhysiologicalOutputs(
        metabolic_rate_met=m_t,
        heat_storage_rate_wm2=s_t,
        skin_wettedness=omega,
        boundary_defense_factor=f_def,
        thermal_history_intensity=thi,
        psychological_offset=lam,
    )


__all__ = [
    "PhysiologicalOutputs",
    "SetTmpInputs",
    "delta_set_kelvin",
    "standard_effective_temperature_celsius",
    "dubois_body_surface_area_m2",
    "metabolic_rate_transient_met",
    "metabolic_decay_gradient_met_per_s",
    "heat_storage_transient_wm2",
    "skin_wettedness_from_fluxes",
    "skin_wettedness_refined",
    "boundary_defense_logistic",
    "retention_alpha_for_set_step",
    "thermal_history_intensity",
    "psychological_offset_lambda",
    "evaluate_physiological_state",
]
