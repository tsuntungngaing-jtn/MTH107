"""
Expectancy-Adjusted Thermal Model (EATM) orchestration on top of pythermalcomfort.

SET inputs for Thermal History Intensity use ``set_tmp``. The physical vote
``PMV_dynamic`` follows the report/CBE dynamic bracket multiplied by the ISO 7730
first-law thermal load L (W/m^2), not the steady ``pmv_ppd_iso`` shortcut.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pythermalcomfort.models import pmv_ppd_iso, two_nodes_gagge

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import METABOLIC_PLATFORM_MET, TAU_EXPECTANCY_S
from modules.environmental import (
    environmental_bundle,
    saturation_vapor_pressure_magnus_pa,
)
from modules.pmv_dynamic_iso import iso7730_heat_losses_wm2, pmv_dynamic_from_load
from modules.physiological import (
    SetTmpInputs,
    boundary_defense_logistic,
    delta_set_kelvin,
    heat_storage_transient_wm2,
    metabolic_rate_transient_met,
    psychological_offset_lambda,
    skin_wettedness_from_fluxes,
    thermal_history_intensity,
)
from modules.radiative import clip_view_factors, interpersonal_view_factor, wall_view_factor
from utils.solvers import solve_clothing_surface_temperature_coupled


@dataclass(frozen=True)
class EATMTimePointResult:
    """State snapshot along the transient carriage exposure."""

    time_s: float
    metabolic_met: float
    t_mr_prime_c: float
    t_clothing_c: float
    local_velocity_ms: float
    local_rh_percent: float
    corrected_clo: float
    skin_wettedness: float
    skin_wettedness_gagge: float
    delta_set_fixed_k: float
    thi: float
    boundary_defense: float
    lambda_offset: float
    thermal_load_wm2: float
    heat_storage_exponential_wm2: float
    pmv_sensitivity_bracket: float
    pmv_dynamic: float
    pmv_eatm: float


@dataclass(frozen=True)
class TransitScenarioParameters:
    """Scenario: hot platform -> conditioned high-density carriage."""

    platform_tdb_c: float = 34.0
    platform_tr_c: float = 34.0
    platform_v_ms: float = 0.45
    platform_rh_percent: float = 65.0
    platform_met: float = METABOLIC_PLATFORM_MET
    platform_clo: float = 0.5

    carriage_tdb_c: float = 26.0
    carriage_wall_c: float = 25.0
    carriage_nominal_v_ms: float = 0.45
    carriage_rh_env_percent: float = 55.0
    occupant_density_per_m2: float = 6.0
    clothing_base_clo: float = 0.5

    time_horizon_s: float = 900.0
    time_step_s: float = 10.0
    space_index: int = 1


def _relative_humidity_percent_from_vapor_pair(t_air_c: float, p_vap_pa: float) -> float:
    p_sat = saturation_vapor_pressure_magnus_pa(t_air_c)
    if p_sat <= 0.0:
        return 50.0
    rh = 100.0 * p_vap_pa / p_sat
    return max(5.0, min(98.0, rh))


def simulate_platform_to_carriage(
    params: Optional[TransitScenarioParameters] = None,
) -> List[EATMTimePointResult]:
    """
    March the coupled EATM correction along a fixed-density carriage dwell.

    Delta SET anchoring
    --------------------
    ``delta_set_fixed`` is evaluated once at carriage entry (t -> 0+) as
    SET(carriage microclimate) - SET(platform). This encodes the **initial thermal
    memory shock** carried by THI while the exponential habituation term applies
    the expectancy decay in time.
    """

    p = params or TransitScenarioParameters()
    times: List[float] = []
    t = 0.0
    while t <= p.time_horizon_s + 1e-9:
        times.append(t)
        t += p.time_step_s

    platform_inputs = SetTmpInputs(
        tdb=p.platform_tdb_c,
        tr=p.platform_tr_c,
        v=p.platform_v_ms,
        rh=p.platform_rh_percent,
        met=p.platform_met,
        clo=p.platform_clo,
    )

    twop = two_nodes_gagge(
        tdb=p.platform_tdb_c,
        tr=p.platform_tr_c,
        v=p.platform_v_ms,
        rh=p.platform_rh_percent,
        met=p.platform_met,
        clo=p.platform_clo,
    )
    omega_lag = float(twop.w)

    f_pp = interpersonal_view_factor(p.occupant_density_per_m2)
    f_pw = wall_view_factor(p.occupant_density_per_m2)
    f_pp_c, f_pw_c = clip_view_factors(f_pp, f_pw)

    met0 = metabolic_rate_transient_met(0.0)
    env0 = environmental_bundle(
        p.carriage_nominal_v_ms,
        p.occupant_density_per_m2,
        p.carriage_tdb_c,
        p.carriage_rh_env_percent / 100.0,
        met0,
        p.clothing_base_clo,
        omega_lag,
    )
    _, t_mr0 = solve_clothing_surface_temperature_coupled(
        ta_celsius=p.carriage_tdb_c,
        t_wall_celsius=p.carriage_wall_c,
        icl_clo=env0.corrected_clothing_insulation_clo,
        met=met0,
        work_met=0.0,
        f_pp=f_pp_c,
        f_pwall=f_pw_c,
        local_air_velocity_ms=env0.local_air_velocity_ms,
    )
    rh0 = _relative_humidity_percent_from_vapor_pair(
        p.carriage_tdb_c, env0.vapor_pressure_local_pa
    )
    carriage_inputs0 = SetTmpInputs(
        tdb=p.carriage_tdb_c,
        tr=t_mr0,
        v=env0.local_air_velocity_ms,
        rh=rh0,
        met=met0,
        clo=env0.corrected_clothing_insulation_clo,
    )
    delta_set_fixed = delta_set_kelvin(platform_inputs, carriage_inputs0)

    results: List[EATMTimePointResult] = []
    for time_s in times:
        met = metabolic_rate_transient_met(time_s)
        env = environmental_bundle(
            p.carriage_nominal_v_ms,
            p.occupant_density_per_m2,
            p.carriage_tdb_c,
            p.carriage_rh_env_percent / 100.0,
            met,
            p.clothing_base_clo,
            omega_lag,
        )
        t_cl, t_mr = solve_clothing_surface_temperature_coupled(
            ta_celsius=p.carriage_tdb_c,
            t_wall_celsius=p.carriage_wall_c,
            icl_clo=env.corrected_clothing_insulation_clo,
            met=met,
            work_met=0.0,
            f_pp=f_pp_c,
            f_pwall=f_pw_c,
            local_air_velocity_ms=env.local_air_velocity_ms,
        )
        rh_loc = _relative_humidity_percent_from_vapor_pair(
            p.carriage_tdb_c, env.vapor_pressure_local_pa
        )

        two = two_nodes_gagge(
            tdb=p.carriage_tdb_c,
            tr=t_mr,
            v=env.local_air_velocity_ms,
            rh=rh_loc,
            met=met,
            clo=env.corrected_clothing_insulation_clo,
        )
        gagge_w = float(two.w)
        if two.e_max and float(two.e_max) > 1e-9:
            omega_ratio = skin_wettedness_from_fluxes(float(two.e_skin), float(two.e_max))
        else:
            omega_ratio = gagge_w
        omega_lag = omega_ratio

        losses = iso7730_heat_losses_wm2(
            tdb=p.carriage_tdb_c,
            tr=t_mr,
            vr=env.local_air_velocity_ms,
            vapor_pressure_pa=env.vapor_pressure_local_pa,
            met=met,
            clo=env.corrected_clothing_insulation_clo,
            wme=0.0,
        )
        thermal_load = losses.thermal_load
        s_exp = heat_storage_transient_wm2(time_s)
        pmv_dyn, bracket = pmv_dynamic_from_load(met, thermal_load)

        thi = thermal_history_intensity(
            delta_set_fixed,
            time_s,
            p.space_index,
            TAU_EXPECTANCY_S,
            step_kind="cooling",
        )
        f_def = boundary_defense_logistic(omega_ratio)
        lam = psychological_offset_lambda(thi, omega_ratio)
        pmv_eatm = pmv_dyn - lam

        results.append(
            EATMTimePointResult(
                time_s=time_s,
                metabolic_met=met,
                t_mr_prime_c=t_mr,
                t_clothing_c=t_cl,
                local_velocity_ms=env.local_air_velocity_ms,
                local_rh_percent=rh_loc,
                corrected_clo=env.corrected_clothing_insulation_clo,
                skin_wettedness=omega_ratio,
                skin_wettedness_gagge=gagge_w,
                delta_set_fixed_k=delta_set_fixed,
                thi=thi,
                boundary_defense=f_def,
                lambda_offset=lam,
                thermal_load_wm2=thermal_load,
                heat_storage_exponential_wm2=s_exp,
                pmv_sensitivity_bracket=bracket,
                pmv_dynamic=pmv_dyn,
                pmv_eatm=pmv_eatm,
            )
        )

    return results


def pmv_iso_baseline_reference(
    tdb: float = 26.0,
    tr: float = 26.0,
    vr: float = 0.45,
    rh_percent: float = 55.0,
    met: float = 1.2,
    clo: float = 0.5,
) -> float:
    """
    Reference ISO PMV without crowding corrections (CBE / pythermalcomfort baseline).

    Used for benchmarking against the expectancy-adjusted trajectory.
    """

    return float(pmv_ppd_iso(tdb=tdb, tr=tr, vr=vr, rh=rh_percent, met=met, clo=clo).pmv)


__all__ = [
    "EATMTimePointResult",
    "TransitScenarioParameters",
    "pmv_iso_baseline_reference",
    "simulate_platform_to_carriage",
]
