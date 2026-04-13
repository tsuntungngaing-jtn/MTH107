"""
Scalar Newton-Raphson solvers for nonlinear thermal-balance equations.

The clothing-surface balance couples convective losses with Stefan-Boltzmann
radiation to a density-corrected mean radiant temperature T_mr'. Substituting
the T^4-weighted definition of T_mr' (see the radiative module) yields a scalar
residual R(T_cl)=0 that is solved with Newton's method and an analytically
derived derivative dR/dT_cl.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Tuple

from scipy.optimize import newton as scipy_newton

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    EMISSIVITY_CLOTHING,
    METABOLIC_W_PER_M2_PER_MET,
    NEWTON_MAX_ITERATIONS,
    NEWTON_TOLERANCE,
    STEFAN_BOLTZMANN,
)


def newton_raphson_1d(
    residual: Callable[[float], float],
    derivative: Callable[[float], float],
    x0: float,
    *,
    tol: float = NEWTON_TOLERANCE,
    max_iter: int = NEWTON_MAX_ITERATIONS,
    xmin: float = -80.0,
    xmax: float = 80.0,
) -> float:
    """
    Solve f(x)=0 using Newton-Raphson with bracket clamping on each iterate.

    Raises
    ------
    RuntimeError
        If the iteration fails to converge or the derivative is numerically singular.
    """

    x = min(max(x0, xmin), xmax)
    for _ in range(max_iter):
        fx = residual(x)
        if abs(fx) < tol:
            return x
        dfx = derivative(x)
        if not math.isfinite(dfx) or abs(dfx) < 1e-14:
            raise RuntimeError("Newton-Raphson stalled: singular or non-finite derivative.")
        step = fx / dfx
        x_next = x - step
        if not math.isfinite(x_next):
            raise RuntimeError("Newton-Raphson produced a non-finite iterate.")
        x = min(max(x_next, xmin), xmax)
    raise RuntimeError("Newton-Raphson did not converge within max_iter.")


def _clothing_area_factor_fcl(iclo: float) -> float:
    """ISO 7730 clothing area factor as a function of intrinsic clothing insulation."""

    if iclo <= 0.078:
        return 1.0 + 1.29 * iclo
    return 1.05 + 0.645 * iclo


def _skin_temperature_celsius(met: float, work_met: float) -> float:
    """Proxy mean skin temperature (deg C) from metabolic activity (met)."""

    h_wm2 = METABOLIC_W_PER_M2_PER_MET * (met - work_met)
    return 35.7 - 0.028 * h_wm2


def solve_clothing_surface_temperature_coupled(
    ta_celsius: float,
    t_wall_celsius: float,
    icl_clo: float,
    met: float,
    work_met: float,
    f_pp: float,
    f_pwall: float,
    local_air_velocity_ms: float,
    *,
    emissivity: float = EMISSIVITY_CLOTHING,
    sigma: float = STEFAN_BOLTZMANN,
    tol: float = NEWTON_TOLERANCE,
    max_iter: int = NEWTON_MAX_ITERATIONS,
) -> Tuple[float, float]:
    """
    Solve for clothing surface temperature T_cl and corrected T_mr' with SB radiation.

    The mean radiant temperature follows the fourth-power mixture used in the
    EATM radiative network:

        T_mr'^4 = F_pp * T_cl^4 + F_pwall * T_wall^4 + (1 - F_pp - F_pwall) * T_a^4

    with all temperatures expressed in kelvin on the right-hand side. The clothing
    node balance reads

        (T_sk - T_cl) / (0.155 * I_cl) = f_cl * ( h_c (T_cl - T_a)
            + epsilon * sigma * (T_cl^4 - T_mr'^4(T_cl)) ),

    where h_c = 8.3 * V_local^0.6 (forced convection, ISO-style).

    Parameters
    ----------
    ta_celsius, t_wall_celsius:
        Dry-bulb air temperature and effective wall radiative temperature (deg C).
    icl_clo:
        Corrected intrinsic clothing insulation (clo) including crowding / moisture.
    met, work_met:
        Total metabolic rate and external mechanical work (met).
    f_pp, f_pwall:
        Interpersonal and wall view factors after density corrections.
    local_air_velocity_ms:
        Local air speed at the body (m/s).

    Returns
    -------
    tuple[float, float]
        (T_cl, T_mr_prime) both in degrees Celsius.
    """

    if icl_clo <= 0.0:
        raise ValueError("icl_clo must be positive.")
    if local_air_velocity_ms < 0.0:
        raise ValueError("local_air_velocity_ms must be non-negative.")
    if f_pp < 0.0 or f_pwall < 0.0:
        raise ValueError("View factors must be non-negative.")
    if f_pp + f_pwall >= 1.0:
        raise ValueError("Require F_pp + F_pwall < 1 so the air channel weight stays positive.")

    fcl = _clothing_area_factor_fcl(icl_clo)
    r_cl_e = 0.155 * icl_clo
    tsk = _skin_temperature_celsius(met, work_met)
    hc = 8.3 * (local_air_velocity_ms**0.6) if local_air_velocity_ms > 0.0 else 0.0

    ta_k = ta_celsius + 273.15
    tw_k = t_wall_celsius + 273.15
    c0 = f_pwall * tw_k**4 + (1.0 - f_pp - f_pwall) * ta_k**4

    def residual(tcl_c: float) -> float:
        tk = tcl_c + 273.15
        tmrp_k4 = f_pp * tk**4 + c0
        q_conv = hc * (tcl_c - ta_celsius)
        q_rad = emissivity * sigma * (tk**4 - tmrp_k4)
        q_out = fcl * (q_conv + q_rad)
        q_in = (tsk - tcl_c) / r_cl_e
        return q_in - q_out

    def derivative(tcl_c: float) -> float:
        tk = tcl_c + 273.15
        dtmrp_k4_dtcl = 4.0 * f_pp * tk**3
        dqrad_dtcl = emissivity * sigma * (4.0 * tk**3 - dtmrp_k4_dtcl)
        dqconv_dtcl = hc
        dqout_dtcl = fcl * (dqconv_dtcl + dqrad_dtcl)
        dqin_dtcl = -1.0 / r_cl_e
        return dqin_dtcl - dqout_dtcl

    t_guess = 0.5 * (ta_celsius + tsk)
    t_cl = float(
        scipy_newton(
            residual,
            t_guess,
            fprime=derivative,
            tol=tol,
            maxiter=max_iter,
            disp=False,
        )
    )

    tk = t_cl + 273.15
    tmrp_k4 = f_pp * tk**4 + c0
    t_mr_prime = tmrp_k4**0.25 - 273.15
    return t_cl, t_mr_prime


def solve_clothing_surface_temperature_constant_tr(
    ta_celsius: float,
    t_mr_celsius: float,
    icl_clo: float,
    met: float,
    work_met: float,
    local_air_velocity_ms: float,
    *,
    emissivity: float = EMISSIVITY_CLOTHING,
    sigma: float = STEFAN_BOLTZMANN,
    tol: float = NEWTON_TOLERANCE,
    max_iter: int = NEWTON_MAX_ITERATIONS,
) -> float:
    """
    Solve the clothing node when T_mr is fixed (no interpersonal T^4 feedback).

    This reduces to a single-node Stefan-Boltzmann balance and is useful for
    diagnostics or for coupling strategies that lag T_mr'.
    """

    if icl_clo <= 0.0:
        raise ValueError("icl_clo must be positive.")
    fcl = _clothing_area_factor_fcl(icl_clo)
    r_cl_e = 0.155 * icl_clo
    tsk = _skin_temperature_celsius(met, work_met)
    hc = 8.3 * (local_air_velocity_ms**0.6) if local_air_velocity_ms > 0.0 else 0.0
    tr_k = t_mr_celsius + 273.15

    def residual(tcl_c: float) -> float:
        tk = tcl_c + 273.15
        q_conv = hc * (tcl_c - ta_celsius)
        q_rad = emissivity * sigma * (tk**4 - tr_k**4)
        return (tsk - tcl_c) / r_cl_e - fcl * (q_conv + q_rad)

    def derivative(tcl_c: float) -> float:
        tk = tcl_c + 273.15
        dqrad = emissivity * sigma * 4.0 * tk**3
        return -1.0 / r_cl_e - fcl * (hc + dqrad)

    t_guess = 0.5 * (ta_celsius + tsk)
    return float(
        scipy_newton(
            residual,
            t_guess,
            fprime=derivative,
            tol=tol,
            maxiter=max_iter,
            disp=False,
        )
    )


__all__ = [
    "newton_raphson_1d",
    "solve_clothing_surface_temperature_coupled",
    "solve_clothing_surface_temperature_constant_tr",
]
