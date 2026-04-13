"""
ISO 7730 first-law thermal load and dynamic PMV bracket (CBE-consistent kernel).

The dynamic vote follows the same structural form used in the CBE optimized PMV
implementation: a metabolic sensitivity bracket multiplied by the thermal load
L (W/m^2), where L is the residual internal heat after all loss pathways.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    METABOLIC_W_PER_M2_PER_MET,
    PMV_DYNAMIC_BRACKET_INTERCEPT,
    PMV_DYNAMIC_EXPONENT_PER_WM2,
    PMV_DYNAMIC_SENSITIVITY_A,
)


@dataclass(frozen=True)
class Iso7730HeatLossesWM2:
    """Dry and latent loss components (all in W/m^2) on a DuBois basis."""

    internal_production: float
    skin_diffusion: float  # hl1 — insensible evaporative / diffusion term in ISO kernel
    regulatory_sweat: float  # hl2 — regulatory sweating term (zero below threshold)
    latent_respiration: float  # E_re — hl3
    dry_respiration: float  # C_re — hl4
    radiation: float  # R — hl5
    convection: float  # C — hl6
    thermal_load: float  # L = M - W - sum(losses to air/skin/radiation as partitioned)


def _clothing_area_factor_fcl_from_clo(clo: float) -> float:
    icl = 0.155 * clo
    if icl <= 0.078:
        return 1.0 + 1.29 * icl
    return 1.05 + 0.645 * icl


def _tcl_and_losses_iso7730_kernel(
    tdb: float,
    tr: float,
    vr: float,
    vapor_pressure_pa: float,
    met: float,
    clo: float,
    wme: float,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Clothing surface temperature and ISO-style loss splits (CBE optimized kernel).

    Returns (tcl_c, hl1, hl2, hl3, hl4, hl5, hl6, mw) with hl* and mw in W/m^2.
    """

    pa = float(vapor_pressure_pa)
    icl = 0.155 * clo
    m = met * METABOLIC_W_PER_M2_PER_MET
    w = wme * METABOLIC_W_PER_M2_PER_MET
    mw = m - w

    f_cl = _clothing_area_factor_fcl_from_clo(clo)
    hcf = 12.1 * math.sqrt(max(vr, 1e-8))
    hc = hcf
    taa = tdb + 273.15
    tra = tr + 273.15
    t_cla = taa + (35.5 - tdb) / (3.5 * icl + 0.1)

    p1 = icl * f_cl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = (308.7 - 0.028 * mw) + (p2 * ((tra / 100.0) ** 4))
    xn = t_cla / 100.0
    xf = t_cla / 50.0
    eps = 0.00015

    for _ in range(151):
        xf = (xf + xn) / 2.0
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        hc = max(hcn, hcf)
        xn = (p5 + p4 * hc - p2 * xf**4) / (100.0 + p3 * hc)
        if abs(xn - xf) <= eps:
            break
    else:
        raise RuntimeError("Clothing surface temperature iteration did not converge.")

    tcl = 100.0 * xn - 273.15

    hl1 = 3.05 * 0.001 * (5733.0 - (6.99 * mw) - pa)
    hl2 = 0.42 * (mw - METABOLIC_W_PER_M2_PER_MET) if mw > METABOLIC_W_PER_M2_PER_MET else 0.0
    hl3 = 1.7e-5 * m * (5867.0 - pa)
    hl4 = 0.0014 * m * (34.0 - tdb)
    hl5 = 3.96 * f_cl * (xn**4 - (tra / 100.0) ** 4)
    hl6 = f_cl * hc * (tcl - tdb)

    return tcl, hl1, hl2, hl3, hl4, hl5, hl6, mw


def iso7730_heat_losses_wm2(
    tdb: float,
    tr: float,
    vr: float,
    vapor_pressure_pa: float,
    met: float,
    clo: float,
    wme: float = 0.0,
) -> Iso7730HeatLossesWM2:
    """
    Partition the ISO 7730 heat balance for use in L(t) = M - W - (R + C + E_sk + C_re + E_re).

    Notes
    -----
    The kernel groups insensible skin moisture and regulatory sweating into hl1
    and hl2 (CBE reference). Their sum corresponds to the total skin-side latent
    pathway E_sk in the conceptual first-law statement.
    """

    tcl, hl1, hl2, hl3, hl4, hl5, hl6, mw = _tcl_and_losses_iso7730_kernel(
        tdb, tr, vr, vapor_pressure_pa, met, clo, wme
    )
    e_sk = hl1 + hl2
    r = hl5
    c = hl6
    c_re = hl4
    e_re = hl3
    losses = e_sk + r + c + c_re + e_re
    thermal_load = mw - losses
    return Iso7730HeatLossesWM2(
        internal_production=mw,
        skin_diffusion=hl1,
        regulatory_sweat=hl2,
        latent_respiration=e_re,
        dry_respiration=c_re,
        radiation=r,
        convection=c,
        thermal_load=thermal_load,
    )


def pmv_dynamic_from_load(met: float, thermal_load_wm2: float) -> Tuple[float, float]:
    """
    Expectancy-adjusted physical vote (dynamic PMV bracket times thermal load).

        PMV_dynamic = (a * exp(b * m) + c0) * L

    where m is the absolute metabolic rate in W/m^2 (DuBois basis), identical to
    the bracket argument used in the CBE optimized ISO PMV kernel.
    """

    m = met * METABOLIC_W_PER_M2_PER_MET
    bracket = PMV_DYNAMIC_SENSITIVITY_A * math.exp(PMV_DYNAMIC_EXPONENT_PER_WM2 * m) + PMV_DYNAMIC_BRACKET_INTERCEPT
    return bracket * thermal_load_wm2, bracket


def iso7730_heat_losses_dict_for_docs(
    tdb: float,
    tr: float,
    vr: float,
    vapor_pressure_pa: float,
    met: float,
    clo: float,
    wme: float = 0.0,
) -> Dict[str, float]:
    """Flattened component map for diagnostics and plotting."""

    h = iso7730_heat_losses_wm2(tdb, tr, vr, vapor_pressure_pa, met, clo, wme)
    return {
        "M_minus_W": h.internal_production,
        "E_sk": h.skin_diffusion + h.regulatory_sweat,
        "E_sk_diffusion": h.skin_diffusion,
        "E_sk_sweat": h.regulatory_sweat,
        "R": h.radiation,
        "C": h.convection,
        "C_re": h.dry_respiration,
        "E_re": h.latent_respiration,
        "L": h.thermal_load,
    }


__all__ = [
    "Iso7730HeatLossesWM2",
    "iso7730_heat_losses_dict_for_docs",
    "iso7730_heat_losses_wm2",
    "pmv_dynamic_from_load",
]
