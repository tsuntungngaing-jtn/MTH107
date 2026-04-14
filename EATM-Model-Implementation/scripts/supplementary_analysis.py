"""
Supplementary EATM figures: heat-partition stack and omega sensitivity.

Figure A: Stacked area chart for D=6 showing the evolution of
convection (C), radiation (R), and skin-side latent heat loss (E_sk)
over the carriage dwell, highlighting the progressive dominance of E_sk.

Figure B: Sensitivity of meltdown timing to crowding, plotting the time
required for skin wettedness omega to reach the critical threshold
omega_crit = 0.25 as occupant density D varies from 0 to 9 pass/m^2.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.eatm import TransitScenarioParameters, simulate_platform_to_carriage
from modules.environmental import saturation_vapor_pressure_magnus_pa
from modules.pmv_dynamic_iso import iso7730_heat_losses_dict_for_docs


def _simulate_density_series(
    density: float,
    time_horizon_s: float = 900.0,
    time_step_s: float = 10.0,
) -> Dict[str, np.ndarray]:
    """Run one density case and return arrays for omega and heat partition."""

    params = TransitScenarioParameters(
        occupant_density_per_m2=density,
        time_horizon_s=time_horizon_s,
        time_step_s=time_step_s,
    )
    series = simulate_platform_to_carriage(params)

    time_s = np.array([r.time_s for r in series], dtype=float)
    time_min = time_s / 60.0
    omega = np.array([r.skin_wettedness for r in series], dtype=float)

    # Recover ISO 7730 partition (C, R, E_sk) using the same microclimate states
    c_vals: List[float] = []
    r_vals: List[float] = []
    esk_vals: List[float] = []
    for r in series:
        p_sat = saturation_vapor_pressure_magnus_pa(params.carriage_tdb_c)
        p_vap_local = p_sat * (r.local_rh_percent / 100.0)
        losses = iso7730_heat_losses_dict_for_docs(
            tdb=params.carriage_tdb_c,
            tr=r.t_mr_prime_c,
            vr=r.local_velocity_ms,
            vapor_pressure_pa=p_vap_local,
            met=r.metabolic_met,
            clo=r.corrected_clo,
            wme=0.0,
        )
        c_vals.append(losses["C"])
        r_vals.append(losses["R"])
        esk_vals.append(losses["E_sk"])

    return {
        "time_min": time_min,
        "omega": omega,
        "C": np.array(c_vals, dtype=float),
        "R": np.array(r_vals, dtype=float),
        "E_sk": np.array(esk_vals, dtype=float),
    }


def _first_omega_crossing(
    time_min: np.ndarray,
    omega: np.ndarray,
    threshold: float = 0.25,
) -> Optional[Tuple[float, float]]:
    """Return first (time, omega) where omega >= threshold, if any."""

    crossing = np.where(omega >= threshold)[0]
    if crossing.size == 0:
        return None
    idx = int(crossing[0])
    return float(time_min[idx]), float(omega[idx])


def create_heat_partition_stack_figure(
    density_data: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Stacked area chart of C, R, and E_sk for a single density case."""

    time_min = density_data["time_min"]
    c = density_data["C"]
    r = density_data["R"]
    esk = density_data["E_sk"]

    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    colors = {
        "C": "#3182bd",  # blue
        "R": "#9e9ac8",  # violet
        "E_sk": "#fc9272",  # warm red/orange
    }

    ax.stackplot(
        time_min,
        c,
        r,
        esk,
        labels=[
            r"Convection, $C$",
            r"Radiation, $R$",
            r"Skin latent, $E_{\mathrm{sk}}$",
        ],
        colors=[colors["C"], colors["R"], colors["E_sk"]],
        alpha=0.9,
    )

    # Highlight the transition to E_sk dominance using a vertical marker at
    # the time where E_sk first exceeds the combined dry channels.
    total_dry = c + r
    dominance_idx = np.where(esk >= total_dry)[0]
    if dominance_idx.size > 0:
        j = int(dominance_idx[0])
        t_dom = float(time_min[j])
        ax.axvline(
            t_dom,
            color="black",
            linestyle="--",
            linewidth=1.3,
            alpha=0.8,
        )
        ax.text(
            t_dom,
            0.98 * float((c + r + esk).max()),
            r"$E_{\mathrm{sk}}$ dominant",
            rotation=90,
            va="top",
            ha="right",
            fontsize=9,
        )

    ax.set_xlabel(r"Time in carriage, $t\;(\mathrm{min})$")
    ax.set_ylabel(r"Heat loss components, $q\;(\mathrm{W/m^2})$")
    ax.set_title(
        r"Transition of first-law channels under crowding: "
        r"$E_{\mathrm{sk}}$ emerges as the dominant pathway (D=6)",
        pad=10,
    )
    ax.grid(True, alpha=0.25)
    ax.set_xlim(time_min.min(), time_min.max())
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def create_omega_sensitivity_figure(
    densities: List[int],
    density_series: Dict[int, Dict[str, np.ndarray]],
    out_path: Path,
    omega_crit: float = 0.25,
    horizon_min: float = 15.0,
) -> None:
    """Line plot of time-to-meltdown as a function of density D."""

    times_to_crit: List[float] = []
    for d in densities:
        series = density_series[d]
        crossing = _first_omega_crossing(series["time_min"], series["omega"], threshold=omega_crit)
        if crossing is None:
            times_to_crit.append(np.nan)
        else:
            t_c, _ = crossing
            times_to_crit.append(t_c)

    d_array = np.array(densities, dtype=float)
    t_array = np.array(times_to_crit, dtype=float)
    reached_mask = np.isfinite(t_array)
    censored_mask = ~reached_mask
    t_plot = np.where(reached_mask, t_array, horizon_min)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        d_array[reached_mask],
        t_plot[reached_mask],
        marker="o",
        linewidth=2.2,
        color="#08519c",
        label=rf"Time to $\omega_{{\mathrm{{crit}}}}={omega_crit:.2f}$",
    )
    if np.any(censored_mask):
        ax.scatter(
            d_array[censored_mask],
            t_plot[censored_mask],
            marker="v",
            s=55,
            color="#6a51a3",
            label=rf"No crossing within {horizon_min:.0f} min",
            zorder=4,
        )

    # Emphasize the reference D=6 point, if present.
    if 6 in densities:
        idx6 = densities.index(6)
        if np.isfinite(t_plot[idx6]):
            ax.scatter(
                [d_array[idx6]],
                [t_plot[idx6]],
                color="#de2d26",
                edgecolor="white",
                linewidth=0.9,
                s=60,
                zorder=4,
                label=r"$D=6\;\mathrm{pass/m^2}$ reference",
            )

    ax.set_xlabel(r"Occupant density, $D\;(\mathrm{pass/m^2})$")
    ax.set_ylabel(
        rf"Time to $\omega_{{\mathrm{{crit}}}}={omega_crit:.2f}$, "
        r"$t_{\mathrm{meltdown}}\;(\mathrm{min})$"
    )
    ax.set_title(
        r"Sensitivity of meltdown timing to crowding: "
        r"$t_{\mathrm{meltdown}}$ vs. $D$",
        pad=10,
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(d_array.min() - 0.2, d_array.max() + 0.2)
    ax.set_ylim(0.0, max(horizon_min * 1.05, float(np.nanmax(t_plot)) + 0.5))
    ax.legend(loc="best", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Entry point to generate supplementary analysis figures."""

    # Density D=6 stack figure (heat-partition evolution).
    d_stack = 6.0
    stack_series = _simulate_density_series(d_stack)

    # Sensitivity sweep for D in [0, 9].
    densities_int: List[int] = list(range(0, 10))
    sensitivity_series: Dict[int, Dict[str, np.ndarray]] = {
        d: _simulate_density_series(float(d)) for d in densities_int
    }

    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    stack_path = out_dir / "heat_partition_stack_D6.png"
    omega_sensitivity_path = out_dir / "omega_meltdown_sensitivity.png"

    create_heat_partition_stack_figure(stack_series, stack_path)
    create_omega_sensitivity_figure(
        densities_int,
        sensitivity_series,
        omega_sensitivity_path,
        omega_crit=0.25,
        horizon_min=15.0,
    )

    print(f"Wrote figure: {stack_path}")
    print(f"Wrote figure: {omega_sensitivity_path}")


if __name__ == "__main__":
    main()

