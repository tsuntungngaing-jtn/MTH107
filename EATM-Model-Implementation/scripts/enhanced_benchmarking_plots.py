"""
Publication-quality benchmarking figures for EATM transit scenarios.

Figure 1: PMV_EATM trajectories at multiple crowding densities with
steady-state ISO baseline and sensory-meltdown markers.
Figure 2: CBE-style micro-environment trajectory in (T_air, p_vap,local) space.
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

from modules.eatm import TransitScenarioParameters, pmv_iso_baseline_reference, simulate_platform_to_carriage
from modules.environmental import saturation_vapor_pressure_magnus_pa


def _first_meltdown_crossing(time_min: np.ndarray, omega: np.ndarray, threshold: float = 0.25) -> Optional[Tuple[float, float]]:
    """Return the first (time, omega) where omega crosses threshold."""
    crossing = np.where(omega >= threshold)[0]
    if crossing.size == 0:
        return None
    idx = int(crossing[0])
    return float(time_min[idx]), float(omega[idx])


def _simulate_density_series(density: float) -> Dict[str, np.ndarray]:
    """Run one density case and return plotting arrays."""
    params = TransitScenarioParameters(
        platform_tdb_c=34.0,
        platform_tr_c=34.0,
        platform_v_ms=0.45,
        platform_rh_percent=65.0,
        carriage_tdb_c=26.0,
        carriage_wall_c=25.0,
        carriage_nominal_v_ms=0.45,
        carriage_rh_env_percent=55.0,
        occupant_density_per_m2=density,
        clothing_base_clo=0.5,
        time_horizon_s=900.0,
        time_step_s=10.0,
    )
    series = simulate_platform_to_carriage(params)

    time_min = np.array([r.time_s for r in series], dtype=float) / 60.0
    pmv_eatm = np.array([r.pmv_eatm for r in series], dtype=float)
    omega = np.array([r.skin_wettedness for r in series], dtype=float)
    t_air = np.full_like(time_min, params.carriage_tdb_c, dtype=float)
    p_vap_local_pa = np.array(
        [
            saturation_vapor_pressure_magnus_pa(params.carriage_tdb_c) * (r.local_rh_percent / 100.0)
            for r in series
        ],
        dtype=float,
    )

    return {
        "time_min": time_min,
        "pmv_eatm": pmv_eatm,
        "omega": omega,
        "t_air": t_air,
        "p_vap_local_pa": p_vap_local_pa,
    }


def create_multidensity_pmv_figure(density_data: Dict[float, Dict[str, np.ndarray]], out_path: Path) -> None:
    """Create PMV_EATM benchmarking plot against steady-state baseline."""
    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    colors = plt.get_cmap("tab10")
    markers = {0.0: "o", 4.0: "s", 8.0: "D"}

    baseline_pmv = pmv_iso_baseline_reference(
        tdb=26.0,
        tr=26.0,
        vr=0.45,
        rh_percent=55.0,
        met=1.2,
        clo=0.5,
    )
    ax.axhline(
        baseline_pmv,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=rf"$PMV_{{\mathrm{{ISO,steady}}}}={baseline_pmv:.3f}$",
    )

    for i, density in enumerate(sorted(density_data)):
        data = density_data[density]
        linewidth = 3.0 if density == 8.0 else 2.2
        zorder = 4 if density == 8.0 else 3
        ax.plot(
            data["time_min"],
            data["pmv_eatm"],
            color=colors(i),
            linewidth=linewidth,
            label=rf"$PMV_{{\mathrm{{EATM}}}}(D={int(density)}\,\mathrm{{pass/m^2}})$",
            zorder=zorder,
        )
        crossing = _first_meltdown_crossing(data["time_min"], data["omega"], threshold=0.25)
        if crossing is not None:
            t_c, omega_c = crossing
            pmv_c = float(data["pmv_eatm"][np.where(data["time_min"] == t_c)[0][0]])
            ax.scatter(
                [t_c],
                [pmv_c],
                color=colors(i),
                edgecolor="white",
                linewidth=0.8,
                s=55,
                marker=markers.get(density, "o"),
                zorder=6,
                label=rf"$\omega\geq0.25$ at $t={t_c:.2f}\,\mathrm{{min}}$ ($D={int(density)}$)",
            )
            ax.axvline(
                t_c,
                color=colors(i),
                linestyle=":",
                linewidth=1.1,
                alpha=0.7,
                zorder=2,
            )

    ax.set_xlabel(r"Time in carriage, $t\;(\mathrm{min})$")
    ax.set_ylabel(r"Expectancy-adjusted vote, $PMV_{\mathrm{EATM}}\;(-)$")
    ax.set_title(
        r"Why steady-state PMV is optimistic: crowding drives a persistent gap from "
        r"$PMV_{\mathrm{ISO,steady}}$",
        pad=10,
    )
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0.0, 15.0)
    ax.legend(loc="best", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def create_microenvironment_figure(density_data: Dict[float, Dict[str, np.ndarray]], out_path: Path) -> None:
    """Create CBE-style micro-environment trajectory in Ta-p_vap space."""
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    cmap = plt.get_cmap("magma")

    # Stylized comfort envelope in Ta-p_vap coordinates (approx. RH 30-60% around 24-26 C).
    ta_band = np.linspace(22.0, 27.0, 80)
    p_sat = np.array([saturation_vapor_pressure_magnus_pa(t) for t in ta_band], dtype=float)
    p_low = 0.30 * p_sat
    p_high = 0.60 * p_sat
    ax.fill_between(
        ta_band,
        p_low / 1000.0,
        p_high / 1000.0,
        color="#9ecae1",
        alpha=0.35,
        label=r"Approx. comfort envelope ($30\%-60\%$ RH)",
    )

    for i, density in enumerate(sorted(density_data)):
        data = density_data[density]
        t_air = data["t_air"]
        p_local_kpa = data["p_vap_local_pa"] / 1000.0
        color = cmap(0.2 + 0.3 * i)
        linewidth = 2.7 if density == 8.0 else 2.0
        ax.plot(
            t_air,
            p_local_kpa,
            color=color,
            linewidth=linewidth,
            label=rf"Trajectory, $D={int(density)}\,\mathrm{{pass/m^2}}$",
        )
        ax.scatter(
            [t_air[0]],
            [p_local_kpa[0]],
            color=color,
            marker="o",
            s=40,
            zorder=5,
        )
        ax.scatter(
            [t_air[-1]],
            [p_local_kpa[-1]],
            color=color,
            marker="^",
            s=48,
            zorder=5,
        )

    ax.set_xlabel(r"Air temperature, $T_{\mathrm{air}}\;(^\circ\mathrm{C})$")
    ax.set_ylabel(r"Local vapor pressure, $p_{\mathrm{vap,local}}\;(\mathrm{kPa})$")
    ax.set_title(
        r"Micro-environment trajectory under crowding: drift away from comfort-like humidity state"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    densities: List[float] = [0.0, 4.0, 8.0]
    density_data = {d: _simulate_density_series(d) for d in densities}

    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    pmv_path = out_dir / "pmv_eatm_multidensity_benchmark.png"
    microenv_path = out_dir / "microenvironment_trajectory_cbe_style.png"

    create_multidensity_pmv_figure(density_data, pmv_path)
    create_microenvironment_figure(density_data, microenv_path)

    print(f"Wrote figure: {pmv_path}")
    print(f"Wrote figure: {microenv_path}")

    for density in densities:
        crossing = _first_meltdown_crossing(
            density_data[density]["time_min"],
            density_data[density]["omega"],
            threshold=0.25,
        )
        if crossing is None:
            print(f"D={int(density)}: no omega crossing for omega >= 0.25")
        else:
            print(f"D={int(density)}: first omega crossing at t={crossing[0]:.2f} min")


if __name__ == "__main__":
    main()
