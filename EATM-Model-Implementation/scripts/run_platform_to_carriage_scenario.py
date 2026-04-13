"""
Scenario test: 34 C platform -> 26 C carriage at D = 6 persons/m^2.

Plots PMV_EATM for the first 15 minutes, the ISO first-law thermal load L(t)
with metabolic trajectory M(t), skin wettedness, and the omega = 0.25 marker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.eatm import (
    TransitScenarioParameters,
    pmv_iso_baseline_reference,
    simulate_platform_to_carriage,
)


def main() -> None:
    params = TransitScenarioParameters(
        platform_tdb_c=34.0,
        platform_tr_c=34.0,
        platform_v_ms=0.45,
        platform_rh_percent=65.0,
        carriage_tdb_c=26.0,
        carriage_wall_c=25.0,
        carriage_nominal_v_ms=0.45,
        carriage_rh_env_percent=55.0,
        occupant_density_per_m2=6.0,
        clothing_base_clo=0.5,
        time_horizon_s=900.0,
        time_step_s=10.0,
    )
    series = simulate_platform_to_carriage(params)
    t_min = np.array([r.time_s for r in series]) / 60.0
    pmv_eatm = np.array([r.pmv_eatm for r in series])
    pmv_dyn = np.array([r.pmv_dynamic for r in series])
    omega = np.array([r.skin_wettedness for r in series])
    thermal_load = np.array([r.thermal_load_wm2 for r in series])
    met_series = np.array([r.metabolic_met for r in series])
    bracket = np.array([r.pmv_sensitivity_bracket for r in series])

    pmv_base = pmv_iso_baseline_reference()

    collapse_idx = None
    for i, w in enumerate(omega):
        if w >= 0.25:
            collapse_idx = i
            break
    collapse_t_min = t_min[collapse_idx] if collapse_idx is not None else None

    fig, (ax_pmv, ax_l, ax_w) = plt.subplots(
        3,
        1,
        figsize=(9, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.3, 1.1]},
    )
    ax_pmv.plot(t_min, pmv_eatm, color="#1f77b4", linewidth=2.0, label=r"$PMV_{\mathrm{EATM}}$")
    ax_pmv.plot(
        t_min,
        pmv_dyn,
        color="#ff7f0e",
        linewidth=1.4,
        linestyle="--",
        label=r"$PMV_{\mathrm{dynamic}} = f(M)\cdot L(t)$ (ISO load, no $\lambda$)",
    )
    ax_pmv.axhline(
        pmv_base,
        color="#2ca02c",
        linestyle=":",
        linewidth=1.5,
        label="CBE ISO PMV baseline (uniform 26 C, no crowding)",
    )
    if collapse_t_min is not None:
        ax_pmv.axvline(
            collapse_t_min,
            color="#d62728",
            linestyle="-.",
            linewidth=1.5,
            label=r"Sensory collapse marker ($\omega \geq 0.25$)",
        )
        ax_w.axvline(collapse_t_min, color="#d62728", linestyle="-.", linewidth=1.2)
        ax_l.axvline(collapse_t_min, color="#d62728", linestyle="-.", linewidth=1.0, alpha=0.6)
    ax_pmv.set_ylabel("Predicted Mean Vote (PMV)")
    ax_pmv.set_title(
        "EATM: 34 C platform $\\rightarrow$ 26 C carriage, $D=6$ pass/m$^2$ "
        "(dynamic PMV from metabolic bracket $\\times$ first-law $L$)"
    )
    ax_pmv.grid(True, alpha=0.3)
    ax_pmv.legend(loc="best", fontsize=7)

    ax_l.plot(t_min, thermal_load, color="#17becf", linewidth=1.8, label=r"$L$ (ISO 7730 residual, W/m$^2$)")
    ax_l.set_ylabel(r"Thermal load $L$ (W/m$^2$)")
    ax_l.grid(True, alpha=0.3)
    ax_l.legend(loc="upper left", fontsize=8)
    ax_m = ax_l.twinx()
    ax_m.plot(t_min, met_series, color="#7f7f7f", linewidth=1.5, linestyle="--", label=r"$M(t)$ (met)")
    ax_m.set_ylabel("Metabolic rate (met)", color="#7f7f7f")
    ax_m.tick_params(axis="y", labelcolor="#7f7f7f")
    ax_m.legend(loc="upper right", fontsize=8)
    ax_l.text(
        0.02,
        0.05,
        f"Bracket mean: {float(np.mean(bracket)):.4f} (sensitivity $f(M)$ in W/m$^2$ domain)",
        transform=ax_l.transAxes,
        fontsize=7,
        va="bottom",
    )

    ax_w.plot(t_min, omega, color="#9467bd", linewidth=1.8, label=r"$\omega = E_{sk}/E_{max}$ (for $f_{def}$)")
    ax_w.axhline(0.25, color="#d62728", linestyle="--", linewidth=1.2, label=r"Critical $\omega$")
    ax_w.set_xlabel("Time in carriage (minutes)")
    ax_w.set_ylabel(r"Skin wettedness $\omega$ (-)")
    ax_w.set_ylim(0.0, max(0.35, float(np.max(omega)) * 1.15))
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(loc="best", fontsize=8)
    if collapse_t_min is None:
        ax_w.text(
            0.02,
            0.92,
            r"Threshold $\omega=0.25$ not crossed in this horizon (marker omitted on PMV panel).",
            transform=ax_w.transAxes,
            fontsize=8,
            va="top",
        )

    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pmv_eatm_platform_to_carriage.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote figure to {out_path}")
    print(f"Delta SET (fixed shock, SET units): {series[0].delta_set_fixed_k:.3f} K")
    print(f"CBE baseline PMV (reference): {pmv_base:.3f}")
    print(f"L(t) range (W/m^2): {float(np.min(thermal_load)):.2f} .. {float(np.max(thermal_load)):.2f}")
    print(f"M(t) range (met): {float(np.min(met_series)):.3f} .. {float(np.max(met_series)):.3f}")
    if collapse_t_min is not None:
        print(f"First crossing of omega = 0.25 at t = {collapse_t_min:.2f} min (index {collapse_idx}).")
    else:
        print("Skin wettedness remained below 0.25 for the entire horizon.")


if __name__ == "__main__":
    main()
