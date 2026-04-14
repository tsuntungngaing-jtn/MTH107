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

import matplotlib.animation as animation
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


def create_heat_partition_comparison_figure(
    nominal_data: Dict[str, np.ndarray],
    crowded_data: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Stacked-bar comparison at t=15 min: nominal (D=0) vs crowded (D=6)."""

    labels = ["Nominal (D=0)", "Crowded (D=6)"]
    x = np.arange(2, dtype=float)

    c = np.array([nominal_data["C"][-1], crowded_data["C"][-1]], dtype=float)
    r = np.array([nominal_data["R"][-1], crowded_data["R"][-1]], dtype=float)
    esk = np.array([nominal_data["E_sk"][-1], crowded_data["E_sk"][-1]], dtype=float)
    total = c + r + esk
    sensible = c + r

    colors = {
        "C": "#3182bd",  # blue
        "R": "#9e9ac8",  # violet
        "E_sk": "#fc9272",  # warm red
    }

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    width = 0.58
    b_c = ax.bar(x, c, width=width, color=colors["C"], label=r"Convection, $C$")
    b_r = ax.bar(x, r, width=width, bottom=c, color=colors["R"], label=r"Radiation, $R$")
    b_e = ax.bar(
        x,
        esk,
        width=width,
        bottom=c + r,
        color=colors["E_sk"],
        label=r"Skin latent, $E_{\mathrm{sk}}$",
    )

    # Percentage labels to emphasize pathway distortion at D=6.
    for i in range(2):
        ax.text(
            x[i],
            c[i] / 2.0,
            f"{100.0 * c[i] / total[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
        ax.text(
            x[i],
            c[i] + r[i] / 2.0,
            f"{100.0 * r[i] / total[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
        ax.text(
            x[i],
            c[i] + r[i] + esk[i] / 2.0,
            f"{100.0 * esk[i] / total[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            fontweight="bold",
        )
        ax.text(
            x[i],
            total[i] + 0.9,
            rf"$C+R={sensible[i]:.2f}$ ({100.0 * sensible[i] / total[i]:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.annotate(
        r"Crowding distortion: dry channels ($C+R$) compressed," "\n"
        r"thermal balance forced toward $E_{\mathrm{sk}}$",
        xy=(x[1], total[1]),
        xytext=(1.35, max(total) * 0.80),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "black"},
        fontsize=9,
        ha="left",
        va="center",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Heat-loss component, $q\;(\mathrm{W/m^2})$")
    ax.set_title(
        r"Heat-loss partition comparison at $t=15\;\mathrm{min}$: "
        r"Nominal vs. Crowded microclimate",
        pad=10,
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    # Optional animation: progressive reveal of stacked bars.
    anim_dir = out_path.parent / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    anim_path = anim_dir / "heat_partition_comparison.gif"

    fig_a, ax_a = plt.subplots(figsize=(9.0, 5.6))
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel(r"Heat-loss component, $q\;(\mathrm{W/m^2})$")
    ax_a.set_title(r"Progressive partition reveal: Nominal vs. Crowded", pad=10)
    ax_a.grid(True, axis="y", alpha=0.25)
    ax_a.set_ylim(0.0, float(max(total) * 1.25))

    bar_c = ax_a.bar(x, [0.0, 0.0], width=width, color=colors["C"], label=r"$C$")
    bar_r = ax_a.bar(x, [0.0, 0.0], width=width, bottom=[0.0, 0.0], color=colors["R"], label=r"$R$")
    bar_e = ax_a.bar(x, [0.0, 0.0], width=width, bottom=[0.0, 0.0], color=colors["E_sk"], label=r"$E_{\mathrm{sk}}$")
    ax_a.legend(loc="upper right", frameon=True, fontsize=9)

    def _anim_update(frame: int):
        frac = frame / 40.0
        c_now = c * frac
        r_now = r * frac
        e_now = esk * frac
        for i in range(2):
            bar_c[i].set_height(c_now[i])
            bar_r[i].set_y(c_now[i])
            bar_r[i].set_height(r_now[i])
            bar_e[i].set_y(c_now[i] + r_now[i])
            bar_e[i].set_height(e_now[i])
        return tuple(bar_c) + tuple(bar_r) + tuple(bar_e)

    anim = animation.FuncAnimation(
        fig_a,
        _anim_update,
        frames=41,
        interval=40,
        blit=True,
        repeat=False,
    )
    anim.save(anim_path, writer=animation.PillowWriter(fps=20), dpi=140)
    plt.close(fig_a)


def create_omega_max_sensitivity_figure(
    densities: List[int],
    density_series: Dict[int, Dict[str, np.ndarray]],
    out_path: Path,
    omega_crit: float = 0.25,
) -> None:
    """Line+scatter plot of omega_max versus density D."""

    d_array = np.array(densities, dtype=float)
    omega_max = np.array([float(np.max(density_series[d]["omega"])) for d in densities], dtype=float)

    cross_idx = np.where(omega_max >= omega_crit)[0]
    cross_i = int(cross_idx[0]) if cross_idx.size > 0 else None

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(
        d_array,
        omega_max,
        color="#08519c",
        linewidth=2.2,
        marker="o",
        markersize=5.5,
        label=r"$\omega_{\max}(D)$",
    )
    ax.axhline(
        omega_crit,
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        label=rf"Critical Threshold ({omega_crit:.2f})",
    )

    if cross_i is not None:
        ax.scatter(
            [d_array[cross_i]],
            [omega_max[cross_i]],
            s=75,
            marker="D",
            color="#2ca02c",
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
            label=rf"First crossing at $D={int(d_array[cross_i])}$",
        )
        ax.annotate(
            rf"$D={int(d_array[cross_i])}$ crosses $\omega_{{crit}}$",
            xy=(d_array[cross_i], omega_max[cross_i]),
            xytext=(d_array[cross_i] + 0.8, omega_max[cross_i] + 0.015),
            arrowprops={"arrowstyle": "->", "lw": 1.1},
            fontsize=8,
            ha="left",
            va="bottom",
        )

    ax.set_xlabel(r"Occupant density, $D\;(\mathrm{pass/m^2})$")
    ax.set_ylabel(r"Peak skin wettedness, $\omega_{\max}\;(-)$")
    ax.set_title(
        r"Crowding sensitivity of physiological load: "
        r"$\omega_{\max}$ vs. $D$",
        pad=10,
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(d_array.min() - 0.2, d_array.max() + 0.2)
    ax.set_ylim(max(0.10, float(np.min(omega_max)) - 0.02), float(np.max(omega_max)) + 0.03)
    ax.legend(loc="best", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    # Optional animation: trajectory of omega_max accumulation over density sweep.
    anim_dir = out_path.parent / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    anim_path = anim_dir / "omega_max_vs_density.gif"

    fig_a, ax_a = plt.subplots(figsize=(7.6, 4.8))
    ax_a.axhline(omega_crit, color="#d62728", linestyle="--", linewidth=1.4)
    ax_a.set_xlabel(r"Occupant density, $D\;(\mathrm{pass/m^2})$")
    ax_a.set_ylabel(r"Peak skin wettedness, $\omega_{\max}\;(-)$")
    ax_a.set_title(r"Density sweep animation: $\omega_{\max}(D)$", pad=10)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xlim(d_array.min() - 0.2, d_array.max() + 0.2)
    ax_a.set_ylim(max(0.10, float(np.min(omega_max)) - 0.02), float(np.max(omega_max)) + 0.03)
    (line,) = ax_a.plot([], [], color="#08519c", linewidth=2.2, marker="o", markersize=5.5)

    def _omega_update(frame: int):
        line.set_data(d_array[: frame + 1], omega_max[: frame + 1])
        return (line,)

    anim = animation.FuncAnimation(
        fig_a,
        _omega_update,
        frames=len(d_array),
        interval=260,
        blit=True,
        repeat=False,
    )
    anim.save(anim_path, writer=animation.PillowWriter(fps=4), dpi=140)
    plt.close(fig_a)


def main() -> None:
    """Entry point to generate supplementary analysis figures."""

    # End-state partition comparison: nominal D=0 vs crowded D=6.
    nominal_series = _simulate_density_series(0.0)
    crowded_series = _simulate_density_series(6.0)

    # Sensitivity sweep for D in [0, 9].
    densities_int: List[int] = list(range(0, 10))
    sensitivity_series: Dict[int, Dict[str, np.ndarray]] = {
        d: _simulate_density_series(float(d)) for d in densities_int
    }

    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = out_dir / "heat_partition_comparison.png"
    omega_sensitivity_path = out_dir / "omega_max_vs_density.png"

    create_heat_partition_comparison_figure(nominal_series, crowded_series, comparison_path)
    create_omega_max_sensitivity_figure(
        densities_int,
        sensitivity_series,
        omega_sensitivity_path,
        omega_crit=0.25,
    )

    print(f"Wrote figure: {comparison_path}")
    print(f"Wrote figure: {omega_sensitivity_path}")
    print(f"Wrote animation: {out_dir / 'animations' / 'heat_partition_comparison.gif'}")
    print(f"Wrote animation: {out_dir / 'animations' / 'omega_max_vs_density.gif'}")


if __name__ == "__main__":
    main()

