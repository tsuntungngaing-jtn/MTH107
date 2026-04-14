"""
Generate presentation-ready EATM animations (GIF) using matplotlib.animation.

Core physics:
  - modules/eatm.py: scenario integration + EATM state variables
  - modules/pmv_dynamic_iso.py: ISO heat loss partition (C, R, E_sk)
  - modules/environmental.py: vapor pressure conversions (for psychrometric view)

Scenario:
  Platform (34°C) -> Carriage (26°C), D = 6 pass/m^2, t = 0..15 min, dt = 10 s

Outputs (fps=20, dpi=150):
  1) sensory_meltdown_dual_panel.gif
  2) heat_loss_pathway_shifting.gif
  3) pmv_eatm_tracking.gif
  4) psychrometric_trajectory.gif
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CRITICAL_SKIN_WETTEDNESS
from modules.eatm import TransitScenarioParameters, pmv_iso_baseline_reference, simulate_platform_to_carriage
from modules.environmental import saturation_vapor_pressure_magnus_pa
from modules.pmv_dynamic_iso import iso7730_heat_losses_dict_for_docs


@dataclass(frozen=True)
class ScenarioSeries:
    t_min: np.ndarray
    omega: np.ndarray
    lam: np.ndarray
    pmv_eatm: np.ndarray
    pmv_dynamic: np.ndarray
    thermal_load: np.ndarray
    c_wm2: np.ndarray
    r_wm2: np.ndarray
    esk_wm2: np.ndarray
    p_vap_local_pa: np.ndarray
    t_air_c: float
    omega_crit: float
    collapse_idx: Optional[int]


def _first_crossing_idx(y: np.ndarray, threshold: float) -> Optional[int]:
    idx = np.where(y >= threshold)[0]
    if idx.size == 0:
        return None
    return int(idx[0])


def simulate_core_scenario() -> ScenarioSeries:
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

    t_min = np.array([r.time_s for r in series], dtype=float) / 60.0
    omega = np.array([r.skin_wettedness for r in series], dtype=float)
    lam = np.array([r.lambda_offset for r in series], dtype=float)
    pmv_eatm = np.array([r.pmv_eatm for r in series], dtype=float)
    pmv_dynamic = np.array([r.pmv_dynamic for r in series], dtype=float)
    thermal_load = np.array([r.thermal_load_wm2 for r in series], dtype=float)

    # Vapor pressure (Pa) reconstructed from the *local* RH and saturation at carriage Ta.
    p_sat = saturation_vapor_pressure_magnus_pa(params.carriage_tdb_c)
    p_vap_local_pa = np.array([p_sat * (r.local_rh_percent / 100.0) for r in series], dtype=float)

    # ISO 7730 loss partition (W/m^2): C, R, E_sk.
    c_list = []
    r_list = []
    esk_list = []
    for i, r in enumerate(series):
        losses = iso7730_heat_losses_dict_for_docs(
            tdb=params.carriage_tdb_c,
            tr=float(r.t_mr_prime_c),
            vr=float(r.local_velocity_ms),
            vapor_pressure_pa=float(p_vap_local_pa[i]),
            met=float(r.metabolic_met),
            clo=float(r.corrected_clo),
            wme=0.0,
        )
        c_list.append(losses["C"])
        r_list.append(losses["R"])
        esk_list.append(losses["E_sk"])

    omega_crit = float(CRITICAL_SKIN_WETTEDNESS)
    collapse_idx = _first_crossing_idx(omega, omega_crit)

    return ScenarioSeries(
        t_min=t_min,
        omega=omega,
        lam=lam,
        pmv_eatm=pmv_eatm,
        pmv_dynamic=pmv_dynamic,
        thermal_load=thermal_load,
        c_wm2=np.array(c_list, dtype=float),
        r_wm2=np.array(r_list, dtype=float),
        esk_wm2=np.array(esk_list, dtype=float),
        p_vap_local_pa=p_vap_local_pa,
        t_air_c=float(params.carriage_tdb_c),
        omega_crit=omega_crit,
        collapse_idx=collapse_idx,
    )


def _set_theme() -> None:
    # Paper-like style; keep mathtext (LaTeX-like) to avoid system LaTeX dependency.
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )


def _save_gif(anim: animation.FuncAnimation, out_path: Path, fps: int = 20, dpi: int = 150) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=dpi)


def animation_1_sensory_meltdown(series: ScenarioSeries, out_path: Path, fps: int = 20, dpi: int = 150) -> None:
    """Dual-panel: omega growth + lambda bar that collapses when omega hits omega_crit."""

    _set_theme()
    fig, (ax_w, ax_lam) = plt.subplots(1, 2, figsize=(10.8, 4.0), gridspec_kw={"width_ratios": [2.2, 1.0]})

    ax_w.set_xlim(0.0, float(series.t_min.max()))
    ax_w.set_ylim(0.0, max(0.35, float(series.omega.max()) * 1.15))
    ax_w.set_xlabel(r"Time, $t\;(\mathrm{min})$")
    ax_w.set_ylabel(r"Skin wettedness, $\omega\;(-)$")
    ax_w.set_title(r"Sensory meltdown trigger: $\omega(t)$")
    ax_w.grid(True, alpha=0.3)
    ax_w.axhline(series.omega_crit, color="#d62728", linestyle="--", linewidth=1.6, label=rf"$\omega_{{crit}}={series.omega_crit:.2f}$")

    (omega_line,) = ax_w.plot([], [], color="#9467bd", linewidth=2.2, label=r"$\omega$")
    omega_dot = ax_w.scatter([], [], s=55, color="#9467bd", edgecolor="white", linewidth=0.8, zorder=4)
    collapse_marker = ax_w.scatter([], [], s=85, color="#d62728", edgecolor="white", linewidth=0.9, zorder=5)

    ax_w.legend(loc="lower right", frameon=True)

    # Lambda bar (psychological offset). Lambda can be negative (cooling relief),
    # so we plot it as a signed bar around zero.
    lam_abs_max = max(1e-6, float(np.max(np.abs(series.lam))))
    ax_lam.set_ylim(-lam_abs_max * 1.15, lam_abs_max * 1.15)
    ax_lam.set_xlim(-0.6, 0.6)
    ax_lam.set_xticks([])
    ax_lam.set_ylabel(r"Psychological offset, $\lambda$")
    ax_lam.set_title(r"Boundary-defense collapse ($f_{def}$)")
    ax_lam.grid(True, axis="y", alpha=0.25)
    ax_lam.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)

    bar = ax_lam.bar([0.0], [0.0], width=0.65, color="#1f77b4", alpha=0.85)[0]
    txt = ax_lam.text(0.0, lam_abs_max * 1.05, "", ha="center", va="top", fontsize=9)

    # Presentation effect: after collapse, shrink the bar rapidly to emphasize the trigger.
    def _lambda_display(frame: int) -> float:
        base = float(series.lam[frame])
        if series.collapse_idx is None or frame < series.collapse_idx:
            return base
        k = frame - series.collapse_idx
        return base * np.exp(-0.9 * k)

    def init():
        omega_line.set_data([], [])
        omega_dot.set_offsets(np.empty((0, 2)))
        collapse_marker.set_offsets(np.empty((0, 2)))
        bar.set_height(0.0)
        txt.set_text("")
        return omega_line, omega_dot, collapse_marker, bar, txt

    def update(frame: int):
        t = series.t_min[: frame + 1]
        w = series.omega[: frame + 1]
        omega_line.set_data(t, w)
        omega_dot.set_offsets(np.array([[t[-1], w[-1]]]))

        if series.collapse_idx is not None and frame >= series.collapse_idx:
            collapse_marker.set_offsets(np.array([[series.t_min[series.collapse_idx], series.omega[series.collapse_idx]]]))
        else:
            collapse_marker.set_offsets(np.empty((0, 2)))

        lam_h = _lambda_display(frame)
        bar.set_height(lam_h)

        if series.collapse_idx is not None and frame >= series.collapse_idx:
            txt.set_text(rf"Collapse at $t={series.t_min[series.collapse_idx]:.2f}\,\mathrm{{min}}$")
            bar.set_color("#d62728")
        else:
            txt.set_text(rf"$\lambda(t) = {float(series.lam[frame]):.3f}$")
            bar.set_color("#1f77b4")

        return omega_line, omega_dot, collapse_marker, bar, txt

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(series.t_min), interval=1000 / fps, blit=True, repeat=False)
    fig.tight_layout()
    _save_gif(anim, out_path, fps=fps, dpi=dpi)
    plt.close(fig)


def animation_2_heat_loss_pathways(series: ScenarioSeries, out_path: Path, fps: int = 20, dpi: int = 150) -> None:
    """Dynamic stacked bar of (C, R, E_sk) proportions, with narrative annotation."""

    _set_theme()
    fig, ax = plt.subplots(figsize=(9.6, 4.2))

    q = np.vstack([series.c_wm2, series.r_wm2, series.esk_wm2]).T
    q_sum = np.sum(q, axis=1)
    frac = q / np.clip(q_sum[:, None], 1e-9, None)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel(r"Heat-loss share, $q_i / (C+R+E_{\mathrm{sk}})$")
    ax.set_title(r"Heat loss pathway shifting: $C$ suppressed (shielding) $\Rightarrow$ $E_{\mathrm{sk}}$ share expands")
    ax.grid(True, axis="x", alpha=0.25)

    colors = {"C": "#3182bd", "R": "#9e9ac8", "E_sk": "#fc9272"}
    bC = ax.barh([0.0], [0.0], left=0.0, height=0.55, color=colors["C"], label=r"$C$ (Convection)")[0]
    bR = ax.barh([0.0], [0.0], left=0.0, height=0.55, color=colors["R"], label=r"$R$ (Radiation)")[0]
    bE = ax.barh([0.0], [0.0], left=0.0, height=0.55, color=colors["E_sk"], label=r"$E_{\mathrm{sk}}$ (Evaporation)")[0]

    dot = ax.scatter([], [], s=60, color="black", zorder=5)

    # Fixed scenario annotation for L(t) range.
    L_min = float(np.min(series.thermal_load))
    L_max = float(np.max(series.thermal_load))
    txt = ax.text(
        0.02,
        0.92,
        rf"$L(t)\in[{L_min:.1f},{L_max:.1f}]\,\mathrm{{W/m^2}}$; $D=6$; $T_a=26^\circ\mathrm{{C}}$",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )

    ax.legend(loc="lower right", frameon=True)

    def init():
        bC.set_width(0.0)
        bR.set_width(0.0)
        bE.set_width(0.0)
        bR.set_x(0.0)
        bE.set_x(0.0)
        dot.set_offsets(np.empty((0, 2)))
        return bC, bR, bE, dot, txt

    def update(frame: int):
        fC, fR, fE = map(float, frac[frame])
        bC.set_width(fC)
        bR.set_x(fC)
        bR.set_width(fR)
        bE.set_x(fC + fR)
        bE.set_width(fE)

        # Tracking dot at the *center* of the E_sk segment.
        dot.set_offsets(np.array([[fC + fR + 0.5 * fE, 0.0]]))
        return bC, bR, bE, dot, txt

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(series.t_min), interval=1000 / fps, blit=True, repeat=False)
    fig.tight_layout()
    _save_gif(anim, out_path, fps=fps, dpi=dpi)
    plt.close(fig)


def animation_3_pmv_tracking(series: ScenarioSeries, out_path: Path, fps: int = 20, dpi: int = 150) -> None:
    """Tracking plot: PMV_EATM grows, baseline static, dot tracks and shows convergence to PMV_dynamic."""

    _set_theme()
    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    pmv_base = float(pmv_iso_baseline_reference())

    ax.set_xlim(0.0, float(series.t_min.max()))
    y0 = float(min(series.pmv_eatm.min(), series.pmv_dynamic.min(), pmv_base))
    y1 = float(max(series.pmv_eatm.max(), series.pmv_dynamic.max(), pmv_base))
    ax.set_ylim(y0 - 0.35, y1 + 0.35)

    ax.set_xlabel(r"Time, $t\;(\mathrm{min})$")
    ax.set_ylabel(r"Vote, $PMV\;(-)$")
    ax.set_title(r"$PMV_{\mathrm{EATM}}$ vs. steady baseline: expectancy peak $\rightarrow$ physiological truth")
    ax.grid(True, alpha=0.3)

    ax.axhline(pmv_base, color="#2ca02c", linestyle="--", linewidth=1.6, label=rf"$PMV_{{\mathrm{{ISO,steady}}}}\approx {pmv_base:.2f}$")

    (line_eatm,) = ax.plot([], [], color="#1f77b4", linewidth=2.4, label=r"$PMV_{\mathrm{EATM}}$")
    (line_dyn,) = ax.plot([], [], color="#ff7f0e", linewidth=2.0, linestyle=":", label=r"$PMV_{\mathrm{dynamic}}$")
    dot = ax.scatter([], [], s=65, color="#1f77b4", edgecolor="white", linewidth=0.9, zorder=5)
    vline = ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.35)

    txt = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=9, va="top")
    ax.legend(loc="best", frameon=True)

    def init():
        line_eatm.set_data([], [])
        line_dyn.set_data([], [])
        dot.set_offsets(np.empty((0, 2)))
        vline.set_xdata([0.0, 0.0])
        txt.set_text("")
        return line_eatm, line_dyn, dot, vline, txt

    def update(frame: int):
        t = series.t_min[: frame + 1]
        line_eatm.set_data(t, series.pmv_eatm[: frame + 1])
        line_dyn.set_data(t, series.pmv_dynamic[: frame + 1])
        dot.set_offsets(np.array([[t[-1], float(series.pmv_eatm[frame])]]))
        vline.set_xdata([t[-1], t[-1]])

        txt.set_text(rf"$\lambda(t)={float(series.lam[frame]):.3f}$; $\omega(t)={float(series.omega[frame]):.3f}$")
        return line_eatm, line_dyn, dot, vline, txt

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(series.t_min), interval=1000 / fps, blit=True, repeat=False)
    fig.tight_layout()
    _save_gif(anim, out_path, fps=fps, dpi=dpi)
    plt.close(fig)


def _psychrometric_background(ax, t_min: float = 20.0, t_max: float = 30.0) -> None:
    """Simplified psychrometric background: saturation curve + a few RH guide lines."""

    Ta = np.linspace(t_min, t_max, 200)
    p_sat = np.array([saturation_vapor_pressure_magnus_pa(float(t)) for t in Ta], dtype=float) / 1000.0  # kPa
    ax.plot(Ta, p_sat, color="black", linewidth=1.2, label=r"Saturation, $p_{sat}(T_a)$")

    for rh in [0.3, 0.5, 0.7, 0.9]:
        ax.plot(Ta, rh * p_sat, color="gray", linewidth=0.9, alpha=0.55, linestyle="--")
        ax.text(Ta[-1] + 0.1, rh * p_sat[-1], rf"{int(rh*100)}\% RH", fontsize=8, va="center", color="gray")

    ax.set_xlabel(r"Air temperature, $T_a\;(^\circ\mathrm{C})$")
    ax.set_ylabel(r"Local vapor pressure, $p_{\mathrm{vap,local}}\;(\mathrm{kPa})$")
    ax.grid(True, alpha=0.25)


def animation_4_psychrometric_trajectory(series: ScenarioSeries, out_path: Path, fps: int = 20, dpi: int = 150) -> None:
    """Dynamic psychrometric trajectory: moving point in (Ta, p_vap,local) space."""

    _set_theme()
    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    _psychrometric_background(ax, t_min=20.0, t_max=30.0)

    Ta = float(series.t_air_c)
    p_kpa = series.p_vap_local_pa / 1000.0

    ax.set_xlim(20.0, 30.5)
    ax.set_ylim(0.0, float(np.max(p_kpa) * 1.18))
    ax.set_title(r"Dynamic psychrometric trajectory: crowding-driven rise in $p_{\mathrm{vap,local}}$")

    # Full trajectory line (faint) + animated segment + moving point.
    ax.plot([Ta] * len(p_kpa), p_kpa, color="#9ecae1", linewidth=1.2, alpha=0.45, label=r"Trajectory (full)")
    (seg_line,) = ax.plot([], [], color="#08519c", linewidth=2.4, label=r"Trajectory (animated)")
    dot = ax.scatter([], [], s=70, color="#08519c", edgecolor="white", linewidth=0.9, zorder=5)

    # 20% guide from initial.
    p0 = float(p_kpa[0])
    ax.axhline(p0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axhline(1.2 * p0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(20.2, p0, r"initial $p_{\mathrm{vap,local}}$", fontsize=8, va="bottom", color="gray")
    ax.text(20.2, 1.2 * p0, r"+20\% guide", fontsize=8, va="bottom", color="gray")

    txt = ax.text(0.02, 0.94, "", transform=ax.transAxes, fontsize=9, va="top")
    ax.legend(loc="lower right", frameon=True)

    def init():
        seg_line.set_data([], [])
        dot.set_offsets(np.empty((0, 2)))
        txt.set_text("")
        return seg_line, dot, txt

    def update(frame: int):
        y = p_kpa[: frame + 1]
        x = np.full_like(y, Ta, dtype=float)
        seg_line.set_data(x, y)
        dot.set_offsets(np.array([[Ta, float(y[-1])]]))

        pct = 100.0 * (float(y[-1]) / p0 - 1.0) if p0 > 0 else 0.0
        txt.set_text(rf"$t={series.t_min[frame]:.2f}\,\mathrm{{min}}$;  $p_{{\mathrm{{vap,local}}}}={float(y[-1]):.3f}\,\mathrm{{kPa}}$  ({pct:+.1f}\%)")
        return seg_line, dot, txt

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(series.t_min), interval=1000 / fps, blit=True, repeat=False)
    fig.tight_layout()
    _save_gif(anim, out_path, fps=fps, dpi=dpi)
    plt.close(fig)


def main() -> None:
    series = simulate_core_scenario()

    out_dir = ROOT / "output" / "animations"
    out_dir.mkdir(parents=True, exist_ok=True)

    animation_1_sensory_meltdown(series, out_dir / "sensory_meltdown_dual_panel.gif", fps=20, dpi=150)
    animation_2_heat_loss_pathways(series, out_dir / "heat_loss_pathway_shifting.gif", fps=20, dpi=150)
    animation_3_pmv_tracking(series, out_dir / "pmv_eatm_tracking.gif", fps=20, dpi=150)
    animation_4_psychrometric_trajectory(series, out_dir / "psychrometric_trajectory.gif", fps=20, dpi=150)

    print("Wrote animations to:")
    for p in [
        out_dir / "sensory_meltdown_dual_panel.gif",
        out_dir / "heat_loss_pathway_shifting.gif",
        out_dir / "pmv_eatm_tracking.gif",
        out_dir / "psychrometric_trajectory.gif",
    ]:
        print(f" - {p}")


if __name__ == "__main__":
    main()

