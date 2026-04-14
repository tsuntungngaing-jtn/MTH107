"""
PMV_EATM platform-to-carriage animation for the D = 6 scenario.

This script reuses the run_platform_to_carriage_scenario configuration and
exports an animated GIF showing the PMV_EATM trajectory growing in time, with
the sensory-collapse moment (first omega >= 0.25) highlighted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.eatm import TransitScenarioParameters, pmv_iso_baseline_reference, simulate_platform_to_carriage


def _simulate_d6():
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
    pmv_eatm = np.array([r.pmv_eatm for r in series], dtype=float)
    omega = np.array([r.skin_wettedness for r in series], dtype=float)

    collapse_idx: Optional[int] = None
    for i, w in enumerate(omega):
        if w >= 0.25:
            collapse_idx = i
            break

    collapse_t = t_min[collapse_idx] if collapse_idx is not None else None
    collapse_pmv = pmv_eatm[collapse_idx] if collapse_idx is not None else None

    return t_min, pmv_eatm, omega, collapse_idx, collapse_t, collapse_pmv


def create_pmv_animation(out_path: Path, fps: int = 10) -> None:
    """Create a GIF of the growing PMV_EATM curve with omega-collapse marker."""

    t_min, pmv_eatm, omega, collapse_idx, collapse_t, collapse_pmv = _simulate_d6()
    baseline_pmv = pmv_iso_baseline_reference()

    # Use Matplotlib's internal mathtext (LaTeX-like) so that we do not
    # depend on an external LaTeX installation on the cluster.
    plt.rcParams.update({"text.usetex": False, "font.family": "serif"})

    fig, ax = plt.subplots(figsize=(7.0, 4.3))

    ax.set_xlim(0.0, float(t_min.max()))
    # Include both PMV_EATM trajectory and steady ISO baseline in the y-range,
    # otherwise the green baseline can be clipped out of the rendered GIF.
    y_min_ref = min(float(np.min(pmv_eatm)), float(baseline_pmv))
    y_max_ref = max(float(np.max(pmv_eatm)), float(baseline_pmv))
    y_margin = max(0.3, float(y_max_ref - y_min_ref) * 0.25)
    ax.set_ylim(y_min_ref - y_margin, y_max_ref + y_margin)

    line_pmv, = ax.plot([], [], color="#1f77b4", linewidth=2.2, label=r"$PMV_{\mathrm{EATM}}$")
    collapse_scatter = ax.scatter([], [], s=70, color="#d62728", edgecolor="white", zorder=4)

    ax.axhline(
        baseline_pmv,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.4,
        label=r"$PMV_{\mathrm{ISO,steady}}$ baseline",
    )
    ax.set_xlabel(r"Time in carriage, $t\;(\mathrm{min})$")
    ax.set_ylabel(r"Expectancy-adjusted vote, $PMV_{\mathrm{EATM}}\;(-)$")
    ax.set_title(
        r"Dynamic evolution of $PMV_{\mathrm{EATM}}$ "
        r"(34$^\circ$C platform $\rightarrow$ 26$^\circ$C carriage, $D=6$)",
        pad=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=True)

    text_omega = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )

    n_frames = len(t_min)

    def init():
        line_pmv.set_data([], [])
        collapse_scatter.set_offsets(np.empty((0, 2)))
        text_omega.set_text("")
        return line_pmv, collapse_scatter, text_omega

    def update(frame: int):
        current_t = t_min[: frame + 1]
        current_pmv = pmv_eatm[: frame + 1]
        line_pmv.set_data(current_t, current_pmv)

        if collapse_idx is not None and frame >= collapse_idx:
            collapse_scatter.set_offsets(np.array([[collapse_t, collapse_pmv]]))
            text_omega.set_text(
                rf"$\omega_{{\mathrm{{crit}}}}=0.25$ reached at "
                rf"$t={collapse_t:.2f}\,\mathrm{{min}}$"
            )
        else:
            collapse_scatter.set_offsets(np.empty((0, 2)))
            text_omega.set_text(
                rf"$\omega(t) < 0.25$ for $t \leq {current_t[-1]:.2f}\,\mathrm{{min}}$"
            )

        return line_pmv, collapse_scatter, text_omega

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=1000 / fps,
        blit=True,
        repeat=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer="pillow", dpi=150)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "output"
    out_gif = out_dir / "pmv_eatm_platform_to_carriage.gif"
    create_pmv_animation(out_gif, fps=10)
    print(f"Wrote animation: {out_gif}")


if __name__ == "__main__":
    main()

