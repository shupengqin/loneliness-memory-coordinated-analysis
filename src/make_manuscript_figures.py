"""Create publication-ready figures from the five-cohort model outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_longitudinal_models import COHORTS, DERIVED_DIR, OUT_DIR, reml_meta


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False

FIG_DIR = OUT_DIR / "figures"
BLUE = "#0F4D92"
RED = "#B64342"
TEAL = "#42949E"
NEUTRAL = "#767676"
BLACK = "#272727"
LIGHT = "#D8D8D8"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def forest(
    ax: plt.Axes,
    frame: pd.DataFrame,
    label_column: str,
    color: str,
    xlabel: str,
    xlim: tuple[float, float],
) -> None:
    y = np.arange(len(frame))[::-1]
    for idx, row in enumerate(frame.itertuples(index=False)):
        yi = y[idx]
        row_color = BLACK if getattr(row, label_column) == "Pooled" else color
        marker = "D" if getattr(row, label_column) == "Pooled" else "o"
        size = 4.8 if marker == "D" else 4.2
        ax.plot([row.ci_low, row.ci_high], [yi, yi], color=row_color, lw=1.2)
        ax.plot(row.estimate, yi, marker=marker, ms=size, color=row_color, zorder=3)
    pooled_index = frame.index[frame[label_column].eq("Pooled")]
    if len(pooled_index):
        pooled_y = y[list(frame.index).index(pooled_index[0])]
        ax.axhspan(pooled_y - 0.45, pooled_y + 0.45, color="#F2F2F2", zorder=0)
    ax.axvline(0, color=NEUTRAL, ls="--", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(frame[label_column])
    ax.set_xlabel(xlabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, len(frame) - 0.3)
    ax.grid(axis="x", color=LIGHT, lw=0.5, alpha=0.55)
    ax.tick_params(axis="both", length=2.5, width=0.7)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main_figure() -> None:
    model = pd.read_csv(OUT_DIR / "cohort_model_estimates.csv")
    fixed = pd.read_csv(OUT_DIR / "model_fixed_effects.csv")
    sensitivity = pd.read_csv(OUT_DIR / "sensitivity_meta_results.csv")
    attrition = pd.read_csv(OUT_DIR / "attrition_weighted_meta_results.csv")
    meta = pd.read_csv(OUT_DIR / "meta_analysis_results.csv")

    baseline = fixed[(fixed["model"].eq("core")) & fixed["term"].eq("lonely")].copy()
    baseline = baseline.rename(columns={"term": "unused"})
    baseline_meta = reml_meta(baseline)
    baseline_plot = baseline[["cohort", "estimate", "ci_low", "ci_high"]].copy()
    baseline_plot = pd.concat(
        [
            baseline_plot,
            pd.DataFrame(
                [
                    {
                        "cohort": "Pooled",
                        "estimate": baseline_meta["pooled_estimate"],
                        "ci_low": baseline_meta["ci_low"],
                        "ci_high": baseline_meta["ci_high"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    core = model[model["model"].eq("core")].copy()
    core_meta = meta.loc[meta["model"].eq("core")].iloc[0]
    slope_plot = core[["cohort", "estimate", "ci_low", "ci_high"]].copy()
    slope_plot = pd.concat(
        [
            slope_plot,
            pd.DataFrame(
                [
                    {
                        "cohort": "Pooled",
                        "estimate": core_meta["pooled_estimate"],
                        "ci_low": core_meta["ci_low"],
                        "ci_high": core_meta["ci_high"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    sensitivity_labels = {
        "core": "Primary mixed model",
        "full": "Fully adjusted",
        "immediate_memory": "Immediate recall",
        "delayed_memory": "Delayed recall",
        "followup_adjusted_baseline_memory": "Follow-up + baseline memory",
        "exclude_lowest_baseline_decile": "Exclude lowest memory decile",
        "exclude_first_retest": "Exclude first retest",
        "exclude_followup_within_2y": "Exclude follow-up within 2 years",
        "gee_unweighted": "Unweighted GEE",
        "gee_attrition_ipw": "Attrition IPW",
        "gee_survey_x_attrition_ipw": "Survey x attrition weights",
    }
    robustness_rows = []
    for model_name in ["core", "full"]:
        row = meta.loc[meta["model"].eq(model_name)].iloc[0]
        robustness_rows.append(
            {
                "analysis": sensitivity_labels[model_name],
                "estimate": row["pooled_estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
            }
        )
    for analysis_name in [
        "immediate_memory",
        "delayed_memory",
        "followup_adjusted_baseline_memory",
        "exclude_lowest_baseline_decile",
        "exclude_first_retest",
        "exclude_followup_within_2y",
    ]:
        row = sensitivity.loc[sensitivity["analysis"].eq(analysis_name)].iloc[0]
        robustness_rows.append(
            {
                "analysis": sensitivity_labels[analysis_name],
                "estimate": row["pooled_estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
            }
        )
    for analysis_name in [
        "gee_unweighted",
        "gee_attrition_ipw",
        "gee_survey_x_attrition_ipw",
    ]:
        row = attrition.loc[attrition["analysis"].eq(analysis_name)].iloc[0]
        robustness_rows.append(
            {
                "analysis": sensitivity_labels[analysis_name],
                "estimate": row["pooled_estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
            }
        )
    robustness = pd.DataFrame(robustness_rows)

    timeline_rows = []
    for cohort in COHORTS:
        frame = pd.read_csv(DERIVED_DIR / f"{cohort.lower()}_long.csv.gz")
        for wave, group in frame.groupby("wave"):
            timeline_rows.append(
                {
                    "cohort": cohort,
                    "wave": int(wave),
                    "median_year": group["interview_year"].median(),
                    "n": group["pid"].nunique(),
                }
            )
    timeline = pd.DataFrame(timeline_rows)
    sample_sizes = model[model["model"].eq("core")].set_index("cohort")["participants"]

    fig = plt.figure(figsize=(7.1, 7.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.35], hspace=0.44, wspace=0.48)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    y_positions = np.arange(len(COHORTS))[::-1]
    for yi, cohort in zip(y_positions, COHORTS):
        group = timeline[timeline["cohort"].eq(cohort)].sort_values("median_year")
        ax_a.plot(
            [group["median_year"].min(), group["median_year"].max()],
            [yi, yi],
            color=LIGHT,
            lw=2.2,
            solid_capstyle="round",
        )
        sizes = 9 + 24 * np.sqrt(group["n"] / group["n"].max())
        baseline_wave = group["wave"].min()
        baseline_point = group["wave"].eq(baseline_wave)
        ax_a.scatter(
            group.loc[~baseline_point, "median_year"],
            np.repeat(yi, (~baseline_point).sum()),
            s=sizes[~baseline_point],
            color=BLUE,
            marker="o",
            zorder=3,
        )
        ax_a.scatter(
            group.loc[baseline_point, "median_year"],
            np.repeat(yi, baseline_point.sum()),
            s=sizes[baseline_point],
            color=RED,
            marker="s",
            zorder=4,
        )
        ax_a.text(
            group["median_year"].max() + 0.55,
            yi,
            f"n={int(sample_sizes[cohort]):,}",
            va="center",
            fontsize=6.4,
            color=NEUTRAL,
        )
    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels(COHORTS)
    ax_a.set_xlabel("Calendar year")
    ax_a.set_title("Cohort coverage and repeated assessments", loc="left", fontsize=7.5)
    ax_a.spines["left"].set_visible(False)
    ax_a.tick_params(axis="y", length=0)
    panel_label(ax_a, "a")

    forest(
        ax_b,
        baseline_plot,
        "cohort",
        RED,
        "Baseline memory difference (SD)",
        (-0.42, 0.08),
    )
    ax_b.set_title("Loneliness and baseline memory", loc="left", fontsize=7.5)
    panel_label(ax_b, "b")

    forest(
        ax_c,
        slope_plot,
        "cohort",
        BLUE,
        "Additional 10-year memory change (SD)",
        (-0.18, 0.30),
    )
    ax_c.set_title("Loneliness and longitudinal memory change", loc="left", fontsize=7.5)
    panel_label(ax_c, "c")

    forest(
        ax_d,
        robustness,
        "analysis",
        TEAL,
        "Pooled 10-year slope difference (SD)",
        (-0.16, 0.16),
    )
    ax_d.set_title("Robustness of the binary-exposure slope estimate", loc="left", fontsize=7.5)
    panel_label(ax_d, "d")

    fig.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.08)
    save_figure(fig, "figure1_main_results")

    source = []
    for panel, data, label in [
        ("b", baseline_plot, "cohort"),
        ("c", slope_plot, "cohort"),
        ("d", robustness, "analysis"),
    ]:
        for row in data.itertuples(index=False):
            source.append(
                {
                    "panel": panel,
                    "label": getattr(row, label),
                    "estimate": row.estimate,
                    "ci_low": row.ci_low,
                    "ci_high": row.ci_high,
                }
            )
    pd.DataFrame(source).to_csv(FIG_DIR / "figure1_source_data.csv", index=False)
    timeline.assign(panel="a").to_csv(FIG_DIR / "figure1_timeline_source_data.csv", index=False)


def trajectory_figure() -> None:
    trajectory_rows = []
    for cohort in COHORTS:
        frame = pd.read_csv(DERIVED_DIR / f"{cohort.lower()}_long.csv.gz")
        for (wave, lonely), group in frame.groupby(["wave", "lonely"]):
            trajectory_rows.append(
                {
                    "cohort": cohort,
                    "wave": int(wave),
                    "lonely": int(lonely),
                    "time": group["time"].median(),
                    "mean": group["memory_z"].mean(),
                    "se": group["memory_z"].std(ddof=1) / np.sqrt(len(group)),
                    "n": len(group),
                }
            )
    trajectory = pd.DataFrame(trajectory_rows)

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.5), sharey=True)
    axes = axes.ravel()
    for ax, cohort in zip(axes, COHORTS):
        group = trajectory[trajectory["cohort"].eq(cohort)]
        for lonely, label, color in [(0, "Not lonely", NEUTRAL), (1, "Lonely", RED)]:
            values = group[group["lonely"].eq(lonely)].sort_values("time")
            ax.errorbar(
                values["time"],
                values["mean"],
                yerr=1.96 * values["se"],
                color=color,
                marker="o",
                ms=3.2,
                lw=1.2,
                capsize=1.8,
                label=label,
            )
        ax.axhline(0, color=LIGHT, lw=0.7)
        ax.set_title(cohort, loc="left", fontsize=7.5, fontweight="bold")
        ax.set_xlabel("Years since baseline")
        ax.grid(axis="y", color=LIGHT, lw=0.5, alpha=0.5)
    axes[0].set_ylabel("Episodic memory (baseline SD)")
    axes[3].set_ylabel("Episodic memory (baseline SD)")
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", fontsize=7.5)
    axes[-1].text(
        0.5,
        0.28,
        "Wave-specific observed means\nwith 95% confidence intervals.\nSamples vary across waves;\nlines are not model-adjusted.",
        transform=axes[-1].transAxes,
        ha="center",
        va="center",
        color=NEUTRAL,
        fontsize=7,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.11, hspace=0.42, wspace=0.28)
    save_figure(fig, "figure2_observed_trajectories")
    trajectory.to_csv(FIG_DIR / "figure2_source_data.csv", index=False)


def main() -> None:
    main_figure()
    trajectory_figure()
    print(f"Wrote manuscript figures and source data to {FIG_DIR}")


if __name__ == "__main__":
    main()
