# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# Main Figure 1: study flow -> cross-type inherit -> param inherit
# Main Figure 2: primary associations -> Forest -> param inherit
# Main Figure 3: sensitivity and selection -> Forest/Heatmap -> param inherit
# Appendix 2: observed trajectories -> LineTrend -> param inherit
# Appendix 3: exploratory analyses -> Forest/GroupedBarChart -> param inherit
# RULE: All panels use the validated analysis outputs below; no participant-level data are exported.

"""Create the main and supplementary figures for the coordinated ageing-cohort manuscript.

The script reads only the validated aggregate outputs and the restricted local
derived long files needed for descriptive trajectories. It writes editable
SVG/PDF files, 600 dpi TIFF files, 300 dpi PNG previews, and figure Source Data
CSVs into the manuscript package.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COHORTS = ["CHARLS", "ELSA", "HRS", "MHAS", "SHARE"]
SETTINGS = {
    "CHARLS": "China",
    "ELSA": "England",
    "HRS": "United States",
    "MHAS": "Mexico",
    "SHARE": "14 SHARE countries",
}
COHORT_COLORS = {
    "CHARLS": "#B23B3D",
    "ELSA": "#3F63A7",
    "HRS": "#4F9A9B",
    "MHAS": "#D47A25",
    "SHARE": "#8F6A9B",
}
LONELY_COLOR = "#B23B3D"
NOT_LONELY_COLOR = "#777777"
TEAL = "#4F9A9B"
BLUE = "#3F63A7"
ORANGE = "#D47A25"
CHARCOAL = "#2D333B"
MUTED = "#777777"
GRID = "#D9DEE4"
LIGHT = "#EEF1F4"


def resolve_paths() -> tuple[Path, Path, Path, Path]:
    """Resolve analysis, package, and figure directories for local or repo use."""

    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "figure_source":
        package_dir = script_dir.parent
        analysis_dir = package_dir.parent / "分析输出_数据"
        figure_dir = package_dir / "figures"
        supplementary_figure_dir = package_dir / "supplementary_figures"
    else:
        project_root = script_dir.parent
        package_dir = project_root
        analysis_dir = project_root / "outputs"
        figure_dir = project_root / "outputs" / "figures"
        supplementary_figure_dir = project_root / "outputs" / "supplementary_figures"

    analysis_dir = Path(os.environ.get("GLOBAL_AGEING_ANALYSIS_DIR", analysis_dir))
    figure_dir = Path(os.environ.get("GLOBAL_AGEING_FIGURE_DIR", figure_dir))
    supplementary_figure_dir = Path(
        os.environ.get("GLOBAL_AGEING_SUPPLEMENTARY_FIGURE_DIR", supplementary_figure_dir)
    )
    return analysis_dir, package_dir, figure_dir, supplementary_figure_dir


ANALYSIS_DIR, PACKAGE_DIR, FIGURE_DIR, SUPPLEMENTARY_FIGURE_DIR = resolve_paths()
DERIVED_DIR = ANALYSIS_DIR / "derived"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def read_csv(name: str, subdir: str | None = None) -> pd.DataFrame:
    path = ANALYSIS_DIR / (subdir or "") / name
    if not path.exists():
        raise FileNotFoundError(f"Required analysis output not found: {path}")
    return pd.read_csv(path)


def format_n(value: float | int) -> str:
    return f"{int(round(float(value))):,}"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=CHARCOAL,
    )


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=CHARCOAL)
    for spine in ax.spines.values():
        spine.set_color(CHARCOAL)


def save_figure(fig: plt.Figure, stem: str, directory: Path | None = None) -> None:
    target = directory or FIGURE_DIR
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(target / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(target / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(target / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = CHARCOAL,
    fontsize: float = 7,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.05",
        linewidth=0.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=CHARCOAL,
        linespacing=1.12,
    )


def add_arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=CHARCOAL,
        )
    )


def forest_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    label_column: str,
    color: str,
    xlabel: str,
    xlim: tuple[float, float],
    panel: str | None = None,
    pooled_color: str = CHARCOAL,
    fontsize: float = 6.8,
) -> None:
    frame = frame.reset_index(drop=True)
    y = np.arange(len(frame))[::-1]
    for idx, row in frame.iterrows():
        label = str(row[label_column])
        yi = y[idx]
        row_color = pooled_color if label == "Pooled" else color
        marker = "D" if label == "Pooled" else "o"
        marker_size = 5.2 if label == "Pooled" else 4.4
        ax.plot(
            [row["ci_low"], row["ci_high"]],
            [yi, yi],
            color=row_color,
            linewidth=1.4,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            row["estimate"],
            yi,
            marker=marker,
            markersize=marker_size,
            color=row_color,
            markeredgecolor=row_color,
            zorder=3,
        )
        if label == "Pooled":
            ax.axhspan(yi - 0.46, yi + 0.46, color=LIGHT, zorder=0)
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(frame[label_column].astype(str), fontsize=fontsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, len(frame) - 0.3)
    ax.set_xlabel(xlabel)
    style_axis(ax, grid_axis="x")
    if panel:
        panel_label(ax, panel)


def baseline_pool() -> dict[str, float]:
    source = ANALYSIS_DIR / "r_validation" / "r_baseline_meta_result.csv"
    if source.exists():
        row = pd.read_csv(source).iloc[0]
        return {key: float(row[key]) for key in ["pooled_estimate", "ci_low", "ci_high"]}
    raise FileNotFoundError(
        "The verified baseline meta-analysis output is required for Figure 2: "
        f"{source}"
    )


def build_figure1() -> None:
    flow = read_csv("cohort_flow.csv")
    stage_order = [
        "source rows",
        "age 50+",
        "baseline core complete/direct",
        "at least one later direct memory assessment",
    ]
    stage_labels = {
        "source rows": "Source rows",
        "age 50+": "Age >=50",
        "baseline core complete/direct": "Baseline eligible",
        "at least one later direct memory assessment": "Longitudinal sample",
    }

    fig = plt.figure(figsize=(7.25, 6.45))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.25, 1.45], hspace=0.20)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(-0.55, 4.55)
    ax.set_ylim(0.0, 4.55)
    ax.axis("off")
    panel_label(ax, "A")
    ax.text(
        0,
        4.54,
        "Cohort-specific participant flow",
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=CHARCOAL,
    )
    y_positions = [3.55, 2.72, 1.89, 1.06]
    box_width = 0.78
    box_height = 0.54
    for x, cohort in enumerate(COHORTS):
        color = COHORT_COLORS[cohort]
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.42, 3.99),
                0.84,
                0.33,
                boxstyle="round,pad=0.025,rounding_size=0.04",
                linewidth=0.8,
                edgecolor=CHARCOAL,
                facecolor=color,
            )
        )
        ax.text(
            x,
            4.155,
            cohort,
            ha="center",
            va="center",
            fontsize=7.4,
            fontweight="bold",
            color="white",
        )
        ax.text(
            x,
            3.91,
            SETTINGS[cohort],
            ha="center",
            va="center",
            fontsize=5.7,
            color=MUTED,
        )
        for idx, stage in enumerate(stage_order):
            n = flow.loc[
                (flow["cohort"] == cohort) & (flow["stage"] == stage), "n"
            ].iloc[0]
            face = to_rgba(color, alpha=0.16)
            text = f"{stage_labels[stage]}\nn = {format_n(n)}"
            add_box(
                ax,
                x,
                y_positions[idx],
                box_width,
                box_height,
                text,
                face,
                edgecolor=CHARCOAL,
                fontsize=5.7 if idx < 2 else 5.55,
            )
            if idx < len(stage_order) - 1:
                add_arrow(
                    ax,
                    x,
                    y_positions[idx] - box_height / 2 - 0.03,
                    x,
                    y_positions[idx + 1] + box_height / 2 + 0.03,
                )
    ax.text(
        2.0,
        0.33,
        "Overall: 64,200 baseline-eligible participants -> 53,131 with later respondent-completed memory",
        ha="center",
        va="center",
        fontsize=7,
        color=CHARCOAL,
    )

    ax2 = fig.add_subplot(grid[1, 0])
    ax2.set_xlim(0, 9.0)
    ax2.set_ylim(0, 1.8)
    ax2.axis("off")
    panel_label(ax2, "B")
    ax2.text(
        0,
        1.68,
        "Common longitudinal estimand",
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=CHARCOAL,
    )
    steps = [
        (1.0, "Baseline\nsingle-item\nloneliness", to_rgba(LONELY_COLOR, 0.16)),
        (3.1, "Repeated\nrespondent-completed\nword recall", to_rgba(BLUE, 0.16)),
        (5.25, "Cohort-specific\nlinear mixed-effects\nmodel", to_rgba(TEAL, 0.16)),
        (7.55, "Two-stage\nrandom-effects\nmeta-analysis", to_rgba(ORANGE, 0.16)),
    ]
    for x, text, face in steps:
        add_box(ax2, x, 0.82, 1.65, 0.78, text, face, fontsize=6.8)
    for x1, x2 in zip([1.83, 3.95, 6.12], [2.27, 4.48, 6.70]):
        add_arrow(ax2, x1, 0.82, x2, 0.82)
    ax2.text(
        4.5,
        0.20,
        "Primary contrast: adjusted difference in average memory slope, rescaled to a 10-year interval, between baseline loneliness groups",
        ha="center",
        va="center",
        fontsize=6.5,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.045, right=0.985, top=0.985, bottom=0.035)
    save_figure(fig, "figure1_study_flow")

    source = flow.copy()
    source["setting"] = source["cohort"].map(SETTINGS)
    source["stage_order"] = source["stage"].map({stage: i for i, stage in enumerate(stage_order)})
    source.sort_values(["cohort", "stage_order"]).to_csv(
        FIGURE_DIR / "figure1_source_data.csv", index=False
    )


def build_supplementary_figure1() -> None:
    rows: list[dict[str, float | int | str]] = []
    for cohort in COHORTS:
        frame = pd.read_csv(DERIVED_DIR / f"{cohort.lower()}_long.csv.gz")
        for (wave, lonely), group in frame.groupby(["wave", "lonely"], sort=True):
            n = len(group)
            rows.append(
                {
                    "cohort": cohort,
                    "wave": int(wave),
                    "lonely": int(lonely),
                    "time": float(group["time"].median()),
                    "mean": float(group["memory_z"].mean()),
                    "se": float(group["memory_z"].std(ddof=1) / np.sqrt(n)),
                    "n": n,
                }
            )
    trajectory = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.55), sharey=True)
    axes = axes.ravel()
    for idx, cohort in enumerate(COHORTS):
        ax = axes[idx]
        cohort_frame = trajectory[trajectory["cohort"] == cohort]
        for lonely, label, color in [
            (0, "Not lonely", NOT_LONELY_COLOR),
            (1, "Lonely", LONELY_COLOR),
        ]:
            values = cohort_frame[cohort_frame["lonely"] == lonely].sort_values("time")
            ax.errorbar(
                values["time"],
                values["mean"],
                yerr=1.96 * values["se"],
                color=color,
                marker="o",
                markersize=3.8,
                linewidth=1.55,
                capsize=2.1,
                capthick=0.8,
                label=label,
                zorder=3,
            )
        ax.axhline(0, color=GRID, linewidth=0.9, zorder=0)
        ax.set_title(cohort, loc="left", fontsize=8.7, fontweight="bold")
        ax.set_xlabel("Years since baseline")
        ax.set_ylim(-0.85, 0.34)
        max_time = float(cohort_frame["time"].max())
        ax.set_xlim(-0.6, max_time + max(0.8, max_time * 0.04))
        style_axis(ax, grid_axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        panel_label(ax, chr(ord("A") + idx))
    axes[0].set_ylabel("Episodic memory (baseline SD)")
    axes[3].set_ylabel("Episodic memory (baseline SD)")
    legend_ax = axes[5]
    legend_ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.50, 0.78), fontsize=8)
    legend_ax.text(
        0.50,
        0.32,
        "Observed wave-specific means\nwith 95% confidence intervals.\nSamples vary across waves;\nlines are descriptive, not\ncovariate-adjusted trajectories.",
        transform=legend_ax.transAxes,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=7,
        linespacing=1.35,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.13, hspace=0.46, wspace=0.29)
    save_figure(fig, "supplementary_figure1_observed_trajectories", SUPPLEMENTARY_FIGURE_DIR)
    trajectory.to_csv(
        SUPPLEMENTARY_FIGURE_DIR / "supplementary_figure1_source_data.csv", index=False
    )


def build_figure2() -> None:
    fixed = read_csv("model_fixed_effects.csv")
    model = read_csv("cohort_model_estimates.csv")
    meta = read_csv("meta_analysis_results.csv")
    baseline_meta = baseline_pool()

    baseline = fixed[(fixed["model"] == "core") & (fixed["term"] == "lonely")][
        ["cohort", "estimate", "ci_low", "ci_high"]
    ].copy()
    baseline = pd.concat(
        [
            baseline,
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
    baseline["cohort"] = pd.Categorical(baseline["cohort"], COHORTS + ["Pooled"], ordered=True)
    baseline = baseline.sort_values("cohort").reset_index(drop=True)

    slope = model[model["model"] == "core"][
        ["cohort", "estimate", "ci_low", "ci_high"]
    ].copy()
    pooled_slope = meta.loc[meta["model"] == "core"].iloc[0]
    slope = pd.concat(
        [
            slope,
            pd.DataFrame(
                [
                    {
                        "cohort": "Pooled",
                        "estimate": pooled_slope["pooled_estimate"],
                        "ci_low": pooled_slope["ci_low"],
                        "ci_high": pooled_slope["ci_high"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    slope["cohort"] = pd.Categorical(slope["cohort"], COHORTS + ["Pooled"], ordered=True)
    slope = slope.sort_values("cohort").reset_index(drop=True)

    fig = plt.figure(figsize=(7.25, 3.75))
    grid = fig.add_gridspec(1, 2, wspace=0.60)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    forest_panel(
        ax_a,
        baseline,
        "cohort",
        LONELY_COLOR,
        "Baseline memory difference (SD)",
        (-0.31, 0.02),
        panel="A",
        fontsize=7,
    )
    ax_a.set_title("Baseline memory level", loc="left")
    forest_panel(
        ax_b,
        slope,
        "cohort",
        BLUE,
        "Memory slope difference (SD per 10-year interval)",
        (-0.13, 0.10),
        panel="B",
        fontsize=7,
    )
    ax_b.set_title("Longitudinal memory change", loc="left")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.89, bottom=0.22)
    save_figure(fig, "figure2_primary_associations")

    source_rows = []
    for estimand, frame, source in [
        ("baseline_memory", baseline, "model_fixed_effects.csv; r_baseline_meta_result.csv"),
        ("ten_year_slope", slope, "cohort_model_estimates.csv; meta_analysis_results.csv"),
    ]:
        for row in frame.itertuples(index=False):
            source_rows.append(
                {
                    "estimand": estimand,
                    "label": row.cohort,
                    "estimate": row.estimate,
                    "ci_low": row.ci_low,
                    "ci_high": row.ci_high,
                    "source_file": source,
                }
            )
    pd.DataFrame(source_rows).to_csv(FIGURE_DIR / "figure2_source_data.csv", index=False)


def build_figure3() -> None:
    meta = read_csv("meta_analysis_results.csv")
    sensitivity = read_csv("sensitivity_meta_results.csv")
    weighted = read_csv("attrition_weighted_meta_results.csv")
    fractional = read_csv("bounded_recall_gee_meta_results.csv")
    response = read_csv("attrition_response_by_wave.csv")
    wave_qa = read_csv("wave_qa.csv")
    selection = read_csv("selection_baseline_balance.csv")

    sensitivity_labels = {
        "core": "Primary mixed model",
        "full": "Additional covariate adjustment",
        "immediate_memory": "Immediate recall",
        "delayed_memory": "Delayed recall",
        "followup_adjusted_baseline_memory": "Follow-up + baseline memory",
        "exclude_first_retest": "Exclude first retest",
        "exclude_followup_within_2y": "Exclude follow-up <2 y",
        "exclude_lowest_baseline_decile": "Exclude lowest memory decile",
    }
    rows = []
    for analysis in [
        "core",
        "full",
        "immediate_memory",
        "delayed_memory",
        "followup_adjusted_baseline_memory",
        "exclude_first_retest",
        "exclude_followup_within_2y",
        "exclude_lowest_baseline_decile",
    ]:
        source = meta if analysis in {"core", "full"} else sensitivity
        key = "model" if analysis in {"core", "full"} else "analysis"
        row = source.loc[source[key] == analysis].iloc[0]
        rows.append(
            {
                "analysis": sensitivity_labels[analysis],
                "estimate": row["pooled_estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "source": "meta_analysis_results.csv" if analysis in {"core", "full"} else "sensitivity_meta_results.csv",
            }
        )
    sensitivity_plot = pd.DataFrame(rows)

    weighted_rows = []
    primary = meta.loc[meta["model"] == "core"].iloc[0]
    weighted_rows.append(
        {
            "analysis": "Primary mixed model",
            "estimate": primary["pooled_estimate"],
            "ci_low": primary["ci_low"],
            "ci_high": primary["ci_high"],
            "source": "meta_analysis_results.csv",
        }
    )
    for analysis, label in [
        ("gee_unweighted", "Unweighted GEE"),
        ("gee_attrition_ipw", "Attrition IPW"),
        ("gee_survey_x_attrition_ipw", "Survey x attrition weights"),
    ]:
        row = weighted.loc[weighted["analysis"] == analysis].iloc[0]
        weighted_rows.append(
            {
                "analysis": label,
                "estimate": row["pooled_estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "source": "attrition_weighted_meta_results.csv",
            }
        )
    weighted_plot = pd.DataFrame(weighted_rows)

    response = response.merge(
        wave_qa[["cohort", "wave", "time_median"]], on=["cohort", "wave"], how="left"
    )
    compact_labels = {
        "Baseline episodic memory (SD)": "Baseline memory",
        "Baseline loneliness": "Loneliness",
        "Age (years)": "Age",
        "Women": "Women",
        "Tertiary education": "Tertiary education",
        "Partnered": "Partnered",
    }
    selection["label"] = selection["label"].map(compact_labels)
    heatmap = selection.pivot(index="label", columns="cohort", values="standardized_difference")
    heatmap = heatmap.reindex(
        index=list(compact_labels.values()), columns=COHORTS
    )

    fig = plt.figure(figsize=(7.25, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.30, 1.0], hspace=0.70, wspace=0.62)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    forest_panel(
        ax_a,
        sensitivity_plot,
        "analysis",
        TEAL,
        "Pooled slope difference (SD per 10-year interval)",
        (-0.11, 0.08),
        panel="A",
        fontsize=6.2,
    )
    ax_a.set_title("Outcome, covariate and retest sensitivity", loc="left")
    forest_panel(
        ax_b,
        weighted_plot,
        "analysis",
        ORANGE,
        "Pooled slope difference (SD per 10-year interval)",
        (-0.15, 0.10),
        panel="B",
        fontsize=6.4,
    )
    ax_b.set_title("Observation and survey weighting", loc="left")
    for cohort in COHORTS:
        values = response[response["cohort"] == cohort].sort_values("time_median")
        ax_c.plot(
            values["time_median"],
            values["observed_percent"],
            marker="o",
            markersize=3.2,
            linewidth=1.35,
            color=COHORT_COLORS[cohort],
            label=cohort,
        )
    ax_c.set_title("Observed respondent-completed memory pairs", loc="left")
    ax_c.set_xlabel("Years since baseline")
    ax_c.set_ylabel("Observed among scheduled (%)")
    ax_c.set_ylim(0, 105)
    ax_c.legend(loc="lower left", ncol=2, fontsize=6.5, handlelength=1.8, columnspacing=0.9)
    style_axis(ax_c, grid_axis="y")
    panel_label(ax_c, "C")

    image = ax_d.imshow(
        heatmap.to_numpy(dtype=float),
        cmap="RdBu_r",
        vmin=-0.85,
        vmax=0.85,
        aspect="auto",
    )
    ax_d.set_xticks(np.arange(len(COHORTS)))
    ax_d.set_xticklabels(COHORTS, rotation=35, ha="right", fontsize=6.8)
    ax_d.set_yticks(np.arange(len(heatmap.index)))
    ax_d.set_yticklabels(heatmap.index, fontsize=6.2)
    ax_d.set_title("Selection into repeated testing", loc="left")
    ax_d.set_xlabel("Cohort")
    ax_d.set_ylabel("Baseline variable")
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            value = heatmap.iloc[i, j]
            text_color = "white" if np.isfinite(value) and abs(value) >= 0.5 else CHARCOAL
            ax_d.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.1,
                color=text_color,
            )
    ax_d.set_xticks(np.arange(-0.5, len(COHORTS), 1), minor=True)
    ax_d.set_yticks(np.arange(-0.5, len(heatmap.index), 1), minor=True)
    ax_d.grid(which="minor", color="white", linewidth=0.9)
    ax_d.tick_params(which="minor", bottom=False, left=False)
    ax_d.spines[:].set_visible(False)
    cbar = fig.colorbar(image, ax=ax_d, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Standardized difference", fontsize=6.5)
    panel_label(ax_d, "D")
    fig.text(
        0.5,
        0.015,
        "The bounded fractional-logit result is on a different log-odds scale and is reported in Appendix 8 in the Supplementary Data.",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.13, right=0.96, top=0.94, bottom=0.10)
    save_figure(fig, "figure3_sensitivity_and_selection", FIGURE_DIR)

    source_rows = []
    for family, frame in [("sensitivity", sensitivity_plot), ("weighting", weighted_plot)]:
        for row in frame.itertuples(index=False):
            source_rows.append(
                {
                    "panel": "A" if family == "sensitivity" else "B",
                    "label": row.analysis,
                    "estimate": row.estimate,
                    "ci_low": row.ci_low,
                    "ci_high": row.ci_high,
                    "source_file": row.source,
                }
            )
    response.assign(panel="C").to_csv(
        FIGURE_DIR / "figure3_response_source_data.csv", index=False
    )
    heatmap.reset_index().melt(id_vars="label").rename(
        columns={"variable": "cohort", "value": "standardized_difference"}
    ).assign(panel="D").to_csv(
        FIGURE_DIR / "figure3_selection_source_data.csv", index=False
    )
    pd.DataFrame(source_rows).to_csv(
        FIGURE_DIR / "figure3_source_data.csv", index=False
    )
    fractional.to_csv(
        FIGURE_DIR / "figure3_bounded_recall_source_data.csv", index=False
    )


def build_supplementary_figure2() -> None:
    patterns = read_csv("exposure_pattern_counts.csv")
    pattern_meta = read_csv("exposure_pattern_meta_results.csv")
    lagged = read_csv("lagged_transition_meta_results.csv")
    nonlinear = read_csv("nonlinear_meta_results.csv")
    effect = read_csv("effect_modification_meta_results.csv")

    same = patterns[
        (patterns["rule"] == "same_item")
        & (patterns["cohort"].isin(["CHARLS", "ELSA", "HRS", "MHAS"]))
        & (patterns["pattern"] != "eligible_any_pattern")
    ].copy()
    totals = same.groupby("cohort")["participants"].sum()
    same["percent"] = same.apply(
        lambda row: 100 * row["participants"] / totals[row["cohort"]], axis=1
    )
    pattern_order = ["never", "transient_or_changing", "persistent"]
    pattern_labels = {
        "never": "Never lonely",
        "transient_or_changing": "Transient/changing",
        "persistent": "Persistent",
    }

    repeated_labels = {
        "same_item:persistent_vs_never": "Persistent vs never",
        "same_item:transient_or_changing_vs_never": "Changing vs never",
        "available_signal:persistent_vs_never": "Persistent vs never (mixed)",
        "available_signal:transient_or_changing_vs_never": "Changing vs never (mixed)",
    }
    repeated_rows = []
    for contrast in [
        "same_item:persistent_vs_never",
        "same_item:transient_or_changing_vs_never",
        "available_signal:persistent_vs_never",
        "available_signal:transient_or_changing_vs_never",
    ]:
        row = pattern_meta.loc[pattern_meta["contrast"] == contrast].iloc[0]
        repeated_rows.append(
            {
                "label": repeated_labels[contrast],
                "estimate": row["pooled_estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "source": "exposure_pattern_meta_results.csv",
            }
        )
    repeated_plot = pd.DataFrame(repeated_rows)

    lagged_rows = []
    for rule, label in [("same_item", "Same-item rule"), ("available_signal", "Available signal")]:
        row = lagged.loc[lagged["rule"] == rule].iloc[0]
        lagged_rows.append(
            {
                "label": label,
                "estimate": row["pooled_estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "source": "lagged_transition_meta_results.csv",
            }
        )
    lagged_plot = pd.DataFrame(lagged_rows)

    horizon_rows = nonlinear[nonlinear["contrast"] == "lonely_minus_not_lonely"].copy()
    horizon_plot = horizon_rows.rename(columns={"years": "label"})[
        ["label", "pooled_estimate", "ci_low", "ci_high"]
    ].rename(columns={"pooled_estimate": "estimate"})
    horizon_plot["label"] = horizon_plot["label"].map(lambda value: f"{int(value)} years")
    horizon_plot["source"] = "nonlinear_meta_results.csv"

    effect_labels = {
        "female_vs_male": "Women vs men",
        "per_10_year_older_age": "Per 10 years older",
        "upper_secondary_vs_low_education": "Upper-secondary vs low education",
        "tertiary_vs_low_education": "Tertiary vs low education",
    }
    effect_rows = []
    for contrast in [
        "female_vs_male",
        "per_10_year_older_age",
        "upper_secondary_vs_low_education",
        "tertiary_vs_low_education",
    ]:
        row = effect.loc[effect["contrast"] == contrast].iloc[0]
        effect_rows.append(
            {
                "label": effect_labels[contrast],
                "estimate": row["pooled_estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "source": "effect_modification_meta_results.csv",
            }
        )
    effect_plot = pd.DataFrame(effect_rows)
    effect_plot["display_label"] = effect_plot["label"].replace(
        {
            "Upper-secondary vs low education": "Upper-secondary vs\nlow education",
            "Tertiary vs low education": "Tertiary vs\nlow education",
        }
    )

    fig = plt.figure(figsize=(7.25, 6.20))
    grid = fig.add_gridspec(2, 3, height_ratios=[0.95, 1.1], hspace=0.78, wspace=0.72)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1:])

    x = np.arange(4)
    bottom = np.zeros(4)
    bar_colors = ["#777777", TEAL, ORANGE]
    for pattern, color in zip(pattern_order, bar_colors):
        values = same[same["pattern"] == pattern].set_index("cohort").reindex(
            ["CHARLS", "ELSA", "HRS", "MHAS"]
        )["percent"]
        ax_a.bar(
            x,
            values,
            bottom=bottom,
            width=0.68,
            color=color,
            label=pattern_labels[pattern],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += values.to_numpy()
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["CHARLS", "ELSA", "HRS", "MHAS"], rotation=35, ha="right")
    ax_a.set_ylim(0, 100)
    ax_a.set_ylabel("Participants (%)")
    ax_a.set_title("Same-item exposure patterns", loc="left")
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=6.2, ncol=1)
    style_axis(ax_a, grid_axis="y")
    panel_label(ax_a, "A")

    forest_panel(
        ax_b,
        repeated_plot,
        "label",
        ORANGE,
        "Slope difference (SD per 10-year interval)",
        (-0.35, 0.16),
        panel="B",
        fontsize=5.9,
    )
    ax_b.set_title("Repeated-exposure patterns", loc="left")
    forest_panel(
        ax_c,
        lagged_plot,
        "label",
        BLUE,
        "Next-assessment memory difference (SD)",
        (-0.18, 0.02),
        panel="C",
        fontsize=6.4,
    )
    ax_c.set_title("Lagged adjacent-assessment association", loc="left")
    forest_panel(
        ax_d,
        horizon_plot,
        "label",
        TEAL,
        "Lonely minus not lonely (SD)",
        (-0.34, 0.02),
        panel="D",
        fontsize=6.7,
    )
    ax_d.set_title("Quadratic-time group contrasts", loc="left")
    forest_panel(
        ax_e,
        effect_plot,
        "display_label",
        ORANGE,
        "Difference in slope contrast (SD per 10-year interval)",
        (-0.25, 0.25),
        panel="E",
        fontsize=5.95,
    )
    ax_e.set_title("Effect modification", loc="left")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.94, bottom=0.14)
    save_figure(fig, "supplementary_figure2_exploratory_analyses", SUPPLEMENTARY_FIGURE_DIR)

    sources = []
    for panel, frame in [
        ("B", repeated_plot),
        ("C", lagged_plot),
        ("D", horizon_plot),
        ("E", effect_plot),
    ]:
        for row in frame.itertuples(index=False):
            sources.append(
                {
                    "panel": panel,
                    "label": row.label,
                    "estimate": row.estimate,
                    "ci_low": row.ci_low,
                    "ci_high": row.ci_high,
                    "source_file": row.source,
                }
            )
    SUPPLEMENTARY_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    same.assign(panel="A").to_csv(
        SUPPLEMENTARY_FIGURE_DIR / "supplementary_figure2_pattern_source_data.csv", index=False
    )
    pd.DataFrame(sources).to_csv(
        SUPPLEMENTARY_FIGURE_DIR / "supplementary_figure2_source_data.csv", index=False
    )


def main() -> None:
    build_figure1()
    build_figure2()
    build_figure3()
    build_supplementary_figure1()
    build_supplementary_figure2()
    print(
        f"Wrote three main figures and two supplementary figures to {FIGURE_DIR} and "
        f"{SUPPLEMENTARY_FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()
