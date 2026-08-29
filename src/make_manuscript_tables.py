"""Create five main tables from the validated global ageing analysis outputs."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


def resolve_paths() -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "figure_source":
        package_dir = script_dir.parent
        analysis_dir = package_dir.parent / "分析输出_数据"
    else:
        package_dir = script_dir.parent
        analysis_dir = package_dir / "outputs"
    analysis_dir = Path(os.environ.get("GLOBAL_AGEING_ANALYSIS_DIR", analysis_dir))
    package_dir = Path(os.environ.get("GLOBAL_AGEING_PACKAGE_DIR", package_dir))
    return analysis_dir, package_dir


ANALYSIS_DIR, PACKAGE_DIR = resolve_paths()
TABLE_DIR = PACKAGE_DIR / "tables"


def read_csv(name: str, subdir: str | None = None) -> pd.DataFrame:
    path = ANALYSIS_DIR / (subdir or "") / name
    if not path.exists():
        raise FileNotFoundError(f"Required analysis output not found: {path}")
    return pd.read_csv(path)


def n(value: float | int) -> str:
    return f"{int(round(float(value))):,}"


def one(value: float | int) -> str:
    return f"{float(value):.1f}"


def two(value: float | int) -> str:
    return f"{float(value):.2f}"


def three(value: float | int) -> str:
    return f"{float(value):.3f}"


def p_value(value: float | int | str) -> str:
    if value is None or str(value).strip().upper() in {"NA", "NAN", ""}:
        return "NA"
    value = float(value)
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def markdown_table(frame: pd.DataFrame, note: str) -> str:
    frame = frame.fillna("-").astype(str)
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines) + "\n\n" + note


def write_table(stem: str, frame: pd.DataFrame, title: str, note: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLE_DIR / f"{stem}.csv", index=False)
    content = f"# {title}\n\n{markdown_table(frame, note)}\n"
    (TABLE_DIR / f"{stem}.md").write_text(content, encoding="utf-8")


def table1() -> None:
    source = read_csv("table_s1_cohort_design_and_measurement.csv", "supplementary_tables")
    frame = source[[
        "cohort",
        "setting",
        "baseline_wave",
        "memory_waves",
        "baseline_eligible",
        "longitudinal_sample",
        "retained_percent",
        "maximum_observed_followup_years",
    ]].copy()
    frame.columns = [
        "Cohort",
        "Setting",
        "Baseline wave",
        "Memory waves",
        "Baseline eligible, n",
        "Longitudinal sample, n",
        "Retained (%)",
        "Maximum follow-up (y)",
    ]
    frame["Baseline eligible, n"] = frame["Baseline eligible, n"].map(n)
    frame["Longitudinal sample, n"] = frame["Longitudinal sample, n"].map(n)
    frame["Retained (%)"] = frame["Retained (%)"].map(one)
    frame["Maximum follow-up (y)"] = frame["Maximum follow-up (y)"].map(one)
    write_table(
        "table_1_cohort_design_and_analysis_sets",
        frame,
        "Table 1 | Cohort design and analysis sets",
        "The longitudinal sample included participants contributing at least one later respondent-completed assessment of both immediate and delayed word recall. Exact data-product releases, filenames and proxy rules are provided in Supplementary Table S1.",
    )


def table2() -> None:
    source = read_csv("baseline_descriptive.csv")
    rows = []
    group_labels = {0: "Not lonely", 1: "Lonely"}
    for cohort in ["CHARLS", "ELSA", "HRS", "MHAS", "SHARE"]:
        group = source[source["cohort"] == cohort].sort_values("lonely")
        for row in group.itertuples(index=False):
            rows.append(
                {
                    "Cohort": cohort,
                    "Group": group_labels[int(row.lonely)],
                    "Participants, n": n(row.participants),
                    "Age, mean (SD)": f"{one(row.age_mean)} ({one(row.age_sd)})",
                    "Women (%)": one(row.female_percent),
                    "Tertiary education (%)": one(row.tertiary_education_percent),
                    "Partnered (%)": one(row.partnered_percent),
                    "Baseline memory, mean (SD)": f"{two(row.baseline_memory_mean)} ({two(row.baseline_memory_sd)})",
                    "Median follow-up (y)": one(row.median_followup_years),
                }
            )
    frame = pd.DataFrame(rows)
    write_table(
        "table_2_baseline_characteristics",
        frame,
        "Table 2 | Baseline characteristics by cohort and loneliness group",
        "Characteristics are calculated in the longitudinal analysis sample. Baseline memory is expressed in cohort-specific baseline SD units. Loneliness is the cohort-harmonized binary baseline exposure.",
    )


def table3() -> None:
    fixed = read_csv("model_fixed_effects.csv")
    models = read_csv("cohort_model_estimates.csv")
    meta = read_csv("meta_analysis_results.csv")
    baseline_meta = read_csv("r_baseline_meta_result.csv", "r_validation").iloc[0]
    rows = []
    model_n = models[models["model"] == "core"].set_index("cohort")
    for cohort in ["CHARLS", "ELSA", "HRS", "MHAS", "SHARE"]:
        row = fixed[(fixed["model"] == "core") & (fixed["term"] == "lonely") & (fixed["cohort"] == cohort)].iloc[0]
        rows.append(
            {
                "Estimand": "Baseline memory level",
                "Cohort": cohort,
                "Participants": n(model_n.loc[cohort, "participants"]),
                "Observations": n(model_n.loc[cohort, "observations"]),
                "Estimate": three(row["estimate"]),
                "95% CI": f"{three(row['ci_low'])} to {three(row['ci_high'])}",
                "P value": p_value(row["p_value"]),
                "I2 (%)": "-",
                "95% prediction interval": "-",
            }
        )
    rows.append(
        {
            "Estimand": "Baseline memory level",
            "Cohort": "Pooled",
            "Participants": n(model_n["participants"].sum()),
            "Observations": n(model_n["observations"].sum()),
            "Estimate": three(baseline_meta["pooled_estimate"]),
            "95% CI": f"{three(baseline_meta['ci_low'])} to {three(baseline_meta['ci_high'])}",
            "P value": p_value(baseline_meta["p_value"]),
            "I2 (%)": one(baseline_meta["i2_percent"]),
            "95% prediction interval": f"{three(baseline_meta['prediction_low'])} to {three(baseline_meta['prediction_high'])}",
        }
    )
    for cohort in ["CHARLS", "ELSA", "HRS", "MHAS", "SHARE"]:
        row = models[(models["model"] == "core") & (models["cohort"] == cohort)].iloc[0]
        rows.append(
            {
                "Estimand": "10-year-scaled memory slope difference",
                "Cohort": cohort,
                "Participants": n(row["participants"]),
                "Observations": n(row["observations"]),
                "Estimate": three(row["estimate"]),
                "95% CI": f"{three(row['ci_low'])} to {three(row['ci_high'])}",
                "P value": p_value(row["p_value"]),
                "I2 (%)": "-",
                "95% prediction interval": "-",
            }
        )
    slope_meta = meta[meta["model"] == "core"].iloc[0]
    rows.append(
        {
            "Estimand": "10-year-scaled memory slope difference",
            "Cohort": "Pooled",
            "Participants": n(model_n["participants"].sum()),
            "Observations": n(model_n["observations"].sum()),
            "Estimate": three(slope_meta["pooled_estimate"]),
            "95% CI": f"{three(slope_meta['ci_low'])} to {three(slope_meta['ci_high'])}",
            "P value": p_value(slope_meta["p_value"]),
            "I2 (%)": one(slope_meta["i2_percent"]),
            "95% prediction interval": f"{three(slope_meta['prediction_low'])} to {three(slope_meta['prediction_high'])}",
        }
    )
    write_table(
        "table_3_primary_model_estimates",
        pd.DataFrame(rows),
        "Table 3 | Primary model estimates across five ageing cohorts",
        "Cohort-specific estimates came from core linear mixed-effects models with participant-specific random intercepts and time slopes. Pooled estimates used two-stage REML random-effects meta-analysis with Hartung-Knapp confidence intervals and t-based prediction intervals. Baseline memory level and the memory slope scaled to a 10-year interval are reported on the cohort-specific baseline-SD scale; this scaling does not imply 10 years of observation for every cohort.",
    )


def table4() -> None:
    meta = read_csv("meta_analysis_results.csv")
    sensitivity = read_csv("sensitivity_meta_results.csv")
    weighted = read_csv("attrition_weighted_meta_results.csv")
    bounded = read_csv("bounded_recall_gee_meta_results.csv")
    rows = []
    labels = {
        "core": "Primary mixed model",
        "full": "Fully adjusted model",
        "immediate_memory": "Immediate recall",
        "delayed_memory": "Delayed recall",
        "followup_adjusted_baseline_memory": "Follow-up adjusted for baseline memory",
        "exclude_first_retest": "Exclude first retest",
        "exclude_followup_within_2y": "Exclude follow-up within 2 years",
        "exclude_lowest_baseline_decile": "Exclude lowest baseline-memory decile",
        "gee_unweighted": "Unweighted GEE",
        "gee_attrition_ipw": "Attrition IPW",
        "gee_survey_x_attrition_ipw": "Survey x attrition weights",
        "fractional_logit_recall": "Fractional-logit recalled fraction",
    }
    sources = {
        "core": (meta, "model"),
        "full": (meta, "model"),
        "immediate_memory": (sensitivity, "analysis"),
        "delayed_memory": (sensitivity, "analysis"),
        "followup_adjusted_baseline_memory": (sensitivity, "analysis"),
        "exclude_first_retest": (sensitivity, "analysis"),
        "exclude_followup_within_2y": (sensitivity, "analysis"),
        "exclude_lowest_baseline_decile": (sensitivity, "analysis"),
        "gee_unweighted": (weighted, "analysis"),
        "gee_attrition_ipw": (weighted, "analysis"),
        "gee_survey_x_attrition_ipw": (weighted, "analysis"),
        "fractional_logit_recall": (bounded, "analysis"),
    }
    for analysis in labels:
        source, key = sources[analysis]
        row = source[source[key] == analysis].iloc[0]
        effect_scale = (
            "log odds of recalled fraction per 10-year interval"
            if analysis == "fractional_logit_recall"
            else row["effect_scale"]
            if "effect_scale" in row.index
            else "baseline-SD slope difference rescaled to a 10-year interval"
        )
        rows.append(
            {
                "Analysis": labels[analysis],
                "Effect scale": effect_scale,
                "Pooled estimate (95% CI)": f"{three(row['pooled_estimate'])} ({three(row['ci_low'])} to {three(row['ci_high'])})",
                "95% prediction interval": f"{three(row['prediction_low'])} to {three(row['prediction_high'])}",
                "I2 (%)": one(row["i2_percent"]),
                "P value": p_value(row["p_value"]),
            }
        )
    write_table(
        "table_4_sensitivity_and_weighting",
        pd.DataFrame(rows),
        "Table 4 | Sensitivity and observation-weighting analyses",
        "The primary estimand is the adjusted difference in episodic-memory slope between participants classified as lonely and not lonely at baseline, rescaled to a 10-year interval. The fractional-logit result is on the log-odds scale and is not numerically comparable with the standardized memory-slope estimates. I2 denotes I-squared.",
    )


def table5() -> None:
    source = read_csv("table_s7_exploratory_meta_results.csv", "supplementary_tables")
    family_labels = {
        "Repeated-exposure pattern": "Repeated exposure",
        "Lagged adjacent-assessment association": "Lagged association",
        "Quadratic-time interaction": "Quadratic-time interaction",
        "Quadratic-time group difference": "Quadratic-time horizon",
        "Effect modification": "Effect modification",
    }
    contrast_labels = {
        "same_item:persistent_vs_never": "Same item: persistent vs never",
        "same_item:transient_or_changing_vs_never": "Same item: transient/changing vs never",
        "available_signal:persistent_vs_never": "Available signal: persistent vs never",
        "available_signal:transient_or_changing_vs_never": "Available signal: transient/changing vs never",
        "available_signal": "Available signal",
        "same_item": "Same-item rule",
        "linear loneliness-by-time component": "Linear loneliness-by-time component",
        "quadratic loneliness-by-time component": "Quadratic loneliness-by-time component",
        "female_vs_male": "Women vs men",
        "per_10_year_older_age": "Per 10 years older",
        "upper_secondary_vs_low_education": "Upper-secondary vs low education",
        "tertiary_vs_low_education": "Tertiary vs low education",
    }
    rows = []
    for row in source.itertuples(index=False):
        contrast = str(row.contrast)
        label = contrast_labels.get(contrast, contrast)
        if str(row.analysis_family) == "Quadratic-time group difference":
            match = re.search(r"\d+(?:\.\d+)?", contrast)
            label = f"{match.group(0)}-year horizon" if match else contrast
        rows.append(
            {
                "Analysis family": family_labels.get(str(row.analysis_family), str(row.analysis_family)),
                "Contrast": label,
                "Effect scale": row.effect_scale,
                "k": n(row.k),
                "Estimate (95% CI)": f"{three(row.estimate)} ({three(row.ci_low)} to {three(row.ci_high)})",
                "95% prediction interval": f"{three(row.prediction_low)} to {three(row.prediction_high)}",
                "I2 (%)": one(row.i2_percent),
                "Adjusted P value": p_value(row.adjusted_p_value),
            }
        )
    write_table(
        "table_5_exploratory_analyses",
        pd.DataFrame(rows),
        "Table 5 | Exploratory repeated-exposure, lagged, nonlinear and effect-modification analyses",
        "These analyses were hypothesis-generating. Repeated-exposure groups used future exposure information and continued observation; available-signal analyses mixed loneliness instruments across cohorts; 10-year quadratic-time contrasts required extrapolation in shorter cohorts. The primary 10-year quantity is a slope rescaling, whereas the 10-year quadratic-time horizon is a model-based contrast. These analyses were not used to strengthen the primary claim.",
    )


def main() -> None:
    table1()
    table2()
    table3()
    table4()
    table5()
    print(f"Wrote five main tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
