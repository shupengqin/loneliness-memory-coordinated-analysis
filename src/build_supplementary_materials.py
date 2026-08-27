"""Assemble reviewer-facing supplementary tables from validated analysis outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from project_config import OUTPUT_DIR, PROJECT_ROOT

ROOT = PROJECT_ROOT
OUT = OUTPUT_DIR
SUPP = OUT / "supplementary_tables"
COHORTS = ["CHARLS", "ELSA", "HRS", "MHAS", "SHARE"]


def fmt_number(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):.{digits}f}"


def save(frame: pd.DataFrame, filename: str) -> pd.DataFrame:
    SUPP.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SUPP / filename, index=False)
    return frame


def cohort_design_table() -> pd.DataFrame:
    wave_qa = pd.read_csv(OUT / "wave_qa.csv")
    selection = pd.read_csv(OUT / "selection_to_longitudinal_sample.csv")
    design = pd.DataFrame(
        [
            {
                "cohort": "CHARLS",
                "setting": "China",
                "analysis_data_product": "Harmonized CHARLS Version D (2011-2018)",
                "release_date": "June 2021",
                "source_release_basis": "Waves 1-4",
                "analysis_source_file": "H_CHARLS_D_Data.dta",
                "baseline_wave": 1,
                "memory_waves": "1-4",
                "memory_assessments": 4,
                "recall_words_per_test": 10,
                "baseline_loneliness": "Four-level frequency item; responses 2-4 classified as lonely",
                "repeated_loneliness": "Same single item",
                "proxy_rule": "No harmonized proxy indicator available; completed interview status required",
            },
            {
                "cohort": "ELSA",
                "setting": "England",
                "analysis_data_product": "Harmonized ELSA Version G.3 (2002-2019)",
                "release_date": "June 2023",
                "source_release_basis": "38th ELSA edition (April 2023); waves 1-9",
                "analysis_source_file": "h_elsa_g3.dta",
                "baseline_wave": 1,
                "memory_waves": "1-9",
                "memory_assessments": 9,
                "recall_words_per_test": 10,
                "baseline_loneliness": "Binary harmonized single item",
                "repeated_loneliness": "Same single item",
                "proxy_rule": "Proxy interviews excluded",
            },
            {
                "cohort": "HRS",
                "setting": "United States",
                "analysis_data_product": "RAND HRS Longitudinal File 2020 (V2), final release",
                "release_date": "May 2024",
                "source_release_basis": "Analysis waves 9-13",
                "analysis_source_file": "randhrs1992_2020v2.dta",
                "baseline_wave": 9,
                "memory_waves": "9-13",
                "memory_assessments": 5,
                "recall_words_per_test": 10,
                "baseline_loneliness": "Binary harmonized single item",
                "repeated_loneliness": "Same single item",
                "proxy_rule": "Proxy interviews excluded",
            },
            {
                "cohort": "MHAS",
                "setting": "Mexico",
                "analysis_data_product": "Harmonized MHAS Version C.2 (2001-2019)",
                "release_date": "August 2023",
                "source_release_basis": "Parent-study datasets available July 2020; waves 1-5",
                "analysis_source_file": "H_MHAS_c2.dta",
                "baseline_wave": 1,
                "memory_waves": "1-5",
                "memory_assessments": 5,
                "recall_words_per_test": 8,
                "baseline_loneliness": "Binary harmonized single item",
                "repeated_loneliness": "Same single item",
                "proxy_rule": "Proxy interviews excluded",
            },
            {
                "cohort": "SHARE",
                "setting": "14 countries in SHARE",
                "analysis_data_product": "Harmonized SHARE Version F.2 (2004-2020)",
                "release_date": "April 2024",
                "source_release_basis": "SHARE Release 9.0.0; waves 2 and 4-8",
                "analysis_source_file": "H_SHARE_f2.dta",
                "baseline_wave": 2,
                "memory_waves": "2, 4-8",
                "memory_assessments": 6,
                "recall_words_per_test": 10,
                "baseline_loneliness": "Binary harmonized single item",
                "repeated_loneliness": "Later three-item scale available only as an exploratory fallback",
                "proxy_rule": "Proxy interviews and proxy-derived memory values excluded",
            },
        ]
    )
    max_followup = wave_qa.groupby("cohort", observed=True)["time_max"].max()
    design["maximum_observed_followup_years"] = design["cohort"].map(max_followup)
    design = design.merge(selection, on="cohort", how="left")
    return save(design, "table_s1_cohort_design_and_measurement.csv")


def sample_flow_table() -> pd.DataFrame:
    flow = pd.read_csv(OUT / "cohort_flow.csv")
    pivot = flow.pivot(index="cohort", columns="stage", values="n").reset_index()
    selection = pd.read_csv(OUT / "selection_to_longitudinal_sample.csv")
    result = pivot.merge(selection, on="cohort", how="left")
    result = result.rename(
        columns={
            "baseline core complete/direct": "baseline eligible with respondent-completed baseline memory",
            "at least one later direct memory assessment": "longitudinal sample with later respondent-completed memory",
        }
    )
    result["excluded_between_baseline_and_repeat_percent"] = (
        100
        * result["excluded_before_repeat_assessment"]
        / result["baseline_eligible"]
    )
    return save(result, "table_s2_sample_flow.csv")


def baseline_table() -> pd.DataFrame:
    baseline = pd.read_csv(OUT / "baseline_descriptive.csv")
    baseline["group"] = baseline["lonely"].map({0: "Not lonely", 1: "Lonely"})
    columns = [
        "cohort",
        "group",
        "participants",
        "age_mean",
        "age_sd",
        "female_percent",
        "tertiary_education_percent",
        "partnered_percent",
        "baseline_memory_mean",
        "baseline_memory_sd",
        "median_observations",
        "median_followup_years",
    ]
    return save(baseline[columns], "table_s3_baseline_characteristics.csv")


def observation_table() -> pd.DataFrame:
    response = pd.read_csv(OUT / "attrition_response_by_wave.csv")
    proxy = pd.read_csv(OUT / "proxy_memory_audit.csv")
    proxy = proxy[
        [
            "cohort",
            "wave",
            "proxy_variable_available",
            "n_proxy_yes",
            "n_memory_pair_proxy_yes",
        ]
    ]
    result = response.merge(proxy, on=["cohort", "wave"], how="left")
    charls = result["cohort"].eq("CHARLS")
    result.loc[charls, "proxy_variable_available"] = False
    result.loc[charls, ["n_proxy_yes", "n_memory_pair_proxy_yes"]] = np.nan
    result["proxy_indicator_status"] = np.where(
        result["proxy_variable_available"].eq(True), "Available", "Not available"
    )
    return save(result, "table_s4_wave_observation_and_proxy_audit.csv")


def primary_results_table() -> pd.DataFrame:
    meta = pd.read_csv(OUT / "meta_analysis_results.csv").rename(
        columns={"model": "analysis"}
    )
    sensitivity = pd.read_csv(OUT / "sensitivity_meta_results.csv")
    attrition = pd.read_csv(OUT / "attrition_weighted_meta_results.csv")
    bounded = pd.read_csv(OUT / "bounded_recall_gee_meta_results.csv")
    keep = [
        "analysis",
        "k",
        "pooled_estimate",
        "ci_low",
        "ci_high",
        "p_value",
        "tau2_reml",
        "i2_percent",
        "prediction_low",
        "prediction_high",
        "method",
    ]
    result = pd.concat(
        [meta[keep], sensitivity[keep], attrition[keep], bounded[keep]],
        ignore_index=True,
    )
    scale = np.select(
        [
            result["analysis"].eq("fractional_logit_recall"),
            result["analysis"].eq("standardized_loneliness"),
        ],
        [
            "log odds of recalled fraction per 10 years",
            "baseline-SD units per 10 years per exposure SD",
        ],
        default="baseline-SD units per 10 years",
    )
    result.insert(1, "effect_scale", scale)
    return save(result, "table_s5_primary_and_sensitivity_meta_results.csv")


def selection_table() -> pd.DataFrame:
    balance = pd.read_csv(OUT / "selection_baseline_balance.csv")
    return save(balance, "table_s6_selection_balance.csv")


def exploratory_table() -> pd.DataFrame:
    rows = []
    patterns = pd.read_csv(OUT / "exposure_pattern_meta_results.csv")
    for row in patterns.itertuples(index=False):
        rows.append(
            {
                "analysis_family": "Repeated-exposure pattern",
                "contrast": row.contrast,
                "effect_scale": "baseline-SD difference in 10-year memory change",
                "k": row.k,
                "estimate": row.pooled_estimate,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "i2_percent": row.i2_percent,
                "prediction_low": row.prediction_low,
                "prediction_high": row.prediction_high,
                "adjusted_p_value": np.nan,
            }
        )
    lagged = pd.read_csv(OUT / "lagged_transition_meta_results.csv")
    for row in lagged.itertuples(index=False):
        rows.append(
            {
                "analysis_family": "Lagged adjacent-assessment association",
                "contrast": row.rule,
                "effect_scale": "baseline-SD difference at the next assessment",
                "k": row.k,
                "estimate": row.pooled_estimate,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "i2_percent": row.i2_percent,
                "prediction_low": row.prediction_low,
                "prediction_high": row.prediction_high,
                "adjusted_p_value": np.nan,
            }
        )
    coefficients = pd.read_csv(OUT / "nonlinear_coefficient_meta_results.csv")
    coefficient_labels = {
        "time10:lonely": (
            "linear loneliness-by-time component",
            "baseline-SD units per 10 years",
        ),
        "time10_sq:lonely": (
            "quadratic loneliness-by-time component",
            "baseline-SD units per squared 10-year unit",
        ),
    }
    for row in coefficients.loc[
        coefficients["term"].isin(coefficient_labels)
    ].itertuples(index=False):
        contrast, effect_scale = coefficient_labels[row.term]
        rows.append(
            {
                "analysis_family": "Quadratic-time interaction",
                "contrast": contrast,
                "effect_scale": effect_scale,
                "k": row.k,
                "estimate": row.pooled_estimate,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "i2_percent": row.i2_percent,
                "prediction_low": row.prediction_low,
                "prediction_high": row.prediction_high,
                "adjusted_p_value": np.nan,
            }
        )
    nonlinear = pd.read_csv(OUT / "nonlinear_meta_results.csv")
    for row in nonlinear.itertuples(index=False):
        rows.append(
            {
                "analysis_family": "Quadratic-time group difference",
                "contrast": f"{row.years:g} years",
                "effect_scale": "baseline-SD group difference at stated horizon",
                "k": row.k,
                "estimate": row.pooled_estimate,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "i2_percent": row.i2_percent,
                "prediction_low": row.prediction_low,
                "prediction_high": row.prediction_high,
                "adjusted_p_value": np.nan,
            }
        )
    modifiers = pd.read_csv(OUT / "effect_modification_meta_results.csv")
    for row in modifiers.itertuples(index=False):
        rows.append(
            {
                "analysis_family": "Effect modification",
                "contrast": row.contrast,
                "effect_scale": "difference in the baseline-SD 10-year slope contrast",
                "k": row.k,
                "estimate": row.pooled_estimate,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "i2_percent": row.i2_percent,
                "prediction_low": row.prediction_low,
                "prediction_high": row.prediction_high,
                "adjusted_p_value": row.p_value_fdr,
            }
        )
    return save(pd.DataFrame(rows), "table_s7_exploratory_meta_results.csv")


def death_coverage_table() -> pd.DataFrame:
    coverage = pd.read_csv(OUT / "death_coverage_audit.csv")
    return save(coverage, "table_s8_death_information_coverage.csv")


def render_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    display = frame[columns].copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: fmt_number(value, digits))
    headers = [str(column).replace("_", " ") for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        values = [
            str(value).replace("|", "\\|").replace("\n", " ") for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(tables: dict[str, pd.DataFrame]) -> None:
    s1 = tables["s1"]
    s2 = tables["s2"]
    s3 = tables["s3"]
    s4 = tables["s4"]
    s5 = tables["s5"]
    s6 = tables["s6"]
    s7 = tables["s7"]
    s8 = tables["s8"]

    text = f"""# Supplementary Information

## Supplementary Methods

### Analysis-set selection and observation weighting

The baseline-eligible population comprised participants with complete core baseline variables and a respondent-completed baseline memory assessment. The primary longitudinal analysis additionally required at least one later respondent-completed assessment of both immediate and delayed recall. Proxy observations were excluded when a harmonized proxy indicator was available; CHARLS eligibility instead required completed-interview status and valid recall scores because the selected harmonized file lacked that indicator. We compared baseline characteristics between participants who did and did not enter the longitudinal sample using standardized mean differences. Observation weights were estimated in the full baseline-eligible participant-by-wave panel from baseline loneliness, memory, age, sex, education, partnered status, survey wave and SHARE country. These weights address selection associated with measured baseline predictors, but they do not identify outcomes after death or remove selection through unmeasured health deterioration.

### Bounded recall sensitivity analysis

To test whether results depended on treating the standardized memory composite as an unbounded Gaussian outcome, we divided the sum of immediate and delayed recall by the cohort-specific maximum possible total. Cohort-specific fractional-logit generalized estimating equations used a logit mean model, participant clustering, exchangeable working correlation and robust standard errors. The loneliness-by-time coefficient was pooled using the same two-stage REML and Hartung-Knapp procedure. This coefficient is on a log-odds scale and is therefore not numerically comparable with the primary standardized mean slope.

### Exploratory repeated-exposure analyses

Repeated-exposure patterns and lagged adjacent-assessment associations were treated as hypothesis-generating. Pattern classifications depended on later exposure observations and should not be interpreted as baseline prognostic groups or causal exposure regimes. The same-item analysis excluded SHARE because repeated harmonized single-item loneliness was unavailable; the available-signal analysis used the later SHARE three-item scale as an explicitly measurement-mixed fallback.

## Supplementary Tables

### Table S1 | Cohort design, measurements and analysis-set retention

{render_table(s1, ['cohort', 'setting', 'analysis_data_product', 'release_date', 'source_release_basis', 'analysis_source_file', 'baseline_wave', 'memory_waves', 'recall_words_per_test', 'baseline_loneliness', 'proxy_rule', 'maximum_observed_followup_years', 'baseline_eligible', 'longitudinal_sample', 'retained_percent'], 1)}

### Table S2 | Participant flow

{render_table(s2, ['cohort', 'source rows', 'age 50+', 'baseline eligible with respondent-completed baseline memory', 'longitudinal sample with later respondent-completed memory', 'retained_percent'], 1)}

### Table S3 | Baseline characteristics by cohort and loneliness group

{render_table(s3, ['cohort', 'group', 'participants', 'age_mean', 'age_sd', 'female_percent', 'tertiary_education_percent', 'partnered_percent', 'baseline_memory_mean', 'baseline_memory_sd', 'median_observations', 'median_followup_years'], 1)}

### Table S4 | Wave-specific observation and proxy audit

{render_table(s4, ['cohort', 'wave', 'scheduled', 'observed', 'observed_percent', 'proxy_indicator_status', 'n_proxy_yes', 'n_memory_pair_proxy_yes'], 1)}

`NA` indicates that the selected CHARLS harmonized file did not provide a proxy indicator; these entries must not be interpreted as zero proxy interviews. For cohorts with an available indicator, `observed` denotes a respondent-completed immediate-and-delayed recall pair after applying the proxy exclusion rule.

### Table S5 | Primary and sensitivity meta-analysis results

{render_table(s5, ['analysis', 'effect_scale', 'pooled_estimate', 'ci_low', 'ci_high', 'i2_percent', 'prediction_low', 'prediction_high'])}

### Table S6 | Baseline differences between participants included in and excluded from the longitudinal sample

Positive standardized differences indicate a higher mean or proportion among included participants.

{render_table(s6, ['cohort', 'label', 'included_mean_or_proportion', 'excluded_mean_or_proportion', 'standardized_difference'])}

### Table S7 | Exploratory repeated-exposure, lagged, nonlinear and effect-modification meta-results

{render_table(s7, ['analysis_family', 'contrast', 'effect_scale', 'k', 'estimate', 'ci_low', 'ci_high', 'i2_percent', 'prediction_low', 'prediction_high', 'adjusted_p_value'])}

### Table S8 | Availability and calendar coverage of harmonized death information

Death counts in this table describe metadata coverage in each full harmonized source file, not mortality incidence in the analytic sample. Differing calendar coverage was the reason death-specific weights were not pooled across cohorts.

{render_table(s8, ['cohort', 'source_rows', 'known_death_year_n', 'known_death_year_percent', 'minimum_known_death_year', 'maximum_known_death_year', 'maximum_analysis_interview_year'], 1)}

## Supplementary Reporting Notes

- `n` denotes participants for participant counts and participant-wave records for observation counts.
- The independent unit for all inferential models was the participant.
- The common primary exposure was a cohort-harmonized single loneliness item; harmonization and cohort-level standardization do not establish measurement invariance across languages or instruments.
- Primary mixed models, selected retest-timing analyses and weighted-GEE conclusions were independently reproduced in R. Exploratory analyses were not prospectively preregistered.
"""
    (ROOT / "supplementary_materials_draft.md").write_text(text, encoding="utf-8")


def main() -> None:
    tables = {
        "s1": cohort_design_table(),
        "s2": sample_flow_table(),
        "s3": baseline_table(),
        "s4": observation_table(),
        "s5": primary_results_table(),
        "s6": selection_table(),
        "s7": exploratory_table(),
        "s8": death_coverage_table(),
    }
    write_markdown(tables)
    print(f"Wrote {len(tables)} supplementary tables to {SUPP}")
    print(f"Wrote {ROOT / 'supplementary_materials_draft.md'}")


if __name__ == "__main__":
    main()
