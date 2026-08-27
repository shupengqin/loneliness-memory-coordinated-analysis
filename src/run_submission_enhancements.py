"""Build reviewer-facing selection and bounded-outcome sensitivity analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE

from audit_longitudinal_feasibility import COHORTS as COHORT_SPECS
from run_longitudinal_models import COHORTS, DERIVED_DIR, OUT_DIR, prepare_frame, reml_meta


FRACTIONAL_LOGIT_FORMULA = (
    "recall_fraction ~ time10 * (lonely + age10 + female + C(education) + "
    "partnered + C(country_code))"
)
TARGET_TERM = "time10:lonely"
RECALL_MAX = {"CHARLS": 10.0, "ELSA": 10.0, "HRS": 10.0, "MHAS": 8.0, "SHARE": 10.0}


def as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype("string").str.lower().map({"true": True, "false": False})


def standardized_difference(included: pd.Series, excluded: pd.Series) -> float:
    pooled_sd = np.sqrt(0.5 * (included.var(ddof=1) + excluded.var(ddof=1)))
    if not np.isfinite(pooled_sd) or pooled_sd == 0:
        return np.nan
    return float((included.mean() - excluded.mean()) / pooled_sd)


def selection_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_rows: list[dict] = []
    balance_rows: list[dict] = []
    variables = {
        "baseline_memory": "Baseline episodic memory (SD)",
        "lonely": "Baseline loneliness",
        "age": "Age (years)",
        "female": "Women",
        "tertiary": "Tertiary education",
        "partnered": "Partnered",
    }

    for cohort in COHORTS:
        panel = pd.read_csv(DERIVED_DIR / f"{cohort.lower()}_attrition_panel.csv.gz")
        panel["observed"] = as_boolean(panel["observed"])
        baseline = panel.sort_values(["pid", "time", "wave"]).drop_duplicates("pid").copy()
        followup_observed = (
            panel.loc[panel["time"].gt(0)]
            .groupby("pid", observed=True)["observed"]
            .any()
        )
        baseline["included"] = baseline["pid"].map(followup_observed).fillna(False).astype(bool)
        baseline["tertiary"] = baseline["education"].eq(3).astype(float)

        cohort_rows.append(
            {
                "cohort": cohort,
                "baseline_eligible": int(len(baseline)),
                "longitudinal_sample": int(baseline["included"].sum()),
                "retained_percent": float(100 * baseline["included"].mean()),
                "excluded_before_repeat_assessment": int((~baseline["included"]).sum()),
            }
        )

        for variable, label in variables.items():
            included = pd.to_numeric(
                baseline.loc[baseline["included"], variable], errors="coerce"
            ).dropna()
            excluded = pd.to_numeric(
                baseline.loc[~baseline["included"], variable], errors="coerce"
            ).dropna()
            balance_rows.append(
                {
                    "cohort": cohort,
                    "variable": variable,
                    "label": label,
                    "included_n": int(len(included)),
                    "excluded_n": int(len(excluded)),
                    "included_mean_or_proportion": float(included.mean()),
                    "excluded_mean_or_proportion": float(excluded.mean()),
                    "standardized_difference": standardized_difference(included, excluded),
                }
            )

    return pd.DataFrame(cohort_rows), pd.DataFrame(balance_rows)


def fit_fractional_logit(cohort: str) -> dict:
    frame = prepare_frame(cohort)
    recall_max = RECALL_MAX[cohort]
    frame["recall_fraction"] = (
        frame["immediate"] + frame["delayed"]
    ) / (2.0 * recall_max)
    model_frame = frame.dropna(
        subset=[
            "recall_fraction",
            "time10",
            "lonely",
            "age10",
            "female",
            "education",
            "partnered",
            "country_code",
        ]
    ).copy()
    counts = model_frame.groupby("pid", observed=True).size()
    model_frame = model_frame[model_frame["pid"].isin(counts[counts >= 2].index)]

    model = GEE.from_formula(
        FRACTIONAL_LOGIT_FORMULA,
        groups="pid",
        data=model_frame,
        cov_struct=Exchangeable(),
        family=Binomial(),
    )
    fit = model.fit(maxiter=200, cov_type="robust")
    estimate = float(fit.params[TARGET_TERM])
    std_error = float(fit.bse[TARGET_TERM])
    critical = float(norm.ppf(0.975))
    return {
        "cohort": cohort,
        "analysis": "fractional_logit_recall",
        "scale": "log odds of the recalled fraction per 10 years",
        "estimate": estimate,
        "std_error": std_error,
        "ci_low": estimate - critical * std_error,
        "ci_high": estimate + critical * std_error,
        "p_value": float(fit.pvalues[TARGET_TERM]),
        "participants": int(model_frame["pid"].nunique()),
        "observations": int(len(model_frame)),
        "converged": bool(fit.converged),
        "dependence_parameter": float(np.atleast_1d(fit.cov_struct.dep_params)[0]),
    }


def death_coverage_audit() -> pd.DataFrame:
    rows: list[dict] = []
    for cohort in COHORTS:
        spec = COHORT_SPECS[cohort]
        labels = pd.io.stata.StataReader(
            spec["file"], convert_categoricals=False
        ).variable_labels()
        columns = [column for column in ["radyear", "radmonth"] if column in labels]
        source = pd.read_stata(
            spec["file"], columns=columns, convert_categoricals=False
        )
        death_year = (
            pd.to_numeric(source["radyear"], errors="coerce").where(
                lambda values: values.between(1900, 2100)
            )
            if "radyear" in source
            else pd.Series(np.nan, index=source.index)
        )
        death_month = (
            pd.to_numeric(source["radmonth"], errors="coerce").where(
                lambda values: values.between(1, 12)
            )
            if "radmonth" in source
            else pd.Series(np.nan, index=source.index)
        )
        long_frame = pd.read_csv(
            DERIVED_DIR / f"{cohort.lower()}_long.csv.gz",
            usecols=["interview_year"],
        )
        rows.append(
            {
                "cohort": cohort,
                "source_rows": int(len(source)),
                "known_death_year_n": int(death_year.notna().sum()),
                "known_death_year_percent": float(100 * death_year.notna().mean()),
                "known_death_month_n": int(death_month.notna().sum()),
                "minimum_known_death_year": float(death_year.min())
                if death_year.notna().any()
                else np.nan,
                "maximum_known_death_year": float(death_year.max())
                if death_year.notna().any()
                else np.nan,
                "maximum_analysis_interview_year": float(long_frame["interview_year"].max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    selection, balance = selection_diagnostics()
    selection.to_csv(OUT_DIR / "selection_to_longitudinal_sample.csv", index=False)
    balance.to_csv(OUT_DIR / "selection_baseline_balance.csv", index=False)
    death_coverage_audit().to_csv(OUT_DIR / "death_coverage_audit.csv", index=False)

    bounded = pd.DataFrame([fit_fractional_logit(cohort) for cohort in COHORTS])
    bounded.to_csv(OUT_DIR / "bounded_recall_gee_cohort_estimates.csv", index=False)
    pooled = pd.DataFrame(
        [{"analysis": "fractional_logit_recall", **reml_meta(bounded)}]
    )
    pooled.to_csv(OUT_DIR / "bounded_recall_gee_meta_results.csv", index=False)

    print("Selection into the longitudinal sample")
    print(selection.to_string(index=False))
    print("\nFractional-logit recall sensitivity")
    print(bounded.to_string(index=False))
    print("\nPooled fractional-logit result")
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
