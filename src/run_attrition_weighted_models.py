"""Estimate observation weights and run weighted GEE sensitivity models."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE

from run_longitudinal_models import COHORTS, DERIVED_DIR, OUT_DIR, reml_meta


OUTCOME_FORMULA = (
    "memory_z ~ time10 * (lonely + age10 + female + C(education) + partnered "
    "+ C(country_code))"
)
TARGET_TERM = "time10:lonely"


def effective_sample_size(weights: pd.Series) -> float:
    """Return the Kish effective sample size for a non-negative weight vector."""

    values = pd.to_numeric(weights, errors="coerce").dropna()
    denominator = float(np.square(values).sum())
    if values.empty or denominator <= 0:
        return float("nan")
    return float(np.square(values.sum()) / denominator)


def load_panel(cohort: str) -> pd.DataFrame:
    panel = pd.read_csv(DERIVED_DIR / f"{cohort.lower()}_attrition_panel.csv.gz")
    if not pd.api.types.is_bool_dtype(panel["observed"]):
        panel["observed"] = (
            panel["observed"].astype("string").str.lower().map({"true": True, "false": False})
        )
    panel["time10"] = panel["time"] / 10.0
    baseline = panel.sort_values("time").drop_duplicates("pid")
    panel["age10"] = (panel["age"] - baseline["age"].mean()) / 10.0
    panel["education"] = panel["education"].astype("category")
    panel["country_code"] = panel["country_code"].astype("category")
    return panel


def estimate_weights(panel: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, list[dict], dict]:
    result = panel.copy()
    result["ipow"] = 1.0
    followup = result[result["time"].gt(0)].copy()
    followup["observed_int"] = followup["observed"].astype(int)

    schedule_terms = "C(wave) + C(country_code)"
    denominator_formula = (
        "observed_int ~ "
        + schedule_terms
        + " + lonely + baseline_memory + age10 + female + C(education) + partnered"
    )
    numerator_formula = "observed_int ~ " + schedule_terms

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        denominator_fit = smf.glm(
            denominator_formula, data=followup, family=sm.families.Binomial()
        ).fit(maxiter=200)
        numerator_fit = smf.glm(
            numerator_formula, data=followup, family=sm.families.Binomial()
        ).fit(maxiter=200)

    denominator_p_raw = denominator_fit.predict(followup)
    numerator_p_raw = numerator_fit.predict(followup)
    denominator_p = np.clip(denominator_p_raw, 0.01, 0.99)
    numerator_p = np.clip(numerator_p_raw, 0.01, 0.99)
    stabilized = numerator_p / denominator_p
    result.loc[followup.index, "ipow"] = stabilized

    observed_followup = result["observed"] & result["time"].gt(0)
    lower, upper = result.loc[observed_followup, "ipow"].quantile([0.01, 0.99])
    result["ipow_truncated"] = result["ipow"].clip(lower=lower, upper=upper)

    baseline_weights = (
        result.sort_values("time").drop_duplicates("pid").set_index("pid")["baseline_weight"]
    )
    positive = baseline_weights[baseline_weights.gt(0)]
    normalized = baseline_weights / positive.mean()
    result["survey_weight"] = result["pid"].map(normalized)
    result["combined_weight"] = result["ipow_truncated"] * result["survey_weight"]

    observed_ipow = result.loc[observed_followup, "ipow_truncated"].dropna()
    observed_combined = result.loc[observed_followup, "combined_weight"].dropna()

    response_rows = []
    for wave, group in result.groupby("wave"):
        response_rows.append(
            {
                "cohort": cohort,
                "wave": int(wave),
                "scheduled": len(group),
                "observed": int(group["observed"].sum()),
                "observed_percent": group["observed"].mean() * 100,
            }
        )
    diagnostics = {
        "cohort": cohort,
        "denominator_converged": bool(denominator_fit.converged),
        "numerator_converged": bool(numerator_fit.converged),
        "ipow_observed_mean": result.loc[observed_followup, "ipow_truncated"].mean(),
        "ipow_observed_sd": result.loc[observed_followup, "ipow_truncated"].std(ddof=1),
        "ipow_observed_min": result.loc[observed_followup, "ipow_truncated"].min(),
        "ipow_observed_max": result.loc[observed_followup, "ipow_truncated"].max(),
        "ipow_truncation_low": lower,
        "ipow_truncation_high": upper,
        "denominator_min_probability": denominator_p.min(),
        "denominator_max_probability": denominator_p.max(),
        "denominator_probability_raw_min": denominator_p_raw.min(),
        "denominator_probability_raw_max": denominator_p_raw.max(),
        "denominator_probability_below_0_05_percent": float(
            100 * (denominator_p_raw < 0.05).mean()
        ),
        "denominator_probability_above_0_95_percent": float(
            100 * (denominator_p_raw > 0.95).mean()
        ),
        "ipow_observed_n": int(len(observed_ipow)),
        "ipow_effective_sample_size": effective_sample_size(observed_ipow),
        "combined_observed_n": int(len(observed_combined)),
        "combined_effective_sample_size": effective_sample_size(observed_combined),
    }
    return result, response_rows, diagnostics


def fit_gee(panel: pd.DataFrame, weight_column: str | None) -> dict:
    observed = panel[panel["observed"] & panel["memory_z"].notna()].copy()
    counts = observed.groupby("pid").size()
    observed = observed[observed["pid"].isin(counts[counts >= 2].index)]
    if weight_column:
        observed = observed.dropna(subset=[weight_column])
        weights = observed[weight_column]
    else:
        weights = None
    model = GEE.from_formula(
        OUTCOME_FORMULA,
        groups="pid",
        data=observed,
        cov_struct=Exchangeable(),
        family=Gaussian(),
        weights=weights,
    )
    fit = model.fit(maxiter=200, cov_type="robust")
    ci = fit.conf_int().loc[TARGET_TERM]
    return {
        "estimate": float(fit.params[TARGET_TERM]),
        "std_error": float(fit.bse[TARGET_TERM]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "p_value": float(fit.pvalues[TARGET_TERM]),
        "participants": int(observed["pid"].nunique()),
        "observations": int(len(observed)),
        "converged": bool(fit.converged),
        "dependence_parameter": float(np.atleast_1d(fit.cov_struct.dep_params)[0]),
    }


def main() -> None:
    estimates = []
    response_rows = []
    weight_diagnostics = []
    for cohort in COHORTS:
        panel, response, diagnostics = estimate_weights(load_panel(cohort), cohort)
        response_rows.extend(response)
        weight_diagnostics.append(diagnostics)
        for analysis, weight_column in [
            ("gee_unweighted", None),
            ("gee_attrition_ipw", "ipow_truncated"),
            ("gee_survey_x_attrition_ipw", "combined_weight"),
        ]:
            result = fit_gee(panel, weight_column)
            estimates.append({"cohort": cohort, "analysis": analysis, **result})
            print(
                f"{cohort} {analysis}: beta={result['estimate']:.4f}, "
                f"95% CI {result['ci_low']:.4f} to {result['ci_high']:.4f}"
            )

    estimate_frame = pd.DataFrame(estimates)
    estimate_frame.to_csv(OUT_DIR / "attrition_weighted_cohort_estimates.csv", index=False)
    pd.DataFrame(response_rows).to_csv(OUT_DIR / "attrition_response_by_wave.csv", index=False)
    pd.DataFrame(weight_diagnostics).to_csv(
        OUT_DIR / "attrition_weight_diagnostics.csv", index=False
    )
    meta_rows = []
    for analysis, group in estimate_frame.groupby("analysis"):
        meta_rows.append({"analysis": analysis, **reml_meta(group)})
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(OUT_DIR / "attrition_weighted_meta_results.csv", index=False)
    print("\nWeighted meta-analyses")
    print(meta.to_string(index=False))


if __name__ == "__main__":
    main()
