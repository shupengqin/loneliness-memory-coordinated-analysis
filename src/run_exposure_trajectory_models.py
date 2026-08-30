"""Run repeated-loneliness, lagged-transition, and nonlinear-time analyses.

These analyses are secondary to the primary baseline-loneliness model.
They use the anonymized derived long files and write separate output tables.
The ``same_item`` rule is limited to repeated single-item loneliness measures;
the ``available_signal`` rule uses SHARE's later three-item scale as an
explicitly labelled exploratory fallback.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.formula.api import mixedlm

from run_longitudinal_models import COHORTS, OUT_DIR, prepare_frame, reml_meta


BASE_COVARIATES = (
    "age10 + female + C(education) + partnered + C(country_code)"
)
PATTERN_LEVELS = ["never", "transient_or_changing", "persistent"]
EXPOSURE_RULES = ["same_item", "available_signal"]


def exposure_mask(frame: pd.DataFrame, rule: str) -> pd.Series:
    if rule == "same_item":
        return frame["loneliness_observed"] & frame["loneliness_instrument"].eq(
            "single_item"
        )
    if rule == "available_signal":
        return frame["loneliness_observed"]
    raise ValueError(f"Unknown exposure rule: {rule}")


def add_exposure_pattern(
    frame: pd.DataFrame, rule: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify participants using available memory-plus-exposure waves.

    A pattern requires a valid baseline exposure and at least one later wave
    with both a direct memory assessment and a valid loneliness signal. The
    mixed category means that both loneliness states were observed; it is not
    interpreted as a perfectly measured transient exposure.
    """

    result = frame.sort_values(["pid", "time", "wave"]).copy()
    valid = exposure_mask(result, rule)
    baseline = result.drop_duplicates("pid").set_index("pid")
    baseline_ok = baseline["lonely_wave"].notna()
    if rule == "same_item":
        baseline_ok &= baseline["loneliness_instrument"].eq("single_item")
    baseline_ok = baseline_ok.to_dict()

    observed = result.loc[valid, ["pid", "time", "lonely_wave"]].copy()
    summary = observed.groupby("pid").agg(
        exposure_waves=("lonely_wave", "size"),
        minimum_loneliness=("lonely_wave", "min"),
        maximum_loneliness=("lonely_wave", "max"),
    )
    summary["baseline_exposure_valid"] = summary.index.map(baseline_ok).fillna(False)
    summary["pattern"] = pd.NA
    eligible = summary["baseline_exposure_valid"] & summary["exposure_waves"].ge(2)
    summary.loc[
        eligible
        & summary["minimum_loneliness"].eq(0)
        & summary["maximum_loneliness"].eq(0),
        "pattern",
    ] = "never"
    summary.loc[
        eligible
        & summary["minimum_loneliness"].eq(1)
        & summary["maximum_loneliness"].eq(1),
        "pattern",
    ] = "persistent"
    summary.loc[
        eligible
        & summary["minimum_loneliness"].eq(0)
        & summary["maximum_loneliness"].eq(1),
        "pattern",
    ] = "transient_or_changing"

    pattern_map = summary["pattern"].to_dict()
    result["exposure_pattern"] = result["pid"].map(pattern_map)
    pattern_dtype = pd.CategoricalDtype(PATTERN_LEVELS, ordered=True)
    result["exposure_pattern"] = result["exposure_pattern"].astype(pattern_dtype)

    count_rows = []
    for pattern in PATTERN_LEVELS:
        count_rows.append(
            {
                "rule": rule,
                "pattern": pattern,
                "participants": int(summary["pattern"].eq(pattern).sum()),
            }
        )
    count_rows.append(
        {
            "rule": rule,
            "pattern": "eligible_any_pattern",
            "participants": int(summary["pattern"].notna().sum()),
        }
    )
    return result, pd.DataFrame(count_rows)


def fit_mixed(
    frame: pd.DataFrame,
    formula: str,
    required: list[str],
    target_terms: list[str],
    random_structures: list[str],
) -> tuple[object, pd.DataFrame, str, list[str]]:
    model_frame = frame.dropna(subset=required).copy()
    counts = model_frame.groupby("pid").size()
    model_frame = model_frame[
        model_frame["pid"].isin(counts[counts.ge(2)].index)
    ].copy()
    if model_frame["pid"].nunique() < 100:
        raise RuntimeError("Fewer than 100 participants with at least two observations")

    warning_messages: list[str] = []
    for re_formula in random_structures:
        for method in ["lbfgs", "powell"]:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = mixedlm(
                        formula,
                        model_frame,
                        groups=model_frame["pid"],
                        re_formula=re_formula,
                        missing="drop",
                    )
                    fit = model.fit(
                        reml=True, method=method, maxiter=500, disp=False
                    )
                    warning_messages.extend(str(item.message) for item in caught)
                if fit.converged and all(term in fit.fe_params for term in target_terms):
                    return fit, model_frame, re_formula, warning_messages
                warning_messages.append(
                    f"{re_formula}/{method}: missing target term or no convergence"
                )
            except Exception as exc:
                warning_messages.append(
                    f"{re_formula}/{method}: {type(exc).__name__}: {exc}"
                )
    raise RuntimeError("No converged model: " + " | ".join(warning_messages))


def effect_row(
    fit: object,
    model_frame: pd.DataFrame,
    cohort: str,
    analysis: str,
    contrast: str,
    term: str,
    random_structure: str,
    warnings_seen: list[str],
) -> dict:
    ci = fit.conf_int().loc[term]
    return {
        "cohort": cohort,
        "analysis": analysis,
        "contrast": contrast,
        "term": term,
        "estimate": float(fit.fe_params[term]),
        "std_error": float(fit.bse_fe[term]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "p_value": float(fit.pvalues[term]),
        "participants": int(model_frame["pid"].nunique()),
        "observations": int(len(model_frame)),
        "random_structure": random_structure,
        "converged": bool(fit.converged),
        "warnings": " | ".join(dict.fromkeys(warnings_seen)),
    }


def pattern_models() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    estimates: list[dict] = []
    counts: list[pd.DataFrame] = []
    failures: list[dict] = []
    formula = (
        "memory_z ~ time10 * (C(exposure_pattern) + "
        f"{BASE_COVARIATES})"
    )
    required = [
        "memory_z",
        "time10",
        "exposure_pattern",
        "age10",
        "female",
        "education",
        "partnered",
        "country_code",
    ]

    for rule in EXPOSURE_RULES:
        for cohort in COHORTS:
            prepared = prepare_frame(cohort)
            frame, count_frame = add_exposure_pattern(prepared, rule)
            count_frame.insert(0, "cohort", cohort)
            counts.append(count_frame)
            eligible = frame["exposure_pattern"].notna()
            if not eligible.any():
                failures.append(
                    {"analysis": "exposure_pattern", "rule": rule, "cohort": cohort, "error": "No eligible patterns"}
                )
                continue
            try:
                fit, model_frame, re_formula, warning_messages = fit_mixed(
                    frame.loc[eligible],
                    formula,
                    required,
                    [
                        "time10:C(exposure_pattern)[T.persistent]",
                        "time10:C(exposure_pattern)[T.transient_or_changing]",
                    ],
                    ["1 + time10", "1"],
                )
                for contrast, term in [
                    (
                        "persistent_vs_never",
                        "time10:C(exposure_pattern)[T.persistent]",
                    ),
                    (
                        "transient_or_changing_vs_never",
                        "time10:C(exposure_pattern)[T.transient_or_changing]",
                    ),
                ]:
                    estimates.append(
                        effect_row(
                            fit,
                            model_frame,
                            cohort,
                            "exposure_pattern",
                            f"{rule}:{contrast}",
                            term,
                            re_formula,
                            warning_messages,
                        )
                    )
                print(
                    f"{cohort} {rule} pattern: N={model_frame['pid'].nunique():,}, "
                    f"persistent={int(count_frame.loc[count_frame['pattern'].eq('persistent'), 'participants'].iloc[0]):,}, "
                    f"transient={int(count_frame.loc[count_frame['pattern'].eq('transient_or_changing'), 'participants'].iloc[0]):,}"
                )
            except Exception as exc:
                failures.append(
                    {
                        "analysis": "exposure_pattern",
                        "rule": rule,
                        "cohort": cohort,
                        "error": repr(exc),
                    }
                )
                print(f"FAILED {cohort} {rule} pattern: {exc}")

    estimate_frame = pd.DataFrame(estimates)
    count_frame = pd.concat(counts, ignore_index=True)
    meta_rows = []
    for contrast, group in estimate_frame.groupby("contrast"):
        if len(group) >= 3:
            meta_rows.append({"contrast": contrast, **reml_meta(group)})
    return estimate_frame, count_frame, pd.DataFrame(meta_rows), pd.DataFrame(failures)


def lagged_pairs(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    result = frame.sort_values(["pid", "time", "wave"]).copy()
    valid = exposure_mask(result, rule)
    grouped = result.groupby("pid", sort=False)
    result["memory_next"] = grouped["memory_z"].shift(-1)
    result["time_next"] = grouped["time"].shift(-1)
    result["wave_next"] = grouped["wave"].shift(-1)
    pairs = result.loc[
        valid & result["memory_next"].notna() & result["time_next"].gt(result["time"])
    ].copy()
    pairs["memory_prev"] = pairs["memory_z"]
    pairs["lonely_prev"] = pairs["lonely_wave"]
    pairs["time_gap10"] = (pairs["time_next"] - pairs["time"]) / 10.0
    pairs["memory_change"] = pairs["memory_next"] - pairs["memory_prev"]
    return pairs


def lagged_models() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    estimates: list[dict] = []
    diagnostics: list[dict] = []
    failures: list[dict] = []
    formula = (
        "memory_next ~ lonely_prev + memory_prev + time_gap10 + "
        f"{BASE_COVARIATES} + C(wave_next)"
    )
    required = [
        "memory_next",
        "lonely_prev",
        "memory_prev",
        "time_gap10",
        "age10",
        "female",
        "education",
        "partnered",
        "country_code",
        "wave_next",
    ]
    for rule in EXPOSURE_RULES:
        for cohort in COHORTS:
            frame = prepare_frame(cohort)
            pairs = lagged_pairs(frame, rule)
            diagnostics.append(
                {
                    "cohort": cohort,
                    "rule": rule,
                    "pairs": len(pairs),
                    "participants": pairs["pid"].nunique(),
                    "median_time_gap": pairs["time_next"].sub(pairs["time"]).median(),
                    "same_wave_step_percent": (pairs["wave_next"].sub(pairs["wave"]).eq(1).mean() * 100) if len(pairs) else np.nan,
                }
            )
            if pairs.empty:
                failures.append(
                    {"analysis": "lagged_transition", "rule": rule, "cohort": cohort, "error": "No eligible pairs"}
                )
                continue
            try:
                # Transition rows are clustered within participant; this exploratory
                # model intentionally uses a participant-specific random intercept only.
                fit, model_frame, re_formula, warning_messages = fit_mixed(
                    pairs,
                    formula,
                    required,
                    ["lonely_prev"],
                    ["1"],
                )
                estimates.append(
                    effect_row(
                        fit,
                        model_frame,
                        cohort,
                        "lagged_transition",
                        rule,
                        "lonely_prev",
                        re_formula,
                        warning_messages,
                    )
                )
                print(
                    f"{cohort} {rule} lagged: beta={fit.fe_params['lonely_prev']:.4f}, "
                    f"pairs={len(model_frame):,}"
                )
            except Exception as exc:
                failures.append(
                    {
                        "analysis": "lagged_transition",
                        "rule": rule,
                        "cohort": cohort,
                        "error": repr(exc),
                    }
                )
                print(f"FAILED {cohort} {rule} lagged: {exc}")

    estimate_frame = pd.DataFrame(estimates)
    meta_rows = []
    for rule, group in estimate_frame.groupby("contrast"):
        if len(group) >= 3:
            meta_rows.append({"rule": rule, **reml_meta(group)})
    return estimate_frame, pd.DataFrame(diagnostics), pd.DataFrame(meta_rows), pd.DataFrame(failures)


def contrast_at_time(fit: object, years: float) -> dict:
    terms = ["lonely", "time10:lonely", "time10_sq:lonely"]
    coefficients = np.array([1.0, years / 10.0, (years / 10.0) ** 2])
    beta = fit.fe_params.loc[terms].to_numpy(dtype=float)
    covariance = fit.cov_params().loc[terms, terms].to_numpy(dtype=float)
    estimate = float(coefficients @ beta)
    variance = float(coefficients @ covariance @ coefficients)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    z_value = estimate / standard_error if standard_error else np.nan
    return {
        "years": years,
        "estimate": estimate,
        "std_error": standard_error,
        "ci_low": estimate - norm.ppf(0.975) * standard_error,
        "ci_high": estimate + norm.ppf(0.975) * standard_error,
        "p_value": float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan,
    }


def nonlinear_models() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    estimates: list[dict] = []
    coefficients: list[dict] = []
    failures: list[dict] = []
    formula = (
        "memory_z ~ time10 + time10_sq + lonely + time10:lonely + "
        "time10_sq:lonely + age10 + female + C(education) + partnered + "
        "C(country_code) + time10:age10 + time10:female + "
        "time10:C(education) + time10:partnered + time10:C(country_code)"
    )
    required = [
        "memory_z",
        "time10",
        "time10_sq",
        "lonely",
        "age10",
        "female",
        "education",
        "partnered",
        "country_code",
    ]
    target_terms = ["lonely", "time10:lonely", "time10_sq:lonely"]
    for cohort in COHORTS:
        frame = prepare_frame(cohort)
        frame["time10_sq"] = frame["time10"] ** 2
        try:
            fit, model_frame, re_formula, warning_messages = fit_mixed(
                frame,
                formula,
                required,
                target_terms,
                ["1 + time10", "1"],
            )
            for term in target_terms:
                ci = fit.conf_int().loc[term]
                coefficients.append(
                    {
                        "cohort": cohort,
                        "term": term,
                        "estimate": float(fit.fe_params[term]),
                        "std_error": float(fit.bse_fe[term]),
                        "ci_low": float(ci.iloc[0]),
                        "ci_high": float(ci.iloc[1]),
                        "p_value": float(fit.pvalues[term]),
                        "participants": int(model_frame["pid"].nunique()),
                        "observations": int(len(model_frame)),
                        "random_structure": re_formula,
                        "converged": bool(fit.converged),
                        "warnings": " | ".join(dict.fromkeys(warning_messages)),
                    }
                )
            for years in [2.0, 5.0, 10.0]:
                estimates.append(
                    {
                        "cohort": cohort,
                        "analysis": "nonlinear_loneliness_difference",
                        "contrast": "lonely_minus_not_lonely",
                        **contrast_at_time(fit, years),
                        "participants": int(model_frame["pid"].nunique()),
                        "observations": int(len(model_frame)),
                        "random_structure": re_formula,
                        "converged": bool(fit.converged),
                        "warnings": " | ".join(dict.fromkeys(warning_messages)),
                    }
                )
            print(
                f"{cohort} nonlinear: baseline={fit.fe_params['lonely']:.4f}, "
                f"linear_interaction={fit.fe_params['time10:lonely']:.4f}, "
                f"quadratic_interaction={fit.fe_params['time10_sq:lonely']:.4f}"
            )
        except Exception as exc:
            failures.append(
                {
                    "analysis": "nonlinear_loneliness_difference",
                    "cohort": cohort,
                    "error": repr(exc),
                }
            )
            print(f"FAILED {cohort} nonlinear: {exc}")

    estimate_frame = pd.DataFrame(estimates)
    coefficient_frame = pd.DataFrame(coefficients)
    meta_rows = []
    for years, group in estimate_frame.groupby("years"):
        if len(group) >= 3:
            meta_rows.append(
                {
                    "years": years,
                    "contrast": "lonely_minus_not_lonely",
                    **reml_meta(group),
                }
            )
    coefficient_meta_rows = []
    for term, group in coefficient_frame.groupby("term"):
        if len(group) >= 3:
            coefficient_meta_rows.append({"term": term, **reml_meta(group)})
    return (
        estimate_frame,
        coefficient_frame,
        pd.DataFrame(meta_rows),
        pd.DataFrame(coefficient_meta_rows),
        pd.DataFrame(failures),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analyses",
        nargs="+",
        choices=["patterns", "lagged", "nonlinear"],
        default=["patterns", "lagged", "nonlinear"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_info = {
        "script": Path(__file__).name,
        "exposure_rules": {
            "same_item": "Repeated single-item loneliness only",
            "available_signal": "Single item where available; SHARE later three-item scale as exploratory fallback",
        },
        "pattern_definition": "Baseline plus at least one later observed memory-and-exposure wave; mixed states labelled transient_or_changing",
    }

    if "patterns" in args.analyses:
        estimates, counts, meta, failures = pattern_models()
        estimates.to_csv(OUT_DIR / "exposure_pattern_cohort_estimates.csv", index=False)
        counts.to_csv(OUT_DIR / "exposure_pattern_counts.csv", index=False)
        meta.to_csv(OUT_DIR / "exposure_pattern_meta_results.csv", index=False)
        failures.to_csv(OUT_DIR / "exposure_pattern_failures.csv", index=False)
        run_info["patterns"] = {"cohort_estimates": len(estimates), "failures": len(failures)}

    if "lagged" in args.analyses:
        estimates, diagnostics, meta, failures = lagged_models()
        estimates.to_csv(OUT_DIR / "lagged_transition_cohort_estimates.csv", index=False)
        diagnostics.to_csv(OUT_DIR / "lagged_transition_diagnostics.csv", index=False)
        meta.to_csv(OUT_DIR / "lagged_transition_meta_results.csv", index=False)
        failures.to_csv(OUT_DIR / "lagged_transition_failures.csv", index=False)
        run_info["lagged"] = {"cohort_estimates": len(estimates), "failures": len(failures)}

    if "nonlinear" in args.analyses:
        estimates, coefficients, meta, coefficient_meta, failures = nonlinear_models()
        estimates.to_csv(OUT_DIR / "nonlinear_cohort_estimates.csv", index=False)
        coefficients.to_csv(OUT_DIR / "nonlinear_coefficients.csv", index=False)
        meta.to_csv(OUT_DIR / "nonlinear_meta_results.csv", index=False)
        coefficient_meta.to_csv(
            OUT_DIR / "nonlinear_coefficient_meta_results.csv", index=False
        )
        failures.to_csv(OUT_DIR / "nonlinear_failures.csv", index=False)
        run_info["nonlinear"] = {"cohort_estimates": len(estimates), "failures": len(failures)}

    (OUT_DIR / "extended_analysis_run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
