"""Run prespecified sensitivity models for the loneliness-memory analysis."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.formula.api import mixedlm

from run_longitudinal_models import COHORTS, OUT_DIR, prepare_frame, reml_meta


BASE_COVARIATES = "age10 + female + C(education) + partnered + C(country_code)"

SENSITIVITIES = {
    "immediate_memory": {
        "outcome": "immediate_z",
        "exposure": "lonely",
        "formula": f"immediate_z ~ time10 * (lonely + {BASE_COVARIATES})",
    },
    "delayed_memory": {
        "outcome": "delayed_z",
        "exposure": "lonely",
        "formula": f"delayed_z ~ time10 * (lonely + {BASE_COVARIATES})",
    },
    "followup_adjusted_baseline_memory": {
        "outcome": "memory_z",
        "exposure": "lonely",
        "formula": (
            f"memory_z ~ time10 * (lonely + baseline_memory + {BASE_COVARIATES})"
        ),
    },
    "exclude_lowest_baseline_decile": {
        "outcome": "memory_z",
        "exposure": "lonely",
        "formula": f"memory_z ~ time10 * (lonely + {BASE_COVARIATES})",
    },
    "standardized_loneliness": {
        "outcome": "memory_z",
        "exposure": "loneliness_z",
        "formula": f"memory_z ~ time10 * (loneliness_z + {BASE_COVARIATES})",
    },
    "exclude_first_retest": {
        "outcome": "memory_z",
        "exposure": "lonely",
        "formula": f"memory_z ~ time10 * (lonely + {BASE_COVARIATES})",
    },
    "exclude_followup_within_2y": {
        "outcome": "memory_z",
        "exposure": "lonely",
        "formula": f"memory_z ~ time10 * (lonely + {BASE_COVARIATES})",
    },
}


def add_sensitivity_variables(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame.sort_values("time").drop_duplicates("pid").set_index("pid")
    for raw, standardized in [("immediate", "immediate_z"), ("delayed", "delayed_z")]:
        mean = baseline[raw].mean()
        sd = baseline[raw].std(ddof=1)
        frame[standardized] = (frame[raw] - mean) / sd
    frame["baseline_memory"] = frame["pid"].map(baseline["memory_z"])
    loneliness_mean = baseline["loneliness_raw"].mean()
    loneliness_sd = baseline["loneliness_raw"].std(ddof=1)
    frame["loneliness_z"] = (
        frame["loneliness_raw"] - loneliness_mean
    ) / loneliness_sd
    return frame


def sensitivity_frame(frame: pd.DataFrame, analysis: str) -> pd.DataFrame:
    result = frame.copy()
    if analysis == "followup_adjusted_baseline_memory":
        result = result[result["time"].gt(0)].copy()
    elif analysis == "exclude_lowest_baseline_decile":
        baseline = result.sort_values("time").drop_duplicates("pid")
        cutoff = baseline["memory_z"].quantile(0.10)
        retained = baseline.loc[baseline["memory_z"].gt(cutoff), "pid"]
        result = result[result["pid"].isin(retained)].copy()
    elif analysis == "exclude_first_retest":
        first_retest = (
            result.loc[result["time"].gt(0)]
            .sort_values(["pid", "time"])
            .drop_duplicates("pid")
        )
        result = result.drop(index=first_retest.index).copy()
    elif analysis == "exclude_followup_within_2y":
        result = result[result["time"].eq(0) | result["time"].gt(2)].copy()
    return result


def fit_one(frame: pd.DataFrame, formula: str, outcome: str, exposure: str) -> dict:
    target = f"time10:{exposure}"
    needed = [
        outcome,
        "time10",
        exposure,
        "age10",
        "female",
        "education",
        "partnered",
        "country_code",
    ]
    if "baseline_memory" in formula:
        needed.append("baseline_memory")
    model_frame = frame.dropna(subset=needed).copy()
    counts = model_frame.groupby("pid").size()
    model_frame = model_frame[model_frame["pid"].isin(counts[counts >= 2].index)]
    if model_frame["pid"].nunique() < 100:
        raise RuntimeError("Fewer than 100 participants with at least two observations")

    warning_messages = []
    fit = None
    random_structure = "random intercept and time slope"
    for re_formula in ["1 + time10", "1"]:
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
                    candidate = model.fit(
                        reml=True, method=method, maxiter=500, disp=False
                    )
                    warning_messages.extend(str(item.message) for item in caught)
                if candidate.converged:
                    fit = candidate
                    random_structure = (
                        "random intercept and time slope"
                        if re_formula != "1"
                        else "random intercept"
                    )
                    break
            except Exception as exc:
                warning_messages.append(
                    f"{re_formula}/{method}: {type(exc).__name__}: {exc}"
                )
        if fit is not None:
            break
    if fit is None:
        raise RuntimeError("Sensitivity model failed to converge")

    ci = fit.conf_int().loc[target]
    return {
        "estimate": float(fit.fe_params[target]),
        "std_error": float(fit.bse_fe[target]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "p_value": float(fit.pvalues[target]),
        "participants": int(model_frame["pid"].nunique()),
        "observations": int(len(model_frame)),
        "random_structure": random_structure,
        "converged": bool(fit.converged),
        "warnings": " | ".join(dict.fromkeys(warning_messages)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", nargs="+", choices=COHORTS, default=COHORTS)
    parser.add_argument(
        "--analyses",
        nargs="+",
        choices=list(SENSITIVITIES),
        default=list(SENSITIVITIES),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_cohorts = args.cohorts
    selected_analyses = args.analyses
    rows = []
    failures = []
    for cohort in selected_cohorts:
        prepared = add_sensitivity_variables(prepare_frame(cohort))
        for analysis, spec in SENSITIVITIES.items():
            if analysis not in selected_analyses:
                continue
            frame = sensitivity_frame(prepared, analysis)
            try:
                result = fit_one(
                    frame, spec["formula"], spec["outcome"], spec["exposure"]
                )
                rows.append({"cohort": cohort, "analysis": analysis, **result})
                print(
                    f"{cohort} {analysis}: beta={result['estimate']:.4f}, "
                    f"95% CI {result['ci_low']:.4f} to {result['ci_high']:.4f}, "
                    f"N={result['participants']:,}"
                )
            except Exception as exc:
                failures.append(
                    {"cohort": cohort, "analysis": analysis, "error": repr(exc)}
                )
                print(f"FAILED {cohort} {analysis}: {exc}")

    estimates = pd.DataFrame(rows)
    estimates_path = OUT_DIR / "sensitivity_cohort_estimates.csv"
    subset_run = (
        set(selected_cohorts) != set(COHORTS)
        or set(selected_analyses) != set(SENSITIVITIES)
    )
    if subset_run and estimates_path.exists():
        previous = pd.read_csv(estimates_path)
        selected_pairs = pd.MultiIndex.from_product(
            [selected_cohorts, selected_analyses], names=["cohort", "analysis"]
        )
        previous_pairs = pd.MultiIndex.from_frame(previous[["cohort", "analysis"]])
        previous = previous[~previous_pairs.isin(selected_pairs)]
        estimates = pd.concat([previous, estimates], ignore_index=True)
    estimates = estimates.sort_values(["analysis", "cohort"])
    estimates.to_csv(estimates_path, index=False)
    pd.DataFrame(failures).to_csv(OUT_DIR / "sensitivity_failures.csv", index=False)

    meta_rows = []
    for analysis, group in estimates.groupby("analysis"):
        if len(group) == len(COHORTS):
            meta_rows.append({"analysis": analysis, **reml_meta(group)})
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(OUT_DIR / "sensitivity_meta_results.csv", index=False)
    print("\nSensitivity meta-analyses")
    print(meta.to_string(index=False))


if __name__ == "__main__":
    main()
