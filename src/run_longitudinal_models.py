"""Fit cohort-specific longitudinal models and pool slope differences.

The estimand is the difference in baseline-standardized episodic-memory slope
associated with baseline loneliness, rescaled to a 10-year interval.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm, t
from statsmodels.formula.api import mixedlm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from project_config import DERIVED_DIR, OUTPUT_DIR, PROJECT_ROOT


ROOT = PROJECT_ROOT
OUT_DIR = OUTPUT_DIR
COHORTS = ["CHARLS", "ELSA", "HRS", "MHAS", "SHARE"]

CORE_FORMULA = (
    "memory_z ~ time10 * (lonely + age10 + female + C(education) + partnered "
    "+ C(country_code))"
)
FULL_FORMULA = (
    "memory_z ~ time10 * (lonely + age10 + female + C(education) + partnered "
    "+ depression_z + diabetes + stroke + C(country_code))"
)
TARGET_TERM = "time10:lonely"


def prepare_frame(cohort: str) -> pd.DataFrame:
    frame = pd.read_csv(DERIVED_DIR / f"{cohort.lower()}_long.csv.gz")
    frame["time10"] = frame["time"] / 10.0
    age_mean = frame.drop_duplicates("pid")["age"].mean()
    frame["age10"] = (frame["age"] - age_mean) / 10.0
    baseline = frame.drop_duplicates("pid")
    depression_mean = baseline["depression_excl_lonely"].mean()
    depression_sd = baseline["depression_excl_lonely"].std(ddof=1)
    frame["depression_z"] = (
        frame["depression_excl_lonely"] - depression_mean
    ) / depression_sd
    frame["education"] = frame["education"].astype("category")
    frame["country_code"] = frame["country_code"].astype("category")
    return frame


def fit_mixed_model(
    frame: pd.DataFrame, formula: str, model_name: str
) -> tuple[object, dict, list[dict]]:
    model_frame = frame.dropna(
        subset=[
            "memory_z",
            "time10",
            "lonely",
            "age10",
            "female",
            "education",
            "partnered",
            "country_code",
        ]
        + (
            ["depression_z", "diabetes", "stroke"]
            if model_name == "full"
            else []
        )
    ).copy()
    eligible_pid = model_frame.groupby("pid").size()
    model_frame = model_frame[model_frame["pid"].isin(eligible_pid[eligible_pid >= 2].index)]

    warning_messages: list[str] = []
    fit = None
    random_structure = "random intercept and time slope"
    for re_formula in ["1 + time10", "1"]:
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
                    reml=True,
                    method="lbfgs",
                    maxiter=500,
                    disp=False,
                )
                warning_messages.extend(str(item.message) for item in caught)
            if candidate.converged:
                fit = candidate
                random_structure = (
                    "random intercept and time slope" if re_formula != "1" else "random intercept"
                )
                break
            warning_messages.append(f"{re_formula}: optimizer did not converge")
        except Exception as exc:  # numerical fallback is recorded in diagnostics
            warning_messages.append(f"{re_formula}: {type(exc).__name__}: {exc}")

    if fit is None:
        raise RuntimeError(f"No converged {model_name} model: {' | '.join(warning_messages)}")

    fixed_rows = []
    ci = fit.conf_int()
    for term, estimate in fit.fe_params.items():
        fixed_rows.append(
            {
                "term": term,
                "estimate": estimate,
                "std_error": fit.bse_fe[term],
                "ci_low": ci.loc[term, 0],
                "ci_high": ci.loc[term, 1],
                "p_value": fit.pvalues[term],
            }
        )

    target = {
        "model": model_name,
        "estimate": float(fit.fe_params[TARGET_TERM]),
        "std_error": float(fit.bse_fe[TARGET_TERM]),
        "ci_low": float(ci.loc[TARGET_TERM, 0]),
        "ci_high": float(ci.loc[TARGET_TERM, 1]),
        "p_value": float(fit.pvalues[TARGET_TERM]),
        "participants": int(model_frame["pid"].nunique()),
        "observations": int(len(model_frame)),
        "converged": bool(fit.converged),
        "random_structure": random_structure,
        "log_likelihood": float(fit.llf),
        "scale": float(fit.scale),
        "warnings": " | ".join(dict.fromkeys(warning_messages)),
    }
    return fit, target, fixed_rows


def reml_meta(estimates: pd.DataFrame) -> dict:
    y = estimates["estimate"].to_numpy(dtype=float)
    variance = np.square(estimates["std_error"].to_numpy(dtype=float))
    k = len(y)

    def objective(tau2: float) -> float:
        weights = 1.0 / (variance + tau2)
        mean = np.sum(weights * y) / np.sum(weights)
        return 0.5 * (
            np.sum(np.log(variance + tau2))
            + np.log(np.sum(weights))
            + np.sum(weights * np.square(y - mean))
        )

    upper = max(float(np.var(y, ddof=1) * 20), float(np.max(variance) * 100), 1e-6)
    optimized = minimize_scalar(objective, bounds=(0, upper), method="bounded")
    tau2 = max(0.0, float(optimized.x))
    weights = 1.0 / (variance + tau2)
    pooled = float(np.sum(weights * y) / np.sum(weights))
    conventional_se = float(np.sqrt(1.0 / np.sum(weights)))
    q_re = float(np.sum(weights * np.square(y - pooled)))
    hk_scale = q_re / (k - 1)
    hk_se = float(np.sqrt(hk_scale / np.sum(weights)))
    hk_critical = float(t.ppf(0.975, df=k - 1))

    fixed_weights = 1.0 / variance
    fixed_mean = np.sum(fixed_weights * y) / np.sum(fixed_weights)
    q = float(np.sum(fixed_weights * np.square(y - fixed_mean)))
    i2 = max(0.0, (q - (k - 1)) / q) * 100 if q > 0 else 0.0
    q_p = float(chi2.sf(q, df=k - 1))
    prediction_critical = float(t.ppf(0.975, df=max(k - 2, 1)))
    prediction_half_width = prediction_critical * np.sqrt(tau2 + conventional_se**2)

    return {
        "k": k,
        "pooled_estimate": pooled,
        "hk_std_error": hk_se,
        "ci_low": pooled - hk_critical * hk_se,
        "ci_high": pooled + hk_critical * hk_se,
        "p_value": float(2 * t.sf(abs(pooled / hk_se), df=k - 1)) if hk_se > 0 else np.nan,
        "tau2_reml": tau2,
        "i2_percent": i2,
        "q": q,
        "q_df": k - 1,
        "q_p_value": q_p,
        "prediction_low": pooled - prediction_half_width,
        "prediction_high": pooled + prediction_half_width,
        "method": "two-stage REML; Hartung-Knapp CI; t-based prediction interval",
    }


def leave_one_out(estimates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in estimates["cohort"]:
        pooled = reml_meta(estimates.loc[estimates["cohort"].ne(cohort)])
        rows.append({"omitted_cohort": cohort, **pooled})
    return pd.DataFrame(rows)


def baseline_summary(frame: pd.DataFrame, cohort: str) -> list[dict]:
    baseline = frame.sort_values("time").drop_duplicates("pid")
    followup = frame.groupby("pid").agg(
        observations=("wave", "size"), followup_years=("time", "max")
    )
    baseline = baseline.join(followup, on="pid")
    rows = []
    for lonely, group in baseline.groupby("lonely"):
        rows.append(
            {
                "cohort": cohort,
                "lonely": int(lonely),
                "participants": len(group),
                "age_mean": group["age"].mean(),
                "age_sd": group["age"].std(ddof=1),
                "female_percent": group["female"].mean() * 100,
                "tertiary_education_percent": group["education"].astype(float).eq(3).mean() * 100,
                "partnered_percent": group["partnered"].mean() * 100,
                "baseline_memory_mean": group["memory_z"].mean(),
                "baseline_memory_sd": group["memory_z"].std(ddof=1),
                "median_observations": group["observations"].median(),
                "median_followup_years": group["followup_years"].median(),
            }
        )
    return rows


def main() -> None:
    targets = []
    fixed_effects = []
    baseline_rows = []
    diagnostics = []

    for cohort in COHORTS:
        frame = prepare_frame(cohort)
        baseline_rows.extend(baseline_summary(frame, cohort))
        for model_name, formula in [("core", CORE_FORMULA), ("full", FULL_FORMULA)]:
            fit, target, fixed = fit_mixed_model(frame, formula, model_name)
            target["cohort"] = cohort
            targets.append(target)
            for row in fixed:
                fixed_effects.append({"cohort": cohort, "model": model_name, **row})
            diagnostics.append(
                {
                    "cohort": cohort,
                    "model": model_name,
                    "converged": fit.converged,
                    "random_structure": target["random_structure"],
                    "warnings": target["warnings"],
                }
            )
            print(
                f"{cohort} {model_name}: beta={target['estimate']:.4f}, "
                f"95% CI {target['ci_low']:.4f} to {target['ci_high']:.4f}, "
                f"N={target['participants']:,}"
            )

    target_frame = pd.DataFrame(targets)
    target_frame.to_csv(OUT_DIR / "cohort_model_estimates.csv", index=False)
    pd.DataFrame(fixed_effects).to_csv(OUT_DIR / "model_fixed_effects.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(OUT_DIR / "baseline_descriptive.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT_DIR / "model_diagnostics.csv", index=False)

    meta_rows = []
    for model_name, group in target_frame.groupby("model"):
        meta_rows.append({"model": model_name, **reml_meta(group)})
        leave_one_out(group).assign(model=model_name).to_csv(
            OUT_DIR / f"leave_one_out_{model_name}.csv", index=False
        )
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(OUT_DIR / "meta_analysis_results.csv", index=False)
    print("\nMeta-analysis")
    print(meta.to_string(index=False))

    run_info = {
        "core_formula": CORE_FORMULA,
        "full_formula": FULL_FORMULA,
        "target_term": TARGET_TERM,
        "effect_unit": "SD difference in memory slope rescaled to a 10-year interval for lonely versus not lonely",
        "software": {
            "python": __import__("sys").version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "statsmodels": __import__("statsmodels").__version__,
            "scipy": __import__("scipy").__version__,
        },
    }
    (OUT_DIR / "analysis_run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
