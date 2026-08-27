"""Test sex, age, and education modification using direct interactions."""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

from run_longitudinal_models import COHORTS, OUT_DIR, prepare_frame, reml_meta


FORMULAS = {
    "sex": (
        "memory_z ~ time10 * (lonely * female + age10 + C(education) + partnered "
        "+ C(country_code))"
    ),
    "age": (
        "memory_z ~ time10 * (lonely * age10 + female + C(education) + partnered "
        "+ C(country_code))"
    ),
    "education": (
        "memory_z ~ time10 * (lonely * C(education) + age10 + female + partnered "
        "+ C(country_code))"
    ),
}


def target_terms(modifier: str, terms: list[str]) -> list[str]:
    if modifier == "sex":
        return [term for term in terms if set(term.split(":")) == {"time10", "lonely", "female"}]
    if modifier == "age":
        return [term for term in terms if set(term.split(":")) == {"time10", "lonely", "age10"}]
    return [
        term
        for term in terms
        if "time10" in term and "lonely" in term and "C(education)[T." in term
    ]


def fit_modifier(frame: pd.DataFrame, modifier: str, formula: str) -> tuple[list[dict], dict]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = mixedlm(
            formula,
            frame,
            groups=frame["pid"],
            re_formula="1 + time10",
            missing="drop",
        )
        fit = model.fit(reml=True, method="lbfgs", maxiter=500, disp=False)
    if not fit.converged:
        raise RuntimeError(f"{modifier} interaction model did not converge")

    ci = fit.conf_int()
    rows = []
    targets = target_terms(modifier, list(fit.fe_params.index))
    if not targets:
        raise RuntimeError(f"No target terms found for {modifier}")
    for term in targets:
        rows.append(
            {
                "modifier": modifier,
                "term": term,
                "estimate": float(fit.fe_params[term]),
                "std_error": float(fit.bse_fe[term]),
                "ci_low": float(ci.loc[term, 0]),
                "ci_high": float(ci.loc[term, 1]),
                "p_value": float(fit.pvalues[term]),
                "participants": int(frame["pid"].nunique()),
                "observations": int(len(frame)),
            }
        )
    diagnostic = {
        "modifier": modifier,
        "converged": bool(fit.converged),
        "warnings": " | ".join(dict.fromkeys(str(item.message) for item in caught)),
    }
    return rows, diagnostic


def harmonized_contrast(term: str) -> str:
    if "female" in term:
        return "female_vs_male"
    if "age10" in term:
        return "per_10_year_older_age"
    if "[T.2" in term:
        return "upper_secondary_vs_low_education"
    if "[T.3" in term:
        return "tertiary_vs_low_education"
    return term


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", nargs="+", choices=COHORTS, default=COHORTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_cohorts = args.cohorts
    rows = []
    diagnostics = []
    for cohort in selected_cohorts:
        frame = prepare_frame(cohort)
        for modifier, formula in FORMULAS.items():
            targets, diagnostic = fit_modifier(frame, modifier, formula)
            for row in targets:
                row["cohort"] = cohort
                row["contrast"] = harmonized_contrast(row["term"])
                rows.append(row)
                print(
                    f"{cohort} {row['contrast']}: beta={row['estimate']:.4f}, "
                    f"95% CI {row['ci_low']:.4f} to {row['ci_high']:.4f}"
                )
            diagnostics.append({"cohort": cohort, **diagnostic})

    estimates = pd.DataFrame(rows)
    estimate_path = OUT_DIR / "effect_modification_cohort_estimates.csv"
    diagnostic_frame = pd.DataFrame(diagnostics)
    diagnostic_path = OUT_DIR / "effect_modification_diagnostics.csv"
    if set(selected_cohorts) != set(COHORTS) and estimate_path.exists():
        previous = pd.read_csv(estimate_path)
        estimates = pd.concat(
            [previous[~previous["cohort"].isin(selected_cohorts)], estimates],
            ignore_index=True,
        )
    if set(selected_cohorts) != set(COHORTS) and diagnostic_path.exists():
        previous_diagnostics = pd.read_csv(diagnostic_path)
        diagnostic_frame = pd.concat(
            [
                previous_diagnostics[
                    ~previous_diagnostics["cohort"].isin(selected_cohorts)
                ],
                diagnostic_frame,
            ],
            ignore_index=True,
        )
    estimates = estimates.sort_values(["contrast", "cohort"])
    estimates.to_csv(estimate_path, index=False)
    diagnostic_frame.to_csv(diagnostic_path, index=False)

    pooled_rows = []
    for contrast, group in estimates.groupby("contrast"):
        pooled_rows.append({"contrast": contrast, **reml_meta(group)})
    pooled = pd.DataFrame(pooled_rows)
    pooled["p_value_fdr"] = multipletests(
        pooled["p_value"].to_numpy(), method="fdr_bh"
    )[1]
    pooled.to_csv(OUT_DIR / "effect_modification_meta_results.csv", index=False)
    print("\nEffect-modification meta-analyses")
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
