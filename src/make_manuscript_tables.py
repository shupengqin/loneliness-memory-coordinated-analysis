"""Create the single main table from the validated global ageing analysis outputs."""

from __future__ import annotations

import os
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


def main_table() -> None:
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
                "I-squared (%)": "-",
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
            "I-squared (%)": one(baseline_meta["i2_percent"]),
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
                "I-squared (%)": "-",
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
            "I-squared (%)": one(slope_meta["i2_percent"]),
            "95% prediction interval": f"{three(slope_meta['prediction_low'])} to {three(slope_meta['prediction_high'])}",
        }
    )
    write_table(
        "table_1_primary_model_estimates",
        pd.DataFrame(rows),
        "Table 1 | Primary model estimates across five ageing cohorts",
        "Cohort-specific estimates came from core linear mixed-effects models with participant-specific random intercepts and time slopes in the selected longitudinal analysis sample. Pooled estimates used two-stage REML random-effects meta-analysis with Hartung-Knapp confidence intervals and t-based prediction intervals. Five cohort-level estimates were pooled; SHARE contributed one estimate after adjustment for country and country-by-time terms. Baseline memory level and the memory slope scaled to a 10-year interval are reported on the cohort-specific baseline-SD scale; this scaling does not imply 10 years of observation for every cohort. A confidence interval crossing zero is not an equivalence test, and heterogeneity statistics based on five units are descriptive.",
    )


def main() -> None:
    main_table()
    print(f"Wrote one main table to {TABLE_DIR}")


if __name__ == "__main__":
    main()
