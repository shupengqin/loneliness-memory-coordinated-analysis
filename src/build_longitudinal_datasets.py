"""Build pseudonymized long files for the five-cohort loneliness-memory study.

Raw Stata files are read only. Derived files contain a cohort-local sequential
participant number rather than the source identifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from audit_longitudinal_feasibility import COHORTS
from project_config import DERIVED_DIR, OUTPUT_DIR, require_input_files


OUT_DIR = OUTPUT_DIR


@dataclass(frozen=True)
class BuildSpec:
    date_kind: str
    education_kind: str
    depression: str
    recall_max: float
    has_proxy: bool


BUILD_SPECS = {
    "CHARLS": BuildSpec("year_month", "three_level", "cesd10", 10.0, False),
    "ELSA": BuildSpec("year", "three_level", "cesd", 10.0, True),
    "HRS": BuildSpec("stata_date", "hrs_five_level", "cesd", 10.0, True),
    "MHAS": BuildSpec("year_month", "three_level", "cesd_m", 8.0, True),
    "SHARE": BuildSpec("year_month", "three_level", "cesd", 10.0, True),
}


def available_columns(path: Path) -> set[str]:
    reader = pd.io.stata.StataReader(path, convert_categoricals=False)
    return set(reader.variable_labels())


def valid_numeric(series: pd.Series, low: float, high: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values.between(low, high))


def education_three_level(series: pd.Series, kind: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if kind == "three_level":
        return values.where(values.isin([1, 2, 3]))
    if kind == "hrs_five_level":
        # RAND HRS: <HS; GED; HS; some college; college+. GED is grouped
        # with completed upper-secondary/vocational education.
        return values.map({1: 1, 2: 2, 3: 2, 4: 2, 5: 3})
    raise ValueError(f"Unknown education coding: {kind}")


def interview_decimal_year(
    frame: pd.DataFrame, wave: int, kind: str
) -> pd.Series:
    if kind == "year_month":
        year = pd.to_numeric(frame[f"r{wave}iwy"], errors="coerce")
        month = valid_numeric(frame[f"r{wave}iwm"], 1, 12)
        # Mid-year fallback avoids assigning January when only the year is known.
        month_fraction = (month.fillna(6.5) - 0.5) / 12.0
        return year + month_fraction
    if kind == "year":
        return pd.to_numeric(frame[f"r{wave}iwindy"], errors="coerce") + 0.5
    if kind == "stata_date":
        value = frame[f"r{wave}iwmid"]
        if pd.api.types.is_datetime64_any_dtype(value):
            return value.dt.year + (value.dt.dayofyear - 0.5) / 365.25
        # Stata daily dates are days since 1960-01-01 when not auto-converted.
        date = pd.Timestamp("1960-01-01") + pd.to_timedelta(
            pd.to_numeric(value, errors="coerce"), unit="D"
        )
        return date.dt.year + (date.dt.dayofyear - 0.5) / 365.25
    raise ValueError(f"Unknown date coding: {kind}")


def loneliness_wave_values(
    frame: pd.DataFrame, name: str, spec: dict, wave: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return a validated loneliness value, binary indicator, and source label.

    SHARE has no repeated single-item loneliness variable in the harmonized
    file after wave 2. Its later three-item scale is retained as an explicitly
    labelled fallback for exploratory repeated-exposure analyses only.
    """

    primary = spec["loneliness"].format(wave=wave)
    candidates = [(primary, "single_item")]
    if name == "SHARE":
        candidates.append((f"r{wave}lnlys3", "three_item"))

    for column, source in candidates:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if name == "CHARLS" and source == "single_item":
            raw = values.where(values.between(1, 4))
            lonely = raw.gt(1).astype(float).where(raw.notna())
        elif source == "three_item":
            raw = values.where(values.between(1, 3))
            lonely = raw.gt(1).astype(float).where(raw.notna())
        else:
            raw = values.where(values.isin([0, 1]))
            lonely = raw.astype(float)
        source_series = pd.Series(source, index=frame.index, dtype="string")
        source_series = source_series.where(raw.notna())
        return raw, lonely, source_series

    empty = pd.Series(np.nan, index=frame.index, dtype=float)
    source_series = pd.Series(pd.NA, index=frame.index, dtype="string")
    return empty, empty.copy(), source_series


def requested_columns(name: str, spec: dict, build: BuildSpec) -> list[str]:
    education_variable = "raeduc" if name == "HRS" else "raeducl"
    columns = [
        spec["id"],
        spec["sex"],
        education_variable,
        spec["weight"],
    ]
    if name == "SHARE":
        columns.append("country")
    baseline = spec["baseline"]
    columns.extend(
        [
            f"r{baseline}mstat",
            f"r{baseline}{build.depression}",
            f"r{baseline}diabe",
            f"r{baseline}stroke",
        ]
    )
    for wave in spec["waves"]:
        columns.extend(
            [
                spec["immediate"].format(wave=wave),
                spec["delayed"].format(wave=wave),
                f"r{wave}iwstat",
                spec["loneliness"].format(wave=wave),
            ]
        )
        if name == "SHARE":
            columns.append(f"r{wave}lnlys3")
        if build.date_kind == "year_month":
            columns.extend([f"r{wave}iwy", f"r{wave}iwm"])
        elif build.date_kind == "year":
            columns.append(f"r{wave}iwindy")
        else:
            columns.append(f"r{wave}iwmid")
        if build.has_proxy:
            columns.append(f"r{wave}proxy")
    columns.extend(
        [
            spec["loneliness"].format(wave=baseline),
            spec["age"].format(wave=baseline),
        ]
    )
    available = available_columns(spec["file"])
    return list(dict.fromkeys(column for column in columns if column in available))


def depression_excluding_loneliness(
    frame: pd.DataFrame,
    name: str,
    baseline: int,
    build: BuildSpec,
    loneliness_raw: pd.Series,
) -> pd.Series:
    score = pd.to_numeric(
        frame[f"r{baseline}{build.depression}"], errors="coerce"
    )
    if name == "CHARLS":
        loneliness_component = loneliness_raw - 1
        result = score - loneliness_component
        return result.where(result.between(0, 27))
    result = score - loneliness_raw
    max_score = 8 if name == "MHAS" else 7
    return result.where(result.between(0, max_score))


def build_cohort(
    name: str, spec: dict
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[dict], dict]:
    build = BUILD_SPECS[name]
    baseline = spec["baseline"]
    columns = requested_columns(name, spec, build)
    frame = pd.read_stata(
        spec["file"], columns=columns, convert_categoricals=False
    )
    if frame[spec["id"]].duplicated().any():
        raise ValueError(f"{name}: source identifier is not unique")

    age = valid_numeric(frame[spec["age"].format(wave=baseline)], 50, 110)
    sex_raw = pd.to_numeric(frame[spec["sex"]], errors="coerce")
    female = sex_raw.map({1: 0, 2: 1})
    education_variable = "raeduc" if name == "HRS" else "raeducl"
    education = education_three_level(frame[education_variable], build.education_kind)
    country_code = (
        pd.to_numeric(frame["country"], errors="coerce")
        if name == "SHARE"
        else pd.Series(1.0, index=frame.index)
    )
    marital = pd.to_numeric(frame[f"r{baseline}mstat"], errors="coerce")
    partnered = marital.isin([1, 2, 3]).astype(float).where(marital.notna())
    weight = pd.to_numeric(frame[spec["weight"]], errors="coerce").where(lambda x: x > 0)

    loneliness_raw, lonely, baseline_loneliness_source = loneliness_wave_values(
        frame, name, spec, baseline
    )

    depression = depression_excluding_loneliness(
        frame, name, baseline, build, loneliness_raw
    )
    diabetes = pd.to_numeric(frame[f"r{baseline}diabe"], errors="coerce").where(
        lambda x: x.isin([0, 1])
    )
    stroke = pd.to_numeric(frame[f"r{baseline}stroke"], errors="coerce").where(
        lambda x: x.isin([0, 1])
    )

    baseline_immediate = valid_numeric(
        frame[spec["immediate"].format(wave=baseline)], 0, build.recall_max
    )
    baseline_delayed = valid_numeric(
        frame[spec["delayed"].format(wave=baseline)], 0, build.recall_max
    )
    baseline_status = pd.to_numeric(
        frame[f"r{baseline}iwstat"], errors="coerce"
    ).eq(1)
    if build.has_proxy:
        baseline_direct = pd.to_numeric(
            frame[f"r{baseline}proxy"], errors="coerce"
        ).eq(0)
    else:
        baseline_direct = pd.Series(True, index=frame.index)

    core = (
        age.notna()
        & female.notna()
        & education.notna()
        & country_code.notna()
        & partnered.notna()
        & lonely.notna()
        & baseline_immediate.notna()
        & baseline_delayed.notna()
        & baseline_status
        & baseline_direct
    )

    long_parts = []
    for wave in spec["waves"]:
        wave_loneliness_raw, wave_lonely, wave_loneliness_source = (
            loneliness_wave_values(frame, name, spec, wave)
        )
        immediate = valid_numeric(
            frame[spec["immediate"].format(wave=wave)], 0, build.recall_max
        )
        delayed = valid_numeric(
            frame[spec["delayed"].format(wave=wave)], 0, build.recall_max
        )
        status = pd.to_numeric(frame[f"r{wave}iwstat"], errors="coerce").eq(1)
        if build.has_proxy:
            direct = pd.to_numeric(frame[f"r{wave}proxy"], errors="coerce").eq(0)
        else:
            direct = pd.Series(True, index=frame.index)
        date = interview_decimal_year(frame, wave, build.date_kind)
        observed = immediate.notna() & delayed.notna() & status & direct & date.notna()
        long_parts.append(
            pd.DataFrame(
                {
                    "source_row": frame.index,
                    "country_code": country_code,
                    "wave": wave,
                    "interview_year": date,
                    "immediate": immediate,
                    "delayed": delayed,
                    "observed": observed,
                    "loneliness_raw_wave": wave_loneliness_raw,
                    "lonely_wave": wave_lonely,
                    "loneliness_observed": wave_lonely.notna(),
                    "loneliness_instrument": wave_loneliness_source,
                }
            )
        )

    long_all = pd.concat(long_parts, ignore_index=True)
    if name == "SHARE":
        scheduled_date = long_all.groupby(
            ["wave", "country_code"], observed=True
        )["interview_year"].transform("median")
    else:
        scheduled_date = long_all.groupby("wave")["interview_year"].transform("median")
    long_all["scheduled_year"] = long_all["interview_year"].fillna(scheduled_date)
    baseline_date = (
        long_all.loc[long_all["wave"].eq(baseline), ["source_row", "interview_year"]]
        .set_index("source_row")["interview_year"]
    )
    long_all["time"] = long_all["scheduled_year"] - long_all["source_row"].map(baseline_date)
    long_all["core"] = long_all["source_row"].map(core)
    attrition_panel = long_all.loc[
        long_all["core"] & long_all["time"].between(0, 30)
    ].copy()
    long_all = long_all.loc[long_all["core"] & long_all["observed"]].copy()
    long_all = long_all.loc[long_all["time"].between(0, 30)].copy()

    followup_counts = (
        long_all.loc[long_all["time"].gt(0)].groupby("source_row").size()
    )
    longitudinal_rows = followup_counts.index
    long_all = long_all.loc[long_all["source_row"].isin(longitudinal_rows)].copy()

    baseline_rows = long_all["wave"].eq(baseline)
    if not baseline_rows.any():
        raise ValueError(f"{name}: no eligible baseline observations")
    im_mean = long_all.loc[baseline_rows, "immediate"].mean()
    im_sd = long_all.loc[baseline_rows, "immediate"].std(ddof=1)
    dl_mean = long_all.loc[baseline_rows, "delayed"].mean()
    dl_sd = long_all.loc[baseline_rows, "delayed"].std(ddof=1)
    long_all["memory_component"] = 0.5 * (
        (long_all["immediate"] - im_mean) / im_sd
        + (long_all["delayed"] - dl_mean) / dl_sd
    )
    component_mean = long_all.loc[baseline_rows, "memory_component"].mean()
    component_sd = long_all.loc[baseline_rows, "memory_component"].std(ddof=1)
    long_all["memory_z"] = (long_all["memory_component"] - component_mean) / component_sd

    retained_rows = pd.Index(long_all["source_row"].unique())
    pid_map = pd.Series(np.arange(1, len(retained_rows) + 1), index=retained_rows)
    long_all["pid"] = long_all["source_row"].map(pid_map).astype(int)
    baseline_covariates = pd.DataFrame(
        {
            "source_row": frame.index,
            "lonely": lonely,
            "loneliness_raw": loneliness_raw,
            "age": age,
            "female": female,
            "education": education,
            "partnered": partnered,
            "depression_excl_lonely": depression,
            "diabetes": diabetes,
            "stroke": stroke,
            "baseline_weight": weight,
        }
    ).set_index("source_row")
    long_all = long_all.join(baseline_covariates, on="source_row")
    long_all.insert(0, "cohort", name)

    all_baseline_rows = pd.Index(attrition_panel["source_row"].unique())
    attrition_pid_map = pd.Series(
        np.arange(1, len(all_baseline_rows) + 1), index=all_baseline_rows
    )
    attrition_panel["pid"] = attrition_panel["source_row"].map(attrition_pid_map).astype(int)
    attrition_panel["memory_component"] = 0.5 * (
        (attrition_panel["immediate"] - im_mean) / im_sd
        + (attrition_panel["delayed"] - dl_mean) / dl_sd
    )
    attrition_panel["memory_z"] = (
        attrition_panel["memory_component"] - component_mean
    ) / component_sd
    attrition_panel = attrition_panel.join(baseline_covariates, on="source_row")
    baseline_memory_map = (
        attrition_panel.loc[attrition_panel["wave"].eq(baseline)]
        .set_index("source_row")["memory_z"]
    )
    attrition_panel["baseline_memory"] = attrition_panel["source_row"].map(
        baseline_memory_map
    )
    attrition_panel.insert(0, "cohort", name)
    attrition_panel = attrition_panel[
        [
            "cohort",
            "pid",
            "wave",
            "time",
            "observed",
            "memory_z",
            "baseline_memory",
            "lonely",
            "age",
            "female",
            "education",
            "partnered",
            "depression_excl_lonely",
            "diabetes",
            "stroke",
            "baseline_weight",
            "country_code",
            "loneliness_raw_wave",
            "lonely_wave",
            "loneliness_observed",
            "loneliness_instrument",
        ]
    ].sort_values(["pid", "time", "wave"])

    long_all = long_all[
        [
            "cohort",
            "pid",
            "wave",
            "time",
            "interview_year",
            "memory_z",
            "immediate",
            "delayed",
            "lonely",
            "loneliness_raw",
            "loneliness_raw_wave",
            "lonely_wave",
            "loneliness_observed",
            "loneliness_instrument",
            "age",
            "female",
            "education",
            "partnered",
            "depression_excl_lonely",
            "diabetes",
            "stroke",
            "baseline_weight",
            "country_code",
        ]
    ].sort_values(["pid", "time", "wave"])

    flow = [
        {"cohort": name, "stage": "source rows", "n": len(frame)},
        {"cohort": name, "stage": "age 50+", "n": int(age.notna().sum())},
        {"cohort": name, "stage": "baseline core complete/direct", "n": int(core.sum())},
        {
            "cohort": name,
            "stage": "at least one later direct memory assessment",
            "n": int(long_all["pid"].nunique()),
        },
    ]
    wave_qa = []
    for wave, group in long_all.groupby("wave"):
        wave_qa.append(
            {
                "cohort": name,
                "wave": int(wave),
                "n_observations": len(group),
                "n_participants": group["pid"].nunique(),
                "time_min": group["time"].min(),
                "time_median": group["time"].median(),
                "time_max": group["time"].max(),
                "memory_mean": group["memory_z"].mean(),
                "memory_sd": group["memory_z"].std(ddof=1),
                "memory_min": group["memory_z"].min(),
                "memory_max": group["memory_z"].max(),
            }
        )
    standardization = {
        "cohort": name,
        "n_baseline": int(baseline_rows.sum()),
        "immediate_mean": im_mean,
        "immediate_sd": im_sd,
        "delayed_mean": dl_mean,
        "delayed_sd": dl_sd,
        "component_mean": component_mean,
        "component_sd": component_sd,
    }
    return long_all, attrition_panel, flow, wave_qa, standardization


def validate_long(name: str, frame: pd.DataFrame) -> None:
    if frame.duplicated(["pid", "wave"]).any():
        raise ValueError(f"{name}: duplicate participant-wave rows")
    if frame[["pid", "time", "memory_z"]].isna().any().any():
        raise ValueError(f"{name}: missing core long-file fields")
    if (frame.groupby("pid")["time"].diff().dropna() <= 0).any():
        raise ValueError(f"{name}: non-increasing follow-up time")
    waves_per_person = frame.groupby("pid").size()
    if waves_per_person.min() < 2:
        raise ValueError(f"{name}: participant with fewer than two observations")


def main() -> None:
    require_input_files(tuple(COHORTS))
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    all_flow: list[dict] = []
    all_wave_qa: list[dict] = []
    all_standardization: list[dict] = []
    manifest: list[dict] = []

    for name, spec in COHORTS.items():
        long_frame, attrition_panel, flow, wave_qa, standardization = build_cohort(name, spec)
        validate_long(name, long_frame)
        output = DERIVED_DIR / f"{name.lower()}_long.csv.gz"
        attrition_output = DERIVED_DIR / f"{name.lower()}_attrition_panel.csv.gz"
        long_frame.to_csv(output, index=False, compression="gzip")
        attrition_panel.to_csv(attrition_output, index=False, compression="gzip")
        all_flow.extend(flow)
        all_wave_qa.extend(wave_qa)
        all_standardization.append(standardization)
        manifest.append(
            {
                "cohort": name,
                "file": str(output),
                "attrition_panel": str(attrition_output),
                "participants": long_frame["pid"].nunique(),
                "baseline_eligible_participants": attrition_panel["pid"].nunique(),
                "observations": len(long_frame),
                "minimum_waves": long_frame.groupby("pid").size().min(),
                "maximum_waves": long_frame.groupby("pid").size().max(),
                "direct_identifiers_retained": False,
            }
        )
        print(
            f"{name}: {long_frame['pid'].nunique():,} participants, "
            f"{len(long_frame):,} observations"
        )

    pd.DataFrame(all_flow).to_csv(OUT_DIR / "cohort_flow.csv", index=False)
    pd.DataFrame(all_wave_qa).to_csv(OUT_DIR / "wave_qa.csv", index=False)
    pd.DataFrame(all_standardization).to_csv(
        OUT_DIR / "baseline_standardization.csv", index=False
    )
    pd.DataFrame(manifest).to_csv(OUT_DIR / "derived_manifest.csv", index=False)
    print(f"\nWrote pseudonymized long files and QA tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
