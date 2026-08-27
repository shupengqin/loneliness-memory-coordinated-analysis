"""Read-only feasibility audit for the five-cohort loneliness-memory study.

The script reads only selected columns from the local Stata files and writes a
small CSV summary in the workspace. It never modifies the source datasets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from project_config import DATA_FILES, OUTPUT_DIR, require_input_files


OUT = OUTPUT_DIR / "longitudinal_feasibility_audit.csv"


COHORTS = {
    "CHARLS": {
        "file": DATA_FILES["CHARLS"],
        "id": "ID",
        "baseline": 1,
        "waves": [1, 2, 3, 4],
        "loneliness": "r{wave}flonel",
        "immediate": "r{wave}imrc",
        "delayed": "r{wave}dlrc",
        "age": "r{wave}agey",
        "sex": "ragender",
        "education": "raeduc_c",
        "weight": "r1wtrespb",
    },
    "ELSA": {
        "file": DATA_FILES["ELSA"],
        "id": "idauniq",
        "baseline": 1,
        "waves": list(range(1, 10)),
        "loneliness": "r{wave}flone",
        "immediate": "r{wave}imrc",
        "delayed": "r{wave}dlrc",
        "age": "r{wave}agey",
        "sex": "ragender",
        "education": "raeduc_e",
        "weight": "r1cwtresp",
    },
    "HRS": {
        "file": DATA_FILES["HRS"],
        "id": "hhidpn",
        "baseline": 9,
        "waves": list(range(9, 14)),
        "loneliness": "r{wave}flone",
        "immediate": "r{wave}imrc",
        "delayed": "r{wave}dlrc",
        "age": "r{wave}agey_m",
        "sex": "ragender",
        "education": "raeduc",
        "weight": "r9wtresp",
    },
    "MHAS": {
        "file": DATA_FILES["MHAS"],
        "id": "rahhidnp",
        "baseline": 1,
        "waves": list(range(1, 6)),
        "loneliness": "r{wave}flone",
        "immediate": "r{wave}imrc8",
        "delayed": "r{wave}dlrc8",
        "age": "r{wave}agey",
        "sex": "ragender",
        "education": "raeducl",
        "weight": "r1wtresp",
    },
    "SHARE": {
        "file": DATA_FILES["SHARE"],
        "id": "mergeid",
        "baseline": 2,
        "waves": [2, 4, 5, 6, 7, 8],
        "loneliness": "r{wave}flone",
        "immediate": "r{wave}imrc",
        "delayed": "r{wave}dlrc",
        "age": "r{wave}agey",
        "sex": "ragender",
        "education": "raeducl",
        "weight": "r2wtresp",
    },
}


def existing_columns(path: Path, requested: list[str]) -> list[str]:
    sample = pd.read_stata(path, iterator=True).read(1)
    available = set(sample.columns)
    return [column for column in requested if column in available]


def complete_pair(frame: pd.DataFrame, immediate: str, delayed: str) -> pd.Series:
    return frame[immediate].notna() & frame[delayed].notna()


def audit_cohort(name: str, spec: dict) -> list[dict]:
    requested = [spec["id"], spec["sex"], spec["education"]]
    for wave in spec["waves"]:
        requested.extend(
            template.format(wave=wave)
            for template in (spec["loneliness"], spec["immediate"], spec["delayed"], spec["age"])
        )
    if isinstance(spec["weight"], str) and "{wave}" in spec["weight"]:
        requested.extend(spec["weight"].format(wave=wave) for wave in spec["waves"])
    else:
        requested.append(spec["weight"])

    columns = existing_columns(spec["file"], list(dict.fromkeys(requested)))
    frame = pd.read_stata(spec["file"], columns=columns)
    baseline = spec["baseline"]

    baseline_loneliness = spec["loneliness"].format(wave=baseline)
    baseline_pair = [
        spec["immediate"].format(wave=baseline),
        spec["delayed"].format(wave=baseline),
    ]
    baseline_core = [baseline_loneliness, *baseline_pair, spec["age"].format(wave=baseline), spec["sex"], spec["education"]]
    baseline_core = [column for column in baseline_core if column in frame]

    baseline_complete = frame[baseline_core].notna().all(axis=1)
    memory_pairs = []
    memory_pair_waves = []
    for wave in spec["waves"]:
        pair = [spec["immediate"].format(wave=wave), spec["delayed"].format(wave=wave)]
        if all(column in frame for column in pair):
            memory_pairs.append(complete_pair(frame, *pair))
            memory_pair_waves.append(wave)
    pair_matrix = pd.concat(memory_pairs, axis=1) if memory_pairs else pd.DataFrame(index=frame.index)
    pair_count = pair_matrix.sum(axis=1) if not pair_matrix.empty else pd.Series(0, index=frame.index)
    subsequent_memory = pd.Series(0, index=frame.index)
    for wave, pair_status in zip(memory_pair_waves, memory_pairs):
        if wave != baseline:
            subsequent_memory = subsequent_memory + pair_status.astype(int)
    weight_column = spec["weight"].format(wave=baseline) if "{wave}" in spec["weight"] else spec["weight"]
    weight_available = frame[weight_column].notna() if weight_column in frame else pd.Series(False, index=frame.index)
    age_column = spec["age"].format(wave=baseline)
    age50 = frame[age_column].ge(50) if age_column in frame else pd.Series(False, index=frame.index)
    eligible = baseline_complete & (subsequent_memory >= 1)

    return [
        {
            "cohort": name,
            "source_file": str(spec["file"]),
            "n_rows": len(frame),
            "n_baseline_loneliness": int(frame[baseline_loneliness].notna().sum()) if baseline_loneliness in frame else 0,
            "n_baseline_memory_pair": int(complete_pair(frame, *baseline_pair).sum()) if all(column in frame for column in baseline_pair) else 0,
            "n_baseline_core_complete": int(baseline_complete.sum()),
            "n_at_least_one_subsequent_memory_pair": int((baseline_complete & (subsequent_memory >= 1)).sum()),
            "n_at_least_two_total_memory_pairs": int((baseline_complete & (pair_count >= 2)).sum()),
            "n_baseline_weight_available": int((baseline_complete & weight_available).sum()),
            "n_baseline_age_50_plus": int((baseline_complete & age50).sum()),
            "n_eligible_age_50_plus": int((eligible & age50).sum()),
            "n_memory_waves_available": int(len(memory_pairs)),
            "baseline_wave": baseline,
            "available_columns": len(columns),
        }
    ]


def main() -> None:
    require_input_files(tuple(COHORTS))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, spec in COHORTS.items():
        rows.extend(audit_cohort(name, spec))
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\\nWrote {OUT}")


if __name__ == "__main__":
    main()
