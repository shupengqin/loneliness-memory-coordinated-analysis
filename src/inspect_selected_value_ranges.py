"""Inspect observed ranges and missingness for selected baseline variables."""

import pandas as pd

from audit_longitudinal_feasibility import COHORTS
from project_config import OUTPUT_DIR, require_input_files

OUT = OUTPUT_DIR / "selected_value_ranges.csv"

SPECS = {
    "CHARLS": (COHORTS["CHARLS"]["file"], ["r1flonel", "r1imrc", "r1dlrc"]),
    "ELSA": (COHORTS["ELSA"]["file"], ["r1flone", "r1imrc", "r1dlrc"]),
    "HRS": (COHORTS["HRS"]["file"], ["r9flone", "r9imrc", "r9dlrc"]),
    "MHAS": (COHORTS["MHAS"]["file"], ["r1flone", "r1imrc8", "r1dlrc8"]),
    "SHARE": (COHORTS["SHARE"]["file"], ["r2flone", "r2imrc", "r2dlrc"]),
}


def main() -> None:
    require_input_files(tuple(SPECS))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cohort, (path, columns) in SPECS.items():
        frame = pd.read_stata(path, columns=columns, convert_categoricals=False)
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            counts = values.value_counts().sort_index()
            rows.append(
                {
                    "cohort": cohort,
                    "variable": column,
                    "n_rows": len(frame),
                    "n_nonmissing": int(values.size),
                    "n_missing": int(frame[column].isna().sum()),
                    "min": float(values.min()) if len(values) else None,
                    "max": float(values.max()) if len(values) else None,
                    "n_unique": int(values.nunique()),
                    "n_negative": int((values < 0).sum()),
                    "common_values": "; ".join(f"{value:g}:{count}" for value, count in counts.head(12).items()),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
