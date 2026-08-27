"""Audit coding and proxy/cognition overlap for selected analysis variables.

The exported tables contain only aggregate counts and category labels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from audit_longitudinal_feasibility import COHORTS
from project_config import OUTPUT_DIR, require_input_files


OUT_DIR = OUTPUT_DIR


def available_columns(path: Path) -> tuple[list[str], dict[str, str]]:
    reader = pd.io.stata.StataReader(path, convert_categoricals=False)
    labels = reader.variable_labels()
    return list(labels), labels


def candidate_columns(spec: dict, available: set[str]) -> list[str]:
    baseline = spec["baseline"]
    candidates = [
        spec["sex"],
        spec["education"],
        "raeducl",
        "raeduc",
        "raeduc_c",
        "raeduc_e",
        "raedyrs",
        spec["loneliness"].format(wave=baseline),
        spec["immediate"].format(wave=baseline),
        spec["delayed"].format(wave=baseline),
        f"r{baseline}iwstat",
        f"r{baseline}proxy",
        f"r{baseline}mpart",
        f"r{baseline}mstat",
        f"r{baseline}cesd",
        f"r{baseline}cesd10",
        f"r{baseline}cesd_m",
        f"r{baseline}eurod",
        f"r{baseline}depres",
        f"r{baseline}depresl",
    ]
    return list(dict.fromkeys(column for column in candidates if column in available))


def main() -> None:
    require_input_files(tuple(COHORTS))
    coding_rows: list[dict] = []
    proxy_rows: list[dict] = []

    for cohort, spec in COHORTS.items():
        columns, labels = available_columns(spec["file"])
        selected = candidate_columns(spec, set(columns))
        raw = pd.read_stata(
            spec["file"], columns=selected, convert_categoricals=False
        )
        displayed = pd.read_stata(
            spec["file"], columns=selected, convert_categoricals=True
        )

        for column in selected:
            pairs = pd.DataFrame(
                {
                    "raw": raw[column],
                    "display": displayed[column].astype("string"),
                }
            )
            counts = pairs.value_counts(dropna=False).reset_index(name="n")
            for row in counts.itertuples(index=False):
                coding_rows.append(
                    {
                        "cohort": cohort,
                        "variable": column,
                        "variable_label": labels.get(column, ""),
                        "raw_value": row.raw,
                        "display_value": row.display,
                        "n": int(row.n),
                    }
                )

        for wave in spec["waves"]:
            immediate = spec["immediate"].format(wave=wave)
            delayed = spec["delayed"].format(wave=wave)
            proxy = f"r{wave}proxy"
            required = [column for column in [immediate, delayed, proxy] if column in columns]
            wave_frame = pd.read_stata(
                spec["file"], columns=required, convert_categoricals=False
            )
            memory_observed = wave_frame[immediate].notna() & wave_frame[delayed].notna()
            proxy_yes = (
                wave_frame[proxy].eq(1)
                if proxy in wave_frame
                else pd.Series(False, index=wave_frame.index)
            )
            proxy_rows.append(
                {
                    "cohort": cohort,
                    "wave": wave,
                    "n_rows": len(wave_frame),
                    "n_memory_pair": int(memory_observed.sum()),
                    "proxy_variable_available": proxy in wave_frame,
                    "n_proxy_yes": int(proxy_yes.sum()),
                    "n_memory_pair_proxy_yes": int((memory_observed & proxy_yes).sum()),
                }
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coding_rows).to_csv(
        OUT_DIR / "selected_coding_labels.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(proxy_rows).to_csv(
        OUT_DIR / "proxy_memory_audit.csv", index=False, encoding="utf-8-sig"
    )
    print(pd.DataFrame(proxy_rows).to_string(index=False))
    print(f"\nWrote aggregate audits to {OUT_DIR}")


if __name__ == "__main__":
    main()
