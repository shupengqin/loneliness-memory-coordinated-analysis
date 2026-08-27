"""Inspect metadata needed for the five-cohort longitudinal analysis.

This script opens the harmonized Stata files in read-only mode and exports
variable labels for a focused set of cognition, interview, proxy, depression,
demographic, and health candidates. No participant-level values are written.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from audit_longitudinal_feasibility import COHORTS
from project_config import OUTPUT_DIR, require_input_files


OUT = OUTPUT_DIR / "harmonized_metadata_candidates.csv"

PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^(ID|idauniq|hhidpn|rahhidnp|mergeid|ragender|raeduc.*)$",
        r"^r\d+(agey|agey_m|iwy|iwm|iwmid|iwendy|iwindy|iwstat|proxy|.*proxy.*)$",
        r"^r\d+(flone|flonel|imrc|dlrc|imrc8|dlrc8|lnly.*|lblonely.*)$",
        r"^r\d+(cesd.*|eurod.*|depres.*|.*depress.*|fsad|fhappy|feffort|fsleep|fgoing|fenjoy.*)$",
        r"^r\d+(mpart|mstat|diab.*|heart.*|stroke.*|wtresp|wtrespb|cwtresp)$",
    )
]


def selected(name: str) -> bool:
    return any(pattern.search(name) for pattern in PATTERNS)


def main() -> None:
    require_input_files(tuple(COHORTS))
    rows: list[dict] = []
    for cohort, spec in COHORTS.items():
        reader = pd.io.stata.StataReader(spec["file"], convert_categoricals=False)
        labels = reader.variable_labels()
        variables = list(labels)

        for variable in variables:
            if selected(variable):
                rows.append(
                    {
                        "cohort": cohort,
                        "variable": variable,
                        "label": labels.get(variable, ""),
                    }
                )

    result = pd.DataFrame(rows).sort_values(["cohort", "variable"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
