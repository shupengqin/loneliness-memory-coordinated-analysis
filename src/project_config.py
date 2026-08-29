"""Shared paths for the five-cohort loneliness-memory analysis.

Raw cohort files are intentionally kept outside version control. Set
``GLOBAL_AGEING_DATA_ROOT`` to a directory containing the five expected files,
or set an individual ``<COHORT>_DATA_FILE`` environment variable when a file
is stored elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_configured_output_root = os.environ.get("GLOBAL_AGEING_OUTPUT_ROOT")
OUTPUT_DIR = (
    Path(_configured_output_root).expanduser()
    if _configured_output_root
    else PROJECT_ROOT / "outputs"
)
DERIVED_DIR = OUTPUT_DIR / "derived"

_configured_root = os.environ.get("GLOBAL_AGEING_DATA_ROOT") or os.environ.get(
    "AGEING_DATA_ROOT"
)
DATA_ROOT = (
    Path(_configured_root).expanduser()
    if _configured_root
    else PROJECT_ROOT / "data" / "raw"
)


def _data_file(cohort: str, filename: str) -> Path:
    configured = os.environ.get(f"{cohort}_DATA_FILE")
    return Path(configured).expanduser() if configured else DATA_ROOT / filename


DATA_FILES = {
    "CHARLS": _data_file("CHARLS", "H_CHARLS_D_Data.dta"),
    "ELSA": _data_file("ELSA", "h_elsa_g3.dta"),
    "HRS": _data_file("HRS", "randhrs1992_2020v2.dta"),
    "MHAS": _data_file("MHAS", "H_MHAS_c2.dta"),
    "SHARE": _data_file("SHARE", "H_SHARE_f2.dta"),
}


def require_input_files(cohorts: list[str] | tuple[str, ...]) -> None:
    """Raise a helpful error before analysis when a required file is absent."""

    missing = [
        f"{cohort}: {DATA_FILES[cohort]}"
        for cohort in cohorts
        if not DATA_FILES[cohort].is_file()
    ]
    if missing:
        details = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            "Required cohort data files were not found. Set "
            "GLOBAL_AGEING_DATA_ROOT or the cohort-specific *_DATA_FILE "
            f"variables. Missing files:\n{details}"
        )
