# Loneliness and episodic-memory change across five ageing cohorts

This repository contains the analysis code for a coordinated longitudinal study of baseline loneliness and episodic-memory change in CHARLS, ELSA, HRS, MHAS and SHARE.

The primary estimand is the difference in 10-year change in cohort-standardized episodic memory between participants classified as lonely and not lonely at baseline. The workflow also contains prespecified sensitivity analyses, attrition-weighted models, exploratory repeated-exposure analyses, five main figures, five main tables, supplementary tables and independent R checks.

## Data are not included

The repository does not redistribute raw cohort data. The five parent studies have their own access conditions and data-use terms. Obtain the permitted harmonized files from their official sources before running the analysis.

The following files are expected by default in data/raw/:

| Cohort | Expected filename |
| --- | --- |
| CHARLS | H_CHARLS_D_Data.dta |
| ELSA | h_elsa_g3.dta |
| HRS | randhrs1992_2020v2.dta |
| MHAS | H_MHAS_c2.dta |
| SHARE | H_SHARE_f2.dta |

For files stored elsewhere, set GLOBAL_AGEING_DATA_ROOT to a directory containing these five files. Individual files can instead be configured with CHARLS_DATA_FILE, ELSA_DATA_FILE, HRS_DATA_FILE, MHAS_DATA_FILE and SHARE_DATA_FILE. The configuration is implemented in src/project_config.py.

Example in PowerShell:

~~~powershell
$env:GLOBAL_AGEING_DATA_ROOT = "D:\path\to\permitted\harmonized\files"
~~~

Do not put raw files, participant-level derived files or credentials in this repository. Generated files under outputs/ are ignored by Git for the same reason.

## Environment

Use Python 3.10 or newer and install the Python dependencies:

~~~powershell
python -m pip install -r requirements.txt
~~~

The R validation scripts require R with the lme4 and geepack packages. R validation is optional and should be run only after the Python outputs have been generated.

## Reproduce the analysis

Run the commands from the repository root in the following order:

~~~powershell
python src/audit_longitudinal_feasibility.py
python src/audit_selected_codings.py
python src/inspect_harmonized_metadata.py
python src/inspect_selected_value_ranges.py

python src/build_longitudinal_datasets.py
python src/run_longitudinal_models.py
python src/run_sensitivity_models.py
python src/run_attrition_weighted_models.py
python src/run_submission_enhancements.py
python src/run_exposure_trajectory_models.py

python src/make_manuscript_figures.py
python src/make_manuscript_tables.py
python src/build_supplementary_materials.py
~~~

The independent R checks can then be run from the repository root:

~~~powershell
Rscript src/validate_models_in_r.R
Rscript src/validate_weighted_gee_in_r.R
~~~

The first four scripts are read-only audits of the source files. Dataset construction writes cohort-local sequential participant identifiers to the derived files; these identifiers are pseudonyms, not a guarantee of anonymity. Treat all generated participant-level files as restricted research data.

## Repository layout

~~~text
src/       analysis and validation scripts
data/      data-use instructions; raw files are ignored
outputs/   generated tables, figures and derived data; contents are ignored
~~~

The primary analysis is the baseline-loneliness mixed model. Repeated-exposure, lagged-transition, nonlinear-time, bounded-recall and weighting analyses are secondary or sensitivity analyses and should be interpreted according to the methods recorded in the scripts and generated supplementary materials. The manuscript figure script writes five main figures with editable SVG/PDF outputs, 600 dpi TIFF files, 300 dpi PNG previews and aggregate figure Source Data. The manuscript table script writes five main tables as CSV and Markdown files.

## Version control and release

This repository is intended to hold code and reproducibility instructions. Before making it public, inspect the Git-tracked file list and confirm that no raw data, individual-level output, private author information or credential has been added. A code license and a frozen release should be selected by the authors before publication.
