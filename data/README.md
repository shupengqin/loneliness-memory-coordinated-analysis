# Data placement

Raw cohort files are not included in this repository and must be obtained under the access terms of CHARLS, ELSA, HRS, MHAS and SHARE.

By default, place the five permitted harmonized Stata files directly in data/raw/ using the names below:

~~~text
data/raw/H_CHARLS_D_Data.dta
data/raw/h_elsa_g3.dta
data/raw/randhrs1992_2020v2.dta
data/raw/H_MHAS_c2.dta
data/raw/H_SHARE_f2.dta
~~~

Alternatively, set GLOBAL_AGEING_DATA_ROOT or the cohort-specific *_DATA_FILE environment variables documented in the repository README. Never commit raw files or participant-level extracts.
