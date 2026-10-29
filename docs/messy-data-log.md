# Messy-data log — UCI Diabetes cohort

## Cohort
- Raw: 101,766 encounters, 50 columns, 71,518 unique patients
- Excluded expired / hospice: 2,423 (discharge_disposition_id 11, 13, 14, 19, 20, 21)
- Cohort: 99,343 encounters, <30 rate 0.1139 (11,314 / 99,343)
- File: data/processed/cohort.csv, built by sql/cohort_uci.sql via src/build_cohort.py

## Missingness encoding
- UCI uses "?" for missing, not blank. pandas isna() misses it unless na_values="?",
  DuckDB needs nullstr='?'. Without this, weight/payer/specialty look 0% missing.
- Verified with "?": weight 0.9686, max_glu_serum 0.9475, A1Cresult 0.8328,
  medical_specialty 0.4908, payer_code 0.3956, race 0.0223, diag_3 0.0140, diag_2 0.0035, diag_1 0.0002

## Decisions (locked for Step 3)
- Drop for v1: weight, payer_code, medical_specialty — too high to impute honestly
- Keep as "Not measured" categories: max_glu_serum, A1Cresult — "None" means test not performed, not missing to fill
- Race 2.2% missing: keep as Unknown category later, do not mode-fill
- Median, not mean, only for low-missing numeric features, fit on train only
- No interpolation for UCI — encounters are not a clean time series, no admission dates

## Other flags
- DtypeWarning for payer_code: mixed types. Fixed with low_memory=False in pandas, nullstr handling in DuckDB
- Discharge code 18 = NULL / unknown: 3,691 rows in raw. Kept in cohort, flagged, not excluded (unknown != expired)
- Diag chapter "Other" = 18,038 in cohort. Includes symptom codes 780-799, skin 680-709,
  other endocrine 240-279 except diabetes. Can split in Step 3 if needed
- Duplicate encounter_id: 0. Patients repeat: 71,518 patients < 101,766 encounters, so split by patient in Step 3, not by row
- Target: readmitted has 3 levels NO / >30 / <30. Binarized to 1 = <30 only, 0 = NO + >30.
  0 means "not readmitted within 30 days", not "never readmitted"

## EDA findings
- Age rate is non-monotonic: [20-30) highest at 14.31% (n=1,649), [50-60) lowest adult at 9.77%,
  [70-80) 12.06%, [80-90) 12.57%. [0-10) n=160 only, rate noisy. Do not treat age as simple linear risk
- Prior inpatient gradient is steep and monotonic: 0 prior = 8.59%, 1 = 13.25%, 2 = 17.93%, 3+ = 26.40%
- Primary diagnosis is usually circulatory (29,585), not diabetes (8,661). Diabetes is cohort background, not usually primary reason
