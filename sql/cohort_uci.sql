-- Cohort for 30-day readmission model
-- Source: UCI Diabetes 130-US Hospitals, data/raw/diabetic_data.csv
-- Exclusions: expired / hospice discharges cannot be readmitted
--   11 = Expired, 13/14 = Hospice, 19/20/21 = Expired hospice
--   Raw 101,766 - 2,423 excluded = 99,343 cohort
-- Dropped for missingness (Step 1 plan): weight, payer_code, medical_specialty
-- Target: readmitted_30d = 1 if readmitted = '<30', else 0

SELECT
  encounter_id,
  patient_nbr,
  race,
  gender,
  age,
  time_in_hospital,
  num_lab_procedures,
  num_procedures,
  num_medications,
  number_outpatient,
  number_emergency,
  number_inpatient,
  number_diagnoses,
  max_glu_serum,
  A1Cresult,
  insulin,
  change,
  diabetesMed,
  CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END AS readmitted_30d,
  CASE
    WHEN diag_1 LIKE '250%' THEN 'Diabetes'
    WHEN TRY_CAST(SPLIT_PART(diag_1, '.', 1) AS INTEGER) BETWEEN 390 AND 459 THEN 'Circulatory'
    WHEN TRY_CAST(SPLIT_PART(diag_1, '.', 1) AS INTEGER) BETWEEN 460 AND 519 THEN 'Respiratory'
    WHEN TRY_CAST(SPLIT_PART(diag_1, '.', 1) AS INTEGER) BETWEEN 520 AND 579 THEN 'Digestive'
    WHEN TRY_CAST(SPLIT_PART(diag_1, '.', 1) AS INTEGER) BETWEEN 710 AND 739 THEN 'Musculoskeletal'
    WHEN TRY_CAST(SPLIT_PART(diag_1, '.', 1) AS INTEGER) BETWEEN 580 AND 629 THEN 'Genitourinary'
    WHEN TRY_CAST(SPLIT_PART(diag_1, '.', 1) AS INTEGER) BETWEEN 140 AND 239 THEN 'Neoplasms'
    WHEN TRY_CAST(SPLIT_PART(diag_1, '.', 1) AS INTEGER) BETWEEN 800 AND 999 THEN 'Injury'
    WHEN TRY_CAST(SPLIT_PART(diag_1, '.', 1) AS INTEGER) BETWEEN 1 AND 139 THEN 'Infectious'
    WHEN diag_1 LIKE 'V%' OR diag_1 LIKE 'E%' THEN 'V/E codes'
    ELSE 'Other'
  END AS diag_1_chapter
FROM read_csv('data/raw/diabetic_data.csv', header = true, nullstr = '?')
WHERE discharge_disposition_id NOT IN (11, 13, 14, 19, 20, 21);
