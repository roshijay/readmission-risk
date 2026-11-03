# 30-Day Readmission Risk — Post-Discharge Priority Board

## Business question, in plain words
After a patient leaves hospital, will they come back within 30 days? A small follow-up team cannot call everyone, so this project ranks discharged patients by risk, so the team calls the highest-risk people first.

In medical terms: 30-day all-cause readmission risk triage (all-cause = readmitted for any reason, not just the original problem).

## What I did
1. **Cohort (SQL / DuckDB):** UCI Diabetes 130-US Hospitals, 1999–2008. Raw 101,766 to encounters 99,343 after removing patients who died or went to hospice (they cannot be readmitted). Primary diagnosis grouped from ICD-9 codes to chapters (Circulatory, Respiratory, etc.).
2. **Model:** Patient-grouped split (same patient never in both train and test, overlap 0). Features: prior hospital stays, length of stay, age, main diagnosis, and others 51 columns after encoding. Compared 3 models on held-out test.
3. **Dashboard (Streamlit):** Post-Discharge Priority Board with a capacity slider, diagnosis filter, colour-coded priority (High / Medium / Low), and plain + EMR language so clinicians and board members can both read it. Riyadh hospital trends are shown separately, for context only.
4. **Discharge plan (RAG-lite):** For a high-risk patient, retrieve guideline chunks (AHRQ, ADA, etc.) matching their risk drivers, then build a templated checklist with a citation for every bullet. No free-text medical generation, so no invented medicines or doses. Groundedness eval: pass.

## Did it work? Held-out test results
- Random guessing: PR-AUC 0.1119
- Logistic Regression (baseline): PR-AUC 0.1941, Precision@100 0.51
- **HistGradientBoosting (chosen): PR-AUC 0.2030, Precision@100 0.59, Precision@200 0.475**
- Random Forest (comparison): PR-AUC 0.1705, Precision@100 0.38

In plain words: out of the top 100 patients flagged as highest risk, 59 actually come back within 30 days, vs. about 11 if you picked 100 at random. It helps prioritize, it does not perfectly predict.

## How to run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Download UCI diabetic_data.csv to data/raw/ and Riyadh 2022 file to data/raw/riyadh.csv (Kaggle login required for Riyadh)
python src/build_cohort.py
python src/train_models.py
python src/predict_test.py
streamlit run app/app.py
```

## Data sources

- UCI Diabetes 130-US Hospitals 1999–2008, http://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008 , de-identified, public, raw data is gitignored and not redistributed.
- Riyadh Hospital Admissions, Kaggle, Jan 2022–Sep 2024, 24,000 hourly rows (2022 file, not the 2020 file), hourly aggregates, cardiorespiratory only, for trends only, not for patient-level modeling.

## Honest limitations: read this before using it

- UCI is 1999–2008, ICD-9 era, diabetes cohort only. Practice, medicines, and coding have changed. Do not use for real clinical decisions without re-checking on recent local data.
- Model ranks risk, it does not prove that calling patients reduces readmissions. No causal claim is made. Next step in real life would be silent-mode pilot: score, but don't act, compare to what actually happens.
- RAG plan is decision support only, not medical advice. Clinician to review before use. Templated generation prevents invented doses in this version, but retrieval relevance must still be checked.
- Riyadh rates must not be compared directly to UCI rates: they come from a different country, population, and grain (hourly aggregates vs. patient encounters).

