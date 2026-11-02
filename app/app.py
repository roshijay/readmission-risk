import streamlit as st
import pandas as pd

st.set_page_config(page_title="Post-Discharge Priority Board", layout="wide")
st.title("Post-Discharge Priority Board")
st.write("30-day all-cause readmission risk triage — in plain words: ranking patients by how likely they are to come back to hospital for any reason within 30 days of discharge, so follow-up teams contact the highest-risk people first.")

with st.expander("About this board + Glossary — for clinicians and board members"):
    st.write("Basis: US hospital data for diabetes patients, 1999–2008, 99,343 stays after excluding discharge disposition = expired / hospice (patients who died or went to hospice cannot be readmitted). Risk is estimated from prior inpatient encounters, length of stay (LOS), age group, and primary diagnosis, among other factors. Tested on patients the model had never seen, grouped by patient so the same person is never in both training and testing.")
    st.write("Glossary (EMR term — plain meaning):")
    st.write("- 30-day all-cause readmission — readmitted for any reason within 30 days of discharge, not just the original problem")
    st.write("- Care transition — the move from hospital to home / primary care, where follow-up often breaks down")
    st.write("- Discharge disposition — where the patient went after discharge (home, skilled nursing facility, hospice, etc.)")
    st.write("- LOS (Length of Stay) — days in hospital for that stay")
    st.write("- Medication reconciliation (med rec) — pharmacist check that discharge medicines match home medicines, no duplicates or dangerous interactions")
    st.write("- PCP (Primary Care Provider) — the patient's regular doctor")
    st.write("- ICD-9 — the diagnosis coding system used in this 1999–2008 data (ICD-10 is used today, chapters are similar)")
    st.write("- Encounter — one hospital stay / visit, not one patient. One patient can have multiple encounters.")
    st.write("- MRN (Medical Record Number) — the real hospital ID in an EMR. The Patient ID here is de-identified, NOT a real MRN.")
    st.write("Safety: this is a ranking tool for scheduled follow-up, not emergency triage. If a patient has chest pain, severe breathlessness, or other emergency symptoms now, seek emergency care immediately — do not wait for a follow-up call.")

pred = pd.read_csv("data/processed/test_predictions.csv")
riyadh = pd.read_csv("data/raw/riyadh.csv", parse_dates=["admission_date"])

diag_labels = {
    "Circulatory": "Circulatory (ICD-9 390-459) — Heart / vascular",
    "Respiratory": "Respiratory (ICD-9 460-519) — Lungs / breathing",
    "Digestive": "Digestive (ICD-9 520-579) — Stomach / gut",
    "Diabetes": "Diabetes (ICD-9 250) — Diabetes as primary diagnosis",
    "Musculoskeletal": "Musculoskeletal (ICD-9 710-739) — Bones / joints / muscles",
    "Genitourinary": "Genitourinary (ICD-9 580-629) — Kidneys / urinary",
    "Neoplasms": "Neoplasms (ICD-9 140-239) — Tumours",
    "Injury": "Injury (ICD-9 800-999) — Injury / complications of care",
    "Infectious": "Infectious (ICD-9 001-139) — Infection",
    "V/E codes": "V/E codes — External causes / factors influencing health",
    "Other": "Other — Symptoms, skin, endocrine, etc.",
}
pred["Main diagnosis"] = pred["diag_1_chapter"].map(diag_labels).fillna(pred["diag_1_chapter"])

def priority_for(risk, diag):
    urgent_diag = diag in ["Circulatory", "Respiratory"]
    if risk >= 0.50 or (risk >= 0.30 and urgent_diag):
        return "🔴 High priority — Urgent follow-up"
    if risk >= 0.30 or urgent_diag:
        return "🟡 Medium priority — Follow up soon"
    return "🟢 Low priority — Routine follow-up"

def recommendation_for(risk, diag):
    urgent_diag = diag in ["Circulatory", "Respiratory"]
    if risk >= 0.50 and urgent_diag:
        return "Contact within 24h: post-discharge phone follow-up, medication reconciliation (med rec), PCP follow-up in 7 days — plain: call in 1 day, check medicines, book regular doctor in a week"
    if risk >= 0.50:
        return "Contact within 48h: post-discharge phone follow-up, medication reconciliation (med rec), PCP follow-up in 7 days — plain: call in 2 days, check medicines, book regular doctor in a week"
    if risk >= 0.30:
        return "Contact within 1 week: care transition check, PCP follow-up in 14 days — plain: check in within a week, book regular doctor within 2 weeks"
    return "Routine discharge follow-up within 2 weeks: standard discharge instructions, PCP as needed — plain: normal follow-up, no urgent call needed"

pred["Priority"] = [priority_for(r, d) for r, d in zip(pred["risk"], pred["diag_1_chapter"])]
pred["Recommendation"] = [recommendation_for(r, d) for r, d in zip(pred["risk"], pred["diag_1_chapter"])]
pred["Risk %"] = (pred["risk"] * 100).round(0).astype(int)

tab1, tab2, tab3 = st.tabs(["Priority list", "Hospital trends", "How accurate is this?"])

with tab1:
    st.header("Priority list — highest priority at the top")
    st.write("Priority = risk score + main diagnosis. Cardiac (Circulatory) and breathing (Respiratory) cases move up a level, because those conditions deteriorate faster if follow-up is missed.")
    capacity = st.slider("How many patients can the team follow up? (Follow-up capacity)", 20, 200, 100, step=10)
    diag_options = ["All diagnoses"] + sorted(pred["Main diagnosis"].unique().tolist())
    selected_diag = st.selectbox("Filter by main diagnosis (ICD-9 chapter) — click and type to search:", diag_options, index=0)
    queue = pred.copy()
    if selected_diag != "All diagnoses":
        queue = queue[queue["Main diagnosis"] == selected_diag]
    top = queue.head(capacity)

    col1, col2, col3 = st.columns(3)
    col1.metric("Patients shown", len(top))
    col2.metric("High priority", int(top["Priority"].str.startswith("🔴").sum()))
    col3.metric("Actually readmitted in this group", f"{top['readmitted_30d'].mean():.0%}")

    display = top.rename(columns={
        "patient_nbr": "Patient ID (de-identified, not MRN)",
        "age": "Age group",
        "time_in_hospital": "LOS — Days in hospital",
        "number_inpatient": "Prior inpatient encounters",
    })
    st.dataframe(display[["Priority", "Patient ID (de-identified, not MRN)", "Age group", "Main diagnosis", "LOS — Days in hospital", "Prior inpatient encounters", "Risk %", "Recommendation"]],
                 hide_index=True, use_container_width=True)

    with st.expander("Show more details (for audit, not for the call list)"):
        st.write("Gender, race, emergency visits, and true outcome (readmitted_30d = 1 if readmitted within 30 days, else 0) are here for checking the model.")
        st.dataframe(top[["patient_nbr", "gender", "race", "number_emergency", "readmitted_30d"]],
                     hide_index=True, use_container_width=True)

with tab2:
    st.header("Hospital trends — Riyadh, Jan 2022 to Sep 2024")
    st.info("Different place, different patients, different data shape (hourly aggregates, not patient-level) than the US diabetes model above. Use for pattern context only — do not compare its rates directly to the US model.")
    st.write("So what: admission volume (how many admissions) rises and falls by month and by condition. This helps plan staffing, not predict an individual patient.")
    riyadh["month"] = riyadh["admission_date"].dt.to_period("M").dt.to_timestamp()
    monthly = riyadh.groupby("month")[["admission_count", "readmission_count"]].sum()
    st.subheader("Admissions per month (admission_count)")
    st.line_chart(monthly["admission_count"])
    by_condition = riyadh.groupby("condition_type")[["admission_count", "readmission_count"]].sum().sort_values("admission_count", ascending=False)
    st.subheader("Admissions by condition (condition_type)")
    st.bar_chart(by_condition["admission_count"])
    with st.expander("Show the numbers behind the charts"):
        st.dataframe(by_condition, use_container_width=True)

with tab3:
    st.header("How accurate is this?")
    st.write("Plain version: out of the top 100 patients it flags as highest risk, 59 actually come back within 30 days. If you picked 100 patients at random, only about 11 would come back.")
    st.write("Technical version: HistGradientBoosting PR-AUC 0.2030 (random = 0.1119), ROC-AUC 0.6360, Brier 0.0960. PR-AUC = area under precision-recall curve, best for low-prevalence outcomes. Brier = calibration error, lower is better.")
    st.warning("Limitations: trained on US hospital data 1999–2008, diabetes cohort only, ICD-9 era. Do not use for real clinical decisions without re-checking on recent local data and running in silent mode first (score, but don't act, and compare to what actually happens).")
    with st.expander("Show all model comparisons + metric definitions"):
        st.write("HistGradientBoosting (chosen): PR-AUC 0.2030, ROC-AUC 0.6360, Brier 0.0960, Precision@100 0.59, Precision@200 0.475")
        st.write("Logistic Regression (baseline, interpretable scorecard): PR-AUC 0.1941, ROC-AUC 0.6316, Brier 0.0967, Precision@100 0.51")
        st.write("Random Forest (comparison, 100 trees voting): PR-AUC 0.1705, ROC-AUC 0.6007, Brier 0.0995, Precision@100 0.38")
        st.write("Precision@100 = out of the top 100 flagged, how many actually readmitted. Precision@200 = same for top 200.")

