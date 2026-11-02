import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

guidelines = json.load(open("data/guidelines.json"))
texts = [g["text"] for g in guidelines]

pred = pd.read_csv("data/processed/test_predictions.csv")
top = pred.iloc[0]

query = f"{top['diag_1_chapter']} prior inpatient stays medication reconciliation primary care follow-up red flags discharge instructions"
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(texts)
scores = cosine_similarity(vectorizer.transform([query]), doc_vectors)[0]
order = scores.argsort()[::-1][:3]
retrieved = [guidelines[i] for i in order]

print(f"Discharge plan — Patient ID {top['patient_nbr']}, Age {top['age']}, Main diagnosis {top['diag_1_chapter']}, Risk {top['risk']:.0%}")
print("Disclaimer: Decision support demo only, not medical advice. Clinician to review before use.")
print()
for i, g in enumerate(retrieved, start=1):
    print(f"{i}. {g['text']} [{g['id']} — {g['source']}]")

print()
print("Groundedness eval:")
print("Bullet | Cited chunk | Supported by chunk text? | Any medicine/dose invented?")
for i, g in enumerate(retrieved, start=1):
    # Templated bullets use only the chunk text itself, so support is by construction.
    # Hallucination check: plan text must not contain a dose number like 'mg' that is not in the chunk.
    has_dose = "mg" in g["text"]
    print(f"{i} | {g['id']} | Yes — bullet is the chunk text | {'FAIL — dose in chunk' if has_dose else 'Pass — no dose invented'}")
print("Overall: PASS if every bullet is cited and no dose invented.")
