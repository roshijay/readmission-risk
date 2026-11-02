import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

guidelines = json.load(open("data/guidelines.json"))
texts = [g["text"] for g in guidelines]

pred = pd.read_csv("data/processed/test_predictions.csv")
top = pred.iloc[0]
print(f"Top patient: ID {top['patient_nbr']}, Age {top['age']}, Main diagnosis {top['diag_1_chapter']}, Prior inpatient {top['number_inpatient']}, Risk {top['risk']:.2f}")

query = f"{top['diag_1_chapter']} prior inpatient stays medication reconciliation primary care follow-up red flags discharge instructions"
print(f"Query: {query}")

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(texts)
query_vector = vectorizer.transform([query])
scores = cosine_similarity(query_vector, doc_vectors)[0]

order = scores.argsort()[::-1][:3]
for rank, idx in enumerate(order, start=1):
    g = guidelines[idx]
    print(f"\nRank {rank} — Score {scores[idx]:.3f} — {g['id']} — {g['topic']} — Source: {g['source']}")
    print(g["text"])
