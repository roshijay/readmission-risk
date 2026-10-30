import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

df = pd.read_csv("data/processed/cohort.csv")

numeric = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
           "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
categorical = ["age", "gender", "race", "diag_1_chapter", "max_glu_serum", "A1Cresult",
               "insulin", "change", "diabetesMed"]
features = numeric + categorical

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df["patient_nbr"]))
train = df.iloc[train_idx]
test = df.iloc[test_idx]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

def precision_at_k(y_true, y_prob, k):
    order = np.argsort(-y_prob)[:k]
    return y_true[order].mean()

rf = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])
rf.fit(train[features], train["readmitted_30d"])
prob = rf.predict_proba(test[features])[:, 1]
y_test = test["readmitted_30d"]

print("Random Forest:")
print(f"  PR-AUC: {average_precision_score(y_test, prob):.4f}")
print(f"  ROC-AUC: {roc_auc_score(y_test, prob):.4f}")
print(f"  Brier: {brier_score_loss(y_test, prob):.4f}")
print(f"  Precision@100: {precision_at_k(y_test.values, prob, 100):.4f}")
print(f"  Precision@200: {precision_at_k(y_test.values, prob, 200):.4f}")
