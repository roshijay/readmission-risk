import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
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

def evaluate(name, model, X_test, y_test):
    prob = model.predict_proba(X_test)[:, 1]
    print(f"\n{name}:")
    print(f"  PR-AUC: {average_precision_score(y_test, prob):.4f}")
    print(f"  ROC-AUC: {roc_auc_score(y_test, prob):.4f}")
    print(f"  Brier: {brier_score_loss(y_test, prob):.4f}")
    print(f"  Precision@100: {precision_at_k(y_test.values, prob, 100):.4f}")
    print(f"  Precision@200: {precision_at_k(y_test.values, prob, 200):.4f}")

X_train = train[features]
y_train = train["readmitted_30d"]
X_test = test[features]
y_test = test["readmitted_30d"]

logreg = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000))
])
logreg.fit(X_train, y_train)
evaluate("Logistic Regression (baseline)", logreg, X_test, y_test)

hgb = Pipeline([
    ("preprocess", preprocess),
    ("model", HistGradientBoostingClassifier(random_state=42))
])
hgb.fit(X_train, y_train)
evaluate("HistGradientBoosting (challenger)", hgb, X_test, y_test)
