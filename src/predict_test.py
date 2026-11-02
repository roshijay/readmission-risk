import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv("data/processed/cohort.csv")

numeric = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
           "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
categorical = ["age", "gender", "race", "diag_1_chapter", "max_glu_serum", "A1Cresult",
               "insulin", "change", "diabetesMed"]
features = numeric + categorical

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df["patient_nbr"]))
train = df.iloc[train_idx]
test = df.iloc[test_idx].copy()

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

model = Pipeline([
    ("preprocess", preprocess),
    ("model", HistGradientBoostingClassifier(random_state=42))
])
model.fit(train[features], train["readmitted_30d"])
test["risk"] = model.predict_proba(test[features])[:, 1]

out = test[["patient_nbr", "age", "gender", "race", "diag_1_chapter",
            "time_in_hospital", "number_inpatient", "number_emergency",
            "risk", "readmitted_30d"]].sort_values("risk", ascending=False)
out.to_csv("data/processed/test_predictions.csv", index=False)
print(f"Saved rows: {len(out)}")
print(out.head(5).to_string(index=False))
