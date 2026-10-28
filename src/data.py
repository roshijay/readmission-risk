""" Load and verify the UCI diabetes readmission data."""

import pandas as pd 
from pathlib import Path 

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "diabetic_data.csv"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path, na_values="?", low_memory=False)
    return df 

def check_data(df):
    print(f"Shape (rows,columns): {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nTarget distribution (readmitted):")
    print(df["readmitted"].value_counts(dropna=False))
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    print(f"\nBinarized <30 rate: {df['readmitted_30d'].mean():.4f} (expect ~0.1116)")
    print("\nTop 10 columns by missing fraction:")
    print(df.isna().mean().sort_values(ascending=False).head(10))
    print(f"\nDuplicate encounter_id rows: {df['encounter_id'].duplicated().sum()}")
    print(f"Unique patients: {df['patient_nbr'].nunique()}")
    print(f"Unique encounters: {df['encounter_id'].nunique()}")
    return True

if __name__ == "__main__":
    df = load_data()
    check_data(df)
    print("\nCheck complete. If shapes and rates match the guide, Step 1 passes.")