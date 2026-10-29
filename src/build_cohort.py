import duckdb

sql = open("sql/cohort_uci.sql").read()
df = duckdb.sql(sql).df()
df.to_csv("data/processed/cohort.csv", index=False)
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print(f"Cohort <30 rate: {df['readmitted_30d'].mean():.4f}")

