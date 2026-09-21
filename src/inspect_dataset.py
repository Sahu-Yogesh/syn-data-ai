import pandas as pd

# Load dataset
df = pd.read_csv("data/telco_customer_churn.csv")

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())


print("\n" + "=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)

print("Duplicate rows:", df.duplicated().sum())


print("\n" + "=" * 70)
print("UNIQUE VALUES")
print("=" * 70)

for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")


print("\n" + "=" * 70)
print("CHURN DISTRIBUTION")
print("=" * 70)

print(df["Churn"].value_counts())

print("\nChurn percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)