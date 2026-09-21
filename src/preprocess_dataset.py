import pandas as pd

# ============================================================
# 1. Load original dataset
# ============================================================

input_file = "data/telco_customer_churn.csv"

df = pd.read_csv(input_file)

print("=" * 70)
print("ORIGINAL DATASET")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 2. Check TotalCharges before conversion
# ============================================================

print("\n" + "=" * 70)
print("TOTALCHARGES BEFORE CONVERSION")
print("=" * 70)

print("Data type:", df["TotalCharges"].dtype)

# Convert blank/whitespace values to missing values
df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True)

print("Missing TotalCharges:", df["TotalCharges"].isna().sum())


# ============================================================
# 3. Convert TotalCharges to numeric
# ============================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nTotalCharges data type after conversion:")
print(df["TotalCharges"].dtype)

print("Missing TotalCharges after conversion:")
print(df["TotalCharges"].isna().sum())


# ============================================================
# 4. Remove rows with missing TotalCharges
# ============================================================

rows_before = len(df)

df = df.dropna(subset=["TotalCharges"])

rows_after = len(df)

print("\n" + "=" * 70)
print("ROWS REMOVED")
print("=" * 70)

print("Rows before cleaning:", rows_before)
print("Rows after cleaning:", rows_after)
print("Rows removed:", rows_before - rows_after)


# ============================================================
# 5. Remove customerID
# ============================================================

df = df.drop(columns=["customerID"])

print("\n" + "=" * 70)
print("CUSTOMER ID")
print("=" * 70)

print("customerID removed because it is an identifier,")
print("not a meaningful predictive feature for synthetic generation.")


# ============================================================
# 6. Check duplicates after cleaning
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATES AFTER REMOVING CUSTOMER ID")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Duplicate rows found:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed:", duplicate_count)
else:
    print("No duplicate rows found.")


# ============================================================
# 7. Display final data types
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ============================================================
# 8. Save cleaned dataset
# ============================================================

output_file = "data/telco_cleaned.csv"

df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("CLEANING COMPLETED")
print("=" * 70)

print("Cleaned dataset saved to:")
print(output_file)

print("\nFinal shape:", df.shape)