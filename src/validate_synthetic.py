import pandas as pd

REAL_PATH = "data/telco_cleaned.csv"
SYNTHETIC_PATH = "output/synthetic_telco.csv"

print("=" * 70)
print("SYNTHETIC DATA VALIDATION")
print("=" * 70)

# Load datasets
real = pd.read_csv(REAL_PATH)
synthetic = pd.read_csv(SYNTHETIC_PATH)

print("\n1. DATASET SHAPES")
print("-" * 70)
print("Real dataset:", real.shape)
print("Synthetic dataset:", synthetic.shape)

# --------------------------------------------------
# Column comparison
# --------------------------------------------------

print("\n2. COLUMN COMPARISON")
print("-" * 70)

real_columns = list(real.columns)
synthetic_columns = list(synthetic.columns)

print("Same columns:", real_columns == synthetic_columns)

if real_columns != synthetic_columns:
    print("\nReal columns:")
    print(real_columns)

    print("\nSynthetic columns:")
    print(synthetic_columns)


# --------------------------------------------------
# Data type comparison
# --------------------------------------------------

print("\n3. DATA TYPE COMPARISON")
print("-" * 70)

dtype_comparison = pd.DataFrame({
    "Real": real.dtypes.astype(str),
    "Synthetic": synthetic.dtypes.astype(str)
})

print(dtype_comparison)

print(
    "\nAll data types match:",
    (dtype_comparison["Real"] == dtype_comparison["Synthetic"]).all()
)


# --------------------------------------------------
# Missing values
# --------------------------------------------------

print("\n4. MISSING VALUES")
print("-" * 70)

real_missing = real.isnull().sum().sum()
synthetic_missing = synthetic.isnull().sum().sum()

print("Real dataset missing values:", real_missing)
print("Synthetic dataset missing values:", synthetic_missing)


# --------------------------------------------------
# Duplicate rows
# --------------------------------------------------

print("\n5. DUPLICATE ROWS")
print("-" * 70)

real_duplicates = real.duplicated().sum()
synthetic_duplicates = synthetic.duplicated().sum()

print("Real dataset duplicates:", real_duplicates)
print("Synthetic dataset duplicates:", synthetic_duplicates)


# --------------------------------------------------
# Unique values
# --------------------------------------------------

print("\n6. UNIQUE VALUE COMPARISON")
print("-" * 70)

unique_comparison = pd.DataFrame({
    "Real_Unique": real.nunique(),
    "Synthetic_Unique": synthetic.nunique()
})

print(unique_comparison)


# --------------------------------------------------
# Final result
# --------------------------------------------------

print("\n" + "=" * 70)
print("BASIC VALIDATION COMPLETED")
print("=" * 70)