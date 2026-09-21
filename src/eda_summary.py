import pandas as pd
import os

# Load cleaned dataset
df = pd.read_csv("data/telco_cleaned.csv")

os.makedirs("output/eda", exist_ok=True)

# ============================================================
# Numerical summary
# ============================================================

numerical_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

numerical_summary = df[numerical_columns].describe().T

numerical_summary.to_csv(
    "output/eda/numerical_summary.csv"
)

# ============================================================
# Categorical summary
# ============================================================

categorical_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "Churn"
]

categorical_results = []

for column in categorical_columns:

    counts = df[column].value_counts()

    percentages = df[column].value_counts(
        normalize=True
    ) * 100

    for category in counts.index:

        categorical_results.append({
            "Column": column,
            "Category": category,
            "Count": counts[category],
            "Percentage": percentages[category]
        })

categorical_summary = pd.DataFrame(
    categorical_results
)

categorical_summary.to_csv(
    "output/eda/categorical_summary.csv",
    index=False
)

print("=" * 70)
print("EDA SUMMARY CREATED")
print("=" * 70)

print("\nNumerical summary:")
print(numerical_summary)

print("\nCategorical summary saved to:")
print("output/eda/categorical_summary.csv")

print("\nNumerical summary saved to:")
print("output/eda/numerical_summary.csv")