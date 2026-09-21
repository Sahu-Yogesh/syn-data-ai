import pandas as pd
import numpy as np
import os


# --------------------------------------------------
# File paths
# --------------------------------------------------

REAL_PATH = "data/telco_cleaned.csv"
SYNTHETIC_PATH = "output/synthetic_telco.csv"

OUTPUT_DIR = "output/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

real = pd.read_csv(REAL_PATH)
synthetic = pd.read_csv(SYNTHETIC_PATH)


print("=" * 70)
print("REAL vs SYNTHETIC STATISTICAL COMPARISON")
print("=" * 70)


# --------------------------------------------------
# Numerical columns
# --------------------------------------------------

numerical_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


print("\n1. NUMERICAL STATISTICS")
print("-" * 70)


results = []

for column in numerical_columns:

    real_mean = real[column].mean()
    synthetic_mean = synthetic[column].mean()

    real_std = real[column].std()
    synthetic_std = synthetic[column].std()

    real_median = real[column].median()
    synthetic_median = synthetic[column].median()

    real_min = real[column].min()
    synthetic_min = synthetic[column].min()

    real_max = real[column].max()
    synthetic_max = synthetic[column].max()

    mean_difference = synthetic_mean - real_mean

    if real_mean != 0:
        mean_difference_percent = (
            abs(mean_difference) / abs(real_mean)
        ) * 100
    else:
        mean_difference_percent = 0

    results.append({
        "Column": column,
        "Real_Mean": real_mean,
        "Synthetic_Mean": synthetic_mean,
        "Real_Std": real_std,
        "Synthetic_Std": synthetic_std,
        "Real_Median": real_median,
        "Synthetic_Median": synthetic_median,
        "Real_Min": real_min,
        "Synthetic_Min": synthetic_min,
        "Real_Max": real_max,
        "Synthetic_Max": synthetic_max,
        "Mean_Difference": mean_difference,
        "Mean_Difference_Percent": mean_difference_percent
    })


numerical_results = pd.DataFrame(results)

print(
    numerical_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# --------------------------------------------------
# Save numerical comparison
# --------------------------------------------------

numerical_results.to_csv(
    f"{OUTPUT_DIR}/numerical_comparison.csv",
    index=False
)


# --------------------------------------------------
# Categorical comparison
# --------------------------------------------------

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


print("\n\n2. CATEGORICAL DISTRIBUTION COMPARISON")
print("-" * 70)


categorical_results = []

for column in categorical_columns:

    categories = sorted(
        set(real[column].dropna().unique())
        |
        set(synthetic[column].dropna().unique())
    )

    for category in categories:

        real_percentage = (
            (real[column] == category).mean() * 100
        )

        synthetic_percentage = (
            (synthetic[column] == category).mean() * 100
        )

        difference = synthetic_percentage - real_percentage

        categorical_results.append({
            "Column": column,
            "Category": category,
            "Real_Percentage": real_percentage,
            "Synthetic_Percentage": synthetic_percentage,
            "Difference": difference,
            "Absolute_Difference": abs(difference)
        })


categorical_results = pd.DataFrame(
    categorical_results
)


print(
    categorical_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# --------------------------------------------------
# Save categorical comparison
# --------------------------------------------------

categorical_results.to_csv(
    f"{OUTPUT_DIR}/categorical_comparison.csv",
    index=False
)


# --------------------------------------------------
# Overall categorical error
# --------------------------------------------------

mean_categorical_difference = (
    categorical_results["Absolute_Difference"].mean()
)


print("\n\n3. OVERALL SUMMARY")
print("-" * 70)

print(
    "Average absolute categorical percentage difference:",
    round(mean_categorical_difference, 4),
    "%"
)


print("\nFiles saved:")

print(
    "output/evaluation/numerical_comparison.csv"
)

print(
    "output/evaluation/categorical_comparison.csv"
)


print("\n" + "=" * 70)
print("STATISTICAL COMPARISON COMPLETED")
print("=" * 70)