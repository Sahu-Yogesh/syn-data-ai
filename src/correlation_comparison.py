import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


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
print("REAL vs SYNTHETIC CORRELATION COMPARISON")
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


# --------------------------------------------------
# Calculate correlations
# --------------------------------------------------

real_corr = real[numerical_columns].corr()
synthetic_corr = synthetic[numerical_columns].corr()


print("\nREAL DATASET CORRELATION MATRIX")
print("-" * 70)
print(real_corr.round(4))


print("\nSYNTHETIC DATASET CORRELATION MATRIX")
print("-" * 70)
print(synthetic_corr.round(4))


# --------------------------------------------------
# Difference matrix
# --------------------------------------------------

difference_matrix = (
    synthetic_corr - real_corr
).abs()


print("\nABSOLUTE CORRELATION DIFFERENCE")
print("-" * 70)
print(difference_matrix.round(4))


# --------------------------------------------------
# Mean correlation difference
# --------------------------------------------------

# Remove diagonal because a variable's correlation
# with itself is always 1.0

mask = ~np.eye(
    len(numerical_columns),
    dtype=bool
)

mean_difference = difference_matrix.values[mask].mean()

print(
    "\nAverage absolute correlation difference:",
    round(mean_difference, 4)
)


# --------------------------------------------------
# Save matrices
# --------------------------------------------------

real_corr.to_csv(
    f"{OUTPUT_DIR}/real_correlation_matrix.csv"
)

synthetic_corr.to_csv(
    f"{OUTPUT_DIR}/synthetic_correlation_matrix.csv"
)

difference_matrix.to_csv(
    f"{OUTPUT_DIR}/correlation_difference_matrix.csv"
)


# --------------------------------------------------
# Create heatmap function
# --------------------------------------------------

def create_heatmap(
    matrix,
    title,
    filename
):

    plt.figure(figsize=(8, 6))

    plt.imshow(
        matrix,
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(matrix.columns)),
        matrix.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(matrix.index)),
        matrix.index
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/{filename}",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# Create heatmaps
# --------------------------------------------------

create_heatmap(
    real_corr,
    "Real Dataset Correlation",
    "real_correlation_heatmap.png"
)

create_heatmap(
    synthetic_corr,
    "Synthetic Dataset Correlation",
    "synthetic_correlation_heatmap.png"
)

create_heatmap(
    difference_matrix,
    "Correlation Difference",
    "correlation_difference_heatmap.png"
)


# --------------------------------------------------
# Final output
# --------------------------------------------------

print("\nFiles saved:")
print("output/evaluation/real_correlation_matrix.csv")
print("output/evaluation/synthetic_correlation_matrix.csv")
print("output/evaluation/correlation_difference_matrix.csv")
print("output/evaluation/real_correlation_heatmap.png")
print("output/evaluation/synthetic_correlation_heatmap.png")
print("output/evaluation/correlation_difference_heatmap.png")

print("\n" + "=" * 70)
print("CORRELATION COMPARISON COMPLETED")
print("=" * 70)