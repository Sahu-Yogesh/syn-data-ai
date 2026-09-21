import pandas as pd
import numpy as np
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1. FILE PATHS
# ============================================================

REAL_PATH = "data/telco_cleaned.csv"
SYNTHETIC_PATH = "output/synthetic_telco.csv"
OUTPUT_DIR = "output/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

real_data = pd.read_csv(REAL_PATH)
synthetic_data = pd.read_csv(SYNTHETIC_PATH)

print("Real dataset shape:", real_data.shape)
print("Synthetic dataset shape:", synthetic_data.shape)


# ============================================================
# 3. MAKE SURE BOTH DATASETS HAVE SAME COLUMNS
# ============================================================

if list(real_data.columns) != list(synthetic_data.columns):
    raise ValueError("Real and synthetic datasets do not have matching columns.")

print("Columns match: True")


# ============================================================
# 4. EXACT DUPLICATE CHECK
# ============================================================

# Convert every row into a comparable tuple
real_rows = set(map(tuple, real_data.astype(str).values))
synthetic_rows = set(map(tuple, synthetic_data.astype(str).values))

exact_matches = real_rows.intersection(synthetic_rows)

exact_match_count = len(exact_matches)

exact_match_percentage = (
    exact_match_count / len(synthetic_data)
) * 100

print("\n" + "=" * 60)
print("EXACT RECORD MATCHING")
print("=" * 60)

print("Exact matching records:", exact_match_count)
print("Percentage of synthetic records with exact match:",
      round(exact_match_percentage, 4), "%")


# ============================================================
# 5. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# ============================================================

numeric_columns = real_data.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = real_data.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ============================================================
# 6. PREPARE DATA FOR SIMILARITY ANALYSIS
# ============================================================

# Numerical values are standardized.
# Categorical values are converted using one-hot encoding.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_columns
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            ),
            categorical_columns
        )
    ]
)


# Fit transformation using REAL data
real_transformed = preprocessor.fit_transform(real_data)

# Apply same transformation to synthetic data
synthetic_transformed = preprocessor.transform(synthetic_data)


print("\nData transformation completed.")


# ============================================================
# 7. FIND CLOSEST REAL RECORD FOR EACH SYNTHETIC RECORD
# ============================================================

print("\nFinding nearest real records...")

nearest_neighbors = NearestNeighbors(
    n_neighbors=1,
    metric="euclidean"
)

nearest_neighbors.fit(real_transformed)

distances, indices = nearest_neighbors.kneighbors(
    synthetic_transformed
)

nearest_distances = distances[:, 0]


# ============================================================
# 8. NORMALIZE DISTANCES
# ============================================================

# Convert distances into a 0–1 similarity score.
# Higher score = more similar.

max_distance = nearest_distances.max()

if max_distance == 0:
    similarity_scores = np.ones(len(nearest_distances))
else:
    similarity_scores = 1 - (
        nearest_distances / max_distance
    )

similarity_percentages = similarity_scores * 100


# ============================================================
# 9. SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("RECORD SIMILARITY RESULTS")
print("=" * 60)

print("Minimum distance:",
      round(nearest_distances.min(), 4))

print("Average distance:",
      round(nearest_distances.mean(), 4))

print("Maximum distance:",
      round(nearest_distances.max(), 4))

print("Average similarity:",
      round(similarity_percentages.mean(), 2), "%")

print("Maximum similarity:",
      round(similarity_percentages.max(), 2), "%")


# ============================================================
# 10. VERY CLOSE RECORD COUNTS
# ============================================================

# These thresholds are used only as practical screening levels.
# They are NOT formal privacy guarantees.

thresholds = [90, 95, 99]

print("\nSimilarity thresholds:")

for threshold in thresholds:

    count = np.sum(
        similarity_percentages >= threshold
    )

    percentage = (
        count / len(similarity_percentages)
    ) * 100

    print(
        f"{threshold}% or higher similarity: "
        f"{count} records ({percentage:.2f}%)"
    )


# ============================================================
# 11. SAVE DETAILED RESULTS
# ============================================================

results = pd.DataFrame({
    "Synthetic_Record_Number":
        range(1, len(synthetic_data) + 1),

    "Closest_Real_Record_Index":
        indices[:, 0],

    "Distance":
        nearest_distances,

    "Similarity_Percentage":
        similarity_percentages
})

results_path = os.path.join(
    OUTPUT_DIR,
    "record_similarity_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print("\nDetailed similarity results saved to:")
print(results_path)


# ============================================================
# 12. SAVE SUMMARY
# ============================================================

summary = {
    "Real_Records": len(real_data),
    "Synthetic_Records": len(synthetic_data),
    "Exact_Matching_Records": exact_match_count,
    "Exact_Match_Percentage": exact_match_percentage,
    "Minimum_Distance": nearest_distances.min(),
    "Average_Distance": nearest_distances.mean(),
    "Maximum_Distance": nearest_distances.max(),
    "Average_Similarity_Percentage": similarity_percentages.mean(),
    "Maximum_Similarity_Percentage": similarity_percentages.max()
}

for threshold in thresholds:

    count = np.sum(
        similarity_percentages >= threshold
    )

    percentage = (
        count / len(similarity_percentages)
    ) * 100

    summary[
        f"Similarity_{threshold}_Percent_or_Higher_Count"
    ] = count

    summary[
        f"Similarity_{threshold}_Percent_or_Higher_Percentage"
    ] = percentage


summary_df = pd.DataFrame(
    [summary]
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "privacy_similarity_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("\nSummary saved to:")
print(summary_path)


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("PRIVACY & SIMILARITY EVALUATION COMPLETED")
print("=" * 60)

print("\nIMPORTANT:")
print(
    "Similarity analysis is a screening measure. "
    "It does NOT prove differential privacy or guarantee "
    "that the synthetic dataset is private."
)