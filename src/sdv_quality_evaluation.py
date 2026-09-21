import pandas as pd
import os

from sdv.metadata import Metadata
from sdv.evaluation import run_diagnostic, evaluate_quality


# --------------------------------------------------
# File paths
# --------------------------------------------------

REAL_PATH = "data/telco_cleaned.csv"
SYNTHETIC_PATH = "output/synthetic_telco.csv"
METADATA_PATH = "models/telco_metadata.json"

OUTPUT_DIR = "output/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("SDV SYNTHETIC DATA QUALITY EVALUATION")
print("=" * 70)


# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

print("\nLoading real dataset...")

real_data = pd.read_csv(REAL_PATH)

print("Real dataset shape:", real_data.shape)


print("\nLoading synthetic dataset...")

synthetic_data = pd.read_csv(SYNTHETIC_PATH)

print("Synthetic dataset shape:", synthetic_data.shape)


# --------------------------------------------------
# 2. Load saved metadata
# --------------------------------------------------

print("\nLoading metadata...")

metadata = Metadata.load_from_json(
    filepath=METADATA_PATH
)

print("Metadata loaded successfully.")


# --------------------------------------------------
# 3. Run diagnostic
# --------------------------------------------------

print("\nRunning SDV diagnostic evaluation...")
print("Please wait...")

diagnostic = run_diagnostic(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)


print("\n" + "=" * 70)
print("DIAGNOSTIC RESULTS")
print("=" * 70)

print(diagnostic)


# --------------------------------------------------
# 4. Run quality evaluation
# --------------------------------------------------

print("\nRunning SDV quality evaluation...")
print("Please wait...")

quality_report = evaluate_quality(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)


print("\n" + "=" * 70)
print("QUALITY REPORT")
print("=" * 70)

print(quality_report)


# --------------------------------------------------
# 5. Get overall quality score
# --------------------------------------------------

overall_score = quality_report.get_score()

print("\n" + "=" * 70)
print("OVERALL SYNTHETIC DATA QUALITY SCORE")
print("=" * 70)

print(
    "Overall Quality Score:",
    round(overall_score, 4)
)


# --------------------------------------------------
# 6. Get property scores
# --------------------------------------------------

print("\n" + "=" * 70)
print("QUALITY PROPERTY SCORES")
print("=" * 70)

property_scores = quality_report.get_properties()

print(property_scores)


# --------------------------------------------------
# 7. Save reports
# --------------------------------------------------

diagnostic_path = (
    f"{OUTPUT_DIR}/sdv_diagnostic_report.pkl"
)

quality_path = (
    f"{OUTPUT_DIR}/sdv_quality_report.pkl"
)

diagnostic.save(
    filepath=diagnostic_path
)

quality_report.save(
    filepath=quality_path
)


print("\nReports saved:")

print(
    "output/evaluation/sdv_diagnostic_report.pkl"
)

print(
    "output/evaluation/sdv_quality_report.pkl"
)


print("\n" + "=" * 70)
print("SDV QUALITY EVALUATION COMPLETED")
print("=" * 70)