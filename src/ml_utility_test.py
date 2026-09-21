import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# --------------------------------------------------
# File paths
# --------------------------------------------------

REAL_PATH = "data/telco_cleaned.csv"
SYNTHETIC_PATH = "output/synthetic_telco.csv"

OUTPUT_DIR = "output/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("MACHINE LEARNING UTILITY EVALUATION")
print("=" * 70)


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

real = pd.read_csv(REAL_PATH)
synthetic = pd.read_csv(SYNTHETIC_PATH)

print("\nReal dataset:", real.shape)
print("Synthetic dataset:", synthetic.shape)


# --------------------------------------------------
# Separate features and target
# --------------------------------------------------

TARGET = "Churn"

X_real = real.drop(columns=[TARGET])
y_real = real[TARGET]

X_synthetic = synthetic.drop(columns=[TARGET])
y_synthetic = synthetic[TARGET]


# --------------------------------------------------
# Split REAL data
# --------------------------------------------------

X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real,
    y_real,
    test_size=0.20,
    random_state=42,
    stratify=y_real
)


print("\nReal training records:", len(X_train_real))
print("Real testing records:", len(X_test_real))


# --------------------------------------------------
# Identify column types
# --------------------------------------------------

categorical_columns = X_real.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X_real.select_dtypes(
    exclude=["object"]
).columns.tolist()


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# --------------------------------------------------
# Create classifier
# --------------------------------------------------

def create_model():

    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ]
    )

    return model


# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name
):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        pos_label="Yes"
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="Yes"
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="Yes"
    )

    # Convert target to binary for ROC-AUC
    y_test_binary = (
        y_test == "Yes"
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test_binary,
        probabilities
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["No", "Yes"]
    )

    print("\n" + "-" * 70)
    print(model_name)
    print("-" * 70)

    print(
        "Accuracy:",
        round(accuracy, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall:",
        round(recall, 4)
    )

    print(
        "F1-score:",
        round(f1, 4)
    )

    print(
        "ROC-AUC:",
        round(roc_auc, 4)
    )

    print("\nConfusion Matrix:")
    print(matrix)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc
    }


# --------------------------------------------------
# Experiment A
# Train on REAL data
# --------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENT A: TRAIN ON REAL DATA")
print("=" * 70)

real_model = create_model()

real_model.fit(
    X_train_real,
    y_train_real
)

real_results = evaluate_model(
    real_model,
    X_test_real,
    y_test_real,
    "Real → Real"
)


# --------------------------------------------------
# Experiment B
# Train on SYNTHETIC data
# Test on REAL data
# --------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENT B: TRAIN ON SYNTHETIC DATA")
print("=" * 70)

synthetic_model = create_model()

synthetic_model.fit(
    X_synthetic,
    y_synthetic
)

synthetic_results = evaluate_model(
    synthetic_model,
    X_test_real,
    y_test_real,
    "Synthetic → Real"
)


# --------------------------------------------------
# Compare results
# --------------------------------------------------

results = pd.DataFrame(
    [
        real_results,
        synthetic_results
    ]
)


print("\n" + "=" * 70)
print("ML UTILITY COMPARISON")
print("=" * 70)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# --------------------------------------------------
# Calculate utility ratios
# --------------------------------------------------

real_f1 = real_results["F1"]
synthetic_f1 = synthetic_results["F1"]

real_auc = real_results["ROC_AUC"]
synthetic_auc = synthetic_results["ROC_AUC"]

f1_retention = (
    synthetic_f1 / real_f1
) * 100

auc_retention = (
    synthetic_auc / real_auc
) * 100


print("\n" + "-" * 70)
print("UTILITY RETENTION")
print("-" * 70)

print(
    "F1-score retention:",
    round(f1_retention, 2),
    "%"
)

print(
    "ROC-AUC retention:",
    round(auc_retention, 2),
    "%"
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

results.to_csv(
    f"{OUTPUT_DIR}/ml_utility_results.csv",
    index=False
)


print("\nResults saved to:")
print(
    "output/evaluation/ml_utility_results.csv"
)


print("\n" + "=" * 70)
print("ML UTILITY EVALUATION COMPLETED")
print("=" * 70)