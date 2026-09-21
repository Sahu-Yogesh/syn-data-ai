import pandas as pd
import os
import time

from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer


# --------------------------------------------------
# 1. Load cleaned real dataset
# --------------------------------------------------

DATA_PATH = "data/telco_cleaned.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "output"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("CTGAN SYNTHETIC DATA GENERATION")
print("=" * 70)

data = pd.read_csv(DATA_PATH)

print("\nReal dataset loaded successfully.")
print("Shape:", data.shape)
print("Columns:", list(data.columns))


# --------------------------------------------------
# 2. Detect metadata
# --------------------------------------------------

print("\nDetecting metadata...")

metadata = Metadata.detect_from_dataframe(
    data=data,
    table_name="telco_customers"
)

print("Metadata detected successfully.")


# --------------------------------------------------
# 3. Save metadata
# --------------------------------------------------

METADATA_PATH = os.path.join(MODEL_DIR, "telco_metadata.json")

metadata.save_to_json(
    filepath=METADATA_PATH
)

print("Metadata saved to:")
print(METADATA_PATH)


# --------------------------------------------------
# 4. Create CTGAN model
# --------------------------------------------------

print("\nCreating CTGAN model...")

synthesizer = CTGANSynthesizer(
    metadata,
    epochs=300,
    verbose=True
)

print("CTGAN model created.")


# --------------------------------------------------
# 5. Train CTGAN
# --------------------------------------------------

print("\nStarting CTGAN training...")
print("This may take some time.")

start_time = time.time()

synthesizer.fit(data)

end_time = time.time()

training_time = end_time - start_time

print("\nCTGAN training completed successfully.")

print(
    "Training time:",
    round(training_time / 60, 2),
    "minutes"
)


# --------------------------------------------------
# 6. Save trained model
# --------------------------------------------------

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ctgan_telco.pkl"
)

synthesizer.save(
    filepath=MODEL_PATH
)

print("\nTrained CTGAN model saved to:")
print(MODEL_PATH)


# --------------------------------------------------
# 7. Generate synthetic data
# --------------------------------------------------

print("\nGenerating synthetic dataset...")

synthetic_data = synthesizer.sample(
    num_rows=len(data)
)

print("Synthetic data generated successfully.")

print("Synthetic dataset shape:")
print(synthetic_data.shape)


# --------------------------------------------------
# 8. Save synthetic dataset
# --------------------------------------------------

SYNTHETIC_PATH = os.path.join(
    OUTPUT_DIR,
    "synthetic_telco.csv"
)

synthetic_data.to_csv(
    SYNTHETIC_PATH,
    index=False
)

print("\nSynthetic dataset saved to:")
print(SYNTHETIC_PATH)


# --------------------------------------------------
# 9. Display sample
# --------------------------------------------------

print("\nFirst 5 synthetic records:")
print(synthetic_data.head())

print("\n" + "=" * 70)
print("CTGAN PROCESS COMPLETED")
print("=" * 70)