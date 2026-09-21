import pandas as pd

from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer


# 1. Load the real dataset
data = pd.read_csv("data/sample_customers.csv")

print("Real dataset:")
print(data)
print("\nDataset shape:", data.shape)


# 2. Create metadata
metadata = Metadata.detect_from_dataframe(
    data=data,
    table_name="customers"
)

print("\nMetadata created successfully.")


# 3. Create CTGAN model
synthesizer = CTGANSynthesizer(
    metadata,
    epochs=10
)

print("\nTraining CTGAN...")


# 4. Train the model
synthesizer.fit(data)

print("CTGAN training completed successfully.")


# 5. Generate synthetic data
synthetic_data = synthesizer.sample(
    num_rows=10
)

print("\nSynthetic dataset:")
print(synthetic_data)


# 6. Save synthetic data
synthetic_data.to_csv(
    "output/test_synthetic_customers.csv",
    index=False
)

print("\nSynthetic dataset saved successfully.")