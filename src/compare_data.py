import pandas as pd

# Load real data
real_data = pd.read_csv("data/sample_customers.csv")

# Load synthetic data
synthetic_data = pd.read_csv("output/test_synthetic_customers.csv")

print("=" * 60)
print("REAL DATASET")
print("=" * 60)

print(real_data)

print("\nReal dataset shape:")
print(real_data.shape)


print("\n" + "=" * 60)
print("SYNTHETIC DATASET")
print("=" * 60)

print(synthetic_data)

print("\nSynthetic dataset shape:")
print(synthetic_data.shape)


print("\n" + "=" * 60)
print("REAL DATA TYPES")
print("=" * 60)

print(real_data.dtypes)


print("\n" + "=" * 60)
print("SYNTHETIC DATA TYPES")
print("=" * 60)

print(synthetic_data.dtypes)