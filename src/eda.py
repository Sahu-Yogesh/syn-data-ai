import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# 1. Load cleaned dataset
# ============================================================

df = pd.read_csv("data/telco_cleaned.csv")

print("=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("Dataset shape:", df.shape)


# ============================================================
# 2. Basic statistics
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL SUMMARY")
print("=" * 70)

print(df.describe())


# ============================================================
# 3. Churn distribution
# ============================================================

print("\n" + "=" * 70)
print("CHURN DISTRIBUTION")
print("=" * 70)

print(df["Churn"].value_counts())
print("\nPercentage:")
print(df["Churn"].value_counts(normalize=True) * 100)


# ============================================================
# 4. Create EDA output folder
# ============================================================

os.makedirs("output/eda", exist_ok=True)


# ============================================================
# 5. Churn distribution chart
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "output/eda/churn_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 6. Contract distribution
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract"
)

plt.title("Contract Type Distribution")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "output/eda/contract_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 7. Internet service distribution
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="InternetService"
)

plt.title("Internet Service Distribution")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "output/eda/internet_service_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 8. Monthly charges distribution
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="MonthlyCharges",
    bins=30,
    kde=True
)

plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "output/eda/monthly_charges_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. Tenure distribution
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="tenure",
    bins=30,
    kde=True
)

plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "output/eda/tenure_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. Churn vs Contract
# ============================================================

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "output/eda/churn_by_contract.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. Churn vs Monthly Charges
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig(
    "output/eda/monthly_charges_by_churn.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. Churn vs Tenure
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Tenure by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.tight_layout()

plt.savefig(
    "output/eda/tenure_by_churn.png",
    dpi=300
)

plt.close()


print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)

print("Charts saved in:")
print("output/eda/")