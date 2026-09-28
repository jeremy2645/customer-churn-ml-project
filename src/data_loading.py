import pandas as pd

# Load the dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 1. Dataset size
print("Dataset shape:")
print(df.shape)

# 2. Column information
print("\nDataset information:")
print(df.info())

# 3. Missing values
print("\nMissing values:")
print(df.isnull().sum())

# 4. Inspect TotalCharges
print("\nTotalCharges sample values:")
print(df["TotalCharges"].head(20).to_list())

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check missing TotalCharges
print("\nMissing TotalCharges after conversion:")
print(df["TotalCharges"].isna().sum())

# Remove rows where TotalCharges is missing
df = df.dropna(subset=["TotalCharges"])

# Check the new dataset size
print("\nDataset shape after cleaning:")
print(df.shape)

# Check for duplicate customers
duplicate_customers = df["customerID"].duplicated().sum()

print("\nDuplicate customer IDs:")
print(duplicate_customers)

# Check unique values in important categorical columns

print("\nContract types:")
print(df["Contract"].unique())

print("\nInternet service types:")
print(df["InternetService"].unique())

print("\nPayment methods:")
print(df["PaymentMethod"].unique())

print("\nChurn values:")
print(df["Churn"].unique())

# Check churn distribution

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nChurn percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# Summary of numerical columns

print("\nNumerical data summary:")
print(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe())

# Churn by contract type

print("\nChurn by contract type:")
print(pd.crosstab(df["Contract"], df["Churn"]))

print("\nChurn percentage by contract type:")
print(pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100)

# Churn by customer tenure

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
)

print("\nChurn by tenure group:")
print(pd.crosstab(df["TenureGroup"], df["Churn"]))

print("\nChurn percentage by tenure group:")
print(pd.crosstab(df["TenureGroup"], df["Churn"], normalize="index") * 100)

# Churn by monthly charges

df["MonthlyChargeGroup"] = pd.cut(
    df["MonthlyCharges"],
    bins=[0, 40, 70, 100, 120],
    labels=["$0-40", "$41-70", "$71-100", "$101-120"]
)

print("\nChurn by monthly charge group:")
print(pd.crosstab(df["MonthlyChargeGroup"], df["Churn"]))

print("\nChurn percentage by monthly charge group:")
print(
    pd.crosstab(
        df["MonthlyChargeGroup"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# Churn by internet service

print("\nChurn by internet service:")
print(pd.crosstab(df["InternetService"], df["Churn"]))

print("\nChurn percentage by internet service:")
print(
    pd.crosstab(
        df["InternetService"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# Churn by internet service

print("\nChurn by internet service:")
print(pd.crosstab(df["InternetService"], df["Churn"]))

print("\nChurn percentage by internet service:")
print(
    pd.crosstab(
        df["InternetService"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# Churn by internet service

print("\nChurn by internet service:")
print(pd.crosstab(df["InternetService"], df["Churn"]))

print("\nChurn percentage by internet service:")
print(
    pd.crosstab(
        df["InternetService"],
        df["Churn"],
        normalize="index"
    ) * 100
)

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate churn percentage by internet service
churn_by_internet = (
    pd.crosstab(
        df["InternetService"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# Create chart
churn_by_internet["Yes"].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("visual/churn_by_internet_service.png", dpi=300)
plt.show()

# Calculate churn percentage by contract type
churn_by_contract = (
    pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# Create chart
churn_by_contract["Yes"].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

# Save chart
plt.savefig("visual/churn_by_contract.png", dpi=300)

plt.show()

# Calculate churn percentage by tenure group
churn_by_tenure = (
    pd.crosstab(
        df["TenureGroup"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# Create chart
churn_by_tenure["Yes"].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Churn Rate by Customer Tenure")
plt.xlabel("Tenure Group")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

# Save chart
plt.savefig("visual/churn_by_tenure.png", dpi=300)

plt.show()

# Save the cleaned dataset
df.to_csv("data/cleaned_customer_churn.csv", index=False)

print("\nCleaned dataset saved successfully!")