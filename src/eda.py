import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/cleaned_customer_churn.csv")

print("Dataset shape:")
print(df.shape)


# -----------------------------
# 1. Churn by Contract
# -----------------------------

churn_by_contract = (
    pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100
)

print("\nChurn percentage by contract:")
print(churn_by_contract)

churn_by_contract["Yes"].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("visual/churn_by_contract.png", dpi=300)
plt.show()


# -----------------------------
# 2. Churn by Tenure
# -----------------------------

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
)

churn_by_tenure = (
    pd.crosstab(
        df["TenureGroup"],
        df["Churn"],
        normalize="index"
    ) * 100
)

print("\nChurn percentage by tenure:")
print(churn_by_tenure)

churn_by_tenure["Yes"].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Churn Rate by Customer Tenure")
plt.xlabel("Tenure Group")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("visual/churn_by_tenure.png", dpi=300)
plt.show()


# -----------------------------
# 3. Churn by Internet Service
# -----------------------------

churn_by_internet = (
    pd.crosstab(
        df["InternetService"],
        df["Churn"],
        normalize="index"
    ) * 100
)

print("\nChurn percentage by internet service:")
print(churn_by_internet)

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