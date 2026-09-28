import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. LOAD DATA
# ==================================================

df = pd.read_csv(
    "data/cleaned_customer_churn.csv"
)

print("Customer churn data loaded!")
print("Customers:", len(df))


# ==================================================
# 2. CHURN BY CONTRACT
# ==================================================

contract_analysis = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="Churn Rate")
)

contract_analysis = contract_analysis.sort_values(
    "Churn Rate",
    ascending=False
)

print("\nChurn by Contract:")
print(contract_analysis)


# ==================================================
# 3. CHURN BY TENURE GROUP
# ==================================================

tenure_analysis = (
    df.groupby("TenureGroup", observed=True)["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="Churn Rate")
)

print("\nChurn by Tenure:")
print(tenure_analysis)


# ==================================================
# 4. CHURN BY INTERNET SERVICE
# ==================================================

internet_analysis = (
    df.groupby("InternetService")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="Churn Rate")
)

internet_analysis = internet_analysis.sort_values(
    "Churn Rate",
    ascending=False
)

print("\nChurn by Internet Service:")
print(internet_analysis)


# ==================================================
# 5. CHURN BY MONTHLY CHARGES
# ==================================================

charge_analysis = (
    df.groupby("MonthlyChargeGroup", observed=True)["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="Churn Rate")
)

print("\nChurn by Monthly Charges:")
print(charge_analysis)


# ==================================================
# 6. CREATE BUSINESS INSIGHTS TABLE
# ==================================================

insights = pd.DataFrame({

    "Business Area": [
        "Contract",
        "Tenure",
        "Internet Service",
        "Monthly Charges"
    ],

    "Observation": [
        "Month-to-month customers show higher churn than customers on longer contracts.",
        "Customers with shorter tenure show higher churn rates.",
        "Fiber optic customers show higher churn than DSL and customers without internet service.",
        "Higher monthly-charge groups generally show higher churn than the lowest-charge group."
    ],

    "Business Action": [
        "Consider retention offers or contract incentives for month-to-month customers.",
        "Create an onboarding and early-retention program for newer customers.",
        "Investigate service experience and support issues among fiber customers.",
        "Review pricing, perceived value and service experience for higher-charge customers."
    ]
})

print("\n========================================")
print("BUSINESS INSIGHTS")
print("========================================")

print(insights.to_string(index=False))


# ==================================================
# 7. SAVE BUSINESS INSIGHTS
# ==================================================

insights.to_csv(
    "data/business_insights.csv",
    index=False
)

print("\nBusiness insights saved successfully!")


# ==================================================
# 8. SAVE ANALYSIS TABLES
# ==================================================

contract_analysis.to_csv(
    "data/churn_by_contract.csv",
    index=False
)

tenure_analysis.to_csv(
    "data/churn_by_tenure.csv",
    index=False
)

internet_analysis.to_csv(
    "data/churn_by_internet_service.csv",
    index=False
)

charge_analysis.to_csv(
    "data/churn_by_monthly_charge.csv",
    index=False
)

print("Analysis tables saved successfully!")