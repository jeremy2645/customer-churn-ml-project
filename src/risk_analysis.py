import pandas as pd


# ==================================================
# 1. LOAD DATA
# ==================================================

df = pd.read_csv(
    "data/cleaned_customer_churn.csv"
)


# ==================================================
# 2. CREATE BUSINESS RISK SCORE
# ==================================================

df["RiskScore"] = 0


# Month-to-month contract
df.loc[
    df["Contract"] == "Month-to-month",
    "RiskScore"
] += 2


# Short tenure
df.loc[
    df["tenure"] <= 12,
    "RiskScore"
] += 2


# High monthly charges
df.loc[
    df["MonthlyCharges"] >= 70,
    "RiskScore"
] += 1


# Fiber optic
df.loc[
    df["InternetService"] == "Fiber optic",
    "RiskScore"
] += 1


# Electronic check
df.loc[
    df["PaymentMethod"] == "Electronic check",
    "RiskScore"
] += 1


# ==================================================
# 3. CREATE RISK CATEGORY
# ==================================================

def classify_risk(score):

    if score >= 5:
        return "High Risk"

    elif score >= 3:
        return "Medium Risk"

    else:
        return "Low Risk"


df["RiskCategory"] = df["RiskScore"].apply(
    classify_risk
)


# ==================================================
# 4. RISK SUMMARY
# ==================================================

risk_summary = (
    df.groupby("RiskCategory", observed=True)
    .agg(
        Customers=("customerID", "count"),
        AverageMonthlyCharges=("MonthlyCharges", "mean"),
        ActualChurnRate=("Churn", lambda x: (x == "Yes").mean() * 100)
    )
    .reset_index()
)


print("\n========================================")
print("CUSTOMER RISK ANALYSIS")
print("========================================")

print(risk_summary)


# ==================================================
# 5. SAVE RESULTS
# ==================================================

df.to_csv(
    "data/customer_risk_analysis.csv",
    index=False
)

risk_summary.to_csv(
    "data/risk_summary.csv",
    index=False
)

print("\nRisk analysis saved successfully!")