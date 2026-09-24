import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_excel("data/telco_churn.xlsx")


# ==========================================
# 2. Basic Dataset Information
# ==========================================

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Information:")
df.info()


# ==========================================
# 3. Target Variable Analysis
# ==========================================

print("\nChurn Distribution:")
print(df["Churn Label"].value_counts())

print("\nChurn Percentage:")
print(
    df["Churn Label"].value_counts(normalize=True) * 100
)


# ==========================================
# 4. Missing Values
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 5. Unique Values
# ==========================================

print("\nNumber of Unique Values:")
print(df.nunique())


# ==========================================
# 6. Total Charges Data Cleaning
# ==========================================

print("\nTotal Charges Data Type Before Conversion:")
print(df["Total Charges"].dtype)

print("\nBlank Total Charges:")
print(
    (df["Total Charges"].astype(str).str.strip() == "").sum()
)


# Convert Total Charges to numeric
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)


print("\nTotal Charges Data Type After Conversion:")
print(df["Total Charges"].dtype)

print("\nMissing Total Charges After Conversion:")
print(df["Total Charges"].isna().sum())

print("\nRows With Missing Total Charges:")
print(
    df[
        df["Total Charges"].isna()
    ][
        [
            "Tenure Months",
            "Monthly Charges",
            "Total Charges"
        ]
    ]
)


# ==========================================
# 7. Total Charges Statistics
# ==========================================

print("\nTotal Charges Descriptive Statistics:")
print(df["Total Charges"].describe())


# ==========================================
# 8. Churn by Contract
# ==========================================

print("\nChurn by Contract:")

contract_churn = (
    pd.crosstab(
        df["Contract"],
        df["Churn Label"],
        normalize="index"
    ) * 100
)

print(contract_churn)


# ==========================================
# 9. Churn by Internet Service
# ==========================================

print("\nChurn by Internet Service:")

internet_churn = (
    pd.crosstab(
        df["Internet Service"],
        df["Churn Label"],
        normalize="index"
    ) * 100
)

print(internet_churn)


# ==========================================
# 10. Churn by Tech Support
# ==========================================

print("\nChurn by Tech Support:")

support_churn = (
    pd.crosstab(
        df["Tech Support"],
        df["Churn Label"],
        normalize="index"
    ) * 100
)

print(support_churn)


# ==========================================
# 11. Churn by Tenure
# ==========================================

df["Tenure Group"] = pd.cut(
    df["Tenure Months"],
    bins=[-1, 12, 24, 48, 72],
    labels=[
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49-72 months"
    ]
)

print("\nChurn by Tenure:")

tenure_churn = (
    pd.crosstab(
        df["Tenure Group"],
        df["Churn Label"],
        normalize="index"
    ) * 100
)

print(tenure_churn)


# ==========================================
# 12. Monthly Charges Analysis
# ==========================================

print("\nMonthly Charges Statistics:")
print(df["Monthly Charges"].describe())

print("\nMonthly Charges by Churn:")
print(
    df.groupby("Churn Label")[
        "Monthly Charges"
    ].describe()
)


# ==========================================
# 13. Monthly Charges Visualization
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Churn Label",
    y="Monthly Charges",
    data=df
)

plt.title("Monthly Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.show()

