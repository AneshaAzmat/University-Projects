import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 1-3 Dataset analysis
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nStatistics:\n", df.describe())

# 4. Four visualizations in ONE window
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

sns.countplot(x="Churn", data=df, ax=ax[0,0])
ax[0,0].set_title("Churn Distribution")

sns.histplot(df["tenure"], bins=30, ax=ax[0,1])
ax[0,1].set_title("Tenure Distribution")

sns.boxplot(x="Churn", y="MonthlyCharges", data=df, ax=ax[1,0])
ax[1,0].set_title("Monthly Charges vs Churn")

sns.countplot(x="Contract", hue="Churn", data=df, ax=ax[1,1])
ax[1,1].set_title("Contract vs Churn")
ax[1,1].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.show()

# 5. Correlation
num = df.select_dtypes(include=np.number)
print("\nCorrelation with Churn:\n",
      num.corr()["Churn"].sort_values(ascending=False))

plt.figure(figsize=(8,6))
sns.heatmap(num.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 6-8 Important features and outliers
features = ["tenure", "MonthlyCharges", "TotalCharges",
            "Contract", "InternetService", "PaymentMethod",
            "TechSupport", "OnlineSecurity", "SeniorCitizen"]

print("\nSelected Features:\n", features)

for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
    Q1, Q3 = df[c].quantile([.25, .75])
    IQR = Q3 - Q1
    out = ((df[c] < Q1-1.5*IQR) | (df[c] > Q3+1.5*IQR)).sum()
    print(c, "outliers:", out)

# 9. Models
print("\nLogistic Regression: suitable for binary Churn prediction.")
print("Random Forest: suitable for nonlinear relationships and feature interactions.")