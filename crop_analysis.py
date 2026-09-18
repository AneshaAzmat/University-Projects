import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Crop_recommendation.csv")

print("Records:",len(df)," Features:",len(df.columns))
print("Columns:",list(df.columns))
print("\nTypes:\n",df.dtypes)
print("\nStatistics:\n",df[["N","P","K"]].describe())
print("\nCrops:",df.label.unique().tolist())

def check(n,p,k):
    if n+p+k < 150: return "Low"
    elif n+p+k < 300: return "Medium"
    else: return "High"

df["condition"] = df.apply(lambda x: check(x.N,x.P,x.K),axis=1)

for x in ["Low","Medium","High"]:
    print(x,(df.condition==x).sum())

print("\nFirst 5 results:")
for x in df.condition.tolist()[:5]:
    print(x)

print("\nRice average:")
print(df[df.label=="rice"][["N","P","K","temperature","humidity","ph","rainfall"]].mean())

# Graphs
df.label.value_counts().plot(kind="bar",figsize=(10,4))
plt.title("Crop Distribution")
plt.tight_layout()
plt.show()

sns.heatmap(df[["N","P","K","temperature","humidity","ph","rainfall"]].corr(),annot=True)
plt.title("Feature Correlation")
plt.show()

# Error example:
# print(df["N"].mean(
print("\nCorrected:",df.N.mean())

print("\nObservations:")
print("1. Crops have different nutrient and environmental requirements.")
print("2. Soil/environment data can support better crop selection.")