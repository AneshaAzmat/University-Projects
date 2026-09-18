import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Crop_recommendation.csv")

# Rule-based decision using N, P, K
def decision(n, p, k):
    if n >= 50 and p >= 30 and k >= 30:
        return "High"
    elif n >= 20 and p >= 15 and k >= 15:
        return "Medium"
    else:
        return "Low"

# Process 10 records
results = []

for i in range(10):
    r = df.iloc[i]
    d = decision(r.N, r.P, r.K)
    results.append(d)
    print(i+1, "N:",r.N, "P:",r.P, "K:",r.K,
          "Decision:",d, "Crop:",r.label)

# Evaluation
print("\nDecision Counts:")
print(pd.Series(results).value_counts())

# Simple graph
pd.Series(results).value_counts().plot(kind="bar")
plt.title("Rule-Based Agricultural Decisions")
plt.xlabel("Decision")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig("decision_graph.png", dpi=300)
plt.show()

# Limitations
print("\nLimitations:")
print("1. Fixed rules may not work well for all crops and conditions.")
print("2. Rules cannot learn complex patterns from the dataset.")
print("3. ML models such as Decision Tree, Random Forest and XGBoost can learn patterns automatically.")