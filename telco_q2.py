import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load and clean
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Feature Engineering
df["AverageMonthlySpending"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

services = ["PhoneService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies"]
df["ServiceCount"] = (df[services] == "Yes").sum(axis=1)

df["CustomerSpendingCategory"] = pd.cut(
    df["MonthlyCharges"],
    bins=[0, 40, 80, 200],
    labels=["Low", "Medium", "High"]
)

# Features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})

# Encoding and scaling
cat = X.select_dtypes(include=["object", "category"]).columns
num = X.select_dtypes(include=["int64", "float64"]).columns

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    )
}

results = []
matrices = {}

# Train and evaluate
for name, model in models.items():

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    results.append([
        name,
        accuracy_score(y_test, pred),
        precision_score(y_test, pred),
        recall_score(y_test, pred),
        f1_score(y_test, pred)
    ])

    matrices[name] = confusion_matrix(y_test, pred)

# Results
result_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
)

print("\nMODEL COMPARISON")
print(result_df.round(3))

# Confusion matrices
print("\nCONFUSION MATRICES")
for name, matrix in matrices.items():
    print("\n" + name)
    print(matrix)

# Performance graph
result_df.set_index("Model").plot(kind="bar", figsize=(9, 5))
plt.title("Logistic Regression vs Random Forest")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Confusion matrix graphs
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

for i, (name, matrix) in enumerate(matrices.items()):
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax[i])
    ax[i].set_title(name)
    ax[i].set_xlabel("Predicted")
    ax[i].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# Better model based on F1-score
best = result_df.loc[result_df["F1-Score"].idxmax(), "Model"]

print("\nBETTER MODEL:", best)
print("The model with the higher F1-score performs better.")

# Feature engineering effect
print("\nFEATURE ENGINEERING EFFECT")
print("AverageMonthlySpending = average monthly spending.")
print("ServiceCount = number of subscribed services.")
print("CustomerSpendingCategory = Low, Medium and High spending.")