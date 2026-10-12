
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

print("Missing values:", 0)

# Split and preprocess
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model and predict
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

# Evaluation
metrics = [accuracy_score(y_test, pred), precision_score(y_test, pred),
           recall_score(y_test, pred), f1_score(y_test, pred)]

print("\nAccuracy :", round(metrics[0], 4))
print("Precision:", round(metrics[1], 4))
print("Recall   :", round(metrics[2], 4))
print("F1-score :", round(metrics[3], 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# All graphs in one window
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Breast Cancer Classification - Logistic Regression",
             fontsize=14, fontweight="bold")

sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d",
            cmap="Blues", xticklabels=data.target_names,
            yticklabels=data.target_names, ax=ax[0, 0])
ax[0, 0].set_title("Confusion Matrix")
ax[0, 0].set_xlabel("Predicted")
ax[0, 0].set_ylabel("Actual")

ax[0, 1].bar(["Accuracy", "Precision", "Recall", "F1-score"],
             metrics, color="teal")
ax[0, 1].set_title("Performance Metrics")
ax[0, 1].set_ylim(0, 1.05)
ax[0, 1].tick_params(axis="x", rotation=15)

fpr, tpr, _ = roc_curve(y_test, prob)
ax[1, 0].plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
ax[1, 0].plot([0, 1], [0, 1], "--", color="gray")
ax[1, 0].set_title("ROC Curve")
ax[1, 0].set_xlabel("False Positive Rate")
ax[1, 0].set_ylabel("True Positive Rate")
ax[1, 0].legend()

sns.countplot(x=data.target_names[pred], hue=data.target_names[pred],
              legend=False, ax=ax[1, 1])
ax[1, 1].set_title("Predicted Tumor Classes")
ax[1, 1].set_xlabel("Tumor Type")
ax[1, 1].set_ylabel("Number of Predictions")

plt.tight_layout()
plt.show()
