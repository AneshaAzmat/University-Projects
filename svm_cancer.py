
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix)

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Same split as Question 1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train both models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000)),
    "SVM": make_pipeline(StandardScaler(), SVC())
}

scores = []
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = [
        accuracy_score(y_test, pred),
        precision_score(y_test, pred),
        recall_score(y_test, pred),
        f1_score(y_test, pred)
    ]
    scores.append(metrics)

    print("\n", name)
    print("Accuracy, Precision, Recall, F1:", 
          [round(v, 4) for v in metrics])
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

    if name == "SVM":
        sns.heatmap(confusion_matrix(y_test, pred), annot=True,
                    fmt="d", cmap="Blues",
                    xticklabels=data.target_names,
                    yticklabels=data.target_names, ax=ax[1])
        ax[1].set_title("SVM Confusion Matrix")
        ax[1].set_xlabel("Predicted")
        ax[1].set_ylabel("Actual")

# Compare models
labels = ["Accuracy", "Precision", "Recall", "F1-score"]
for i, name in enumerate(models):
    ax[0].plot(labels, scores[i], marker="o", label=name)

ax[0].set_ylim(0.8, 1.01)
ax[0].set_title("Model Performance Comparison")
ax[0].legend()
ax[0].tick_params(axis="x", rotation=15)

print("\nBetter model by accuracy:",
      list(models.keys())[scores[1][0] > scores[0][0]])

plt.tight_layout()
plt.show()
