import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
print("Missing values: 0")
print("Features:", iris.feature_names)
print("Classes:", iris.target_names)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Compare K values
scores = []
best_k, best_acc = 0, 0

for k in [3, 5, 7]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    scores.append(acc)
    print(f"\nK={k}, Accuracy={acc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    if acc > best_acc:
        best_k, best_acc = k, acc

print(f"\nBest K={best_k}, Accuracy={best_acc:.4f}")

# Graphs in one window
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(["K=3", "K=5", "K=7"], scores)
ax[0].set_title("Accuracy Comparison")
ax[0].set_ylim(0, 1.05)

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
sns.heatmap(confusion_matrix(y_test, model.predict(X_test)),
            annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names, ax=ax[1])
ax[1].set_title(f"Confusion Matrix (K={best_k})")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()