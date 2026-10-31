
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import *

# 1. Load Dataset
df = pd.read_csv("iris.csv")

print("First Five Rows:\n", df.head())
print("\nDataset Statistics:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# 2. Select Features and Target
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

# 3. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train Decision Tree Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Generate Predictions
y_pred = model.predict(X_test)

print("\nPredictions:\n", y_pred)

# 6. Evaluation Metrics
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))

print("\nClassification Report:\n",
      classification_report(y_test, y_pred))

# 7. First Four Graphs in One Window
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# Graph 1: Species Distribution
sns.countplot(data=df, x="Species", ax=ax[0, 0])
ax[0, 0].set_title("Species Distribution")

# Graph 2: Petal Measurements
sns.scatterplot(data=df, x="PetalLengthCm",
                y="PetalWidthCm", hue="Species", ax=ax[0, 1])
ax[0, 1].set_title("Petal Measurements")

# Graph 3: Correlation Heatmap
sns.heatmap(X.corr(), annot=True, cmap="coolwarm", ax=ax[1, 0])
ax[1, 0].set_title("Feature Correlation")

# Graph 4: Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_, ax=ax[1, 1])
ax[1, 1].set_title("Confusion Matrix")
ax[1, 1].set_xlabel("Predicted")
ax[1, 1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# 8. Decision Tree in Separate Window
plt.figure(figsize=(16, 9))

plot_tree(model, feature_names=X.columns,
          class_names=model.classes_,
          filled=True, rounded=True)

plt.title("Decision Tree Classification")
plt.show()
