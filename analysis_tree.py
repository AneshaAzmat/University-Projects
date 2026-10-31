
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import *

# Smaller graph settings
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 8


# ==========================================
# 1. DECISION TREE - BREAST CANCER
# ==========================================

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("\n========== BREAST CANCER CLASSIFICATION ==========")
print("Dataset: Breast Cancer Wisconsin")
print("Algorithm: Decision Tree Classifier")
print("\nFirst Five Records:\n", X.head())
print("\nDataset Statistical Summary:\n", X.describe())
print("\nTotal Missing Values:", X.isnull().sum().sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
p = dt.predict(X_test)

print("\n--- Model Evaluation Results ---")
print("Accuracy:", accuracy_score(y_test, p))
print("Precision:", precision_score(y_test, p, average="weighted"))
print("Recall:", recall_score(y_test, p, average="weighted"))
print("F1 Score:", f1_score(y_test, p, average="weighted"))
print("\nDetailed Classification Report:")
print(classification_report(y_test, p, target_names=data.target_names))

fig, ax = plt.subplots(2, 2, figsize=(8, 5))

sns.countplot(x=pd.Series(y).map({0: "Malignant", 1: "Benign"}), ax=ax[0, 0])
ax[0, 0].set_title("Cancer Class Distribution")

sns.scatterplot(x=X["mean radius"], y=X["mean texture"], hue=y, ax=ax[0, 1])
ax[0, 1].set_title("Mean Radius vs Mean Texture")
ax[0, 1].set_xlabel("Mean Radius")
ax[0, 1].set_ylabel("Mean Texture")

sns.heatmap(confusion_matrix(y_test, p), annot=True, fmt="d", cmap="Blues",
            xticklabels=data.target_names, yticklabels=data.target_names, ax=ax[1, 0])
ax[1, 0].set_title("Confusion Matrix")
ax[1, 0].set_xlabel("Predicted Class")
ax[1, 0].set_ylabel("Actual Class")

sns.barplot(x=dt.feature_importances_, y=X.columns, ax=ax[1, 1])
ax[1, 1].set_title("Feature Importance")
ax[1, 1].set_xlabel("Importance Score")
ax[1, 1].set_ylabel("Features")

plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plot_tree(dt, feature_names=X.columns, class_names=data.target_names,
          filled=True, max_depth=3, fontsize=7)
plt.title("Decision Tree Model Structure")
plt.tight_layout()
plt.show()


# ==========================================
# 2. RANDOM FOREST - WINE
# ==========================================

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("\n========== WINE CLASSIFICATION ==========")
print("Dataset: Wine Recognition")
print("Algorithm: Random Forest Classifier")
print("\nFirst Five Records:\n", X.head())
print("\nDataset Statistical Summary:\n", X.describe())
print("\nTotal Missing Values:", X.isnull().sum().sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
p = rf.predict(X_test)

print("\n--- Model Evaluation Results ---")
print("Accuracy:", accuracy_score(y_test, p))
print("Precision:", precision_score(y_test, p, average="weighted"))
print("Recall:", recall_score(y_test, p, average="weighted"))
print("F1 Score:", f1_score(y_test, p, average="weighted"))
print("\nDetailed Classification Report:")
print(classification_report(y_test, p, target_names=data.target_names))

fig, ax = plt.subplots(2, 2, figsize=(8, 5))

sns.countplot(x=pd.Series(y).map({0: "Class 0", 1: "Class 1", 2: "Class 2"}), ax=ax[0, 0])
ax[0, 0].set_title("Wine Class Distribution")

sns.scatterplot(x=X["alcohol"], y=X["color_intensity"], hue=y, ax=ax[0, 1])
ax[0, 1].set_title("Alcohol vs Color Intensity")
ax[0, 1].set_xlabel("Alcohol")
ax[0, 1].set_ylabel("Color Intensity")

sns.heatmap(confusion_matrix(y_test, p), annot=True, fmt="d", cmap="Blues",
            xticklabels=data.target_names, yticklabels=data.target_names, ax=ax[1, 0])
ax[1, 0].set_title("Confusion Matrix")
ax[1, 0].set_xlabel("Predicted Class")
ax[1, 0].set_ylabel("Actual Class")

sns.barplot(x=rf.feature_importances_, y=X.columns, ax=ax[1, 1])
ax[1, 1].set_title("Feature Importance")
ax[1, 1].set_xlabel("Importance Score")
ax[1, 1].set_ylabel("Features")

plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plot_tree(rf.estimators_[0], feature_names=X.columns,
          class_names=data.target_names, filled=True, max_depth=3, fontsize=7)
plt.title("Random Forest - First Decision Tree")
plt.tight_layout()
plt.show()


# ==========================================
# 3. GRADIENT BOOSTING - DIABETES
# ==========================================

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("\n========== DIABETES PREDICTION ==========")
print("Dataset: Diabetes")
print("Algorithm: Gradient Boosting Regressor")
print("\nFirst Five Records:\n", X.head())
print("\nDataset Statistical Summary:\n", X.describe())
print("\nTotal Missing Values:", X.isnull().sum().sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
p = gb.predict(X_test)

print("\n--- Model Evaluation Results ---")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, p))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, p))
print("R-Squared Score (R2):", r2_score(y_test, p))

fig, ax = plt.subplots(2, 2, figsize=(8, 5))

sns.histplot(y, kde=True, ax=ax[0, 0])
ax[0, 0].set_title("Diabetes Target Distribution")
ax[0, 0].set_xlabel("Disease Progression")

sns.heatmap(X.corr(), cmap="coolwarm", ax=ax[0, 1])
ax[0, 1].set_title("Feature Correlation")

sns.scatterplot(x=y_test, y=p, ax=ax[1, 0])
ax[1, 0].set_title("Actual vs Predicted Values")
ax[1, 0].set_xlabel("Actual Values")
ax[1, 0].set_ylabel("Predicted Values")

sns.barplot(x=gb.feature_importances_, y=X.columns, ax=ax[1, 1])
ax[1, 1].set_title("Feature Importance")
ax[1, 1].set_xlabel("Importance Score")
ax[1, 1].set_ylabel("Features")

plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plot_tree(gb.estimators_[0, 0], feature_names=X.columns,
          filled=True, max_depth=3, fontsize=7)
plt.title("Gradient Boosting - First Regression Tree")
plt.tight_layout()
plt.show()

print("\n========== ANALYSIS COMPLETED SUCCESSFULLY ==========")
print("All Three Machine Learning Models Have Been Evaluated.")
