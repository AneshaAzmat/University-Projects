
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

# 1. Load Dataset
df = pd.read_csv("ai4i2020.csv")

print("Dataset Shape:", df.shape)

# 2. Data Preprocessing
df.drop_duplicates(inplace=True)

df.drop(columns=[
    'UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
], inplace=True)

X = df.drop(columns=['Machine failure'])
y = df['Machine failure']

# Encode categorical column
X = pd.get_dummies(X, columns=['Type'], dtype=int)

# Fix XGBoost invalid feature names
X.columns = X.columns.str.replace('[', '', regex=False)
X.columns = X.columns.str.replace(']', '', regex=False)

print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Records:", df.duplicated().sum())
print("\nFailure Distribution:\n", y.value_counts())

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Handle class imbalance
scale = (y_train == 0).sum() / (y_train == 1).sum()

# 4. Define Three XGBoost Models
models = {
    "Model 1": XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        scale_pos_weight=scale, eval_metric='logloss',
        random_state=42, n_jobs=1
    ),

    "Model 2": XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale, eval_metric='logloss',
        random_state=42, n_jobs=1
    ),

    "Model 3": XGBClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.03,
        scale_pos_weight=scale, eval_metric='logloss',
        random_state=42, n_jobs=1
    )
}

# 5. Five-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

print("\nCROSS VALIDATION RESULTS")

for name, model in models.items():
    scores = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring=['f1', 'recall'],
        n_jobs=1
    )

    mean_f1 = scores['test_f1'].mean()
    mean_recall = scores['test_recall'].mean()

    results.append({
        'Configuration': name,
        'F1 Score': mean_f1,
        'Recall': mean_recall
    })

    print(name, "| F1:", round(mean_f1, 4),
          "| Recall:", round(mean_recall, 4))

# 6. Select Best Model
results_df = pd.DataFrame(results)

best_name = results_df.loc[
    results_df['F1 Score'].idxmax(), 'Configuration'
]

best_model = models[best_name]

print("\nBest Model:", best_name)

# 7. Train Best Model
best_model.fit(X_train, y_train)

# 8. Predictions
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# 9. Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print("\nFINAL MODEL EVALUATION")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(auc, 4))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)
print("Missed Failures:", fn)

# 11. False Negative Analysis
false_negatives = X_test[(y_test == 1) & (y_pred == 0)]
print("\nTotal Missed Failures:", len(false_negatives))

# 12. Model Comparison
print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

# 13. Save Model
joblib.dump(best_model, "xgboost_predictive_maintenance_model.pkl")
print("\nModel Saved Successfully!")

# 14. Small Graphs in One Window
fig, ax = plt.subplots(2, 3, figsize=(11, 7))

# Failure Distribution
sns.countplot(x=y, ax=ax[0, 0])
ax[0, 0].set_title("Machine Failure Distribution")
ax[0, 0].set_xlabel("Machine Failure")

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0, 1])
ax[0, 1].set_title("Confusion Matrix")
ax[0, 1].set_xlabel("Predicted")
ax[0, 1].set_ylabel("Actual")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
ax[0, 2].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
ax[0, 2].plot([0, 1], [0, 1], linestyle='--')
ax[0, 2].set_title("ROC Curve")
ax[0, 2].legend()

# Feature Importance
importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False).head(10)

importance.sort_values().plot(kind='barh', ax=ax[1, 0])
ax[1, 0].set_title("Top 10 Feature Importance")

# Model Comparison
results_df.set_index('Configuration')[
    ['F1 Score', 'Recall']
].plot(kind='bar', ax=ax[1, 1])

ax[1, 1].set_title("Model Comparison")
ax[1, 1].set_ylim(0, 1)

# Hide unused graph
ax[1, 2].axis('off')

plt.tight_layout()
plt.show()
