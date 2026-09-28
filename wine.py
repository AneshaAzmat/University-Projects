import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset
df = pd.read_csv("winequality-red.csv")

print("Original Shape:", df.shape)
print("\nFeatures:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nTarget: quality")
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nStatistics:\n", df.describe())

# Remove duplicates
df = df.drop_duplicates()

# Missing values
df = df.fillna(df.median(numeric_only=True))

# Features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Feature ranges
print("\nFeature Ranges:\n", X.max() - X.min())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# StandardScaler Pipeline
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_train_std = pipe.fit_transform(X_train)
X_test_std = pipe.transform(X_test)

# MinMaxScaler
minmax = MinMaxScaler()
X_train_minmax = minmax.fit_transform(X_train)
X_test_minmax = minmax.transform(X_test)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)
print("\nProcessed Training:\n", X_train_std[:5])
print("\nProcessed Testing:\n", X_test_std[:5])

# Compare original and transformed
print("\nOriginal First Row:\n", X_train.iloc[0].values)
print("\nStandardized First Row:\n", X_train_std[0])
print("\nMinMax First Row:\n", X_train_minmax[0])

# Graph 1: Alcohol vs Quality
plt.figure(figsize=(6,4))
plt.scatter(df["alcohol"], df["quality"])
plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.title("Alcohol vs Wine Quality")
plt.show()

# Graph 2: Acidity vs Quality
plt.figure(figsize=(6,4))
plt.scatter(df["fixed acidity"], df["quality"])
plt.xlabel("Fixed Acidity")
plt.ylabel("Quality")
plt.title("Acidity vs Wine Quality")
plt.show()