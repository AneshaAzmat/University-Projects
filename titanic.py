import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")
df = df.drop_duplicates()

# Basic information
print("Records & Features:", df.shape)
print("\nFeature Names:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nTarget: Survived")
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nStatistics:\n", df.describe())

# Graphs
plt.figure(figsize=(12, 4))

# 1. Survival
plt.subplot(1, 3, 1)
s = df["Survived"].value_counts().sort_index()
plt.bar(["Not Survived", "Survived"], [s.get(0, 0), s.get(1, 0)])
plt.title("Passenger Survival")
plt.ylabel("Passengers")

# 2. Gender
plt.subplot(1, 3, 2)
g = df["Sex"].value_counts()
plt.bar(["Male", "Female"], [g.get("male", 0), g.get("female", 0)])
plt.title("Passengers by Gender")
plt.ylabel("Passengers")

# 3. Age Distribution
plt.subplot(1, 3, 3)
plt.hist(df["Age"].dropna(), bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Passengers")

plt.tight_layout()
plt.show()

# Input features and target
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]

# Numerical and categorical features
num = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
cat = ["Sex", "Embarked"]

# Preprocessing
pre = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), num),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ]), cat)
])

# 80:20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
X_train = pre.fit_transform(X_train)
X_test = pre.transform(X_test)

print("\nFinal Training Shape:", X_train.shape)
print("Final Testing Shape:", X_test.shape)
print("\nPreprocessing Completed Successfully.")