import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

# Basic information
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nData Types:\n", df.dtypes)
print("\nTarget: Species")
print("\nStatistics:\n", df.describe())

# NumPy analysis
f = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[f].values

print("\nMean:", np.mean(X, axis=0))
print("Min:", np.min(X, axis=0))
print("Max:", np.max(X, axis=0))
print("Std:", np.std(X, axis=0))

# Pandas analysis
print("\nSelected Columns:\n", df[["PetalLengthCm", "PetalWidthCm", "Species"]].head())
print("\nPetal Length > 5:\n", df[df["PetalLengthCm"] > 5])
print("\nGroup Mean:\n", df.groupby("Species")[f].mean())

# Graph 1 - Setosa
s = df[df["Species"].str.contains("setosa", case=False)]
plt.figure("Iris Setosa")
plt.scatter(s["SepalLengthCm"], s["SepalWidthCm"], s=70)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Setosa")

# Graph 2 - Versicolor
v = df[df["Species"].str.contains("versicolor", case=False)]
plt.figure("Iris Versicolor")
plt.plot(v["PetalLengthCm"], v["PetalWidthCm"], "o-")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Iris Versicolor")

# Graph 3 - Virginica
g = df[df["Species"].str.contains("virginica", case=False)]
plt.figure("Iris Virginica")
plt.bar(f, g[f].mean())
plt.xticks(rotation=20)
plt.ylabel("Average")
plt.title("Iris Virginica")

plt.show()