import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("iris.csv")

X = df.iloc[:, :4]
y = df.iloc[:, 4]

# Supervised
plt.figure(figsize=(8, 6))

for flower in y.unique():
    d = df[y == flower]
    plt.scatter(d.iloc[:, 2], d.iloc[:, 3], s=50, label=flower)

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Supervised Learning - Iris Classification")
plt.legend()
plt.grid()

# Unsupervised
df["Cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X.iloc[:, 2], X.iloc[:, 3], c=df["Cluster"], s=50)

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Unsupervised Learning - K-Means Clustering")
plt.grid()

plt.show()