import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()

df = pd.DataFrame(
    {
        "Sepal length": iris.data[:, 0],
        "Sepal width": iris.data[:, 1],
        "Petal length": iris.data[:, 2],
        "Petal width": iris.data[:, 3],
        "Species": iris.target,
    }
)

# print(df.head())

# Replaces the original DataFrame with the new one that excludes row where "Species" is not 0
df = df[df["Species"] != 0]

# print(df.head)

sns.relplot(data=df, x="Sepal length", y="Sepal width", hue="Species")

# Train-Test Split
X = df[df.columns[:2]]
y = df[df.columns[-1]]

X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.5)
print(X_train[:5])
