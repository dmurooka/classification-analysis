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
        "Species": iris.target,
    }
)

print(df.head())

df = df[df["Species"] != 0]

print(df.info())

sns.relplot(data=df, x="Sepal length", y="Sepal width", hue="Species")

# Train-test split
X = df[df.columns[:4]]
y = df[df.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print(X_train)
