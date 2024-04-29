import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn import datasets


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

print(df.head())

# Replaces the original DataFrame with the new one that excludes row where "Species" is not 0
df = df[df["Species"] != 0]

print(df.head)
