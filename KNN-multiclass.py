import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# Load the dataset iris
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
# print(df.info())

# A simple visualization
sns.relplot(data=df, x="Sepal length", y="Sepal width", hue="Species")
