import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


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

print(df.head())

sns.relplot(data=df, x="Sepal length", y="Sepal width", hue="Species")
# plt.show(block=True)

# Train test split
X = df[df.columns[:2]]
y = df[df.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train[:5])

# Train the model
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print(
    "Classification Report:",
)
print(result1)
result2 = accuracy_score(y_test, y_pred)
print("Accuracy:", result2)
