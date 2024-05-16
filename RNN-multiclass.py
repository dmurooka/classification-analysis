import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Setup
iris = datasets.load_iris()

df = pd.DataFrame(
    {
        "Sepal length": iris.data[:, 0],
        "Septal width": iris.data[:, 1],
        "Petal length": iris.data[:, 2],
        "Petal width": iris.data[:, 3],
        "Species": iris.target,
    }
)

print(df.head())

X = df[df.columns[:4]]
y = df[df.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print(X_train[:5])

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train[:5])

# R-Nearest Neighbors
r = 1

classifier = RadiusNeighborsClassifier(radius=r)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:")
print(result1)
result2 = accuracy_score(y_test, y_pred)
print("Accuracy:", result2)


# Best r
def rnn_tuning(r):
    classifier = RadiusNeighborsClassifier(radius=r)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


print(rnn_tuning(2))

rnn_results = pd.DataFrame({"R": np.arange(1, 10, 0.5)})
print(rnn_results["R"])

rnn_results["Accuracy"] = rnn_results["R"].apply(rnn_tuning)

print(rnn_results)


# Optimize weights
def rnn_tuning_uniform(r):
    classifier = RadiusNeighborsClassifier(radius=r, weights="uniform")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def rnn_tunin_distance(k):
    classifier = RadiusNeighborsClassifier(radius=r, weights="distance")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


rnn_results["Uniform"] = rnn_results["R"].apply(rnn_tuning_uniform)
rnn_results["Distance"] = rnn_results["R"].apply(rnn_tunin_distance)

print(rnn_results)
