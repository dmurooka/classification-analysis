import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


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

X = df[df.columns[:4]]
y = df[df.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print(X_train[:5])

# Scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train[:5])

# K-Nearest Neighbors
k = 1

classifier = KNeighborsClassifier(n_neighbors=k)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)

result1 = classification_report(y_test, y_pred)
print("Classification Report:")
print(result1)

result2 = accuracy_score(y_test, y_pred)
print("Accuract", result2)

# Best K


def knn_tuning(k):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


print("KNN Tuning", knn_tuning(1))
print("KNN Tuning", knn_tuning(5))

knn_results = pd.DataFrame({"K": np.arange(1, len(X_train), 5)})

knn_results["Accuracy"] = knn_results["K"].apply(knn_tuning)
print(knn_results)


# Optimize weights
def knn_tuning_uniform(k):
    classifier = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def knn_tuning_distance(k):
    classifier = KNeighborsClassifier(n_neighbors=k, weights="distance")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


knn_results["Uniform"] = knn_results["K"].apply(knn_tuning_uniform)
knn_results["Distance"] = knn_results["K"].apply(knn_tuning_distance)
print(knn_results)
