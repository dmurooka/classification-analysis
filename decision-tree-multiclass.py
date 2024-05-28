import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt


iris = datasets.load_iris()

df = pd.DataFrame(
    {
        "Sepal length": iris.data[:, 0],
        "Sepal width": iris.data[:, 1],
        "Petal legnth": iris.data[:, 2],
        "Petal width": iris.data[:, 3],
        "Species": iris.target,
    }
)

# print(df.head())

sns.relplot(data=df, x="Sepal length", y="Sepal width", hue="Species")

# Train-test split
X = df[df.columns[:4]]
y = df[df.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)
print(y_train)

# Train model
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Evaluate model
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Visualize the result
test_presentation = tree.export_text(classifier)
print(test_presentation)

fig = plt.figure(figsize=(10, 8))
_ = tree.plot_tree(
    classifier,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
)

# Tune the model
classifier = DecisionTreeClassifier(criterion="entropy")
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

text_representation = tree.export_text(classifier)
print(text_representation)

fig = plt.figure(figsize=(10, 8))
_ = tree.plot_tree(
    classifier,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
)

classifier = DecisionTreeClassifier(max_depth=1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

text_representation = tree.export_text(classifier)
print(text_representation)

fig = plt.figure()
_ = tree.plot_tree(
    classifier,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
)

classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

text_representation = tree.export_text(classifier)
print(text_representation)

fig = plt.figure(figsize=(10, 8))
_ = tree.plot_tree(
    classifier,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
)

classifier = DecisionTreeClassifier(max_depth=3)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

text_representation = tree.export_text(classifier)
print(text_representation)


def tree_depth_tuning(d):
    classifier = DecisionTreeClassifier(max_depth=d)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


tree_results = pd.DataFrame({"D": np.arange(1, 10)})

tree_results["Accuracy"] = tree_results["D"].apply(tree_depth_tuning)
print(tree_results)
