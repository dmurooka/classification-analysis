import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


iris = datasets.load_iris()

df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]])
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

print(df.head())
print(df.info())

df = df[df["species"] != 0]

print(df.info())

sns.relplot(x="sepal_length", y="sepal_width", hue="species", data=df)

# Train-Test Split
X = df[df.columns[:2]]
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train[:5])

# Train the model
classifier = LogisticRegression().fit(X, y)
classifier.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = classifier.predict(X_test)

result = confusion_matrix(y_test, y_pred)
print("Confusiont Matrix:")
print(result)
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
