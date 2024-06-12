import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

iris = datasets.load_iris()

df = pd.DataFrame(data=np.c_[iris["data"], iris["target"]])
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

print(df.head())
print(df.info())

# Train-Test Split
from sklearn.model_selection import train_test_split

X = df[df.columns[:-1]]
y = df[df.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train[:5])

# Train the model
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluate the model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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
