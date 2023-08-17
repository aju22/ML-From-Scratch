from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from knn import KNN

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)

clf = KNN()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(f"Predictions: {predictions}")
print(f"Truth: {y_test}")
print()
accuracy = (np.sum(predictions == y_test) / len(y_test)) * 100
print(f"Accuracy: {accuracy}%")
