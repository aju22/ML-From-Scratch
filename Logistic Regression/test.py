import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import matplotlib.pyplot as plt
from logistic import LogisticRegression

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

normalizer = StandardScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train_norm, y_train)

y_pred = clf.predict(X_test_norm)

print(f"Accuracy: {round(clf.accuracy(y_pred, y_test)*100, 3)}\n")
print(f"Model Params:\n\n Weights: {clf.get_param()['w']}\n\n Bias: {clf.get_param()['b']}")
