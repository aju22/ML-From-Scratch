import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from linear import LinearRegression

# make linear regression sample data
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# plot
# fig = plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
# plt.show()

clf = LinearRegression()
clf.fit(X_train, y_train)

print("Accuracy:", clf.score(X_test, y_test))


# plot line
params = clf.get_params()
print("Params:", params)

w = params['w']
b = params['b']
x_line = np.arange(-3, 4, 1)
y_line = w * x_line + b
fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
plt.plot(x_line, y_line, color="r")
plt.show()