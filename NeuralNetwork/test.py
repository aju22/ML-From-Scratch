import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from nn import NN

# Load data
digits = load_digits()

X = digits.data

# One-hot encode labels
y = np.zeros((len(X), 10))
for i, label in enumerate(digits.target):
    y[i, label] = 1.0

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

##Reshape Data
X_train = X_train.reshape((X_train.shape[0], 64, 1))
X_test = X_test.reshape((X_test.shape[0], 64, 1))
y_train = y_train.reshape((y_train.shape[0], 10, 1))
y_test = y_test.reshape((y_test.shape[0], 10, 1))


##Train Model
model = NN(features=[64, 16, 10], lr=0.1)
model.show()
model.train(X_train, y_train, epochs=100, loss="bce", verbose=True)

##Predict
y_pred = model.predict(X_test)

##Accuracy
accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print(f"Accuracy: {accuracy*100:.2f}%")

##Plot with predicted labels
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='binary')
    ax.set(title = f"Predicted Label: {np.argmax(y_pred[i])}")
plt.show()







