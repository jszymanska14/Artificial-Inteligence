import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine


class KNNClassifier:
    def __init__(self, k=3, p=2):
        self.k = k
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _minkowski_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = []
        for x in X_test:
            distances = [self._minkowski_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)


def accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))


def optimize_k(X_train, X_val, y_train, y_val, k_values):
    errors = []
    for k in k_values:
        knn = KNNClassifier(k=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        errors.append(1 - accuracy(y_val, y_pred))
    return k_values, errors


def optimize_p(X_train, X_val, y_train, y_val, p_values):
    scores = []
    for p in p_values:
        knn = KNNClassifier(k=3, p=p)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        scores.append(accuracy(y_val, y_pred))
    return p_values, scores



data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


k_values = list(range(1, 21))
k_results, k_errors = optimize_k(X_train, X_val, y_train, y_val, k_values)
plt.plot(k_values, k_errors, marker='o')
plt.xlabel('Liczba sąsiadów k')
plt.ylabel('Błąd klasyfikacji')
plt.title('Optymalizacja k')
plt.show()


p_values = [1, 1.5, 2, 3, 4]
p_results, p_scores = optimize_p(X_train, X_val, y_train, y_val, p_values)
plt.plot(p_values, p_scores, marker='o')
plt.xlabel('Wartość p w metryce Minkowskiego')
plt.ylabel('Dokładność klasyfikacji')
plt.title('Optymalizacja p')
plt.show()
