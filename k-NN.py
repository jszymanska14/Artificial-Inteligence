import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_wine

class KNNClassifier:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _compute_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'minkowski':
            p = 3
            return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)
        else:
            raise ValueError("Nieobsługiwana metryka. Wybierz: 'euclidean', 'manhattan' lub 'minkowski'.")

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = []
        for x in X_test:
            distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)

    def confusion_matrix_custom(self, y_true, y_pred):
        unique_classes = np.unique(y_true)
        matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
        for i, actual in enumerate(unique_classes):
            for j, predicted in enumerate(unique_classes):
                matrix[i, j] = np.sum((y_true == actual) & (y_pred == predicted))
        return matrix

def accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

def precision(y_true, y_pred, average='macro'):
    unique_classes = np.unique(y_true)
    precisions = []
    for cls in unique_classes:
        TP = np.sum((y_pred == cls) & (y_true == cls))
        FP = np.sum((y_pred == cls) & (y_true != cls))
        precisions.append(TP / (TP + FP) if (TP + FP) > 0 else 0)
    if average == 'macro':
        return np.mean(precisions)
    elif average == 'micro':
        TP_total = np.sum(y_pred == y_true)
        FP_total = np.sum(y_pred != y_true)
        return TP_total / (TP_total + FP_total)
    else:
        raise ValueError("Obsługiwane wartości average: 'macro', 'micro'")

def recall(y_true, y_pred, average='macro'):
    unique_classes = np.unique(y_true)
    recalls = []
    for cls in unique_classes:
        TP = np.sum((y_pred == cls) & (y_true == cls))
        FN = np.sum((y_pred != cls) & (y_true == cls))
        recalls.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
    if average == 'macro':
        return np.mean(recalls)
    elif average == 'micro':
        TP_total = np.sum(y_pred == y_true)
        FN_total = np.sum(y_pred != y_true)
        return TP_total / (TP_total + FN_total)
    else:
        raise ValueError("Obsługiwane wartości average: 'macro', 'micro'")

def f1_score(y_true, y_pred, average='macro'):
    prec = precision(y_true, y_pred, average)
    rec = recall(y_true, y_pred, average)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0


data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def optimize_k(X_train, X_val, y_train, y_val, k_values):
    errors = []
    for k in k_values:
        knn = KNNClassifier(k=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        errors.append(1 - accuracy(y_val, y_pred))
    return k_values, errors

k_values = list(range(1, 21))
k_results, k_errors = optimize_k(X_train, X_val, y_train, y_val, k_values)
plt.plot(k_values, k_errors, marker='o')
plt.xlabel('Liczba sąsiadów k')
plt.ylabel('Błąd klasyfikacji')
plt.title('Optymalizacja k')
plt.show()


def optimize_p(X_train, X_val, y_train, y_val, p_values):
    scores = []
    for p in p_values:
        knn = KNNClassifier(k=3, metric='minkowski')
        knn.p = p
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        scores.append(accuracy(y_val, y_pred))
    return p_values, scores

p_values = [1, 1.5, 2, 3, 4]
p_results, p_scores = optimize_p(X_train, X_val, y_train, y_val, p_values)
plt.plot(p_values, p_scores, marker='o')
plt.xlabel('Wartość p w metryce Minkowskiego')
plt.ylabel('Dokładność klasyfikacji')
plt.title('Optymalizacja metryki p')
plt.show()
