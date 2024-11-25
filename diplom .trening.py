import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Загрузка данных
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель
model_sklearn = RandomForestClassifier(n_estimators=100)
model_sklearn.fit(X_train, y_train)

# Предсказание и оценка
y_pred = model_sklearn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Scikit-learn accuracy: {accuracy:.2f}')