import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Загрузка данных
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Однократное кодирование целевых переменных
y = keras.utils.to_categorical(y, num_classes=3)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание модели нейронной сети
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(10, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 класса
])

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=1)

# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Потери: {loss}, Точность: {accuracy}')







