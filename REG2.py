import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Загрузка и предварительная обработка данных
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()
X = california.data  # Измените на california
y = california.target  # Измените на california

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание модели нейронной сети для регрессии
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Выходной слой для регрессии
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)  # verbose=0 для отключения вывода

# Оценка модели
mse = model.evaluate(X_test, y_test)
print(f'Mean Squared Error: {mse:.2f}')


