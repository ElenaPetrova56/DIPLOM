import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import torch
import torch.nn as nn
import torch.optim as optim

# Пример данных
X_train = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])  # Преобразуем в массив NumPy
y_train = np.array([0, 1, 1, 0])  # Преобразуем в массив NumPy

# Scikit-learn
model_sklearn = LogisticRegression()
start_time = time.time()
model_sklearn.fit(X_train, y_train)
sklearn_training_time = time.time() - start_time

# TensorFlow
model_tf = Sequential()
model_tf.add(Input(shape=(2,)))  # Используем Input слой
model_tf.add(Dense(10, activation='relu'))
model_tf.add(Dense(1, activation='sigmoid'))
model_tf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()
model_tf.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
tensorflow_training_time = time.time() - start_time


# PyTorch
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


model_pytorch = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model_pytorch.parameters())

# Преобразуем данные в тензоры
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)

start_time = time.time()
for epoch in range(100):
    for i in range(len(X_train_tensor)):
        inputs = X_train_tensor[i].view(1, -1)
        labels = y_train_tensor[i].view(1, -1)

        optimizer.zero_grad()
        outputs = model_pytorch(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

pytorch_training_time = time.time() - start_time

print(f'Scikit-learn Training Time: {sklearn_training_time:.4f} seconds')
print(f'TensorFlow Training Time: {tensorflow_training_time:.4f} seconds')
print(f'PyTorch Training Time: {pytorch_training_time:.4f} seconds')








