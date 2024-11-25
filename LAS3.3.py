import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDat

# Загрузка данных
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Преобразование в тензоры
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

# Создание DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

# Определение модели
class IrisNN(nn.Module):
def __init__(self):
super(IrisNN, self).__init__()
self.fc1 = nn.Linear(4, 10)


self.fc2 = nn.Linear(10, 10)
self.fc3 = nn.Linear(10, 3) # 3 класса

def forward(self, x):
x = torch.relu(self.fc1(x))
x = torch.relu(self.fc2(x))
x = self.fc3(x)
return x

# Инициализация модели, функции потерь и оптимизатора
model = IrisNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Обучение модели
for epoch in range(100): # 100 эпох
for data in train_loader:
optimizer.zero_grad()
inputs, labels = data
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()

# Оценка модели
with torch.no_grad():
y_test_tensor = torch.LongTensor(y_test)
test_outputs = model(X_test_tensor)
_, predicted = torch.max(test_outputs, 1)

accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
print(f'Accuracy: {accuracy:.2f}')

