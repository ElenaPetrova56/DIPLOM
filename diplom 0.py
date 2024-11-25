import matplotlib.pyplot as plt

# Пример значений MSE
mse_sklearn = 0.2
mse_tensorflow = 0.15
mse_pytorch = 0.1

# Пример данных для графика
libraries = ['Scikit-learn', 'TensorFlow', 'PyTorch']
mse_results = [mse_sklearn, mse_tensorflow, mse_pytorch]

plt.bar(libraries, mse_results)
plt.ylabel('Mean Squared Error')
plt.title('Сравнение MSE для различных библиотек')
plt.show()




