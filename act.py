import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función de activación sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Función para cargar el conjunto de datos desde el archivo CSV
def load_dataset(file_path):
    dataset = pd.read_csv(file_path, header=None)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return X, y

# Clase para la red neuronal multicapa
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.initialize_weights()

    def initialize_weights(self):
        for i in range(1, len(self.layers)):
            # Inicializar pesos aleatoriamente con valores entre -1 y 1
            w = 2 * np.random.rand(self.layers[i - 1], self.layers[i]) - 1
            self.weights.append(w)

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            for i in range(len(X)):
                # Propagación hacia adelante
                activations = [X[i]]
                for j in range(len(self.weights)):
                    z = np.dot(activations[-1], self.weights[j])
                    a = sigmoid(z)
                    activations.append(a)

                # Calcular el error y la derivada del error
                error = y[i] - activations[-1]
                deltas = [error * sigmoid_derivative(activations[-1])]

                # Retropropagación del error
                for j in range(len(self.weights) - 1, 0, -1):
                    error = deltas[-1].dot(self.weights[j].T)
                    delta = error * sigmoid_derivative(activations[j])
                    deltas.append(delta)

                # Actualizar pesos
                deltas.reverse()
                for j in range(len(self.weights)):
                    self.weights[j] += learning_rate * np.outer(activations[j], deltas[j])

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            # Propagación hacia adelante
            activations = [X[i]]
            for j in range(len(self.weights)):
                z = np.dot(activations[-1], self.weights[j])
                a = sigmoid(z)
                activations.append(a)
            predictions.append(activations[-1])

        return np.array(predictions)

# Función para graficar los resultados en 2D
def plot_results(X, y, predictions):
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', label='Real')
    plt.scatter(X[:, 0], X[:, 1], c=predictions, marker='x', label='Predicción')
    plt.legend()
    plt.show()

# Archivo CSV con los datos
dataset_file = 'concentlite.csv'

# Cargar el conjunto de datos
X, y = load_dataset(dataset_file)

# Definir la arquitectura de la red neuronal 
input_size = X.shape[1]
output_size = 1
hidden_layer_size = 1000
layers = [input_size, hidden_layer_size, output_size]

# Crear la red neuronal
nn = NeuralNetwork(layers)

# Entrenar la red neuronal
learning_rate = 0.1
epochs = 1000
nn.train(X, y, learning_rate, epochs)

# Realizar predicciones
predictions = nn.predict(X)

# Graficar los resultados en 2D
plot_results(X, y, predictions)
