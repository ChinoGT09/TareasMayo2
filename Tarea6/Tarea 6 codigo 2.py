import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_entradas, tasa_aprendizaje=0.1, epocas=100):
        self.pesos = np.random.rand(num_entradas + 1)  # +1 para el bias (ahora inicializado aleatoriamente)
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas
        self.errores = []
    
    def funcion_activacion(self, x):
        """Función de activación escalón (devuelve 0 o 1)"""
        return 1 if x >= 0 else 0
    
    def predict(self, entradas):
        """Realiza una predicción"""
        suma = np.dot(entradas, self.pesos[1:]) + self.pesos[0]  # Producto punto + bias
        return self.funcion_activacion(suma)
    
    def entrenar(self, X, y):
        """Entrena el perceptrón"""
        for _ in range(self.epocas):
            error_epoca = 0
            for entradas, etiqueta in zip(X, y):
                prediccion = self.predict(entradas)
                error = etiqueta - prediccion
                self.pesos[1:] += self.tasa_aprendizaje * error * entradas
                self.pesos[0] += self.tasa_aprendizaje * error
                error_epoca += int(error != 0.0)
            self.errores.append(error_epoca)
    
    def plot_limite_decision(self, X, y):
        """Visualiza los datos y el límite de decisión"""
        plt.figure(figsize=(10, 6))
        
        # Graficar puntos de datos
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
        
        # Calcular límite de decisión (w0 + w1*x + w2*y = 0)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        Z = self.pesos[0] + self.pesos[1]*xx + self.pesos[2]*yy
        Z = np.where(Z >= 0, 1, 0)
        
        plt.contourf(xx, yy, Z, alpha=0.2, cmap='bwr')
        plt.xlabel('Característica 1')
        plt.ylabel('Característica 2')
        plt.title('Límite de decisión del Perceptrón')
        plt.colorbar(label='Clase')
        plt.show()
        
        # Graficar errores por época
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(self.errores)+1), self.errores, marker='o')
        plt.xlabel('Época')
        plt.ylabel('Número de errores')
        plt.title('Error durante el entrenamiento')
        plt.grid(True)
        plt.show()


# Generar datos linealmente separables
np.random.seed(42)
num_muestras = 100
X = np.random.randn(num_muestras, 2) * 1.5

# Crear límite de decisión lineal (y = 2x + 1)
y = np.where(X[:, 1] > 2*X[:, 0] + 1, 1, 0)

# Crear y entrenar el perceptrón
perceptron = Perceptron(num_entradas=2, tasa_aprendizaje=0.1, epocas=20)
perceptron.entrenar(X, y)

# Evaluar el perceptrón
print("Pesos finales:", perceptron.pesos)
print("\nPredicciones en algunos puntos de prueba:")
print("[0.5, 0.5] ->", perceptron.predict([0.5, 0.5]))
print("[1.0, 3.0] ->", perceptron.predict([1.0, 3.0]))
print("[-1.0, -1.0] ->", perceptron.predict([-1.0, -1.0]))

# Visualizar resultados
perceptron.plot_limite_decision(X, y)