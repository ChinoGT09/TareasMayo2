import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_entradas, tasa_aprendizaje=0.1, epocas=100):
        self.pesos = np.random.rand(num_entradas + 1) * 2 - 1  # Pesos entre -1 y 1
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
                # Actualizar pesos
                self.pesos[1:] += self.tasa_aprendizaje * error * entradas
                self.pesos[0] += self.tasa_aprendizaje * error  # Bias
                error_epoca += int(error != 0.0)
            self.errores.append(error_epoca)
            if error_epoca == 0:  # Convergencia
                break

def generar_datos_linealmente_separables(n=40):
    """Genera datos linealmente separables alrededor de y = 1.5x + 2"""
    np.random.seed(42)
    X = np.random.randn(n, 2) * 3 + 5  # Centrado alrededor de (5,5)
    
    # Crear límite de decisión lineal (y = 1.5x + 2)
    y = np.where(X[:, 1] > 1.5*X[:, 0] + 2, 1, 0)
    
    # Asegurar que haya puntos en ambas clases
    while len(np.unique(y)) < 2:
        y = np.where(X[:, 1] > 1.5*X[:, 0] + 2, 1, 0)
    
    return X, y

def graficar_datos_y_recta(X, y, pesos, titulo="Perceptrón - Límite de Decisión"):
    """Grafica los datos y la recta de decisión"""
    plt.figure(figsize=(10, 8))
    
    # Graficar puntos de datos
    plt.scatter(X[y==0, 0], X[y==0, 1], color='blue', label='Clase 0', alpha=0.7)
    plt.scatter(X[y==1, 0], X[y==1, 1], color='red', label='Clase 1', alpha=0.7)
    
    # Calcular límite de decisión (w0 + w1*x + w2*y = 0)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Generar puntos para la recta
    if pesos[2] != 0:  # Evitar división por cero
        xx = np.array([x_min, x_max])
        yy = (-pesos[0] - pesos[1]*xx) / pesos[2]
        plt.plot(xx, yy, 'k-', label='Límite de decisión', linewidth=2)
    
    plt.xlabel('Característica X1')
    plt.ylabel('Característica X2')
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

def graficar_errores(errores):
    """Grafica la evolución de los errores durante el entrenamiento"""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(errores)+1), errores, marker='o')
    plt.xlabel('Época')
    plt.ylabel('Número de errores')
    plt.title('Error durante el entrenamiento')
    plt.grid(True)
    plt.show()

# Generar datos
X, y = generar_datos_linealmente_separables(40)

# Crear y entrenar el perceptrón
perceptron = Perceptron(num_entradas=2, tasa_aprendizaje=0.01, epocas=100)
perceptron.entrenar(X, y)

# Mostrar resultados
print("Pesos finales del perceptrón:")
print(f"Bias (w0): {perceptron.pesos[0]:.4f}")
print(f"Peso X1 (w1): {perceptron.pesos[1]:.4f}")
print(f"Peso X2 (w2): {perceptron.pesos[2]:.4f}")

# Graficar
graficar_datos_y_recta(X, y, perceptron.pesos)
graficar_errores(perceptron.errores)

# Probar el perceptrón
print("\nPredicciones de ejemplo:")
test_points = np.array([[4, 10], [8, 5], [6, 8], [2, 2]])
for point in test_points:
    pred = perceptron.predict(point)
    print(f"Punto {point} -> Clase {pred}")