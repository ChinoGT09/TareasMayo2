import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Perceptron:
    def __init__(self, num_entradas, tasa_aprendizaje=0.1, epocas=100):
        self.pesos = np.random.rand(num_entradas + 1) * 2 - 1  # Pesos entre -1 y 1
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas
        self.errores_entrenamiento = []
    
    def funcion_activacion(self, x):
        """Función de activación escalón (devuelve 0 o 1)"""
        return 1 if x >= 0 else 0
    
    def predict(self, entradas):
        """Realiza una predicción"""
        suma = np.dot(entradas, self.pesos[1:]) + self.pesos[0]  # Producto punto + bias
        return self.funcion_activacion(suma)
    
    def entrenar(self, X, y):
        """Entrena el perceptrón con los datos de entrenamiento"""
        for _ in range(self.epocas):
            error_epoca = 0
            for entradas, etiqueta in zip(X, y):
                prediccion = self.predict(entradas)
                error = etiqueta - prediccion
                # Actualizar pesos
                self.pesos[1:] += self.tasa_aprendizaje * error * entradas
                self.pesos[0] += self.tasa_aprendizaje * error  # Bias
                error_epoca += int(error != 0.0)
            self.errores_entrenamiento.append(error_epoca)
            if error_epoca == 0:  # Convergencia
                break
    
    def evaluar(self, X_prueba, y_prueba):
        """Evalúa el perceptrón con los datos de prueba"""
        aciertos = 0
        predicciones = []
        for entradas, etiqueta in zip(X_prueba, y_prueba):
            prediccion = self.predict(entradas)
            predicciones.append(prediccion)
            if prediccion == etiqueta:
                aciertos += 1
        precision = aciertos / len(y_prueba)
        return precision, predicciones

def generar_datos_linealmente_separables(n=100):
    """Genera datos linealmente separables alrededor de y = 1.5x + 2"""
    np.random.seed(42)
    X = np.random.randn(n, 2) * 3 + 5  # Centrado alrededor de (5,5)
    
    # Crear límite de decisión lineal (y = 1.5x + 2)
    y = np.where(X[:, 1] > 1.5*X[:, 0] + 2, 1, 0)
    
    # Asegurar que haya puntos en ambas clases
    while len(np.unique(y)) < 2:
        y = np.where(X[:, 1] > 1.5*X[:, 0] + 2, 1, 0)
    
    return X, y

def dividir_conjuntos(X, y, test_size=0.3, random_state=42):
    """Divide los datos en conjuntos de entrenamiento y prueba"""
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba

def graficar_conjuntos(X_entrenamiento, y_entrenamiento, X_prueba, y_prueba, pesos, titulo="Conjuntos de Entrenamiento y Prueba"):
    """Grafica ambos conjuntos de datos y la recta de decisión"""
    plt.figure(figsize=(12, 8))
    
    # Graficar puntos de entrenamiento (con marcador 'o')
    plt.scatter(X_entrenamiento[y_entrenamiento==0, 0], X_entrenamiento[y_entrenamiento==0, 1], 
                color='blue', marker='o', label='Entrenamiento Clase 0', alpha=0.7)
    plt.scatter(X_entrenamiento[y_entrenamiento==1, 0], X_entrenamiento[y_entrenamiento==1, 1], 
                color='red', marker='o', label='Entrenamiento Clase 1', alpha=0.7)
    
    # Graficar puntos de prueba (con marcador 'x')
    plt.scatter(X_prueba[y_prueba==0, 0], X_prueba[y_prueba==0, 1], 
                color='blue', marker='x', s=100, label='Prueba Clase 0', alpha=0.7)
    plt.scatter(X_prueba[y_prueba==1, 0], X_prueba[y_prueba==1, 1], 
                color='red', marker='x', s=100, label='Prueba Clase 1', alpha=0.7)
    
    # Calcular límite de decisión (w0 + w1*x + w2*y = 0)
    x_min, x_max = min(X_entrenamiento[:,0].min(), X_prueba[:,0].min()) - 1, \
                   max(X_entrenamiento[:,0].max(), X_prueba[:,0].max()) + 1
    y_min, y_max = min(X_entrenamiento[:,1].min(), X_prueba[:,1].min()) - 1, \
                   max(X_entrenamiento[:,1].max(), X_prueba[:,1].max()) + 1
    
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

# Generar y dividir los datos
X, y = generar_datos_linealmente_separables(100)
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = dividir_conjuntos(X, y, test_size=0.3)

# Crear y entrenar el perceptrón
perceptron = Perceptron(num_entradas=2, tasa_aprendizaje=0.01, epocas=100)
perceptron.entrenar(X_entrenamiento, y_entrenamiento)

# Evaluar el perceptrón
precision, predicciones = perceptron.evaluar(X_prueba, y_prueba)

# Mostrar resultados
print("Pesos finales del perceptrón:")
print(f"Bias (w0): {perceptron.pesos[0]:.4f}")
print(f"Peso X1 (w1): {perceptron.pesos[1]:.4f}")
print(f"Peso X2 (w2): {perceptron.pesos[2]:.4f}")
print(f"\nPrecisión en conjunto de prueba: {precision:.2%}")

# Graficar
graficar_conjuntos(X_entrenamiento, y_entrenamiento, X_prueba, y_prueba, perceptron.pesos)
graficar_errores(perceptron.errores_entrenamiento)

# Mostrar algunas predicciones
print("\nPredicciones en conjunto de prueba (primeros 5 puntos):")
for i in range(5):
    print(f"Punto {X_prueba[i]} -> Real: {y_prueba[i]}, Predicho: {predicciones[i]}")