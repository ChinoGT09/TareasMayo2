import numpy as np

class Perceptron:
    def __init__(self, num_entradas, tasa_aprendizaje=0.1, epocas=100):
        self.pesos = np.zeros(num_entradas + 1)  # +1 para el bias
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas
    
    def funcion_activacion(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, entradas):
        # Producto punto entre pesos y entradas, más el bias
        suma = np.dot(entradas, self.pesos[1:]) + self.pesos[0]
        return self.funcion_activacion(suma)
    
    def entrenar(self, X_entrenamiento, y_entrenamiento):
        for _ in range(self.epocas):
            for entradas, etiqueta in zip(X_entrenamiento, y_entrenamiento):
                prediccion = self.predict(entradas)
                # Actualizar pesos y bias
                self.pesos[1:] += self.tasa_aprendizaje * (etiqueta - prediccion) * entradas
                self.pesos[0] += self.tasa_aprendizaje * (etiqueta - prediccion)


# Ejemplo de uso:
if __name__ == "__main__":
    # Datos de entrenamiento para la compuerta OR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    
    # Crear y entrenar el perceptrón
    perceptron = Perceptron(num_entradas=2)
    perceptron.entrenar(X, y)
    
    # Probar el perceptrón
    print("Predicciones:")
    for entrada in X:
        print(f"{entrada} -> {perceptron.predict(entrada)}")