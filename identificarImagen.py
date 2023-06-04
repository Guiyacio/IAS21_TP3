# Importar las bibliotecas necesarias

import numpy as np
import cv2

# Función para entrenar el modelo de Hopfield

def entrenar_hopfield(patrones):
        
    num_neuronas = patrones.shape[1]  # Obtener el número de neuronas
    num_patrones = patrones.shape[0]  # Obtener el número de patrones
    # Inicializar la matriz de pesos
    pesos = np.zeros((num_neuronas, num_neuronas))  
    
    # Calcular los pesos sumando los productos externos de los patrones
    for i in range(num_patrones):
        patron = patrones[i, :]
        pesos += np.outer(patron, patron)

    # Establecer la diagonal de los pesos en cero
    np.fill_diagonal(pesos, 0)  
    
    return pesos

# Función para recuperar un patrón utilizando el modelo de Hopfield

def recuperar_patron(patron_inicial, pesos, max_iteraciones):    
    
    patron_recuperado = np.copy(patron_inicial)  # Copiar el patrón inicial
    num_neuronas = len(patron_inicial)  # Obtener el número de neuronas
    
    num_filas = int(np.sqrt(num_neuronas))  # Calcular el número de filas de la imagen
    num_columnas = int(np.sqrt(num_neuronas))  # Calcular el número de columnas de la imagen
    
    for iteracion in range(max_iteraciones):
        print("Iteración:", iteracion + 1)
        
        # Imprimir el patrón actual
        print("Patrón actual:")
        print(patron_recuperado.reshape((num_filas, num_columnas)))
        
        for i in range(num_neuronas):
            activacion = np.dot(pesos[i, :], patron_recuperado)
            patron_recuperado[i] = 1 if activacion > 0 else 0
        
        # Imprimir el patrón actualizado
        print("Patrón actualizado:")
        print(patron_recuperado.reshape((num_filas, num_columnas)))
        print("-------------------")
        
        if np.array_equal(patron_recuperado, patron_inicial):
            print("Se alcanzó un patrón estable.")
            break
        
    return patron_recuperado, num_filas, num_columnas

# Leer las imágenes y convertirlas en matrices
imagen_circulo = cv2.imread("circulo1.jpg", cv2.IMREAD_GRAYSCALE)
imagen_ruido = cv2.imread("ruido1.jpg", cv2.IMREAD_GRAYSCALE)

# Imprimir la imagen original
print("Imagen original:")
print(imagen_circulo)

# Imprimir la imagen con ruido
print("Imagen con ruido:")
print(imagen_ruido)

# Aplicar un umbral para convertir las imágenes en patrones binarios
umbral = 127
patron_circulo = np.where(imagen_circulo > umbral, 1, 0).flatten()
patron_ruido = np.where(imagen_ruido > umbral, 1, 0).flatten()

# Crear la lista de patrones de entrenamiento
patrones_entrenamiento = np.array([patron_circulo])

# Entrenar el modelo de Hopfield
pesos_hopfield = entrenar_hopfield(patrones_entrenamiento)

# Especificar el patrón inicial como la imagen con ruido
patron_inicial = patron_ruido

max_iteraciones = 5

# Recuperar el patrón utilizando el modelo de Hopfield
patron_recuperado, num_filas, num_columnas = recuperar_patron(patron_inicial, pesos_hopfield, max_iteraciones)

# Convertir el patrón recuperado en una imagen
imagen_recuperada = np.reshape(patron_recuperado, (num_filas, num_columnas)) * 255

# Guardar la imagen recuperada en un archivo
cv2.imwrite("salida1.jpg", imagen_recuperada)

# Imprimir la imagen de salida
print("Imagen de salida:")
print(imagen_recuperada)
