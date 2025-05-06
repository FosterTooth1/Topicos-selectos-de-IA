import pickle
from csv import reader

# Función para preprocesar nuevos datos (igual que en el entrenamiento)
def preprocesar_datos(nombre_archivo, diccionario_clases=None):
    # Cargar datos
    conjunto = []
    with open(nombre_archivo, 'r') as f:
        lector_csv = reader(f)
        for fila in lector_csv:
            if fila:  # Saltar filas vacías
                conjunto.append(fila)
    
    # Convertir columnas numéricas
    for i in range(len(conjunto[0])-1):
        for fila in conjunto:
            fila[i] = float(fila[i].strip())
    
    # Convertir clase si existe diccionario
    if diccionario_clases:
        for fila in conjunto:
            fila[-1] = diccionario_clases[fila[-1]]
    
    return conjunto

# Funciones de predicción  
# Función que realiza una predicción usando un árbol de decisión
def predecir_con_arbol(nodo, fila):
    if fila[nodo['indice']] < nodo['valor']:
        if isinstance(nodo['izquierda'], dict):
            return predecir_con_arbol(nodo['izquierda'], fila)
        else:
            return nodo['izquierda']
    else:
        if isinstance(nodo['derecha'], dict):
            return predecir_con_arbol(nodo['derecha'], fila)
        else:
            return nodo['derecha']
        
# Función que realiza una predicción usando el bosque aleatorio
def predecir_con_bosque(bosque, fila):
    """Realiza una predicción promediando las predicciones de todos los árboles."""
    predicciones = [predecir_con_arbol(arbol, fila) for arbol in bosque]
    return max(set(predicciones), key=predicciones.count)

# Cargar modelo guardado
with open('modelo_final.pkl', 'rb') as f:
    modelo = pickle.load(f)
    bosque = modelo['modelo']
    diccionario_clases = modelo['diccionario_clases']

# Cargar nuevos datos (ejemplo)
nuevos_datos = preprocesar_datos('nuevos_datos.csv', diccionario_clases)

# Función para hacer predicciones
def predecir(nuevos_datos):
    predicciones = []
    for fila in nuevos_datos:
        # Hacer copia para no modificar original
        fila_prediccion = fila.copy()
        fila_prediccion[-1] = None  # Eliminar clase si existe
        prediccion = predecir_con_bosque(bosque, fila_prediccion)
        predicciones.append(prediccion)
    return predicciones

# Obtener y mostrar predicciones
predicciones_numericas = predecir(nuevos_datos)
predicciones_originales = [diccionario_clases[p] for p in predicciones_numericas]

print("\nPredicciones:")
for real, pred in zip(nuevos_datos, predicciones_originales):
    print(f"Datos: {real[:-1]} -> Predicción: {pred}")