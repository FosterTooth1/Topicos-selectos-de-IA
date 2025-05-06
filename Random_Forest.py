# Importar bibliotecas necesarias para operaciones matematicas y pruebas
from random import seed, randrange
from csv import reader
from math import sqrt

# Función para cargar un archivo CSV
# Carga un archivo CSV y lo devuelve como lista de listas
def cargar_csv(nombre_archivo):
    conjunto_datos = list()
    with open(nombre_archivo, 'r') as archivo:
        lector_csv = reader(archivo)
        for fila in lector_csv:
            if not fila:  #Se satlan filas vacías
                continue
            conjunto_datos.append(fila)
    return conjunto_datos

# Conversión de datos a tipos numéricos
#Función que convierte una columna de strings a valores flotantes
def convertir_columna_a_float(conjunto_datos, indice_columna):
    for fila in conjunto_datos:
        fila[indice_columna] = float(fila[indice_columna].strip())

# Función que codifica una columna categórica a valores enteros únicos (para la clasificacion y saber a qué clase pertenece cada fila)
def convertir_columna_a_entero(conjunto_datos, indice_columna):
    valores_clase = [fila[indice_columna] for fila in conjunto_datos]
    valores_unicos = set(valores_clase)
    diccionario_codificacion = dict()
    for indice, valor in enumerate(valores_unicos):
        diccionario_codificacion[valor] = indice
    for fila in conjunto_datos:
        fila[indice_columna] = diccionario_codificacion[fila[indice_columna]]
    return diccionario_codificacion  #Diccionario de codificación para referencia

# Función que divide el dataset en 'k' particiones para realixar validación cruzada.
def dividir_validacion_cruzada(conjunto_datos, num_particiones):
    particiones = list()
    copia_conjunto = list(conjunto_datos)
    tamano_particion = int(len(copia_conjunto) / num_particiones)
    for _ in range(num_particiones):
        particion = list()
        while len(particion) < tamano_particion:
            indice_aleatorio = randrange(len(copia_conjunto))
            particion.append(copia_conjunto.pop(indice_aleatorio))
        particiones.append(particion)
    return particiones

# Métricas de evaluación
# Función que calcula la exactitud de las predicciones comparando los valores reales y los predichos
def calcular_exactitud(reales, predichos):
    correctos = sum(1 for i in range(len(reales)) if reales[i] == predichos[i])
    return (correctos / float(len(reales))) * 100.0

#Evaluación del algoritmo
# Función que evalúa un algoritmo usando validación cruzada y devuelve las puntuaciones
def evaluar_algoritmo(conjunto_datos, algoritmo, num_particiones, *args):
    particiones = dividir_validacion_cruzada(conjunto_datos, num_particiones)
    puntuaciones = list()
    for particion in particiones:
        conjunto_entrenamiento = list(particiones)
        conjunto_entrenamiento.remove(particion)
        # Aplanamos la lista de listas a una sola lista
        conjunto_entrenamiento = sum(conjunto_entrenamiento, [])
        conjunto_prueba = list()
        # Preparamos el conjunto de prueba ocultando las etiquetas reales
        for fila in particion:
            copia_fila = list(fila)
            conjunto_prueba.append(copia_fila)
            copia_fila[-1] = None  #Ocultamos la etiqueta real
        predichos = algoritmo(conjunto_entrenamiento, conjunto_prueba, *args)
        reales = [fila[-1] for fila in particion]
        exactitud = calcular_exactitud(reales, predichos)
        puntuaciones.append(exactitud)
    return puntuaciones

# Funciones para construcción del árbol
# Funcion que divide el dataset en dos grupos basados en un valor de característica
# Esta función se utiliza para dividir el dataset en dos grupos, uno a la izquierda y otro a la derecha, según un valor de característica
def dividir_por_caracteristica(indice, valor, conjunto_datos):
    izquierda, derecha = list(), list()
    for fila in conjunto_datos:
        if fila[indice] < valor:
            izquierda.append(fila)
        else:
            derecha.append(fila)
    return izquierda, derecha

# Función que calcula el índice de Gini para evaluar la calidad de una división
# El índice de Gini mide la impureza de un conjunto de datos, donde 0 es puro y 1 es impuro
def calcular_indice_gini(grupos, clases):
    total_muestras = sum(len(grupo) for grupo in grupos)
    gini = 0.0
    for grupo in grupos:
        tamano_grupo = len(grupo)
        if tamano_grupo == 0:
            continue
        suma_probabilidad = 0.0
        for clase in clases:
            proporcion = [fila[-1] for fila in grupo].count(clase) / tamano_grupo
            suma_probabilidad += proporcion ** 2
        gini += (1.0 - suma_probabilidad) * (tamano_grupo / total_muestras)
    return gini

# Función que encuentra la mejor división del conjunto de datos
# Esta función evalúa todas las posibles divisiones y selecciona la que minimiza el índice de Gini
# Se basa en la idea de que una buena división separa las clases de manera efectiva
def encontrar_mejor_division(conjunto_datos, num_caracteristicas):
    clases_unicas = list(set(fila[-1] for fila in conjunto_datos))
    mejor_indice, mejor_valor, mejor_gini, mejor_grupos = 999, 999, 999, None
    caracteristicas = list()
    
    # Seleccionamos las características aleatorias sin repetición
    while len(caracteristicas) < num_caracteristicas:
        indice_aleatorio = randrange(len(conjunto_datos[0])-1)
        if indice_aleatorio not in caracteristicas:
            caracteristicas.append(indice_aleatorio)
    
    # Evaluamos todas las posibles divisiones para cada característica seleccionada
    for indice in caracteristicas:
        for fila in conjunto_datos:
            grupos = dividir_por_caracteristica(indice, fila[indice], conjunto_datos)
            gini_actual = calcular_indice_gini(grupos, clases_unicas)
            if gini_actual < mejor_gini:
                mejor_indice, mejor_valor, mejor_gini, mejor_grupos = indice, fila[indice], gini_actual, grupos
    return {'indice': mejor_indice, 'valor': mejor_valor, 'grupos': mejor_grupos}

# Función que crea un nodo terminal
# Esta función se utiliza para crear un nodo terminal en el árbol de decisión, que representa una clase final
def crear_nodo_terminal(grupo):
    resultados = [fila[-1] for fila in grupo]
    return max(set(resultados), key=resultados.count)

# Funcion que divide recursivamente los nodos del árbol
def dividir_nodo(nodo, profundidad_max, tamano_min, num_caracteristicas, profundidad_actual):
    izquierda, derecha = nodo['grupos']
    del(nodo['grupos'])  # Eliminamos los grupos ya procesados
    
    # Caso: no hay división posible
    if not izquierda or not derecha:
        nodo['izquierda'] = nodo['derecha'] = crear_nodo_terminal(izquierda + derecha)
        return
    
    # Verificamos la profundidad máxima
    if profundidad_actual >= profundidad_max:
        nodo['izquierda'] = crear_nodo_terminal(izquierda)
        nodo['derecha'] = crear_nodo_terminal(derecha)
        return
    
    # Procesamos le hijo izquierdo
    if len(izquierda) <= tamano_min:
        nodo['izquierda'] = crear_nodo_terminal(izquierda)
    else:
        nodo['izquierda'] = encontrar_mejor_division(izquierda, num_caracteristicas)
        dividir_nodo(nodo['izquierda'], profundidad_max, tamano_min, num_caracteristicas, profundidad_actual+1)
    
    # Procesamos el hijo derecho
    if len(derecha) <= tamano_min:
        nodo['derecha'] = crear_nodo_terminal(derecha)
    else:
        nodo['derecha'] = encontrar_mejor_division(derecha, num_caracteristicas)
        dividir_nodo(nodo['derecha'], profundidad_max, tamano_min, num_caracteristicas, profundidad_actual+1)

# Función que construye el árbol de decisión
def construir_arbol(conjunto_entrenamiento, profundidad_max, tamano_min, num_caracteristicas):
    raiz = encontrar_mejor_division(conjunto_entrenamiento, num_caracteristicas)
    dividir_nodo(raiz, profundidad_max, tamano_min, num_caracteristicas, 1)
    return raiz

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
        
# Función que crea una muestra bootstrap (muestra aleatoria con reemplazo)
def crear_muestra_bootstrap(conjunto_datos, proporcion_muestra):
    muestra = list()
    tamano_muestra = round(len(conjunto_datos) * proporcion_muestra)
    while len(muestra) < tamano_muestra:
        indice_aleatorio = randrange(len(conjunto_datos))
        muestra.append(conjunto_datos[indice_aleatorio])
    return muestra

# Función que realiza una predicción usando el bosque aleatorio
def predecir_con_bosque(bosque, fila):
    """Realiza una predicción promediando las predicciones de todos los árboles."""
    predicciones = [predecir_con_arbol(arbol, fila) for arbol in bosque]
    return max(set(predicciones), key=predicciones.count)

# Algoritmo principal: Random Forest
def random_forest(entrenamiento, prueba, profundidad_max, tamano_min, proporcion_muestra, num_arboles, num_caracteristicas):
    """Construye un bosque aleatorio y realiza predicciones."""
    bosque = list()
    for _ in range(num_arboles):
        muestra = crear_muestra_bootstrap(entrenamiento, proporcion_muestra)
        # Imprimir el progreso de la construcción del árbol
        print(f'Construyendo arbol {_+1}/{num_arboles}...')
        arbol = construir_arbol(muestra, profundidad_max, tamano_min, num_caracteristicas)
        bosque.append(arbol)
    predicciones = [predecir_con_bosque(bosque, fila) for fila in prueba]
    return predicciones

#Configuración y ejecución
if __name__ == "__main__":
    # Configuramos la semilla para el analisis de resultados
    seed(2)
    
    # Cargamos y preparamos los datos
    nombre_archivo = 'Dataset_Final_SE.csv' 
    conjunto_datos = cargar_csv(nombre_archivo)
    
    # Convertimos las columnas numéricas (ajustar índices según tu dataset)
    for indice_columna in range(len(conjunto_datos[0])-1):
        convertir_columna_a_float(conjunto_datos, indice_columna)
    
    # Convertimos la columna de clase a enteros
    convertir_columna_a_entero(conjunto_datos, len(conjunto_datos[0])-1)
    
    # Parámetros
    num_particiones = 5
    profundidad_max = 10
    tamano_min = 1
    proporcion_muestra = 1.0
    num_caracteristicas = int(sqrt(len(conjunto_datos[0])-1))
    
    # Evaluamos con diferentes números de árboles
    print('Evaluando Random Forest con validacion cruzada...')
    for num_arboles in [1]:
        puntuaciones = evaluar_algoritmo(
            conjunto_datos, 
            random_forest, 
            num_particiones,
            profundidad_max,
            tamano_min,
            proporcion_muestra,
            num_arboles,
            num_caracteristicas
        )
        print(f'Árboles: {num_arboles}')
        print(f'Puntuaciones: {puntuaciones}')
        print(f'Exactitud Promedio: {sum(puntuaciones)/float(len(puntuaciones)):.3f}%')