from random import seed, randrange, shuffle
from csv import reader
from math import sqrt
import os
import pickle

# Carga del CSV
# Esta función carga un archivo CSV y devuelve una lista de listas con los datos.
def cargar_csv(nombre_archivo):
    datos = []
    with open(nombre_archivo, 'r') as f:
        lector = reader(f)
        for fila in lector:
            if fila:
                datos.append([float(x.strip()) for x in fila[:-1]] + [fila[-1]])
    return datos


# Codifica la última columna (clase) a enteros
def codificar_clases(datos):
    valores = [fila[-1] for fila in datos]
    unicos = list(set(valores))
    mapa = {v:i for i,v in enumerate(unicos)}
    for fila in datos:
        fila[-1] = mapa[fila[-1]]
    return mapa

# Calcula la exactitud de las predicciones
# Esta función compara las predicciones con los valores reales y devuelve la exactitud en porcentaje.
def exactitud(reales, predichos):
    aciertos = sum(1 for r,p in zip(reales, predichos) if r==p)
    return aciertos / len(reales) * 100.0

# Funciones del random forest
# Esta función divide los datos en dos grupos según un índice y un valor.
# Por ejemplo, si el índice es 0 y el valor es 5, se dividirán los datos en dos grupos:
# uno con los registros donde la primera columna es menor que 5 y otro donde es mayor o igual a 5.  
def dividir_por_caracteristica(ind, val, datos):
    izq = [r for r in datos if r[ind] < val]
    der = [r for r in datos if r[ind] >= val]
    return izq, der

# Calcula el índice de Gini para un conjunto de grupos
# Esta función mide la impureza de los grupos generados por una división.
# Un índice de Gini más bajo indica una mejor división.
def gini(grupos, clases):
    total = sum(len(g) for g in grupos)
    imp = 0.0
    for g in grupos:
        tam = len(g)
        if tam == 0: continue
        s = 0.0
        for c in clases:
            p = sum(1 for r in g if r[-1]==c) / tam
            s += p*p
        imp += (1 - s) * (tam/total)
    return imp

# Esta función encuentra la mejor división de los datos según el índice de Gini.
# Selecciona aleatoriamente un número de características y encuentra la mejor división  
# entre ellas. Devuelve un diccionario con la mejor división encontrada.
def mejor_division(datos, num_feat):
    clases = list(set(r[-1] for r in datos))
    mejor = {'gini':1e9}
    n_feats = []
    while len(n_feats) < num_feat:
        i = randrange(len(datos[0]) - 1)
        if i not in n_feats: n_feats.append(i)
    for ind in n_feats:
        for r in datos:
            grupos = dividir_por_caracteristica(ind, r[ind], datos)
            ig = gini(grupos, clases)
            if ig < mejor['gini']:
                mejor = {'indice':ind, 'valor':r[ind], 'gini':ig, 'grupos':grupos}
    return mejor

# Esta función determina la clase terminal de un grupo de datos.
# Toma el grupo de datos y devuelve la clase más frecuente.
def terminal(grupo):
    labels = [r[-1] for r in grupo]
    return max(set(labels), key=labels.count)

# Esta función divide un nodo en dos subnodos.
# Si el nodo es terminal o ha alcanzado la profundidad máxima, se convierte en un nodo terminal.
def dividir_nodo(nodo, prof_max, tam_min, num_feat, prof):
    izq, der = nodo['grupos']
    del nodo['grupos']
    if not izq or not der:
        nodo['izq'] = nodo['der'] = terminal(izq+der)
        return
    if prof >= prof_max:
        nodo['izq'], nodo['der'] = terminal(izq), terminal(der)
        return
    if len(izq) <= tam_min:
        nodo['izq'] = terminal(izq)
    else:
        nodo['izq'] = mejor_division(izq, num_feat)
        dividir_nodo(nodo['izq'], prof_max, tam_min, num_feat, prof+1)
    if len(der) <= tam_min:
        nodo['der'] = terminal(der)
    else:
        nodo['der'] = mejor_division(der, num_feat)
        dividir_nodo(nodo['der'], prof_max, tam_min, num_feat, prof+1)

# Esta función construye el árbol de decisión.
# Toma los datos, la profundidad máxima, el tamaño mínimo de los grupos y el número de características.
def construir_arbol(datos, prof_max, tam_min, num_feat):
    raiz = mejor_division(datos, num_feat)
    dividir_nodo(raiz, prof_max, tam_min, num_feat, 1)
    return raiz

# Esta función predice la clase de una fila de datos utilizando el árbol de decisión.
# Toma el nodo raíz y la fila de datos y recorre el árbol hasta llegar a un nodo terminal.
def predecir_arbol(nodo, fila):
    if fila[nodo['indice']] < nodo['valor']:
        return predecir_arbol(nodo['izq'], fila) if isinstance(nodo['izq'], dict) else nodo['izq']
    else:
        return predecir_arbol(nodo['der'], fila) if isinstance(nodo['der'], dict) else nodo['der']

# Esta función crea una muestra aleatoria de los datos.
# Toma los datos y la proporción de la muestra y devuelve una lista de filas aleatorias.
def bootstrap(datos, prop):
    n = round(len(datos)*prop)
    return [datos[randrange(len(datos))] for _ in range(n)]

# Esta función construye el bosque aleatorio.
# Toma los datos de entrenamiento y prueba, la profundidad máxima, el tamaño mínimo de los grupos,
# la proporción de la muestra, el número de árboles y el número de características.
def random_forest(train, test, prof_max, tam_min, prop_boot, n_trees, n_feats):
    bosque = []
    for i in range(n_trees):
        muestra = bootstrap(train, prop_boot)
        print(f"Construyendo árbol {i+1}/{n_trees}")
        arbol = construir_arbol(muestra, prof_max, tam_min, n_feats)
        bosque.append(arbol)
    predicciones = []
    for f in test:
        votos = [predecir_arbol(a, f) for a in bosque]
        pred = max(set(votos), key=votos.count)
        predicciones.append(pred)
    return bosque, predicciones


if __name__ == "__main__":
    seed(2)
    archivo = 'Dataset_Final_SE.csv'
    datos = cargar_csv(archivo)
    cod_map = codificar_clases(datos)

    # Parámetros de datos de entrenamiento y prueba:
    porc_entrenamiento = 0.8  
    porc_datos = 1.0 
    modelo_path       = 'random_forest_model.pkl'

    # Mezclar y recortar datos
    shuffle(datos)
    total = int(len(datos)*porc_datos)
    sample = datos[:total]

    # División de datos
    # Separa los datos en conjuntos de entrenamiento y prueba
    corte = int(len(sample)*porc_entrenamiento)
    train_set = sample[:corte]
    test_set  = sample[corte:]

    # Parámetros del bosque
    
    # prof_max: Profundidad máxima del árbol
    prof_max       = 15
    # tam_min: Tamaño mínimo de los grupos
    tam_min        = 1
    # prop_boot: Proporción de la muestra
    prop_boot      = 1.0
    # n_trees: Número de árboles en el bosque
    n_trees        = 100
    # n_feats: Número de características a considerar
    n_feats        = int(sqrt(len(train_set[0])-1))

# Carga el modelo si existe, de lo contrario entrena y guarda el modelo
    if os.path.exists(modelo_path):
        print(f"Cargando modelo desde '{modelo_path}'...")
        with open(modelo_path, 'rb') as f:
            bosque = pickle.load(f)
        # Solo hacemos predicción
        _, preds = bosque, [max(set([predecir_arbol(a, f) for a in bosque]), key=lambda v: [predecir_arbol(a, f) for a in bosque].count) for f in test_set]
    else:
        # Entrenar y guardar
        bosque, preds = random_forest(train_set, test_set, prof_max, tam_min, prop_boot, n_trees, n_feats)
        print(f"Guardando modelo en '{modelo_path}'...")
        with open(modelo_path, 'wb') as f:
            pickle.dump(bosque, f)

    # Evaluación
    reales = [r[-1] for r in test_set]
    acc = exactitud(reales, preds)
    print(f"Exactitud: {acc:.3f}%")