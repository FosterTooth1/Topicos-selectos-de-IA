import joblib
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime

def cargar_CSV(nombre_archivo):
    """
    Carga un archivo CSV y devuelve un DataFrame de pandas.
    """
    try:
        df = pd.read_csv(nombre_archivo)
        return df
    except FileNotFoundError:
        print(f"El archivo {nombre_archivo} no se encuentra.")
        return None

def eliminar_filas_tipos(df, filas_a_conservar):
    """
    Elimina las filas de la columna "tipo" dentro del dataframe, 
    conservando solo las filas especificadas, y devuelve copia explícita.
    """
    if df is not None:
        if 'tipo' in df.columns:
            df_filtrado = df[df['tipo'].isin(filas_a_conservar)].copy()
            return df_filtrado
        else:
            print("La columna 'tipo' no se encuentra en el DataFrame.")
            return None
    else:
        print("El DataFrame es None, no se pueden eliminar filas.")
        return None

def modificar_tiempo_estancia_promedio(df, tiempo_estancia):
    """
    Modifica el tiempo de estancia promedio en un DataFrame. 
    Hace copia explícita antes de asignar la nueva columna.
    """
    if df is not None:
        if 'tipo' in df.columns:
            df = df.copy()
            # Asignar con loc para mayor claridad:
            df.loc[:, 'Tiempo_Estancia_Promedio'] = df['tipo'].map(tiempo_estancia)
            return df
        else:
            print("La columna 'tipo' no se encuentra en el DataFrame.")
            return None
    else:
        print("El DataFrame es None, no se puede modificar el tiempo de estancia promedio.")
        return None
    
    # Funciones para cambiar formato de los datos HORA y DIRECCION VIENTO
def cambiarFormatoHora(fecha_hora_str):
    """
    Convierte una cadena de fecha y hora en formato "YYYY-MM-DD HH:MM" a minutos desde medianoche.
    Devuelve el total de minutos transcurridos desde la medianoche.
    Ejemplo: "2023-10-01 14:30" -> 870
    """
    dt = datetime.strptime(fecha_hora_str, "%Y-%m-%d %H:%M")
    return dt.hour * 60 + dt.minute

def cambiarFormatoViento(dir):
    """ Convierte una dirección de viento en texto a un valor numérico en grados.
    Devuelve -1 si es "CALM" (sin viento).
    Ejemplo: "N" -> 0, "NE" -> 45, "CALM" -> -1
    """
    mapa = {"CALM": -1, "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90,
            "ESE": 112.5, "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5,
            "SW": 225, "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5}
    return mapa.get(dir)

def realizar_predicciones(modelo, df_lugares_filtrado, api_key):
    """
    Para cada fila de df_lugares_filtrado:
      - Verifica latitud/longitud válidas
      - Llama a la API de weatherapi.com current.json
      - Extrae y transforma localtime, temp_c, dewpoint_c, humidity, wind_dir, wind_kph
      - Construye DataFrame con ['Time', 'Temperature', 'Dew Point', 'Humidity', 'Wind', 'Wind Speed']
      - Llama modelo.predict, mapea índice a texto con lista condiciones
      - Añade columna 'Prediccion'
    """
    condiciones = [
        'Nublado', 'Considerablemente nublado', 'Despejado', 'Niebla', 'Bruma',
        'Lluvia intensa', 'Tormenta electrica intensa', 'Neblina', 'Lluvia', 'Truenos'
    ]
    predicciones = []
    # Recorremos con iterrows sobre copia para no modificar original
    for idx, fila in df_lugares_filtrado.iterrows():
        lat = fila.get('latitud')
        lon = fila.get('longitud')
        # Validar lat/lon: no nulos, rango correcto, y evitar (0,0) si es un dato inválido
        if pd.isnull(lat) or pd.isnull(lon):
            print(f"[Aviso] Fila índice {idx}: latitud/longitud faltante, se omite predicción.")
            predicciones.append(None)
            continue
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception:
            print(f"[Aviso] Fila índice {idx}: lat/long no convertible a float: ({lat}, {lon}).")
            predicciones.append(None)
            continue
        # Rango válido
        if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
            print(f"[Aviso] Fila índice {idx}: lat/lon fuera de rango: ({lat_f}, {lon_f}).")
            predicciones.append(None)
            continue
        # Evitar (0,0) como coordenadas inválidas
        if lat_f == 0.0 and lon_f == 0.0:
            print(f"[Aviso] Fila índice {idx}: coordenadas (0.0,0.0) se consideran inválidas, se omite.")
            predicciones.append(None)
            continue

        location = f"{lat_f}, {lon_f}"
        url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=no"
        try:
            response = requests.get(url, timeout=10)
        except Exception as e:
            print(f"[Error] Fila índice {idx}: excepción en requests.get para {location}: {e}")
            predicciones.append(None)
            continue

        if response.status_code != 200:
            print(f"[Error] Fila índice {idx}: status_code={response.status_code} en la petición para {location}.")
            predicciones.append(None)
            continue

        try:
            data = response.json()
        except ValueError:
            print(f"[Error] Fila índice {idx}: no se pudo parsear JSON para {location}.")
            predicciones.append(None)
            continue

        # Extraemos los campos necesarios, manejando posibles faltantes
        try:
            hora_API = data['location']['localtime']  # "YYYY-MM-DD HH:MM"
            temperatura = data['current']['temp_c']
            puntoRocio = data['current']['dewpoint_c']
            humedad = data['current']['humidity']
            dirViento_API = data['current']['wind_dir']
            velocidadViento = data['current']['wind_kph']
        except KeyError as e:
            print(f"[Error] Fila índice {idx}: faltante clave en JSON: {e}")
            predicciones.append(None)
            continue
        # Transformar hora y viento:
        try:
            hora_num = cambiarFormatoHora(hora_API)
        except Exception as e:
            print(f"[Error] Fila índice {idx}: cambiarFormatoHora fallo con '{hora_API}': {e}")
            predicciones.append(None)
            continue

        dir_viento_num = cambiarFormatoViento(dirViento_API)
        if dir_viento_num is None:
            # Si no se reconoce la dirección, mostrar aviso
            print(f"[Aviso] Fila índice {idx}: dirección de viento '{dirViento_API}' no reconocida.")
        # Construir DataFrame de un solo registro
        columnas = ['Time', 'Temperature', 'Dew Point', 'Humidity', 'Wind', 'Wind Speed']
        fila_predict = pd.DataFrame([[
            hora_num,
            temperatura,
            puntoRocio,
            humedad,
            dir_viento_num,
            velocidadViento
        ]], columns=columnas)

        # Llamar a predict
        try:
            pred_array = modelo.predict(fila_predict)
            if len(pred_array) == 0:
                print(f"[Error] Fila índice {idx}: modelo.predict devolvió array vacío.")
                predicciones.append(None)
                continue
            pred_idx = pred_array[0]
        except Exception as e:
            print(f"[Error] Fila índice {idx}: excepción en modelo.predict: {e}")
            predicciones.append(None)
            continue

        # Mapear a texto si es índice válido
        if isinstance(pred_idx, (int, np.integer)) and 0 <= pred_idx < len(condiciones):
            etiqueta = condiciones[pred_idx]
        else:
            print(f"[Aviso] Fila índice {idx}: índice predicción {pred_idx} fuera de rango o no entero.")
            etiqueta = str(pred_idx)
        predicciones.append(etiqueta)

    # Asignar columna en copia
    df = df_lugares_filtrado.copy()
    df.loc[:, 'Prediccion'] = predicciones
    return df

def crear_matriz_distancias(df_lugares):
    """
    Crea una matriz de distancias entre lugares turísticos usando la fórmula de Haversine.
    Devuelve un DataFrame cuadrado con índices y columnas como IDs de lugares.
    """
    ids = df_lugares['id'].values
    lats = df_lugares['latitud'].astype(float).values
    lons = df_lugares['longitud'].astype(float).values

    # Convertir grados a radianes
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)

    # Preparar matrices 
    # lat1 tendrá forma (N,1), lat2 (1,N), similar para lon
    lat1 = lat_rad[:, np.newaxis]
    lat2 = lat_rad[np.newaxis, :]
    lon1 = lon_rad[:, np.newaxis]
    lon2 = lon_rad[np.newaxis, :]

    # Diferencias
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Fórmula de Haversine
    # a = sin^2(dlat/2) + cos(lat1)*cos(lat2)*sin^2(dlon/2)
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    # c = 2 * arcsin(min(1, sqrt(a)))  — usamos arctan2 para estabilidad
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))

    # Radio de la Tierra en km
    R = 6371.0
    dist_matrix = R * c  # forma (N, N)

    # Construir DataFrame con índices y columnas = IDs
    df_dist = pd.DataFrame(dist_matrix, index=ids, columns=ids)
    
    return df_dist

def decimal_to_hhmm(val):
    """
    Convierte un valor en horas decimales (por ejemplo 11.5) a cadena "HH:MM".
    """
    horas = int(val)
    minutos = int(round((val - horas) * 60))
    # Ajuste si el redondeo llega a 60
    if minutos >= 60:
        horas += minutos // 60
        minutos = minutos % 60
    return f"{horas:02d}:{minutos:02d}"


def optimizar_ruta(df_matriz_distancias, df_lugares_filtrado, municipio_inicio, hora_inicio, hora_fin, tiempo_estimado):
    """
    Optimiza la ruta para visitar la mayor cantidad de lugares dentro de las horas disponibles
    y minimizar la distancia recorrida, usando recocido simulado sobre permutaciones,
    con inicialización greedy y temperatura inicial basada en desviación estándar de vecinos.

    Parámetros:
    - df_matriz_distancias: DataFrame cuadrado con distancias penalizadas entre lugares. Índices y columnas deben ser IDs.
    - df_lugares_filtrado: DataFrame con columnas:
        - 'id': identificador que coincide con índices de df_matriz_distancias
        - 'latitud', 'longitud': coordenadas
        - 'Tiempo_Estancia_Promedio': tiempo en horas de estancia para cada lugar.
        - 'Hora_Apertura' y 'Hora_Cierre': ventanas de tiempo para cada lugar.
    - municipio_inicio: Series o dict con 'latitud', 'longitud' del punto de partida.
    - hora_inicio: float, hora inicio del viaje en decimal.
    - hora_fin: float, hora fin del viaje en decimal.
    - tiempo_estimado: float, horas de tiempo "extra" por viaje.

    Retorna:
    - best_route_info: DataFrame con ['id','arrival','departure','travel_distance_prev','travel_time_prev']
    - best_score: tupla (count_visitas, total_distance)
    """
    # Velocidad promedio km/h
    velocidad_promedio_kmph = 50.0

    # Lista de IDs y longitud
    lugares = list(df_lugares_filtrado['id'].astype(int).values)
    n = len(lugares)
    if n == 0:
        print("No hay lugares para optimizar la ruta.")
        return None, None

    if len(set(lugares)) != n:
        print("[Error] IDs de lugares no únicos.")
        return None, None

    # Construir matriz de distancias alineada
    dist_matrix = np.zeros((n, n))
    for i, id_i in enumerate(lugares):
        for j, id_j in enumerate(lugares):
            dist_matrix[i, j] = df_matriz_distancias.loc[int(id_i), int(id_j)]

    # Calcular distancias desde inicio a cada lugar
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
        return R * c

    try:
        lat0 = float(municipio_inicio['latitud'])
        lon0 = float(municipio_inicio['longitud'])
    except Exception:
        print("[Aviso] Coordenadas de inicio inválidas.")
        lat0 = lon0 = None

    dist_start = np.zeros(n)
    if lat0 is not None and lon0 is not None:
        for i, row in enumerate(df_lugares_filtrado.itertuples()):
            try:
                lat_i = float(row.latitud)
                lon_i = float(row.longitud)
                dist_start[i] = haversine(lat0, lon0, lat_i, lon_i)
            except Exception:
                dist_start[i] = np.inf
    else:
        dist_start[:] = np.inf

    # Ventanas
    has_window = ('Hora_Apertura' in df_lugares_filtrado.columns and 'Hora_Cierre' in df_lugares_filtrado.columns)
    apertura = np.zeros(n)
    cierre = np.full(n, 24.0)
    if has_window:
        for i, row in enumerate(df_lugares_filtrado.itertuples()):
            ha = getattr(row, 'Hora_Apertura')
            hc = getattr(row, 'Hora_Cierre')
            def parse_hora(x):
                if pd.isnull(x):
                    return 0.0
                if isinstance(x, (int, float)):
                    return float(x)
                if isinstance(x, str):
                    try:
                        dt = datetime.strptime(x.strip(), "%H:%M")
                        return dt.hour + dt.minute/60.0
                    except Exception:
                        try:
                            return float(x)
                        except:
                            return 0.0
                return 0.0
            apertura[i] = parse_hora(ha)
            cierre[i] = parse_hora(hc)

    # Tiempos de estancia
    stay_times = np.zeros(n)
    for i, row in enumerate(df_lugares_filtrado.itertuples()):
        val = getattr(row, 'Tiempo_Estancia_Promedio', None)
        try:
            stay_times[i] = float(val)
        except Exception:
            stay_times[i] = 0.0

    # Evaluar ruta: simula hasta hora_fin, devuelve (count, total_distance)
    def evaluar_ruta(order):
        current_time = hora_inicio
        total_distance = 0.0
        count = 0
        # Primer lugar
        idx0 = order[0]
        d0 = dist_start[idx0]
        t0 = d0/velocidad_promedio_kmph + tiempo_estimado
        arrival0 = current_time + t0
        if has_window:
            if arrival0 + stay_times[idx0] > cierre[idx0]:
                return (0, np.inf)
            if arrival0 < apertura[idx0]:
                arrival0 = apertura[idx0]
        departure0 = arrival0 + stay_times[idx0]
        if departure0 > hora_fin:
            return (0, np.inf)
        current_time = departure0
        total_distance += d0
        count = 1
        # Siguientes
        for prev, curr in zip(order[:-1], order[1:]):
            dij = dist_matrix[prev, curr]
            tij = dij/velocidad_promedio_kmph + tiempo_estimado
            arrival = current_time + tij
            if has_window:
                if arrival + stay_times[curr] > cierre[curr]:
                    break
                if arrival < apertura[curr]:
                    arrival = apertura[curr]
            departure = arrival + stay_times[curr]
            if departure > hora_fin:
                break
            current_time = departure
            total_distance += dij
            count += 1
        return (count, total_distance)

    # Cost function: minimizar => -count * BIG + total_distance
    BIG = n * (df_matriz_distancias.values.max() + 1)

    # Construir solución inicial greedy
    unvisited = set(range(n))
    order_greedy = []
    current_time = hora_inicio
    current_idx = None  # desde inicio
    # elegir primer lugar: mínimo dist_start factible
    best_first = None
    best_d0 = np.inf
    for i in range(n):
        d0 = dist_start[i]
        t0 = d0/velocidad_promedio_kmph + tiempo_estimado
        arrival = current_time + t0
        if has_window:
            if arrival + stay_times[i] > cierre[i]:
                continue
            if arrival < apertura[i]:
                arrival_mod = apertura[i]
            else:
                arrival_mod = arrival
        else:
            arrival_mod = arrival
        departure = arrival_mod + stay_times[i]
        if departure <= hora_fin and d0 < best_d0:
            best_d0 = d0
            best_first = i
    if best_first is not None:
        order_greedy.append(best_first)
        unvisited.remove(best_first)
        # actualizar tiempo y posición
        arrival = current_time + best_d0/velocidad_promedio_kmph + tiempo_estimado
        if has_window and arrival < apertura[best_first]:
            arrival = apertura[best_first]
        current_time = arrival + stay_times[best_first]
        current_idx = best_first

        # iterar greedy siguientes
        while unvisited:
            mejor = None
            best_dist = np.inf
            for j in unvisited:
                d = dist_matrix[current_idx, j]
                t_viaje = d/velocidad_promedio_kmph + tiempo_estimado
                arrival_j = current_time + t_viaje
                if has_window:
                    if arrival_j + stay_times[j] > cierre[j]:
                        continue
                    if arrival_j < apertura[j]:
                        arrival_j_mod = apertura[j]
                    else:
                        arrival_j_mod = arrival_j
                else:
                    arrival_j_mod = arrival_j
                departure_j = arrival_j_mod + stay_times[j]
                if departure_j <= hora_fin and d < best_dist:
                    best_dist = d
                    mejor = j
            if mejor is None:
                break
            # agregar mejor
            order_greedy.append(mejor)
            unvisited.remove(mejor)
            # actualizar tiempo y posición
            d = best_dist
            arrival = current_time + d/velocidad_promedio_kmph + tiempo_estimado
            if has_window and arrival < apertura[mejor]:
                arrival = apertura[mejor]
            current_time = arrival + stay_times[mejor]
            current_idx = mejor
    else:
        # no hay lugar factible como primero
        order_greedy = list(range(n))
        np.random.shuffle(order_greedy)

    # Completar orden inicial con el resto en orden aleatorio
    remaining = list(unvisited)
    np.random.shuffle(remaining)
    initial_order = order_greedy + remaining

    # Evaluar inicial
    cnt_init, dist_init = evaluar_ruta(initial_order)
    current_order = initial_order.copy()
    current_cost = -cnt_init * BIG + dist_init
    best_order = current_order.copy()
    best_cost = current_cost
    best_score = (cnt_init, dist_init)

    # Calcular temperatura inicial T0 como desviación estándar de diferencias de 100 primeros vecinos
    diffs = []
    for _ in range(100):
        # vecino: swap en la parte visitable
        i, j = np.random.choice(n, 2, replace=False)
        neigh = current_order.copy()
        neigh[i], neigh[j] = neigh[j], neigh[i]
        cnt_n, dist_n = evaluar_ruta(neigh)
        cost_n = -cnt_n * BIG + dist_n
        diffs.append(abs(cost_n - current_cost))
    T0 = np.std(diffs)
    if T0 <= 0:
        T0 = 1.0  # Temperatura inicial minima para evitar división por cero

    # Parám. recocido
    T = T0
    T_end = 1e-5
    alpha = 0.999
    iter_per_T = max(100, n * 10)

    # Simulated annealing
    while T > T_end:
        for _ in range(iter_per_T):
            i, j = np.random.choice(n, 2, replace=False)
            new_order = current_order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
            cnt_new, dist_new = evaluar_ruta(new_order)
            cost_new = -cnt_new * BIG + dist_new
            delta = cost_new - current_cost
            if cost_new < current_cost or np.random.rand() < np.exp(-delta / T):
                current_order = new_order
                current_cost = cost_new
                if cost_new < best_cost:
                    best_order = new_order.copy()
                    best_cost = cost_new
                    best_score = (cnt_new, dist_new)
        T *= alpha

    # Construir DataFrame de la mejor ruta
    cnt_best, _ = best_score
    if cnt_best == 0:
        print("No es posible visitar ningún lugar dentro del tiempo dado.")
        return None, None

    rows = []
    current_time = hora_inicio
    # Primer lugar
    idx0 = best_order[0]
    d0 = dist_start[idx0]
    t0 = d0/velocidad_promedio_kmph + tiempo_estimado
    arrival0 = current_time + t0
    if has_window and arrival0 < apertura[idx0]:
        arrival0 = apertura[idx0]
    departure0 = arrival0 + stay_times[idx0]
    rows.append({'id': lugares[idx0], 'arrival': arrival0, 'departure': departure0,
                 'travel_distance_prev': d0, 'travel_time_prev': t0})
    current_time = departure0
    visits = 1
    for prev, curr in zip(best_order[:-1], best_order[1:]):
        if visits >= cnt_best:
            break
        dij = dist_matrix[prev, curr]
        tij = dij/velocidad_promedio_kmph + tiempo_estimado
        arrival = current_time + tij
        if has_window:
            if arrival + stay_times[curr] > cierre[curr]:
                break
            if arrival < apertura[curr]:
                arrival = apertura[curr]
        departure = arrival + stay_times[curr]
        if departure > hora_fin:
            break
        rows.append({'id': lugares[curr], 'arrival': arrival, 'departure': departure,
                     'travel_distance_prev': dij, 'travel_time_prev': tij})
        current_time = departure
        visits += 1

    best_route_info = pd.DataFrame(rows)
    return best_route_info, best_score
        
def main():
    # Cargar el CSV con la informacion completa de los lugares a visitar
    df_lugares = cargar_CSV('Lugares_Turisticos_Def.csv')
    
    tipos_lugares = df_lugares['tipo'].unique()
    print("Tipos de lugares disponibles:")
    print(tipos_lugares)
    
    # Filtrar los lugares que se quieren visitar
    lugares_quiere_visitar = ['Fonoteca', 'Fototeca'] 
    
    # Eliminar las filas que no son de los tipos especificados
    df_lugares_filtrado = eliminar_filas_tipos(df_lugares, lugares_quiere_visitar)
    
    # Hacer un reset_index para evitar problemas con los indices
    df_lugares_filtrado = df_lugares_filtrado.reset_index(drop=True)
    
    # Definir que dia de la semana que se quiere visitar los lugares
    dia_semana = 'Martes'
    
    # Filtrar por el día de la semana
    df_lugares_filtrado = df_lugares_filtrado[df_lugares_filtrado[dia_semana] == 1]
    
    # Verificar si hay lugares filtrados
    if df_lugares_filtrado.empty:
        print(f"No hay lugares disponibles para visitar el {dia_semana}.")
        return   
    
    # Imprimir el número de lugares filtrados
    print(f"Número de lugares filtrados para visitar el {dia_semana}: {len(df_lugares_filtrado)}")
    
    # Crear la matriz de distancias entre los lugares filtrados
    df_matriz_distancias = crear_matriz_distancias(df_lugares_filtrado)
    
    # Definir el tiempo de estancia promedio por tipo de lugar
    tiempo_estancia = {
        'Biblioteca': 1.5,
        'Casa de artesania': 1.0,
        'Catedral': 1.0,
        'Centro cultural': 2.0,
        'Fonoteca': 1.5,
        'Fototeca': 1.5,
        'Galeria': 1.0,
        'Museo': 2.0,
        'Patrimonio de la Humanidad': 2.5,
        'Teatro': 2.5,
        'Zona Arqueologica': 3.0
    }
    
    # Modificar el tiempo de estancia promedio en el DataFrame filtrado
    df_lugares_filtrado = modificar_tiempo_estancia_promedio(df_lugares_filtrado, tiempo_estancia)
    
    # Cargar modelo para predicciones climaticas
    model_path_local = "random_forest_model.pkl"
    Modelo_RandomForest = joblib.load(model_path_local)

    # Solicitud a la API de weatherapi.com
    api_key = "7f25124e580c4de6a2e00312251205"

    # Añadir la columna con su predicción al CSV
    df_lugares_filtrado = realizar_predicciones(Modelo_RandomForest, df_lugares_filtrado, api_key)
    
    #Establecer penalizaciones por condiciones climáticas
    penalizaciones = {
        'Nublado': 1.1,
        'Considerablemente nublado': 1.2,
        'Despejado': 1.0,
        'Niebla': 1.8,
        'Bruma': 1.3,
        'Lluvia intensa': 1.7,
        'Tormenta electrica intensa': 1.9,
        'Neblina': 1.6,
        'Lluvia': 1.5,
        'Truenos': 1.4
    }
    
    lambda_penalizacion = 1.0
    
    # Multiplicar las distancias por la penalización de la predicción
    for pos, fila in df_lugares_filtrado.reset_index(drop=True).iterrows():
        id_lugar = int(fila['id'])
        pred = fila['Prediccion']
        if pred in penalizaciones:
            penal = penalizaciones[pred] * lambda_penalizacion
            # Multiplicar toda la fila y columna correspondiente:
            mask = df_matriz_distancias.index != id_lugar
            df_matriz_distancias.loc[id_lugar, mask] *= penal
            df_matriz_distancias.loc[mask, id_lugar] *= penal
            # Mantener diagonal a 0:
            df_matriz_distancias.loc[id_lugar, id_lugar] = 0.0
            
    # Mostrar los municipios disponibles
    print("Municipios disponibles para iniciar el recorrido:")
    print(df_lugares_filtrado['nombre_municipio'].unique())
    
    # Elegir un municipio de inicio
    nombre_municipio_inicio = 'San Martín de las Pirámides'
    
    #Cargar el DataFrame de municipios
    df_municipios = cargar_CSV('listado_municipios.csv')
    
    municipio_inicio = df_municipios[df_municipios['nombre_municipio'] == nombre_municipio_inicio].iloc[0]
    
    # Verificar que el municipio de inicio tenga latitud y longitud
    if pd.isnull(municipio_inicio['latitud']) or pd.isnull(municipio_inicio['longitud']):
        print("El municipio de inicio no tiene coordenadas válidas.")
    else:
        print(f"Municipio de inicio: {municipio_inicio['nombre_municipio']} con ID {municipio_inicio['municipio_id']}")
    
    # Elegir la hora de comienzo del viaje
    hora_inicio = 7.00 # 7:00 AM
    
    # Elegir la hora de finalización del viaje
    hora_fin = 20.00 # 8:00 PM
    
    # Tiempo estimado para llegar a cada lugar
    tiempo_estimado = 0.5 # 30 minutos
    
    # Encontrar la ruta optima
    ruta_optima, costo_optimo = optimizar_ruta(
        df_matriz_distancias, 
        df_lugares_filtrado, 
        municipio_inicio, 
        hora_inicio, 
        hora_fin, 
        tiempo_estimado
    )

    print("Ruta óptima encontrada:")
    if ruta_optima is not None:
        count_visitas, dist_total = costo_optimo
        print(f"Cant. visitas: {count_visitas}, distancia total: {dist_total:.2f} km")
        ruta_optima.to_csv('ruta_optima.csv', index=False)
    else:
        print("No se pudo encontrar una ruta óptima viable.")
        
    # Imprimir la ruta con el nombre de los lugares, la hora de llegada, 
    # la hora de salida, la distancia recorrida y el tiempo de viaje entre los lugares
    # Las horas se mostraran en formato HH:MM
    if ruta_optima is not None:
        for index, row in ruta_optima.iterrows():
            id_lugar = row['id']
            lugar_info = df_lugares_filtrado[df_lugares_filtrado['id'] == id_lugar].iloc[0]
            nombre_lugar = lugar_info['nombre']
            arrival_time = decimal_to_hhmm(row['arrival'])
            departure_time = decimal_to_hhmm(row['departure'])
            travel_distance = row['travel_distance_prev']
            travel_time = row['travel_time_prev']
            travel_time_str = decimal_to_hhmm(travel_time)
            print(f"Id: {id_lugar}, Lugar: {nombre_lugar}, Llegada: {arrival_time}, Salida: {departure_time}, "
                    f"Distancia recorrida: {travel_distance:.2f} km, Tiempo de viaje total: {travel_time_str} hrs")
    else:
        print("No se pudo generar la ruta con los detalles de los lugares.")
    

if __name__ == "__main__":
    main()
