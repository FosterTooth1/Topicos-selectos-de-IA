import pandas as pd
import numpy as np
import joblib
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
    Hace copia explícita antes de asignar para evitar SettingWithCopyWarning.
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
    dt = datetime.strptime(fecha_hora_str, "%Y-%m-%d %H:%M")
    return dt.hour * 60 + dt.minute

def cambiarFormatoViento(dir):
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
        # Validar lat/lon: no nulos, rango correcto, y evitar (0,0) si es placeholder
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
        # Evitar (0,0) si sabes que indica ausencia de datos
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

        # Extraer campos con try:
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

        # Transformar hora y viento; asumimos funciones en scope:
        try:
            hora_num = cambiarFormatoHora(hora_API)
        except Exception as e:
            print(f"[Error] Fila índice {idx}: cambiarFormatoHora fallo con '{hora_API}': {e}")
            predicciones.append(None)
            continue

        dir_viento_num = cambiarFormatoViento(dirViento_API)
        if dir_viento_num is None:
            # Si no se reconoce la dirección, loguear aviso pero continuar con None
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

    # Asignar columna en copia para evitar SettingWithCopy
    df = df_lugares_filtrado.copy()
    df.loc[:, 'Prediccion'] = predicciones
    return df

    
def main():
    # Cargar el CSV con la informacion completa de los lugares a visitar
    df_lugares = cargar_CSV('Lugares_Turisticos_Def.csv')
    
    # Cargar el CSV con la informacion de la distancia entre lugares
    df_matriz_distancias = cargar_CSV('Matriz_Distancias.csv')
    
    # Filtrar los lugares que se quieren visitar
    lugares_quiere_visitar = ['Fonoteca', 'Fototeca', 'Museo']
    
    # Eliminar las filas que no son de los tipos especificados
    df_lugares_filtrado = eliminar_filas_tipos(df_lugares, lugares_quiere_visitar)
    
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
    # Rutas locales
    model_path_local = "random_forest_model.pkl"
    encoders_path_local = "label_encoders.pkl"

    # Cargar el modelo y los encoders
    Modelo_RandomForest = joblib.load(model_path_local)
    label_encoders = joblib.load(encoders_path_local)

    # Solicitud a la API
    api_key = "7f25124e580c4de6a2e00312251205"

    # Añadir la columna con su predicción al CSV
    df_lugares_filtrado = realizar_predicciones(Modelo_RandomForest, df_lugares_filtrado, api_key)
    print(df_lugares_filtrado.head())
    print(df_lugares_filtrado.info())
    
    df_lugares_filtrado.to_csv('Lugares_Turistico_Def_Predd.csv')
    
    
main()
