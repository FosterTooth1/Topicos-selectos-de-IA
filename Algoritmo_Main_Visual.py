import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, time
import pydeck as pdk

def cargar_CSV(nombre_archivo):
    try:
        df = pd.read_csv(nombre_archivo)
        return df
    except Exception as e:
        st.error(f"No se pudo cargar el CSV '{nombre_archivo}': {e}")
        return None

def eliminar_filas_tipos(df, filas_a_conservar):
    if df is not None and 'tipo' in df.columns:
        return df[df['tipo'].isin(filas_a_conservar)].copy()
    else:
        return None

def modificar_tiempo_estancia_promedio(df, tiempo_estancia):
    if df is not None and 'tipo' in df.columns:
        df2 = df.copy()
        df2.loc[:, 'Tiempo_Estancia_Promedio'] = df2['tipo'].map(tiempo_estancia)
        return df2
    else:
        return None

def cambiarFormatoHora(fecha_hora_str):
    dt = datetime.strptime(fecha_hora_str, "%Y-%m-%d %H:%M")
    return dt.hour * 60 + dt.minute

def cambiarFormatoViento(dir):
    mapa = {"CALM": -1, "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90,
            "ESE": 112.5, "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5,
            "SW": 225, "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5}
    return mapa.get(dir)

@st.cache_data
def realizar_predicciones(_modelo, df_lugares_filtrado, api_key):
    condiciones = [
        'Nublado', 'Considerablemente nublado', 'Despejado', 'Niebla', 'Bruma',
        'Lluvia intensa', 'Tormenta electrica intensa', 'Neblina', 'Lluvia', 'Truenos'
    ]
    predicciones = []
    for idx, fila in df_lugares_filtrado.iterrows():
        lat = fila.get('latitud')
        lon = fila.get('longitud')
        if pd.isnull(lat) or pd.isnull(lon):
            predicciones.append(None); continue
        try:
            lat_f = float(lat); lon_f = float(lon)
        except:
            predicciones.append(None); continue
        if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
            predicciones.append(None); continue
        if lat_f == 0.0 and lon_f == 0.0:
            predicciones.append(None); continue
        location = f"{lat_f},{lon_f}"
        url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=no"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                predicciones.append(None); continue
            data = response.json()
        except:
            predicciones.append(None); continue
        try:
            hora_API = data['location']['localtime']
            temperatura = data['current']['temp_c']
            puntoRocio = data['current']['dewpoint_c']
            humedad = data['current']['humidity']
            dirViento_API = data['current']['wind_dir']
            velocidadViento = data['current']['wind_kph']
        except KeyError:
            predicciones.append(None); continue
        try:
            hora_num = cambiarFormatoHora(hora_API)
        except:
            predicciones.append(None); continue
        dir_viento_num = cambiarFormatoViento(dirViento_API)
        fila_predict = pd.DataFrame([[hora_num, temperatura, puntoRocio, humedad, dir_viento_num, velocidadViento]],
                                    columns=['Time','Temperature','Dew Point','Humidity','Wind','Wind Speed'])
        try:
            pred_array = _modelo.predict(fila_predict)
            if len(pred_array)==0:
                predicciones.append(None); continue
            pred_idx = pred_array[0]
        except:
            predicciones.append(None); continue
        if isinstance(pred_idx, (int, np.integer)) and 0 <= pred_idx < len(condiciones):
            etiqueta = condiciones[pred_idx]
        else:
            etiqueta = str(pred_idx)
        predicciones.append(etiqueta)
    df2 = df_lugares_filtrado.copy()
    df2.loc[:, 'Prediccion'] = predicciones
    return df2

@st.cache_data
def crear_matriz_distancias(df_lugares):
    ids = df_lugares['id'].values
    lats = df_lugares['latitud'].astype(float).values
    lons = df_lugares['longitud'].astype(float).values
    lat_rad = np.radians(lats); lon_rad = np.radians(lons)
    lat1 = lat_rad[:, np.newaxis]; lat2 = lat_rad[np.newaxis, :]
    lon1 = lon_rad[:, np.newaxis]; lon2 = lon_rad[np.newaxis, :]
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0-a)))
    R = 6371.0
    dist_matrix = R * c
    df_dist = pd.DataFrame(dist_matrix, index=ids, columns=ids)
    return df_dist

def decimal_to_hhmm(val):
    horas = int(val)
    minutos = int(round((val - horas)*60))
    if minutos >= 60:
        horas += minutos//60
        minutos = minutos%60
    return f"{horas:02d}:{minutos:02d}"

@st.cache_data
def optimizar_ruta(df_matriz_distancias, df_lugares_filtrado, municipio_inicio, hora_inicio, hora_fin, tiempo_estimado):
    velocidad_promedio_kmph = 50.0
    lugares = list(df_lugares_filtrado['id'].astype(int).values)
    n = len(lugares)
    if n == 0 or len(set(lugares)) != n:
        return None, None
    dist_matrix = np.zeros((n,n))
    for i,id_i in enumerate(lugares):
        for j,id_j in enumerate(lugares):
            dist_matrix[i,j] = df_matriz_distancias.loc[int(id_i), int(id_j)]
    def haversine(lat1, lon1, lat2, lon2):
        R=6371.0
        lat1,lon1,lat2,lon2 = map(np.radians, [lat1,lon1,lat2,lon2])
        dlat=lat2-lat1; dlon=lon2-lon1
        a=np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
        c=2*np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0,1.0-a)))
        return R*c
    try:
        lat0 = float(municipio_inicio['latitud']); lon0 = float(municipio_inicio['longitud'])
    except:
        lat0 = lon0 = None
    dist_start = np.zeros(n)
    if lat0 is not None:
        for i,row in enumerate(df_lugares_filtrado.itertuples()):
            try:
                lat_i = float(row.latitud); lon_i = float(row.longitud)
                dist_start[i] = haversine(lat0, lon0, lat_i, lon_i)
            except:
                dist_start[i] = np.inf
    else:
        dist_start[:] = np.inf
    has_window = ('Hora_Apertura' in df_lugares_filtrado.columns and 'Hora_Cierre' in df_lugares_filtrado.columns)
    apertura = np.zeros(n); cierre = np.full(n, 24.0)
    if has_window:
        for i,row in enumerate(df_lugares_filtrado.itertuples()):
            def parse_hora(x):
                if pd.isnull(x): return 0.0
                if isinstance(x,(int,float)): return float(x)
                if isinstance(x,str):
                    try:
                        dt = datetime.strptime(x.strip(), "%H:%M")
                        return dt.hour + dt.minute/60.0
                    except:
                        try: return float(x)
                        except: return 0.0
                return 0.0
            apertura[i] = parse_hora(getattr(row,'Hora_Apertura'))
            cierre[i] = parse_hora(getattr(row,'Hora_Cierre'))
    stay_times = np.zeros(n)
    for i,row in enumerate(df_lugares_filtrado.itertuples()):
        val = getattr(row,'Tiempo_Estancia_Promedio',None)
        try: stay_times[i]=float(val)
        except: stay_times[i]=0.0
    def evaluar_ruta(order):
        current_time = hora_inicio; total_distance = 0.0; count = 0
        idx0 = order[0]; d0 = dist_start[idx0]
        t0 = d0/velocidad_promedio_kmph + tiempo_estimado
        arrival0 = current_time + t0
        if has_window:
            if arrival0 + stay_times[idx0] > cierre[idx0]: return (0, np.inf)
            if arrival0 < apertura[idx0]: arrival0 = apertura[idx0]
        departure0 = arrival0 + stay_times[idx0]
        if departure0 > hora_fin: return (0, np.inf)
        current_time = departure0; total_distance += d0; count = 1
        for prev,curr in zip(order[:-1], order[1:]):
            dij = dist_matrix[prev, curr]
            tij = dij/velocidad_promedio_kmph + tiempo_estimado
            arrival = current_time + tij
            if has_window:
                if arrival + stay_times[curr] > cierre[curr]: break
                if arrival < apertura[curr]: arrival = apertura[curr]
            departure = arrival + stay_times[curr]
            if departure > hora_fin: break
            current_time = departure; total_distance += dij; count += 1
        return (count, total_distance)
    BIG = n * (df_matriz_distancias.values.max() + 1)
    unvisited = set(range(n)); order_greedy = []; current_time = hora_inicio; current_idx = None
    best_first=None; best_d0=np.inf
    for i in range(n):
        d0 = dist_start[i]; t0 = d0/velocidad_promedio_kmph + tiempo_estimado
        arrival = current_time + t0
        if has_window:
            if arrival + stay_times[i] > cierre[i]: continue
            arrival_mod = apertura[i] if arrival < apertura[i] else arrival
        else:
            arrival_mod = arrival
        departure = arrival_mod + stay_times[i]
        if departure <= hora_fin and d0 < best_d0:
            best_d0 = d0; best_first = i
    if best_first is not None:
        order_greedy.append(best_first); unvisited.remove(best_first)
        arrival = current_time + best_d0/velocidad_promedio_kmph + tiempo_estimado
        if has_window and arrival < apertura[best_first]: arrival = apertura[best_first]
        current_time = arrival + stay_times[best_first]; current_idx = best_first
        while unvisited:
            mejor=None; best_dist=np.inf
            for j in unvisited:
                d = dist_matrix[current_idx,j]
                arrival_j = current_time + d/velocidad_promedio_kmph + tiempo_estimado
                if has_window:
                    if arrival_j + stay_times[j] > cierre[j]: continue
                    if arrival_j < apertura[j]: arrival_j_mod = apertura[j]
                    else: arrival_j_mod = arrival_j
                else:
                    arrival_j_mod = arrival_j
                departure_j = arrival_j_mod + stay_times[j]
                if departure_j <= hora_fin and d < best_dist:
                    best_dist = d; mejor = j
            if mejor is None: break
            order_greedy.append(mejor); unvisited.remove(mejor)
            arrival = current_time + best_dist/velocidad_promedio_kmph + tiempo_estimado
            if has_window and arrival < apertura[mejor]: arrival = apertura[mejor]
            current_time = arrival + stay_times[mejor]; current_idx = mejor
    else:
        order_greedy = list(range(n)); np.random.shuffle(order_greedy)
    remaining = list(unvisited); np.random.shuffle(remaining)
    initial_order = order_greedy + remaining
    cnt_init, dist_init = evaluar_ruta(initial_order)
    current_order = initial_order.copy(); current_cost = -cnt_init * BIG + dist_init
    best_order = current_order.copy(); best_cost = current_cost; best_score = (cnt_init, dist_init)
    diffs = []
    for _ in range(100):
        i,j = np.random.choice(n,2,replace=False)
        neigh = current_order.copy(); neigh[i],neigh[j] = neigh[j],neigh[i]
        cnt_n, dist_n = evaluar_ruta(neigh)
        cost_n = -cnt_n * BIG + dist_n
        diffs.append(abs(cost_n - current_cost))
    T0 = np.std(diffs)
    if T0 <= 0: T0 = 1.0
    T=T0; T_end=1e-5; alpha=0.999; iter_per_T = max(100, n*10)
    while T > T_end:
        for _ in range(iter_per_T):
            i,j = np.random.choice(n,2,replace=False)
            new_order = current_order.copy(); new_order[i],new_order[j] = new_order[j],new_order[i]
            cnt_new, dist_new = evaluar_ruta(new_order)
            cost_new = -cnt_new * BIG + dist_new
            delta = cost_new - current_cost
            if cost_new < current_cost or np.random.rand() < np.exp(-delta / T):
                current_order = new_order; current_cost = cost_new
                if cost_new < best_cost:
                    best_order = new_order.copy(); best_cost = cost_new; best_score = (cnt_new, dist_new)
        T *= alpha
    cnt_best, _ = best_score
    if cnt_best == 0:
        return None, None
    rows = []
    current_time = hora_inicio
    idx0 = best_order[0]; d0 = dist_start[idx0]; t0 = d0/velocidad_promedio_kmph + tiempo_estimado
    arrival0 = current_time + t0
    if has_window and arrival0 < apertura[idx0]: arrival0 = apertura[idx0]
    departure0 = arrival0 + stay_times[idx0]
    rows.append({'id': lugares[idx0], 'arrival': arrival0, 'departure': departure0,
                 'travel_distance_prev': d0, 'travel_time_prev': t0})
    current_time = departure0; visits=1
    for prev,curr in zip(best_order[:-1], best_order[1:]):
        if visits >= cnt_best: break
        dij = dist_matrix[prev, curr]; tij = dij/velocidad_promedio_kmph + tiempo_estimado
        arrival = current_time + tij
        if has_window:
            if arrival + stay_times[curr] > cierre[curr]: break
            if arrival < apertura[curr]: arrival = apertura[curr]
        departure = arrival + stay_times[curr]
        if departure > hora_fin: break
        rows.append({'id': lugares[curr], 'arrival': arrival, 'departure': departure,
                     'travel_distance_prev': dij, 'travel_time_prev': tij})
        current_time = departure; visits += 1
    best_route_info = pd.DataFrame(rows)
    return best_route_info, best_score

# Carga de datos preexistentes
CSV_LUGARES = 'Lugares_Turisticos_Def.csv'
CSV_MUNICIPIOS = 'listado_municipios.csv'
MODEL_PATH = 'random_forest_model.pkl'
API_KEY = '7f25124e580c4de6a2e00312251205'

@st.cache_data
def cargar_datos_iniciales():
    df_lugares = cargar_CSV(CSV_LUGARES)
    df_municipios = cargar_CSV(CSV_MUNICIPIOS)
    modelo = None
    try:
        modelo = joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo desde '{MODEL_PATH}': {e}")
    return df_lugares, df_municipios, modelo

# Interfaz Streamlit
st.title("Planificador de Ruta Turística")

df_lugares, df_municipios, Modelo_RandomForest = cargar_datos_iniciales()
if df_lugares is None or df_municipios is None or Modelo_RandomForest is None:
    st.error("Revisa que los archivos CSV y el modelo estén en las rutas correctas.")
    st.stop()

# Mostrar tipos disponibles
if 'tipo' not in df_lugares.columns:
    st.error("El CSV de lugares no tiene columna 'tipo'.")
    st.stop()
tipos = sorted(df_lugares['tipo'].dropna().unique().tolist())
st.sidebar.header("Selección de tipos de lugar")
tipos_seleccionados = st.sidebar.multiselect("Elige tipos a visitar:", tipos, default=tipos[:2] if tipos else [])

# Modificar tiempo por tipo
st.sidebar.header("Tiempo de estancia promedio (horas)")
tiempo_estancia = {}
for t in tipos_seleccionados:
    tiempo_estancia[t] = st.sidebar.number_input(f"{t} (hrs)", min_value=0.0, max_value=24.0, value=1.5, step=0.1)

# Día de la semana
dias = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
dia_semana = st.sidebar.selectbox("Día de la semana para visitar", dias, index=0)

# Lambda penalización
lambda_penalizacion = st.sidebar.slider("Lambda de penalización", 1.0, 10.0, 1.0, step=0.1)

# Filtrar por tipo y día
df_filtrado_tipo = eliminar_filas_tipos(df_lugares, tipos_seleccionados)
if df_filtrado_tipo is None:
    st.error("Error al filtrar por tipo.")
    st.stop()
if dia_semana not in df_filtrado_tipo.columns:
    st.warning(f"No existe columna '{dia_semana}' en el CSV; no se filtrará por día.")
    df_filtrado_dia = df_filtrado_tipo
else:
    df_filtrado_dia = df_filtrado_tipo[df_filtrado_tipo[dia_semana] == 1]

# Municipio de inicio
st.sidebar.header("Municipio de inicio")
municipios_disp = sorted(df_filtrado_dia['nombre_municipio'].dropna().unique().tolist())
nombre_municipio_inicio = None
if municipios_disp:
    nombre_municipio_inicio = st.sidebar.selectbox("Selecciona municipio de inicio:", municipios_disp)
else:
    st.sidebar.warning(f"No hay municipios tras filtrar por tipo y día '{dia_semana}'.")

# Horas de inicio/fin
hora_inicio_time = st.sidebar.time_input("Hora inicio", value=time(hour=7, minute=0))
hora_fin_time = st.sidebar.time_input("Hora fin", value=time(hour=20, minute=0))
hora_inicio = hora_inicio_time.hour + hora_inicio_time.minute/60.0
hora_fin = hora_fin_time.hour + hora_fin_time.minute/60.0
# Tiempo estimado llegada fijo o modificable
tiempo_estimado = st.sidebar.number_input("Tiempo de holgadura promedio (hrs)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

if st.sidebar.button("Calcular ruta óptima"):
    if df_filtrado_dia.empty:
        st.error(f"No hay lugares disponibles para visitar el {dia_semana}.")
        st.stop()
    st.write(f"Número de lugares tras filtro: {len(df_filtrado_dia)}")
    df_matriz = crear_matriz_distancias(df_filtrado_dia)
    df_lugares_filtrado = modificar_tiempo_estancia_promedio(df_filtrado_dia, tiempo_estancia)
    with st.spinner("Realizando predicciones climáticas..."):
        df_lugares_filtrado = realizar_predicciones(Modelo_RandomForest, df_lugares_filtrado, API_KEY)
    st.write("Predicciones añadidas.")
    penalizaciones = {
        'Nublado': 1.1, 'Considerablemente nublado': 1.2, 'Despejado': 1.0,
        'Niebla': 1.8, 'Bruma': 1.3, 'Lluvia intensa': 1.7,
        'Tormenta electrica intensa': 1.9, 'Neblina': 1.6,
        'Lluvia': 1.5, 'Truenos': 1.4
    }
    for _, fila in df_lugares_filtrado.iterrows():
        id_lugar = int(fila['id'])
        pred = fila['Prediccion']
        if pred in penalizaciones:
            penal = penalizaciones[pred] * lambda_penalizacion
            mask = df_matriz.index != id_lugar
            df_matriz.loc[id_lugar, mask] *= penal
            df_matriz.loc[mask, id_lugar] *= penal
            df_matriz.loc[id_lugar, id_lugar] = 0.0
    if nombre_municipio_inicio:
        df_muni = df_municipios[df_municipios['nombre_municipio'] == nombre_municipio_inicio]
        if df_muni.empty or pd.isnull(df_muni.iloc[0].get('latitud')) or pd.isnull(df_muni.iloc[0].get('longitud')):
            st.error("El municipio de inicio no tiene coordenadas válidas.")
            st.stop()
        municipio_inicio = df_muni.iloc[0]
    else:
        st.error("No se seleccionó municipio de inicio.")
        st.stop()
    with st.spinner("Optimizando ruta..."):
        ruta_optima, costo_optimo = optimizar_ruta(df_matriz, df_lugares_filtrado, municipio_inicio,
                                                   hora_inicio, hora_fin, tiempo_estimado)
    if ruta_optima is None:
        st.warning("No se pudo encontrar una ruta óptima viable.")
        st.stop()
    count_visitas, dist_total = costo_optimo
    st.success(f"Ruta óptima: {count_visitas} visitas, distancia total ≈ {dist_total:.2f} km")
    tabla = []
    for _, row in ruta_optima.iterrows():
        id_lugar = row['id']
        lugar_info = df_lugares_filtrado[df_lugares_filtrado['id'] == id_lugar].iloc[0]
        nombre_lugar = lugar_info['nombre']
        arrival_time = decimal_to_hhmm(row['arrival'])
        departure_time = decimal_to_hhmm(row['departure'])
        travel_distance = row['travel_distance_prev']
        travel_time = row['travel_time_prev']
        travel_time_str = decimal_to_hhmm(travel_time)
        tabla.append({
            'Id': id_lugar,
            'Lugar': nombre_lugar,
            'Llegada': arrival_time,
            'Salida': departure_time,
            'Distancia (km)': f"{travel_distance:.2f}",
            'Tiempo viaje': travel_time_str
        })
    df_result = pd.DataFrame(tabla)
    st.dataframe(df_result)
    coords = []
    for _, row in ruta_optima.iterrows():
        id_lugar = row['id']
        lugar_info = df_lugares_filtrado[df_lugares_filtrado['id'] == id_lugar].iloc[0]
        coords.append((float(lugar_info['latitud']), float(lugar_info['longitud'])))
    lat0 = float(municipio_inicio['latitud']); lon0 = float(municipio_inicio['longitud'])
    # DataFrame para st.map 
    df_map = pd.DataFrame(coords, columns=['lat','lon'])
    # st.map(df_map)
    # Crear segmentos de línea entre puntos: desde inicio al primero, luego entre cada par
    segments = []
    if coords:
        # desde inicio al primero
        segments.append({'path': [[lon0, lat0], [coords[0][1], coords[0][0]]]})
        # entre cada par:
        for i in range(len(coords)-1):
            start = coords[i]   # (lat, lon)
            end = coords[i+1]
            segments.append({'path': [[start[1], start[0]], [end[1], end[0]]]})
    # Debug opcional:
    #st.write("Segments:", segments)
    # Capa de línea: PathLayer en lugar de LineLayer
    layer_line = pdk.Layer(
        "PathLayer",
        data=segments,
        get_path="path",
        get_width=10,             # grosor en unidades de píxeles
        width_units="pixels",     # especifica que el ancho es en píxeles
        get_color=[0, 255, 0], # coloR verde
        pickable=False,
    )
    # Capa marcador inicio
    marcador_inicio = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame([{'lat': lat0, 'lon': lon0}]),
        get_position='[lon, lat]',
        get_radius=10,          # 10 píxeles
        radius_units="pixels",
        get_fill_color=[255, 0, 0],
        pickable=True
    )
    # Capa marcadores de visita
    marcador_puntos = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position='[lon, lat]',
        get_radius=8,           # 8 píxeles
        radius_units="pixels",
        get_fill_color=[0, 0, 255],
        pickable=True
    )
    # Centrar vista
    all_lats = [lat0] + [c[0] for c in coords]
    all_lons = [lon0] + [c[1] for c in coords]
    midpoint = {
        'latitude': np.mean(all_lats),
        'longitude': np.mean(all_lons)
    }
    view_state = pdk.ViewState(
        latitude=midpoint['latitude'],
        longitude=midpoint['longitude'],
        zoom=10
    )
    st.pydeck_chart(pdk.Deck(layers=[layer_line, marcador_inicio, marcador_puntos], initial_view_state=view_state))
    csv_bytes = df_result.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar ruta óptima CSV", data=csv_bytes, file_name="ruta_optima.csv", mime="text/csv")
