import pandas as pd
import requests
import time

# Leer CSV original
df = pd.read_csv("Lugares_Turisticos_Def.csv", encoding='utf-8-sig')

# Extraer municipios únicos con su ID
df_municipios = (
    df[['municipio_id', 'nombre_municipio']]
    .drop_duplicates()
    .sort_values('municipio_id')
    .reset_index(drop=True)
)

# Agregar ", Estado de México" 
df_municipios['query_municipio'] = df_municipios['nombre_municipio'] + ', Estado de México'

# Preparar headers para Nominatim
headers = {
    'User-Agent': 'MiAppGeocoding/1.0 (tu_email@dominio.com)'
}

def obtener_coordenadas(municipio_query):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': municipio_query,
        'format': 'json',
        'addressdetails': 1,
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
    except requests.exceptions.RequestException as e:
        return None, None
    if response.status_code != 200:
        # print(f"Status {response.status_code} para {municipio_query}: {response.text}")
        return None, None
    data = response.json()
    if data:
        # A veces lat/lon vienen como strings
        lat = data[0].get('lat')
        lon = data[0].get('lon')
        return lat, lon
    return None, None

# Iterar con pausa para no sobrepasar límite de peticiones
latitudes = []
longitudes = []
for idx, row in df_municipios.iterrows():
    municipio_query = row['query_municipio']
    lat, lon = obtener_coordenadas(municipio_query)
    latitudes.append(lat)
    longitudes.append(lon)
    time.sleep(1)  # al menos 1 segundo entre peticiones

df_municipios['latitud'] = latitudes
df_municipios['longitud'] = longitudes

# Mostrar resultados parciales para depurar
print(df_municipios[['municipio_id', 'nombre_municipio', 'latitud', 'longitud']])

# Guardar en CSV final
df_municipios.to_csv('listado_municipios.csv', index=False, encoding='utf-8-sig')
