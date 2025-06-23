# Importar las librerías
from datetime import datetime
import pandas as pd
import joblib
import requests
import json

# Rutas locales a los archivos que bajaste de Drive
model_path_local = "random_forest_model.pkl"
encoders_path_local = "label_encoders.pkl"

# Cargar el modelo y los encoders
Modelo_RandomForest = joblib.load(model_path_local)
label_encoders = joblib.load(encoders_path_local)

# Funciones para cambiar formato de los datos HORA y DIRECCION VIENTO
def cambiarFormatoHora(fecha_hora_str):
    dt = datetime.strptime(fecha_hora_str, "%Y-%m-%d %H:%M")
    return dt.hour * 60 + dt.minute

def cambiarFormatoViento(dir):
    mapa = {"CALM": -1, "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90,
            "ESE": 112.5, "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5,
            "SW": 225, "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5}
    return mapa.get(dir)

# Solicitud a la API
api_key = "7f25124e580c4de6a2e00312251205"
location = "19.503943116188182, -99.14705946287188" # ESCOM

url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=no"
response = requests.get(url)
if response.status_code != 200:
    print(f"Error en la petición: status_code={response.status_code}")
    print("Texto de respuesta:", response.text)
    # Opcional: salir o lanzar excepción
    raise RuntimeError("La API devolvió un error, revisa la clave o la URL.")
try:
    data = response.json()
except ValueError:
    print("No se pudo parsear JSON. Contenido recibido:")
    print(response.text)
    raise
# Verificar que la respuesta contiene los datos esperados
data = response.json()

# Extraer los datos necesarios
lugar = data['location']['name']
pais = data['location']['country']

hora_API = data['location']['localtime']
temperatura = data['current']['temp_c']
puntoRocio = data['current']['dewpoint_c']
humedad = data['current']['humidity']
dirViento_API = data['current']['wind_dir']
velocidadViento = data['current']['wind_kph']

# Cambiar foramto de la hora y direccion obtenidos de la API
hora = cambiarFormatoHora(hora_API)
direccionViento = cambiarFormatoViento(dirViento_API)

# Realizar la prediccion
columnas = ['Time', 'Temperature', 'Dew Point', 'Humidity', 'Wind', 'Wind Speed']
nuevo_dato = [[hora, temperatura, puntoRocio, humedad, direccionViento, velocidadViento]]
nuevo_dato = pd.DataFrame(nuevo_dato, columns=columnas)
prediccion = Modelo_RandomForest.predict(nuevo_dato)

# Definir el nombre de las condiciones
condiciones = ['Nublado', 'Considerablemente nublado', 'Despejado', 'Niebla', 'Bruma',
               'Lluvia intensa', 'Tormenta electrica intensa', 'Neblina', 'Lluvia', 'Truenos']

# Mostrar datos obtenidos ya transformados
print("\nDATOS PARA LA PREDICCIÓN OBTENIDOS DE LA API")
print(f"  País: {pais}, Localidad: {lugar}")
print(f"  Hora: {hora_API}")
print(f"  Direccion Viento: {dirViento_API}, Velocidad viento: {velocidadViento} km/h")
print(f"  Temperatura: {temperatura} °C, Punto de Rocio: {puntoRocio} °C, Humedad: {humedad}%")

# Mostrar ID y clase predicha
print("\nID Predicción:", prediccion)
print(f"Clase predicha: {condiciones[prediccion[0]]}")