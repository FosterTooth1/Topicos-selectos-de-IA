# Sistema de Optimización de Rutas Turísticas en el Estado de México Integrando Predicción Climática

Este proyecto implementa un sistema que **ofrece rutas turísticas en el Estado de México**, considerando tanto la distancia entre puntos de interés como las **condiciones climáticas previstas** mediante modelos de *Machine Learning*.  

El desarrollo forma parte de la materia **Tópicos Selectos de Inteligencia Artificial** y combina técnicas de **algoritmos bioinspirados** y **machine learning** para ofrecer recorridos más seguros y atractivos.  

## Objetivo
Diseñar un sistema que conecte los lugares turísticos más relevantes del Estado de México, **minimizando la distancia total recorrida** y evitando condiciones meteorológicas adversas.  

Se integran:  
- **Modelos de Machine Learning** para predecir el clima.  
- **Algoritmos bioinspirados** para resolver el *Traveling Salesman Problem (TSP)* con restricciones climáticas.  

## Justificación
El Estado de México cuenta con más de 120 municipios y una gran oferta turística. Sin embargo, factores como el clima afectan la seguridad y satisfacción de los visitantes.  

Este sistema:  
- Reduce riesgos asociados a condiciones climáticas adversas.  
- Aumenta la eficiencia logística en recorridos turísticos.  
- Promueve la reactivación económica y cultural.  

## Datos Utilizados
### Lugares turísticos
Fuente: **Sistema de Información Cultural del Estado de México**  
- Datos en CSV con: Nombre, Ubicación (lat/long), Municipio, horarios de apertura/cierre. Disponible en:  https://historico.datos.gob.mx/busca/organization/cultura

### Datos climáticos
Fuente: **Weather Underground (2022-2024)**  
- Variables: temperatura, humedad, viento, precipitación, presión, condiciones meteorológicas, etc.  
- Procesados para entrenar y evaluar modelos de *Machine Learning*.  


## Modelo de Machine Learning
Se evaluaron diferentes algoritmos para **clasificar condiciones meteorológicas (10 categorías)**.
- Se seleccionó "Random Forest" debido a su desempeño.  
- División de datos: 70% entrenamiento, 30% prueba.  
- Dataset total: **80,910 registros**.  
- Se aplicó balanceo de clases para mejorar el rendimiento.  

## Algoritmos Bioinspirados
Se implementaron y compararon cuatro algoritmos para resolver el TSP con 32 puntos turísticos:  
- **Recocido Simulado (RS)**  
- **Búsqueda Tabú (BT)**  
- Otros algoritmos metaheurísticos  

Resultados:  
- **Búsqueda Tabú** obtuvo mejores resultados promedio.  
- **Recocido Simulado** fue elegido por su **simplicidad operativa y menor complejidad paramétrica**.  

## Encuesta de Penalización Climática
Para integrar la percepción de riesgo en la optimización de rutas:  
- Se encuestaron **100 personas**.  
- Se establecieron penalizaciones en función del tipo de clima (ej. lluvia intensa = alto riesgo).  

## Tecnologías
- **Lenguaje C** → Implementación de algoritmos bioinspirados.  
- **Python (Machine Learning)** → Entrenamiento y validación de modelos climáticos.  

## Ejecución
1. Clonar el repositorio
2. Instalar las dependencias necesarias para la visualización: streamlit, pandas, numpy, joblib, requests, datetime, pydeck
3. Ejecutar el archivo "Algoritmo_Main_Visual.py" en Python
