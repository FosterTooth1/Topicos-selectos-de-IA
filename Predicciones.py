import joblib
import pandas as pd

# Rutas locales a los archivos que bajaste de Drive
model_path_local = "random_forest_model.pkl"
encoders_path_local = "label_encoders.pkl"

# Cargar el modelo y los encoders
Modelo_RandomForest = joblib.load(model_path_local)
label_encoders = joblib.load(encoders_path_local)

# Ejemplo: cargar nuevos datos para predecir
df_new = pd.read_csv("Prueba.csv")

# Aplicar codificación usando los encoders cargados
for col, le in label_encoders.items():
    if col in df_new.columns:
        df_new[col] = le.transform(df_new[col])

X_new = df_new.drop("Condition", axis=1, errors='ignore')  # si 'Condition' no existe, ignóralo

# Hacer predicciones
y_pred_new = Modelo_RandomForest.predict(X_new)

# Si quieres convertir las predicciones de vuelta a clases originales:
clases = label_encoders["Condition"].classes_
y_pred_labels = clases[y_pred_new]
print(y_pred_labels)
