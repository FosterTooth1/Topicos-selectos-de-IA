import pandas as pd

df = pd.read_csv('Dataset_Final_SE.csv')

# Dropear la penultima columna que contiene la presión
df = df.drop(df.columns[-2], axis=1)

# Guardar el DataFrame modificado en un nuevo archivo CSV
df.to_csv('Dataset_Final_SE.csv', index=False)