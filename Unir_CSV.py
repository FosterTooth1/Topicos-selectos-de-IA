import os
import re
import pandas as pd

def clean_text(s):
    """
    Limpia cadenas: quita espacios al inicio/final, comillas sobrantes,
    y elimina caracteres no alfanuméricos al inicio si los hay.
    """
    if pd.isna(s):
        return s
    # Convertir a str
    s = str(s).strip()
    # Quitar comillas externas si existen
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    # Eliminar caracteres no alfanuméricos iniciales (permitir letras, números y acentos)
    s = re.sub(r'^[^0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+', '', s)
    # También puede eliminar espacios repetidos en medio
    s = re.sub(r'\s+', ' ', s)
    return s

def procesar_csv(filepath, tipo):
    """
    Lee un CSV, renombra columnas a formato estándar, limpia campos y añade columna 'tipo'.
    Retorna DataFrame.
    """
    df = pd.read_csv(filepath, dtype=str)  # Leer todo como str para limpiar
    # Mapear nombres de columna posibles al estándar:
    # id: puede acabar en "_id"
    # nombre: puede contener "_nombre"
    # municipio_id: contener "municipio" y "id"
    # nombre_municipio: contener "nom" y "mun"
    # longitud: "longitud" o similar
    # latitud: "latitud" o similar
    col_map = {}
    for col in df.columns:
        low = col.lower()
        if 'id' in low and 'municipio' not in low and 'nombre' not in low:
            # asumimos es id original, lo ignoraremos tras limpieza (reemplazaremos)
            col_map[col] = 'orig_id'
        elif 'nombre' in low and 'mun' not in low:
            col_map[col] = 'nombre'
        elif 'municipio' in low and 'id' in low:
            col_map[col] = 'municipio_id'
        elif ('nom' in low and 'mun' in low) or ('municipio' in low and 'nombre' in low):
            col_map[col] = 'nombre_municipio'
        elif 'longitud' in low:
            col_map[col] = 'longitud'
        elif 'latitud' in low:
            col_map[col] = 'latitud'
        else:
            # columna inesperada; la mantenemos con su nombre original o descartamos
            # aquí la descartamos:
            # col_map[col] = col
            pass
    df = df.rename(columns=col_map)
    # Mantener solo las columnas renombradas si existen:
    expected = ['orig_id', 'nombre', 'municipio_id', 'nombre_municipio', 'longitud', 'latitud']
    present = [c for c in expected if c in df.columns]
    df = df[present].copy()
    # Limpiar texto en las columnas de texto
    for c in ['nombre', 'nombre_municipio']:
        if c in df.columns:
            df[c] = df[c].apply(clean_text)
    # Convertir municipio_id a numérico (si falla, dejar NaN)
    if 'municipio_id' in df.columns:
        df['municipio_id'] = pd.to_numeric(df['municipio_id'], errors='coerce').astype('Int64')
    # Convertir longitud/latitud a float (coerce NaN si inválido)
    for c in ['longitud', 'latitud']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # Añadir columna tipo
    df['tipo'] = tipo
    # Eliminamos columna orig_id luego de renumerar externamente
    # La dejamos por ahora para luego asignar nuevo ID
    return df

def unir_csvs_en_carpeta(folder_path, output_path):
    """
    Recorre todos los archivos .csv en folder_path, procesa cada uno y une en uno solo.
    Asigna nuevo id secuencial en columna 'id'.
    """
    all_dfs = []
    # Listar archivos CSV
    for fname in os.listdir(folder_path):
        if fname.lower().endswith('.csv'):
            tipo = os.path.splitext(fname)[0]  # por ejemplo "Biblioteca" o "Casa_artesania"
            # Normalizar tipo: minúsculas y reemplazar '_' por espacio
            tipo = tipo.lower().replace('_', ' ').strip()
            fullpath = os.path.join(folder_path, fname)
            try:
                df = procesar_csv(fullpath, tipo)
                all_dfs.append(df)
                print(f"Procesado: {fname}, filas: {len(df)}")
            except Exception as e:
                print(f"Error al procesar {fname}: {e}")
    # Concatenar
    if not all_dfs:
        print("No se encontraron CSV procesables.")
        return
    merged = pd.concat(all_dfs, ignore_index=True)
    # Asignar nuevo id secuencial empezando en 1
    merged.insert(0, 'id', range(1, len(merged) + 1))
    # Reordenar columnas: id, nombre, municipio_id, nombre_municipio, longitud, latitud, tipo
    cols = ['id']
    for c in ['nombre', 'municipio_id', 'nombre_municipio', 'longitud', 'latitud', 'tipo']:
        if c in merged.columns:
            cols.append(c)
    merged = merged[cols]
    # Guardar resultado
    merged.to_csv(output_path, index=False)
    print(f"CSV unido generado en: {output_path}, total filas: {len(merged)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unir CSVs de lugares turísticos en uno solo")
    parser.add_argument("carpeta", help="Ruta a la carpeta con los CSV")
    parser.add_argument("salida", help="Ruta del CSV de salida, ej. merged.csv")
    args = parser.parse_args()
    unir_csvs_en_carpeta(args.carpeta, args.salida)
