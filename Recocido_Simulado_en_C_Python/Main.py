import ctypes
from ctypes import c_int, c_double, c_char_p, POINTER, Structure, c_char, cast
import os
import matplotlib.pyplot as plt

class ResultadoRecocido(Structure):
    _fields_ = [
        ("recorrido", POINTER(c_int)),
        ("fitness", c_double),
        ("tiempo_ejecucion", c_double),
        ("nombres_ciudades", POINTER(c_char * 50 * 32)),  # Mismo formato que PSO/Genético
        ("longitud_recorrido", c_int),
        ("fitness_generaciones", POINTER(c_double)),
    ]

class AlgoritmoRecocido:
    def __init__(self, ruta_biblioteca):
        self.biblioteca = ctypes.CDLL(ruta_biblioteca)
        
        # Configuración de tipos igual que en Genético/PSO
        self.biblioteca.ejecutar_algoritmo_recocido.restype = POINTER(ResultadoRecocido)
        self.biblioteca.ejecutar_algoritmo_recocido.argtypes = [
            c_int,      # longitud_ruta
            c_int,      # num_generaciones
            c_double,   # tasa_enfriamiento
            c_double,   # temperatura_final
            c_int,      # max_neighbours
            c_int,      # m
            c_char_p,   # nombre_archivo
            c_int       # heuristica
        ]
        self.biblioteca.liberar_resultado.argtypes = [POINTER(ResultadoRecocido)]  # Mismo nombre de función

    def ejecutar(self, longitud_ruta, num_generaciones, tasa_enfriamiento,
               temperatura_final, max_neighbours, m, nombre_archivo, heuristica):
        try:
            nombre_archivo_bytes = nombre_archivo.encode('utf-8')
            
            resultado_ptr = self.biblioteca.ejecutar_algoritmo_recocido(
                c_int(longitud_ruta),
                c_int(num_generaciones),
                c_double(tasa_enfriamiento),
                c_double(temperatura_final),
                c_int(max_neighbours),
                c_int(m),
                nombre_archivo_bytes,
                c_int(heuristica)
            )
            
            if not resultado_ptr:
                raise RuntimeError("Error en ejecución del Recocido")
            
            resultado = resultado_ptr.contents
            
            # Copia de datos
            recorrido = [resultado.recorrido[i] for i in range(resultado.longitud_recorrido)]
            
            nombres_ciudades = []
            ciudades_array = cast(resultado.nombres_ciudades, POINTER(c_char * 50 * 32))
            for i in range(resultado.longitud_recorrido):
                nombre_bytes = bytes(ciudades_array.contents[i])
                nombre = nombre_bytes.decode('utf-8').split('\x00', 1)[0]
                nombres_ciudades.append(nombre)
            
            fitness_hist = [resultado.fitness_generaciones[i] for i in range(num_generaciones)]
            
            salida = {
                'recorrido': recorrido,
                'nombres_ciudades': nombres_ciudades,
                'fitness': resultado.fitness,
                'tiempo_ejecucion': resultado.tiempo_ejecucion,
                'fitness_generaciones': fitness_hist
            }
            
            self.biblioteca.liberar_resultado(resultado_ptr)
            
            return salida
            
        except Exception as e:
            raise RuntimeError(f"Error en Recocido Simulado: {str(e)}")

def main():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    nombre_biblioteca = "recocido.dll" if os.name == 'nt' else "librecocido.so"
    ruta_biblioteca = os.path.join(directorio_actual, nombre_biblioteca)
    
    rs = AlgoritmoRecocido(ruta_biblioteca)
    
    params = {
        'longitud_ruta': 32,
        'num_generaciones': 25000,
        'tasa_enfriamiento': 0.92,
        'temperatura_final': 0.000000001,
        'max_neighbours': 320,
        'm': 3,
        'nombre_archivo': "Distancias_no_head.csv",
        'heuristica': 0
    }
    
    resultado = rs.ejecutar(**params)
    
    print("\nMejor ruta Recocido:")
    #for i, (idx, nombre) in enumerate(zip(resultado['recorrido'], resultado['nombres_ciudades'])):
    #    print(f"{i+1}. {nombre} (índice: {idx})")
    print(f"\nFitness: {resultado['fitness']:.2f}")
    print(f"Tiempo: {resultado['tiempo_ejecucion']:.2f}s")
    
    plt.plot(resultado['fitness_generaciones'])
    plt.title("Evolución del Fitness - Recocido Simulado")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()