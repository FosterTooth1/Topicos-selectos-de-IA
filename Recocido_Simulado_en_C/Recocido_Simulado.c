#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

// Estructura de la solucion
typedef struct
{
    int *ruta;      // Ruta de la solucion
    double fitness; // Fitness de la solucion
} Solucion;

// Estructura para almacenar swaps
typedef struct
{
    int i; // Posición a modificar
    int j; // Posición objetivo del swap
} Swap;

// Estructura para ordenar distancias (Almacena la distancia y el índice)(Usado en la heurística de remoción de abruptos)
typedef struct
{
    double distancia;
    int indice;
} DistanciaOrdenada;

// Funciones principales del Recocido Simulado
// Asigna memoria para la solucion
Solucion *crear_solucion(int tamano, int longitud_permutacion);
// Crea una permutacion aleatoria para la solucion
void crear_permutacion(Solucion *solucion, int longitud_permutacion);
// La heurística se encarga de remover abruptos en la ruta intercamdiando ciudades mal posicionadas
void heuristica_abruptos(int *ruta, int num_ciudades, int m, double **distancias);
// Libera la memoria usada para la solucion
void liberar_solucion(Solucion *solucion);
// Calcula el fitness de la ruta actual
double calcular_fitness(int *ruta, double **distancias, int num_ciudades);
// Genera un vecino de la ruta actual (Operación 2-opt)
void generar_vecino(int *ruta_actual, int *ruta_vecino, int num_ciudades);
// Calcula la probabilidad de aceptación de un nuevo vecino
double probabilidad_aceptacion(double fitness_actual, double fitness_vecino, double temperatura);

// Funciones auxiliares de manipulación de arreglos (Usadas en la heurística de remoción de abruptos)
// Compara dos distancias para ordenarlas
int comparar_distancias(const void *a, const void *b);
// Inserta un elemento en una posición específica del arreglo
void insertar_en_posicion(int *array, int longitud, int elemento, int posicion);
// Elimina un elemento de una posición específica del arreglo
void eliminar_de_posicion(int *array, int longitud, int posicion);


Solucion *crear_solucion(int tamano, int longitud_permutacion)
{
    // Asigna memoria para la solucion
    Solucion *solucion = malloc(sizeof(Solucion));
    solucion->ruta = malloc(longitud_permutacion * sizeof(int));
    solucion->fitness = 0.0;
    return solucion;
}

void crear_permutacion(Solucion *solucion, int longitud_permutacion)
{
    // Inicializa la ruta con valores ordenados
    for (int i = 0; i < longitud_permutacion; i++)
    {
        solucion->ruta[i] = i;
    }

    // Mezcla la ruta utilizando el algoritmo de Fisher-Yates
    for (int i = longitud_permutacion - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        int temp = solucion->ruta[i];
        solucion->ruta[i] = solucion->ruta[j];
        solucion->ruta[j] = temp;
    }
}

// Heurística para remover abruptos en la ruta intercambiando ciudades mal posicionadas
// Recibe un puntero a la ruta, el número de ciudades total (longitud del genotipo), el número de ciudades más cercanas a considerar y la matriz de distancias
// No devuelve nada (todo se hace por referencia)
void heuristica_abruptos(int *ruta, int num_ciudades, int m, double **distancias)
{
    // Inicializamos memoria para un arreglo temporal para la manipulación de rutas
    int *ruta_temp = malloc(num_ciudades * sizeof(int));

    // Inicializamos meemoria para la estructura que sirve para ordenar distancias
    DistanciaOrdenada *dist_ordenadas = malloc(num_ciudades * sizeof(DistanciaOrdenada));

    // Para cada ciudad en la ruta
    for (int i = 0; i < num_ciudades; i++)
    {
        int ciudad_actual = ruta[i];

        // Se obtiene y ordenan las m ciudades más cercanas
        for (int j = 0; j < num_ciudades; j++)
        {
            dist_ordenadas[j].distancia = distancias[ciudad_actual][j];
            dist_ordenadas[j].indice = j;
        }
        qsort(dist_ordenadas, num_ciudades, sizeof(DistanciaOrdenada), comparar_distancias);

        // Encontramos la posición actual de la ciudad en la ruta
        int pos_actual = -1;
        for (int j = 0; j < num_ciudades; j++)
        {
            if (ruta[j] == ciudad_actual)
            {
                pos_actual = j;
                break;
            }
        }

        // Inicializamos el mejor costo con el costo actual
        double mejor_costo = calcular_fitness(ruta, distancias, num_ciudades);
        int mejor_posicion = pos_actual;
        int mejor_vecino = -1;

        // Probamos la inserción con las m ciudades más cercanas
        for (int j = 1; j <= m && j < num_ciudades; j++)
        {
            int ciudad_cercana = dist_ordenadas[j].indice;

            // Encontramos la posición de la ciudad cercana
            int pos_cercana = -1;
            for (int k = 0; k < num_ciudades; k++)
            {
                if (ruta[k] == ciudad_cercana)
                {
                    pos_cercana = k;
                    break;
                }
            }

            if (pos_cercana != -1)
            {
                // Probar inserción antes y después de la ciudad cercana
                for (int posicion_antes_o_despues = 0; posicion_antes_o_despues <= 1; posicion_antes_o_despues++)
                {
                    memcpy(ruta_temp, ruta, num_ciudades * sizeof(int));

                    // Eliminar de posición actual
                    eliminar_de_posicion(ruta_temp, num_ciudades, pos_actual);

                    // Insertar en nueva posición (antes o después de la ciudad cercana)
                    int nueva_pos = pos_cercana + posicion_antes_o_despues;
                    if (nueva_pos > pos_actual)
                        nueva_pos--;
                    if (nueva_pos >= num_ciudades)
                        nueva_pos = num_ciudades - 1;
                    insertar_en_posicion(ruta_temp, num_ciudades, ciudad_actual, nueva_pos);

                    // Evaluar el nuevo costo
                    double nuevo_costo = calcular_fitness(ruta_temp, distancias, num_ciudades);

                    // Actualizar el mejor costo y posición de la ciudad actual si es necesario
                    if (nuevo_costo < mejor_costo)
                    {
                        mejor_costo = nuevo_costo;
                        mejor_posicion = nueva_pos;
                        mejor_vecino = ciudad_cercana;
                    }
                }
            }
        }

        // Si se encontró un mejor vecino, actualizar la ruta
        if (mejor_vecino != -1 && mejor_posicion != pos_actual)
        {
            memcpy(ruta_temp, ruta, num_ciudades * sizeof(int));
            eliminar_de_posicion(ruta_temp, num_ciudades, pos_actual);
            insertar_en_posicion(ruta_temp, num_ciudades, ciudad_actual, mejor_posicion);
            memcpy(ruta, ruta_temp, num_ciudades * sizeof(int));
        }
    }

    // Liberamos memoria
    free(ruta_temp);
    free(dist_ordenadas);
}

// Funciones auxiliares de manipulación de arreglos (Usadas en la heurística de remoción de abruptos)

// Función de comparación para qsort
// Recibe dos punteros a distancia ordenada
// Devuelve un entero que indica la relación entre las distancias
int comparar_distancias(const void *a, const void *b)
{
    DistanciaOrdenada *da = (DistanciaOrdenada *)a;
    DistanciaOrdenada *db = (DistanciaOrdenada *)b;
    if (da->distancia < db->distancia)
        return -1;
    if (da->distancia > db->distancia)
        return 1;
    return 0;
}

// Función para insertar un elemento en una posición específica del array
// Recibe un puntero al array, la longitud del array, el elemento a insertar y la posición
// No devuelve nada (todo se hace por referencia)
void insertar_en_posicion(int *array, int longitud, int elemento, int posicion)
{
    for (int i = longitud - 1; i > posicion; i--)
    {
        array[i] = array[i - 1];
    }
    array[posicion] = elemento;
}

// Función para eliminar un elemento de una posición específica
// Recibe un puntero al array, la longitud del array y la posición
// No devuelve nada (todo se hace por referencia)
void eliminar_de_posicion(int *array, int longitud, int posicion)
{
    for (int i = posicion; i < longitud - 1; i++)
    {
        array[i] = array[i + 1];
    }
}

double calcular_fitness(int *ruta, double **distancias, int num_ciudades)
{
    double total = 0.0;
    for (int i = 0; i < num_ciudades - 1; i++)
    {
        total += distancias[ruta[i]][ruta[i + 1]];
    }
    // Regresar al punto inicial
    total += distancias[ruta[num_ciudades - 1]][ruta[0]];
    return total;
}

void generar_vecino(int *ruta_actual, int *ruta_vecino, int num_ciudades)
{
    memcpy(ruta_vecino, ruta_actual, num_ciudades * sizeof(int));

    // Operación 2-opt (inversión de subruta)
    int i = rand() % (num_ciudades - 1);
    int j = rand() % (num_ciudades - i) + i;

    // Invertir el segmento
    while (i < j)
    {
        int temp = ruta_vecino[i];
        ruta_vecino[i] = ruta_vecino[j];
        ruta_vecino[j] = temp;
        i++;
        j--;
    }
}

double probabilidad_aceptacion(double fitness_actual, double fitness_vecino, double temperatura)
{
    if (fitness_vecino < fitness_actual)
        return 1.0;
    return exp((fitness_actual - fitness_vecino) / temperatura);
}

void liberar_solucion(Solucion *solucion)
{
    free(solucion->ruta);
    free(solucion);
}


int main()
{
    // Iniciamos la medición del tiempo
    time_t inicio = time(NULL);

    srand(time(NULL));
    int longitud_ruta = 32;
    double temperatura_inicial;
    double temperatura_final = 0.000000001;

    // Parámetros adaptativos
    const int max_neighbours = 75; // L(T) = k·N, con k entre 10 y 100; N= 32
    const int max_successes = (int)(0.5 * max_neighbours);

    int num_generaciones = 1000;
    int m = 3;

    // Nombre del archivo con las distancias
    char *nombre_archivo = "Distancias_no_head.csv";

    // Reservamos memoria para la matriz que almacena las distancias
    double **distancias = malloc(longitud_ruta * sizeof(double *));
    for (int i = 0; i < longitud_ruta; i++)
    {
        distancias[i] = malloc(longitud_ruta * sizeof(double));
    }

    // Abrimos el archivo
    FILE *archivo = fopen(nombre_archivo, "r");
    if (!archivo)
    {
        perror("Error al abrir el archivo");
        return 1;
    }

    // Leemos el archivo y llenamos la matriz
    char linea[8192];
    int fila = 0;
    while (fgets(linea, sizeof(linea), archivo) && fila < longitud_ruta)
    {
        char *token = strtok(linea, ",");
        int columna = 0;
        while (token && columna < longitud_ruta)
        {
            distancias[fila][columna] = atof(token);
            token = strtok(NULL, ",");
            columna++;
        }
        fila++;
    }
    fclose(archivo);

    // Creamos un arreglo con los nombres de las ciudades
    char nombres_ciudades[32][19] = {
        "Aguascalientes", "Baja California", "Baja California Sur",
        "Campeche", "Chiapas", "Chihuahua", "Coahuila", "Colima", "Durango",
        "Guanajuato", "Guerrero", "Hidalgo", "Jalisco", "Estado de Mexico",
        "Michoacan", "Morelos", "Nayarit", "Nuevo Leon", "Oaxaca", "Puebla",
        "Queretaro", "Quintana Roo", "San Luis Potosi", "Sinaloa", "Sonora",
        "Tabasco", "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatan",
        "Zacatecas", "CDMX"};

    // Inicializamos la solucion
    Solucion *solucion = crear_solucion(longitud_ruta, longitud_ruta);

    // Creamos una permutacion aleatoria para la solucion
    crear_permutacion(solucion, longitud_ruta);

    // Aplicamos la heurística de remoción de abruptos
    heuristica_abruptos(solucion->ruta, longitud_ruta, m, distancias);

    // Inicialización del recocido
    Solucion *mejor = crear_solucion(longitud_ruta, longitud_ruta);
    Solucion *actual = crear_solucion(longitud_ruta, longitud_ruta);

    // Usar la solución heurística como inicial
    memcpy(actual->ruta, solucion->ruta, longitud_ruta * sizeof(int));
    actual->fitness = calcular_fitness(actual->ruta, distancias, longitud_ruta);

    memcpy(mejor->ruta, actual->ruta, longitud_ruta * sizeof(int));
    mejor->fitness = actual->fitness;

    int *vecino = malloc(longitud_ruta * sizeof(int));

    double suma = 0, suma_cuadrados = 0;
    for (int i = 0; i < 100; i++)
    {
        generar_vecino(actual->ruta, vecino, longitud_ruta);
        // Aplicar la heurística de remoción de abruptos al vecino
        heuristica_abruptos(vecino, longitud_ruta, m, distancias);
        double fit = calcular_fitness(vecino, distancias, longitud_ruta);
        suma += fit;
        suma_cuadrados += fit * fit;
    }
    double desviacion = sqrt((suma_cuadrados - suma * suma / 100) / 99);
    temperatura_inicial = desviacion; // Ajuste empírico

    // Calculo de la temperatura inicial
    // Usando un porcentaje de la mejor solución
    // temperatura_inicial = mejor->fitness * 0.3;

    double temperatura = temperatura_inicial;

    // Ciclo principal de recocido con enfriamiento logarítmico (Béltsman)
    for (int iter = 1; iter <= num_generaciones && temperatura > temperatura_final; iter++)
    {
        // Enfriamiento logarítmico de Béltsman:
        // T_k = T0 / ln(k + 1)
        temperatura = temperatura_inicial / log(iter + 1.0);

        int neighbours = 0;
        int successes = 0;

        // Fase de equilibrio: hasta max_neighbours o max_successes
        while (neighbours < max_neighbours && successes < max_successes)
        {
            generar_vecino(actual->ruta, vecino, longitud_ruta);
            double fit_vecino = calcular_fitness(vecino, distancias, longitud_ruta);

            if (probabilidad_aceptacion(actual->fitness, fit_vecino, temperatura) > ((double)rand() / RAND_MAX))
            {
                // Aceptamos el vecino
                memcpy(actual->ruta, vecino, longitud_ruta * sizeof(int));
                actual->fitness = fit_vecino;
                successes++;

                // Actualizamos el mejor si procede
                if (actual->fitness < mejor->fitness)
                {
                    memcpy(mejor->ruta, actual->ruta, longitud_ruta * sizeof(int));
                    mejor->fitness = actual->fitness;
                }
            }

            neighbours++;
        }

        // Aplicar heurística de remoción de abruptos tras cada enfriamiento
        heuristica_abruptos(actual->ruta,
                            longitud_ruta,
                            m,
                            distancias);
        actual->fitness = calcular_fitness(actual->ruta,
                                           distancias,
                                           longitud_ruta);

        // (Opcional) Mostrar info de cada paso de enfriamiento
        printf("Iter %3d | Temp = %.6f | Neighbors = %3d | Successes = %2d | Mejor = %.2f\n",
               iter, temperatura, neighbours, successes, mejor->fitness);
    }

    // Mostrar tiempo de ejecución
    time_t fin = time(NULL);
    double tiempo_ejecucion = difftime(fin, inicio);
    printf("Tiempo de ejecución: %.2f segundos\n", tiempo_ejecucion);

    // Liberar memoria y mostrar resultados
    free(vecino);
    printf("\nMejor ruta encontrada (%.2f km):\n", mejor->fitness);
    for (int i = 0; i < longitud_ruta; i++)
    {
        printf("%s -> ", nombres_ciudades[mejor->ruta[i]]);
    }
    printf("%s\n", nombres_ciudades[mejor->ruta[0]]);

    liberar_solucion(solucion);
    liberar_solucion(actual);
    liberar_solucion(mejor);
    for (int i = 0; i < longitud_ruta; i++)
    {
        free(distancias[i]);
    }
    free(distancias);
    return 0;
}