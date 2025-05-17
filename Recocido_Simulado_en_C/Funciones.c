#include "Biblioteca.h"


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