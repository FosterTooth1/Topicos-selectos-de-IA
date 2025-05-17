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