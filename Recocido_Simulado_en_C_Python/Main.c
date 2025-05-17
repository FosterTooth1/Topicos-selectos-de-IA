#include "Biblioteca.h"

typedef struct
{
    int *recorrido;
    double fitness;
    double tiempo_ejecucion;
    char (*nombres_ciudades)[50];
    int longitud_recorrido;
    double *fitness_generaciones;
} ResultadoRecocido;

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT ResultadoRecocido *ejecutar_algoritmo_recocido(int longitud_ruta,
                                                      int num_generaciones,
                                                      double tasa_enfriamiento,
                                                      double temperatura_final,
                                                      int max_neighbours,
                                                      int m,
                                                      char *nombre_archivo,
                                                      int heuristica)
{
    // Iniciamos la medición del tiempo
    time_t inicio = time(NULL);

    // Reservamos memoria para la matriz que almacena las distancias
    double **distancias = malloc(longitud_ruta * sizeof(double *));
    for (int i = 0; i < longitud_ruta; i++) {
        distancias[i] = malloc(longitud_ruta * sizeof(double));
    }

    // Abrimos el archivo
    FILE *archivo = fopen(nombre_archivo, "r");
    if (!archivo) {
        perror("Error al abrir el archivo");
        return NULL;
    }

    // Leemos el archivo y llenamos la matriz
    char linea[8192];
    int fila = 0;
    while (fgets(linea, sizeof(linea), archivo) && fila < longitud_ruta) {
        char *token = strtok(linea, ",");
        int columna = 0;
        while (token && columna < longitud_ruta) {
            distancias[fila][columna] = atof(token);
            token = strtok(NULL, ",");
            columna++;
        }
        fila++;
        //free(token);
    }
    fclose(archivo);

    // Creamos un arreglo con los nombres de las ciudades
    const char nombres_ciudades[32][50] = {
        "Aguascalientes", "Baja California", "Baja California Sur",
        "Campeche", "Chiapas", "Chihuahua", "Coahuila", "Colima", "Durango",
        "Guanajuato", "Guerrero", "Hidalgo", "Jalisco", "Estado de Mexico",
        "Michoacan", "Morelos", "Nayarit", "Nuevo Leon", "Oaxaca", "Puebla",
        "Queretaro", "Quintana Roo", "San Luis Potosi", "Sinaloa", "Sonora",
        "Tabasco", "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatan",
        "Zacatecas", "CDMX"
    };

    // 3) Preparar soluciones
    Solucion *sol = crear_solucion(1, longitud_ruta);
    crear_permutacion(sol, longitud_ruta);
    sol->fitness = calcular_fitness(sol->ruta, distancias, longitud_ruta);

    if (heuristica == 1)
    {
        heuristica_abruptos(sol->ruta, longitud_ruta, m, distancias);
        sol->fitness = calcular_fitness(sol->ruta, distancias, longitud_ruta);
    }

    Solucion *actual = crear_solucion(1, longitud_ruta);    
    Solucion *mejor = crear_solucion(1, longitud_ruta);
    
    memcpy(actual->ruta, sol->ruta, longitud_ruta * sizeof(int));
    actual->fitness = calcular_fitness(actual->ruta, distancias, longitud_ruta);
    memcpy(mejor->ruta, actual->ruta, longitud_ruta * sizeof(int));
    mejor->fitness = actual->fitness;

    int *vecino = malloc(longitud_ruta * sizeof(int));

    // 4) Calcular temperatura inicial (desviación típica de 100 muestras)
    double suma = 0, suma2 = 0;
    for (int i = 0; i < 100; i++)
    {
        generar_vecino(actual->ruta, vecino, longitud_ruta);
        if (heuristica == 1)
        {
            heuristica_abruptos(vecino, longitud_ruta, m, distancias);
        }
        double f = calcular_fitness(vecino, distancias, longitud_ruta);
        suma += f;
        suma2 += f * f;
    }
    double desv = sqrt((suma2 - suma * suma / 100) / 99);
    double T0 = desv;
    double T = T0;

    const int max_successes = (int)(0.5 * max_neighbours);

    // 5) Array para histórico de fitness
    double *fitness_generaciones = (double *)malloc(num_generaciones * sizeof(double));

    int k;

    // 6) Bucle de recocido
    for (k = 1; k <= num_generaciones && T > temperatura_final; k++)
    {
        // Enfriamiento logarítmico de Béltsman
        T = T0 / log(k + 1.0);

        int neigh = 0, succ = 0;
        while (neigh < max_neighbours && succ < max_successes)
        {
            generar_vecino(actual->ruta, vecino, longitud_ruta);
            double fv = calcular_fitness(vecino, distancias, longitud_ruta);
            double p = probabilidad_aceptacion(actual->fitness, fv, T);
            if (p > ((double)rand() / RAND_MAX))
            {
                memcpy(actual->ruta, vecino, longitud_ruta * sizeof(int));
                actual->fitness = fv;
                succ++;
                if (fv < mejor->fitness)
                    memcpy(mejor->ruta, vecino, longitud_ruta * sizeof(int)), mejor->fitness = fv;
            }
            neigh++;
        }

        if (heuristica == 1)
            heuristica_abruptos(actual->ruta, longitud_ruta, m, distancias);

        actual->fitness = calcular_fitness(actual->ruta, distancias, longitud_ruta);
        fitness_generaciones[k - 1] = mejor->fitness; // recuerda que ahora k arranca en 1
    }

    // Si no se ha llegado a la última generación, rellenar el resto del histórico
    for (int i = k; i <= num_generaciones; i++) {
        fitness_generaciones[i-1] = mejor->fitness;
    }

    time_t fin = time(NULL);
    double t_total = difftime(fin, inicio);

    // 7) Empaquetar resultado
    ResultadoRecocido* R = (ResultadoRecocido*)malloc(sizeof(ResultadoRecocido));
    R->recorrido = (int*)malloc(longitud_ruta * sizeof(int));
    R->nombres_ciudades = malloc(longitud_ruta * sizeof(char[50]));
    R->fitness = mejor->fitness;
    R->longitud_recorrido = longitud_ruta;
    R->tiempo_ejecucion = t_total;
    R->fitness_generaciones = fitness_generaciones;

    for (int i = 0; i < longitud_ruta; i++) {
        R->recorrido[i] = mejor->ruta[i];
        strncpy(R->nombres_ciudades[i], nombres_ciudades[mejor->ruta[i]], 49);
        R->nombres_ciudades[i][49] = '\0';
    }

    // 8) Limpieza
    liberar_solucion(sol);
    liberar_solucion(actual);
    liberar_solucion(mejor);
    free(vecino);
    for (int i = 0; i < longitud_ruta; i++) {
        free(distancias[i]);
    }
    free(distancias);

    return R;
}

EXPORT void liberar_resultado(ResultadoRecocido *R)
{
    if (R){
    free(R->recorrido);            // Liberar array de enteros
    free(R->nombres_ciudades);     // Liberar array de nombres
    free(R->fitness_generaciones); // Liberar array de doubles
    free(R);    
    }                   // Liberar la estructura principal
}