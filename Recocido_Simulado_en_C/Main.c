#include "Biblioteca.h"

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
    /*
    for (int i = 0; i < longitud_ruta; i++)
    {
        printf("%s -> ", nombres_ciudades[mejor->ruta[i]]);
    }
    printf("%s\n", nombres_ciudades[mejor->ruta[0]]);
    */
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