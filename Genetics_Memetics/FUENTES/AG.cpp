//############################################################################################
// Nombre: AG.cpp                                                                            #
// Autor: Juan Emilio Martinez Manjon                                                        #
// Asignatura: Metaheuristica                                                                #
// Algoritmo: Algoritmos Geneticos                                                           #
//############################################################################################

//-------------------EXPLICACION PROGRAMA-------------------
/*

Vamos a trabajar con tres conjuntos de datos (Iris, Rand, Ecoli y Newthyroid) de los que sabemos lo siguiente:

    - Iris tiene 150 datos en 4 dimensiones y debe ir en 3 clusters
    - Rand tiene 150 datos en 2 dimensiones y debe ir en 3 clusters
    - Ecoli tiene 336 datos en 7 dimensiones y debe ir en 8 clusters
    - Newthyroid tiene 215 datos de 5 dimensiones y debe ir en 3 clusters


Tenemos que asignarle a cada cluster un conjunto de datos n de entre los datos del dataset.
Esto lo haremos con un vector con tantas posiciones como elementos haya en el dataset, y,
donde asignaremos a cada posicion el cluster donde ira cada dato.

Inicialmente, este vector tendra asignaciones completamente aleatorias pero comprobando
que no dejen ningun cluster vacio.

Nuestro algoritmo ira generando un vecindario virtual diferente para cada combinacion
de nuestro vector de indices.

Se explorara dicho vecindario haciendo cambios en nuestro indice y, mediante una funcion objetivo,
veremos si sale rentable cambiar un dato de un centroide a otro.

La funcion objetivo en cuestion se calcula como:

    Desviacion + (Infeasibility * lambda)

    *La desviacion es la media de las distancias Euclideas de cada valor a su correspondiente
    centroide.

    *La infeasibility es el conjunto de restricciones que nuestro vector de indices incumple

    *Lambda es una constante igual a la distancia maxima del dataset entre el numero maximo de
    restricciones. Se usa para ajustar la infeasibility a la desviacion

Esto lo repetimos hasta que hagamos 100000 evaluaciones

/*Vamos a utilizar los siguientes 5 valores de seeds para las pruebas:
    - 545
    - 650
    - 17
    - 1010
    - 1234
*/

//-------------------INCLUDES-------------------

#include <iostream>
#include <vector>
#include <list>
#include <fstream>
#include <limits>
#include <algorithm>
#include <random>
#include <ctime>
#include "random.h"
#include <utility>
using namespace std;


//-------------------STRUCTS Y CLASES-------------------

/*Este struct nos permite pasar la matriz de restricciones a una lista,
donde cada elemento es una instancia de este struct.*/

struct Restriccion{
    int indice1;
    int indice2;
    int valor;
};

/*Este struct nos ayuda a guardar en el mismo sitio un cromosoma junto con su funcion objetivo*/

struct Cromosoma{
    vector<int> elementos;
    float funcion;
    float desviacion;
    int infeasibility;
};

/*Esta clase la usaremos para almacenar de manera mas eficiente, las coordenadas
de los centroides y los datos asignados a cada cluster. Ademas de permitirnos
hacer varias operaciones sobre estos contenedores */
class Centroide {

private:
    int numero; //Etiqueta del cluster
    vector<float> indices; //Coordenadas del centroide
    vector<int> datos_asignados; //Datos asignados al cluster

public:
    Centroide(float max, float min, int num){
        for (int i=0; i<7; i++)
            indices.push_back(Randfloat(min,max));

        numero = num;
    }

    void vaciaDatosAsignados(){
        datos_asignados.clear();
    }

    int getNumero() const{
        return numero;
    }

    void addDato(int indice){
        datos_asignados.push_back(indice);
    }

    bool isContain(int indice){
        if (count(datos_asignados.begin(), datos_asignados.end(), indice))
            return true;
        else
            return false;
    }

    void setPos(int pos, float val){
        indices[pos] = val;
    }

    float getindice(int pos){
        return indices[pos];
    }

    vector<int> getDatosAsignados(){
        return datos_asignados;
    }

};


//-------------------CABECERAS FUNCIONES-------------------

int calculaInfeasibility(vector<int> indices, vector<Restriccion> restrictions);
bool compruebaIndicesCorrectos(vector<int> indices,int num_centroides, vector<int> &indices_vacios);
float calculaDesviacion(vector<Centroide> centroides, vector<list<float>> dataset);
float calculaFuncionObjetivo(vector<int> indices, vector<Centroide> centroides, vector<list<float>> dataset, vector<Restriccion> restrictions,  const float lambda, float &desv, int &inf);
float distanciaMaxima(vector<list<float>> data_set);
void actualizaCentroide(Centroide &a_actualizar, vector<list<float>> data_set, int tipo);
void rellenaDatos(string file_name, vector<list<float>> &data_set, int dimension, float &max, float &min);
void rellenaRestricciones(string file_name, vector<Restriccion> &restrictions, int tam);
void rellenaCentroide(Centroide &centroides, vector<int> indices);
void procesaAGG(vector<Cromosoma> poblacion, vector<Centroide> centroides, vector<list<float>> dataset, vector<Restriccion> restricciones, int num_caracteristicas, const float LAMBDA, int tipo_cruce);
void procesaAGE(vector<Cromosoma> poblacion, vector<Centroide> centroides, vector<list<float>> dataset, vector<Restriccion> restricciones, int num_caracteristicas, const float LAMBDA, int tipo_cruce);
vector<Cromosoma> funcion_cruce_uniforme(vector<Cromosoma> padres, int cruces_esperados, int num_centroides, int &cambios_f_objetivo, vector<Centroide> &centroides, vector<list<float>> dataset, vector<Restriccion> restrictions,const float lambda, int num_caracteristicas);
vector<Cromosoma> funcion_cruce_segmento_fijo(vector<Cromosoma> padres, int cruces_esperados, int num_centroides, int &cambios_f_objetivo, vector<Centroide> &centroides, vector<list<float>> dataset, vector<Restriccion> restrictions,const float lambda, int num_caracteristicas);



//-------------------PROGRAMA PRINCIPAL-------------------


int main(int argc, char** argv){

    if (argc != 6){
        cerr << "Introduzca un numero para inicializar la seed, el porcentaje de restricciones, el tipo de algoritmo, el tipo de cruce y el dataset" << endl;
        exit(1);
    }

    //-------------------DECLARACION VARIABLES Y CONSTANTES-------------------

    //Inicializamos la semilla con el argumento de entrada
    Set_random(atoi(argv[1]));
    srand(atoi(argv[1]));

    //Numero de restricciones de cada dataset

    const float RESTR_IRIS_10 = 1117.0;
    const float RESTR_IRIS_20 = 2235.0;
    const float RESTR_RAND_10 = 1117.0;
    const float RESTR_RAND_20 = 2235.0;
    const float RESTR_ECOLI_10 = 5628.0;
    const float RESTR_ECOLI_20 = 11256.0;
    const float RESTR_NEWTHYROID_10 = 4815.0;
    const float RESTR_NEWTHYROID_20 = 9417.0;

    //Para calcular la funcion objetivo

    vector<list<float>> data_set_iris, data_set_rand, data_set_ecoli, data_set_newthyroid;
    vector<Restriccion> restrictions_10_iris, restrictions_20_iris, restrictions_10_rand,
    restrictions_20_rand, restrictions_10_ecoli, restrictions_20_ecoli, restrictions_10_newthyroid, restrictions_20_newthyroid;
    vector<int> aux_correctos;

    float max_iris = 0, min_iris = numeric_limits<float>::max();
    float max_rand = 0, min_rand = numeric_limits<float>::max();
    float max_ecoli = 0, min_ecoli = numeric_limits<float>::max();
    float max_newthyroid = 0, min_newthyroid = numeric_limits<float>::max();

    vector<Centroide> centroides_iris, centroides_rand, centroides_ecoli, centroides_newthyroid;

    vector<int> indices_iris, indices_rand, indices_ecoli, indices_newthyroid;

    //Poblacion de 50 cromosomas
    vector<Cromosoma> poblacion_iris, poblacion_rand, poblacion_ecoli, poblacion_newthyroid;

    //Para calcular el tiempo de ejecucion
    unsigned t0_iris, t0_rand, t0_ecoli, t0_newthyroid, t1_iris, t1_rand, t1_ecoli, t1_newthyroid;
    double tiempo_iris, tiempo_rand, tiempo_ecoli, tiempo_newthyroid;
    
    //-------------------RELLENADO DE TDA-------------------

    //Vamos a llenar los data_sets con los datos
    rellenaDatos("iris_set.dat",data_set_iris,4,max_iris,min_iris);
    rellenaDatos("rand_set.dat",data_set_rand,2,max_rand,min_rand);
    rellenaDatos("ecoli_set.dat",data_set_ecoli,7,max_ecoli,min_ecoli);
    rellenaDatos("newthyroid_set.dat",data_set_newthyroid,5,max_newthyroid,min_newthyroid);

    //Vamos a rellenar el 10% de restricciones de los datos
    rellenaRestricciones("iris_set_const_10.const",restrictions_10_iris,150);
    rellenaRestricciones("rand_set_const_10.const",restrictions_10_rand,150);
    rellenaRestricciones("ecoli_set_const_10.const",restrictions_10_ecoli,336);
    rellenaRestricciones("newthyroid_set_const_10.const",restrictions_10_newthyroid,215);

    //Vamos a rellenar el 20% de restricciones de los datos
    rellenaRestricciones("iris_set_const_20.const",restrictions_20_iris,150);
    rellenaRestricciones("rand_set_const_20.const",restrictions_20_rand,150);
    rellenaRestricciones("ecoli_set_const_20.const",restrictions_20_ecoli,336);
    rellenaRestricciones("newthyroid_set_const_20.const",restrictions_20_newthyroid,215);


    const float LAMBDA_IRIS_10 = distanciaMaxima(data_set_iris) / RESTR_IRIS_10;
    const float LAMBDA_IRIS_20 = distanciaMaxima(data_set_iris) / RESTR_IRIS_20;
    const float LAMBDA_RAND_10 = distanciaMaxima(data_set_rand) / RESTR_RAND_10;
    const float LAMBDA_RAND_20 = distanciaMaxima(data_set_rand) / RESTR_RAND_20;
    const float LAMBDA_ECOLI_10 = distanciaMaxima(data_set_ecoli) / RESTR_ECOLI_10;
    const float LAMBDA_ECOLI_20 = distanciaMaxima(data_set_ecoli) / RESTR_ECOLI_20;
    const float LAMBDA_NEWTHYROID_10 = distanciaMaxima(data_set_newthyroid) / RESTR_NEWTHYROID_10;
    const float LAMBDA_NEWTHYROID_20 = distanciaMaxima(data_set_newthyroid) / RESTR_NEWTHYROID_20;


    //Generamos los centroides de los conjuntos de datos

    for (int i=0; i<3; i++){
        Centroide aux_iris(max_iris,min_iris,i);
        Centroide aux_rand(max_rand,min_rand,i);
        Centroide aux_newthyroid(max_newthyroid,min_newthyroid,i);
        
        centroides_iris.push_back(aux_iris);
        centroides_rand.push_back(aux_rand);
        centroides_newthyroid.push_back(aux_newthyroid);
    }

    for (int i=0; i<8; i++){
        Centroide aux_ecoli(max_ecoli,min_ecoli,i);
        
        centroides_ecoli.push_back(aux_ecoli);
    }
        

    //-------------------INICIALIZACION POBLACION DE CROMOSOMAS-------------------

    Cromosoma aux;
    for (int c=0; c<50; c++){

        do{
            indices_iris.clear();
            

            for (int i=0; i<150; i++){
                indices_iris.push_back(Randint(0,2));
            }
        
        }while(!compruebaIndicesCorrectos(indices_iris,3, aux_correctos));

        aux.elementos = indices_iris;
        aux.funcion = 0.0;
        poblacion_iris.push_back(aux);


        do{
            indices_rand.clear();
            

            for (int i=0; i<150; i++){
                indices_rand.push_back(Randint(0,2));
            }
        
        }while(!compruebaIndicesCorrectos(indices_rand,3, aux_correctos));


        aux.elementos = indices_rand;
        aux.funcion = 0.0;
        poblacion_rand.push_back(aux);

        do{
            indices_ecoli.clear();
            

            for (int i=0; i<336; i++){
                indices_ecoli.push_back(Randint(0,7));
            }
        
        }while(!compruebaIndicesCorrectos(indices_ecoli,8,aux_correctos));


        aux.elementos = indices_ecoli;
        aux.funcion = 0.0;
        poblacion_ecoli.push_back(aux);

        do{
            indices_newthyroid.clear();
            

            for (int i=0; i<215; i++){
                indices_newthyroid.push_back(Randint(0,2));
            }
        
        }while(!compruebaIndicesCorrectos(indices_newthyroid,3,aux_correctos));


        aux.elementos = indices_newthyroid;
        aux.funcion = 0.0;
        poblacion_newthyroid.push_back(aux);
    }

    

    //Comenzamos procesamiento datasets

    if (atoi(argv[2]) == 10){

        //-------------------ALGORITMO GENETICO GENERACIONAL UNIFORME-------------------
        if (atoi(argv[3]) == 1){ // GENERACIONALES

            if (atoi(argv[4]) == 1){ // CRUCE UNIFORME

                if (atoi(argv[5]) == 1){ //IRIS

                    cout << "IRIS-AGG-UN" << endl;

                    t0_iris = clock();
                    procesaAGG(poblacion_iris, centroides_iris, data_set_iris, restrictions_10_iris, 4, LAMBDA_IRIS_10, 0);
                    t1_iris = clock();
                    tiempo_iris = (double(t1_iris-t0_iris)/CLOCKS_PER_SEC);
                    cout << "Tiempo Iris: " << tiempo_iris << endl << endl;
                }
                else if (atoi(argv[5]) == 2){ //RAND

                    cout << "RAND-AGG-UN" << endl;

                    t0_rand = clock();
                    procesaAGG(poblacion_rand, centroides_rand, data_set_rand, restrictions_10_rand, 2, LAMBDA_RAND_10, 0);
                    t1_rand = clock();
                    tiempo_rand = (double(t1_rand-t0_rand)/CLOCKS_PER_SEC);
                    cout << "Tiempo Rand: " << tiempo_rand << endl << endl;
                }
                else if (atoi(argv[5]) == 3){ //ECOLI

                    cout << "ECOLI-AGG-UN" << endl;

                    t0_ecoli = clock();
                    procesaAGG(poblacion_ecoli, centroides_ecoli, data_set_ecoli, restrictions_10_ecoli, 7, LAMBDA_ECOLI_10, 0);
                    t1_ecoli = clock();
                    tiempo_ecoli = (double(t1_ecoli-t0_ecoli)/CLOCKS_PER_SEC);
                    cout << "Tiempo Ecoli: " << tiempo_ecoli << endl << endl;
                }
                else if (atoi(argv[5]) == 4){ // NEWTHYROID

                    cout << "NEWTHYROID-AGG-UN" << endl;

                    t0_newthyroid = clock();
                    procesaAGG(poblacion_newthyroid, centroides_newthyroid, data_set_newthyroid, restrictions_10_newthyroid, 5, LAMBDA_NEWTHYROID_10, 0);
                    t1_newthyroid = clock();
                    tiempo_newthyroid = (double(t1_newthyroid-t0_newthyroid)/CLOCKS_PER_SEC);
                    cout << "Tiempo Newthyroid: " << tiempo_newthyroid << endl << endl;
                }
            }
            else{ //CRUCE SEGMENTO FIJO
                //-------------------ALGORITMO GENETICO GENERACIONAL SEGMENTO FIJO-------------------

                if (atoi(argv[5]) == 1){

                    cout << "IRIS-AGG-SF" << endl;

                    t0_iris = clock();
                    procesaAGG(poblacion_iris, centroides_iris, data_set_iris, restrictions_10_iris, 4, LAMBDA_IRIS_10, 1);
                    t1_iris = clock();
                    tiempo_iris = (double(t1_iris-t0_iris)/CLOCKS_PER_SEC);
                    cout << "Tiempo Iris: " << tiempo_iris << endl << endl;
                }
                else if (atoi(argv[5]) == 2){

                    cout << "RAND-AGG-SF" << endl;

                    t0_rand = clock();
                    procesaAGG(poblacion_rand, centroides_rand, data_set_rand, restrictions_10_rand, 2, LAMBDA_RAND_10, 1);
                    t1_rand = clock();
                    tiempo_rand = (double(t1_rand-t0_rand)/CLOCKS_PER_SEC);
                    cout << "Tiempo Rand: " << tiempo_rand << endl << endl;
                }
                else if (atoi(argv[5]) == 3){

                    cout << "ECOLI-AGG-SF" << endl;

                    t0_ecoli = clock();
                    procesaAGG(poblacion_ecoli, centroides_ecoli, data_set_ecoli, restrictions_10_ecoli, 7, LAMBDA_ECOLI_10, 1);
                    t1_ecoli = clock();
                    tiempo_ecoli = (double(t1_ecoli-t0_ecoli)/CLOCKS_PER_SEC);
                    cout << "Tiempo Ecoli: " << tiempo_ecoli << endl << endl;
                }
                else if (atoi(argv[5]) == 4){

                    cout << "NEWTHYROID-AGG-SF" << endl;

                    t0_newthyroid = clock();
                    procesaAGG(poblacion_newthyroid, centroides_newthyroid, data_set_newthyroid, restrictions_10_newthyroid, 5, LAMBDA_NEWTHYROID_10, 1);
                    t1_newthyroid = clock();
                    tiempo_newthyroid = (double(t1_newthyroid-t0_newthyroid)/CLOCKS_PER_SEC);
                    cout << "Tiempo Newthyroid: " << tiempo_newthyroid << endl << endl;
                }
            
            }

        }
        else{ //ESTACIONARIOS

        //-------------------ALGORITMO GENETICO ESTACIONARIO UNIFORME-------------------

            if (atoi(argv[4]) == 1){

                if (atoi(argv[5]) == 1){

                    cout << "IRIS-AGE-UN" << endl;

                    t0_iris = clock();
                    procesaAGE(poblacion_iris, centroides_iris, data_set_iris, restrictions_10_iris, 4, LAMBDA_IRIS_10, 0);
                    t1_iris = clock();
                    tiempo_iris = (double(t1_iris-t0_iris)/CLOCKS_PER_SEC);
                    cout << "Tiempo Iris: " << tiempo_iris << endl << endl;
                }
                else if (atoi(argv[5]) == 2){

                    cout << "RAND-AGE-UN" << endl;

                    t0_rand = clock();
                    procesaAGE(poblacion_rand, centroides_rand, data_set_rand, restrictions_10_rand, 2, LAMBDA_RAND_10, 0);
                    t1_rand = clock();
                    tiempo_rand = (double(t1_rand-t0_rand)/CLOCKS_PER_SEC);
                    cout << "Tiempo Rand: " << tiempo_rand << endl << endl;
                }
                else if (atoi(argv[5]) == 3){
                    cout << "ECOLI-AGE-UN" << endl;

                    t0_ecoli = clock();
                    procesaAGE(poblacion_ecoli, centroides_ecoli, data_set_ecoli, restrictions_10_ecoli, 7, LAMBDA_ECOLI_10, 0);
                    t1_ecoli = clock();
                    tiempo_ecoli = (double(t1_ecoli-t0_ecoli)/CLOCKS_PER_SEC);
                    cout << "Tiempo Ecoli: " << tiempo_ecoli << endl << endl;
                }
                else if (atoi(argv[5]) == 4){

                    cout << "NEWTHYROID-AGE-UN" << endl;

                    t0_newthyroid = clock();
                    procesaAGE(poblacion_newthyroid, centroides_newthyroid, data_set_newthyroid, restrictions_10_newthyroid, 5, LAMBDA_NEWTHYROID_10, 0);
                    t1_newthyroid = clock();
                    tiempo_newthyroid = (double(t1_newthyroid-t0_newthyroid)/CLOCKS_PER_SEC);
                    cout << "Tiempo Newthyroid: " << tiempo_newthyroid << endl << endl;
                }
            }
            else{

                //-------------------ALGORITMO GENETICO ESTACIONARIO SEGMENTO FIJO-------------------

                if (atoi(argv[5]) == 1){
                    cout << "IRIS-AGE-SF" << endl;

                    t0_iris = clock();
                    procesaAGE(poblacion_iris, centroides_iris, data_set_iris, restrictions_10_iris, 4, LAMBDA_IRIS_10, 1);
                    t1_iris = clock();
                    tiempo_iris = (double(t1_iris-t0_iris)/CLOCKS_PER_SEC);
                    cout << "Tiempo Iris: " << tiempo_iris << endl << endl;
                }
                else if (atoi(argv[5]) == 2){
                    cout << "RAND-AGE-SF" << endl;

                    t0_rand = clock();
                    procesaAGE(poblacion_rand, centroides_rand, data_set_rand, restrictions_10_rand, 2, LAMBDA_RAND_10, 1);
                    t1_rand = clock();
                    tiempo_rand = (double(t1_rand-t0_rand)/CLOCKS_PER_SEC);
                    cout << "Tiempo Rand: " << tiempo_rand << endl << endl;
                }
                else if (atoi(argv[5]) == 3){

                    cout << "ECOLI-AGE-SF" << endl;

                    t0_ecoli = clock();
                    procesaAGE(poblacion_ecoli, centroides_ecoli, data_set_ecoli, restrictions_10_ecoli, 7, LAMBDA_ECOLI_10, 1);
                    t1_ecoli = clock();
                    tiempo_ecoli = (double(t1_ecoli-t0_ecoli)/CLOCKS_PER_SEC);
                    cout << "Tiempo Ecoli: " << tiempo_ecoli << endl << endl;
                }
                else if (atoi(argv[5]) == 4){

                    cout << "NEWTHYROID-AGE-SF" << endl;

                    t0_newthyroid = clock();
                    procesaAGE(poblacion_newthyroid, centroides_newthyroid, data_set_newthyroid, restrictions_10_newthyroid, 5, LAMBDA_NEWTHYROID_10, 1);
                    t1_newthyroid = clock();
                    tiempo_newthyroid = (double(t1_newthyroid-t0_newthyroid)/CLOCKS_PER_SEC);
                    cout << "Tiempo Newthyroid: " << tiempo_newthyroid << endl << endl;
                }
            }
        }

    }
    else if (atoi(argv[2]) == 20){

        //-------------------ALGORITMO GENETICO GENERACIONAL UNIFORME-------------------
        if (atoi(argv[3]) == 1){ // GENERACIONALES

            if (atoi(argv[4]) == 1){ // CRUCE UNIFORME

                if (atoi(argv[5]) == 1){ //IRIS

                    cout << "IRIS-AGG-UN" << endl;

                    t0_iris = clock();
                    procesaAGG(poblacion_iris, centroides_iris, data_set_iris, restrictions_20_iris, 4, LAMBDA_IRIS_20, 0);
                    t1_iris = clock();
                    tiempo_iris = (double(t1_iris-t0_iris)/CLOCKS_PER_SEC);
                    cout << "Tiempo Iris: " << tiempo_iris << endl << endl;
                }
                else if (atoi(argv[5]) == 2){ //RAND

                    cout << "RAND-AGG-UN" << endl;

                    t0_rand = clock();
                    procesaAGG(poblacion_rand, centroides_rand, data_set_rand, restrictions_20_rand, 2, LAMBDA_RAND_20, 0);
                    t1_rand = clock();
                    tiempo_rand = (double(t1_rand-t0_rand)/CLOCKS_PER_SEC);
                    cout << "Tiempo Rand: " << tiempo_rand << endl << endl;
                }
                else if (atoi(argv[5]) == 3){ //ECOLI

                    cout << "ECOLI-AGG-UN" << endl;

                    t0_ecoli = clock();
                    procesaAGG(poblacion_ecoli, centroides_ecoli, data_set_ecoli, restrictions_20_ecoli, 7, LAMBDA_ECOLI_20, 0);
                    t1_ecoli = clock();
                    tiempo_ecoli = (double(t1_ecoli-t0_ecoli)/CLOCKS_PER_SEC);
                    cout << "Tiempo Ecoli: " << tiempo_ecoli << endl << endl;
                }
                else if (atoi(argv[5]) == 4){ // NEWTHYROID

                    cout << "NEWTHYROID-AGG-UN" << endl;

                    t0_newthyroid = clock();
                    procesaAGG(poblacion_newthyroid, centroides_newthyroid, data_set_newthyroid, restrictions_20_newthyroid, 5, LAMBDA_NEWTHYROID_20, 0);
                    t1_newthyroid = clock();
                    tiempo_newthyroid = (double(t1_newthyroid-t0_newthyroid)/CLOCKS_PER_SEC);
                    cout << "Tiempo Newthyroid: " << tiempo_newthyroid << endl << endl;
                }
            }
            else{ //CRUCE SEGMENTO FIJO
                //-------------------ALGORITMO GENETICO GENERACIONAL SEGMENTO FIJO-------------------

                if (atoi(argv[5]) == 1){

                    cout << "IRIS-AGG-SF" << endl;

                    t0_iris = clock();
                    procesaAGG(poblacion_iris, centroides_iris, data_set_iris, restrictions_20_iris, 4, LAMBDA_IRIS_20, 1);
                    t1_iris = clock();
                    tiempo_iris = (double(t1_iris-t0_iris)/CLOCKS_PER_SEC);
                    cout << "Tiempo Iris: " << tiempo_iris << endl << endl;
                }
                else if (atoi(argv[5]) == 2){

                    cout << "RAND-AGG-SF" << endl;

                    t0_rand = clock();
                    procesaAGG(poblacion_rand, centroides_rand, data_set_rand, restrictions_20_rand, 2, LAMBDA_RAND_20, 1);
                    t1_rand = clock();
                    tiempo_rand = (double(t1_rand-t0_rand)/CLOCKS_PER_SEC);
                    cout << "Tiempo Rand: " << tiempo_rand << endl << endl;
                }
                else if (atoi(argv[5]) == 3){

                    cout << "ECOLI-AGG-SF" << endl;

                    t0_ecoli = clock();
                    procesaAGG(poblacion_ecoli, centroides_ecoli, data_set_ecoli, restrictions_20_ecoli, 7, LAMBDA_ECOLI_20, 1);
                    t1_ecoli = clock();
                    tiempo_ecoli = (double(t1_ecoli-t0_ecoli)/CLOCKS_PER_SEC);
                    cout << "Tiempo Ecoli: " << tiempo_ecoli << endl << endl;
                }
                else if (atoi(argv[5]) == 4){

                    cout << "NEWTHYROID-AGG-SF" << endl;

                    t0_newthyroid = clock();
                    procesaAGG(poblacion_newthyroid, centroides_newthyroid, data_set_newthyroid, restrictions_20_newthyroid, 5, LAMBDA_NEWTHYROID_20, 1);
                    t1_newthyroid = clock();
                    tiempo_newthyroid = (double(t1_newthyroid-t0_newthyroid)/CLOCKS_PER_SEC);
                    cout << "Tiempo Newthyroid: " << tiempo_newthyroid << endl << endl;
                }
            
            }

        }
        else{ //ESTACIONARIOS

        //-------------------ALGORITMO GENETICO ESTACIONARIO UNIFORME-------------------

            if (atoi(argv[4]) == 1){

                if (atoi(argv[5]) == 1){

                    cout << "IRIS-AGE-UN" << endl;

                    t0_iris = clock();
                    procesaAGE(poblacion_iris, centroides_iris, data_set_iris, restrictions_20_iris, 4, LAMBDA_IRIS_20, 0);
                    t1_iris = clock();
                    tiempo_iris = (double(t1_iris-t0_iris)/CLOCKS_PER_SEC);
                    cout << "Tiempo Iris: " << tiempo_iris << endl << endl;
                }
                else if (atoi(argv[5]) == 2){

                    cout << "RAND-AGE-UN" << endl;

                    t0_rand = clock();
                    procesaAGE(poblacion_rand, centroides_rand, data_set_rand, restrictions_20_rand, 2, LAMBDA_RAND_20, 0);
                    t1_rand = clock();
                    tiempo_rand = (double(t1_rand-t0_rand)/CLOCKS_PER_SEC);
                    cout << "Tiempo Rand: " << tiempo_rand << endl << endl;
                }
                else if (atoi(argv[5]) == 3){
                    cout << "ECOLI-AGE-UN" << endl;

                    t0_ecoli = clock();
                    procesaAGE(poblacion_ecoli, centroides_ecoli, data_set_ecoli, restrictions_20_ecoli, 7, LAMBDA_ECOLI_20, 0);
                    t1_ecoli = clock();
                    tiempo_ecoli = (double(t1_ecoli-t0_ecoli)/CLOCKS_PER_SEC);
                    cout << "Tiempo Ecoli: " << tiempo_ecoli << endl << endl;
                }
                else if (atoi(argv[5]) == 4){

                    cout << "NEWTHYROID-AGE-UN" << endl;

                    t0_newthyroid = clock();
                    procesaAGE(poblacion_newthyroid, centroides_newthyroid, data_set_newthyroid, restrictions_20_newthyroid, 5, LAMBDA_NEWTHYROID_20, 0);
                    t1_newthyroid = clock();
                    tiempo_newthyroid = (double(t1_newthyroid-t0_newthyroid)/CLOCKS_PER_SEC);
                    cout << "Tiempo Newthyroid: " << tiempo_newthyroid << endl << endl;
                }
            }
            else{

                //-------------------ALGORITMO GENETICO ESTACIONARIO SEGMENTO FIJO-------------------

                if (atoi(argv[5]) == 1){
                    cout << "IRIS-AGE-SF" << endl;

                    t0_iris = clock();
                    procesaAGE(poblacion_iris, centroides_iris, data_set_iris, restrictions_20_iris, 4, LAMBDA_IRIS_20, 1);
                    t1_iris = clock();
                    tiempo_iris = (double(t1_iris-t0_iris)/CLOCKS_PER_SEC);
                    cout << "Tiempo Iris: " << tiempo_iris << endl << endl;
                }
                else if (atoi(argv[5]) == 2){
                    cout << "RAND-AGE-SF" << endl;

                    t0_rand = clock();
                    procesaAGE(poblacion_rand, centroides_rand, data_set_rand, restrictions_20_rand, 2, LAMBDA_RAND_20, 1);
                    t1_rand = clock();
                    tiempo_rand = (double(t1_rand-t0_rand)/CLOCKS_PER_SEC);
                    cout << "Tiempo Rand: " << tiempo_rand << endl << endl;
                }
                else if (atoi(argv[5]) == 3){

                    cout << "ECOLI-AGE-SF" << endl;

                    t0_ecoli = clock();
                    procesaAGE(poblacion_ecoli, centroides_ecoli, data_set_ecoli, restrictions_20_ecoli, 7, LAMBDA_ECOLI_20, 1);
                    t1_ecoli = clock();
                    tiempo_ecoli = (double(t1_ecoli-t0_ecoli)/CLOCKS_PER_SEC);
                    cout << "Tiempo Ecoli: " << tiempo_ecoli << endl << endl;
                }
                else if (atoi(argv[5]) == 4){

                    cout << "NEWTHYROID-AGE-SF" << endl;

                    t0_newthyroid = clock();
                    procesaAGE(poblacion_newthyroid, centroides_newthyroid, data_set_newthyroid, restrictions_20_newthyroid, 5, LAMBDA_NEWTHYROID_20, 1);
                    t1_newthyroid = clock();
                    tiempo_newthyroid = (double(t1_newthyroid-t0_newthyroid)/CLOCKS_PER_SEC);
                    cout << "Tiempo Newthyroid: " << tiempo_newthyroid << endl << endl;
                }
            }
        }


    }

    
}


//--------------------------------------------------------------------------------------------

/*Funcion para rellenar los vectores de datos leyendo desde un archivo txt*/
void rellenaDatos(string file_name, vector<list<float>> &data_set, int dimension, float &max, float &min){

    ifstream flujo_entrada(file_name);

    if (!flujo_entrada){
        cerr << "No se pudo abrir el archivo deseado";
        exit(1);
    }

    float dato = 0.0;
    char coma = ' ';

    while (!flujo_entrada.eof()){

        list<float> lista_aux;
        for (int i=0; i<dimension; i++){
            flujo_entrada >> dato;

            if (i < dimension-1)
                flujo_entrada >> coma;

            if (dato > max)
                max = dato;

            if (dato < min)
                min = dato;

            lista_aux.push_back(dato);
        }
        data_set.push_back(lista_aux);
    }

}


//--------------------------------------------------------------------------------------------

/*Funcion para rellenar el vector de restricciones con los datos leidos desde archivo*/
void rellenaRestricciones(string file_name, vector<Restriccion> &restrictions, int tam){

    ifstream flujo_entrada(file_name);

    if (!flujo_entrada){
        cerr << "No se pudo abrir el archivo deseado";
        exit(1);
    }

    int dato = 0;
    char coma = ' ';
    vector<vector<int>> matriz;
    Restriccion aux;

    for (int i=0; i<tam; i++){
        vector<int> aux;
        for (int j=0; j<tam; j++){
            flujo_entrada >> dato;

            if (j < tam-1)
                flujo_entrada >> coma;

            aux.push_back(dato);
        }
        matriz.push_back(aux);
    }   


    for (int i = 0; i < tam; i++){
        for (int j = i+1; j < tam; j++){
            if (matriz[i][j] != 0){
                aux.indice1 = i;
                aux.indice2 = j;
                aux.valor = matriz[i][j];
                restrictions.push_back(aux);
            }
        }
    }       
    

}


//--------------------------------------------------------------------------------------------

/*Funcion para actualizar los parametros de un centroide en funcion de sus datos asignados*/
void actualizaCentroide(Centroide &a_actualizar, vector<list<float>> data_set, int tipo){

    vector<float> medias;

    for (int i=0; i<tipo; i++)
        medias.push_back(0.0);

    for (int i=0; i<(a_actualizar.getDatosAsignados()).size(); i++){

        vector<float> aux {begin(data_set[(a_actualizar.getDatosAsignados())[i]]), end(data_set[(a_actualizar.getDatosAsignados())[i]])};

        for (int j=0; j<aux.size(); j++){

            medias[j] += aux[j];
        }
    }

    for (int i=0; i<tipo; i++)
        medias[i] = medias[i] / (float)((a_actualizar.getDatosAsignados()).size());

    for (int j=0; j<tipo; j++)
        a_actualizar.setPos(j,medias[j]);

}

//--------------------------------------------------------------------------------------------

void procesaAGG(vector<Cromosoma> poblacion, vector<Centroide> centroides, vector<list<float>> dataset, vector<Restriccion> restricciones, int num_caracteristicas, const float LAMBDA, int tipo_cruce){

    int evaluaciones_funcion = 50;
    int padre_aleatorio1, padre_aleatorio2;
    float funcion_padre1, funcion_padre2;
    int numero_esperado_cruces = 17; // 25 * 0.7 = Numero de parejas * probabilidad de cruce
    int numero_mutaciones_esperadas;
    int gen_aleatorio, nuevo_indice_aleatorio;
    int sumador_contador;
    vector<int> indices_vacios;
    vector<Cromosoma> padres_cruzados;
    vector<Cromosoma> padres_seleccionados;
    float desv;
    int inf;


    for (int i=0; i<poblacion.size(); i++){

        for (int j=0; j<centroides.size(); j++){
            rellenaCentroide(centroides[j],poblacion[i].elementos);
            actualizaCentroide(centroides[j], dataset, num_caracteristicas);
        }

        poblacion[i].funcion = calculaFuncionObjetivo(poblacion[i].elementos,centroides,dataset,restricciones,LAMBDA,desv,inf);
        poblacion[i].desviacion = desv;
        poblacion[i].infeasibility = inf;
    }


    while (evaluaciones_funcion < 100000){

        //Primero seleccionamos los 50 hijos con torneos binarios       
        padres_seleccionados.clear();

        
        for (int i=0; i<50; i++){

            padre_aleatorio1 = Randint(0,49);
            padre_aleatorio2 = Randint(0,49);

            funcion_padre1 = poblacion[padre_aleatorio1].funcion;
            funcion_padre2 = poblacion[padre_aleatorio2].funcion;


            if (funcion_padre1 < funcion_padre2){
                padres_seleccionados.push_back(poblacion[padre_aleatorio1]);
            }

            else{
                padres_seleccionados.push_back(poblacion[padre_aleatorio2]);
            }

        }
        

        //Ya hemos seleccionado los 50 padres. Procedemos al cruce
        if (tipo_cruce == 0)        //Si cruzamos los padres de la forma uniforme
            padres_cruzados = funcion_cruce_uniforme(padres_seleccionados, numero_esperado_cruces, centroides.size(),sumador_contador,centroides,dataset,restricciones,LAMBDA,num_caracteristicas);

        else                        //Si cruzamos los padres de la forma de segmento fijo
            padres_cruzados = funcion_cruce_segmento_fijo(padres_seleccionados, numero_esperado_cruces, centroides.size(),sumador_contador,centroides,dataset,restricciones,LAMBDA,num_caracteristicas);


        evaluaciones_funcion += sumador_contador;
        

        //Ahora pasamos a mutar

        numero_mutaciones_esperadas = (int)(50*dataset.size()*0.001);

        for (int j = 0; j<numero_mutaciones_esperadas; j++){

            vector<int> aux;

            do{
                aux = padres_cruzados[j].elementos;
                gen_aleatorio = Randint(0,dataset.size()-2);
                nuevo_indice_aleatorio = Randint(0, centroides.size()-1);

                aux[gen_aleatorio] = nuevo_indice_aleatorio;

            }while(!compruebaIndicesCorrectos(aux,centroides.size(),indices_vacios) && (nuevo_indice_aleatorio != (padres_cruzados[j].elementos)[gen_aleatorio]));

            (padres_cruzados[j].elementos)[gen_aleatorio] = nuevo_indice_aleatorio;

            for (int k=0; k<centroides.size(); k++){
                rellenaCentroide(centroides[k],padres_cruzados[j].elementos);
                actualizaCentroide(centroides[k], dataset, num_caracteristicas);
            }

            padres_cruzados[j].funcion = calculaFuncionObjetivo(padres_cruzados[j].elementos,centroides,dataset,restricciones,LAMBDA,desv,inf);
            padres_cruzados[j].desviacion = desv;
            padres_cruzados[j].infeasibility = inf;
            evaluaciones_funcion++;
        }


        //Sustituimos la poblacion anterior por la nueva

        float mejor_funcion_objetivo = numeric_limits<float>::max();
        int mejor_cromosoma;

        for (int i=0; i<poblacion.size(); i++){

            if (poblacion[i].funcion < mejor_funcion_objetivo){
                mejor_funcion_objetivo = poblacion[i].funcion;
                mejor_cromosoma = i;
            }
        }

        float peor_funcion_objetivo = 0.0;
        int peor_cromosoma;

        //Hemos quitado el mejor anterior
        if (padres_cruzados[mejor_cromosoma].funcion != mejor_funcion_objetivo){

            for (int l=0; l<padres_cruzados.size(); l++){

                if (padres_cruzados[l].funcion > peor_funcion_objetivo){
                    peor_funcion_objetivo = padres_cruzados[l].funcion;
                    peor_cromosoma = l;
                }
            }

            padres_cruzados[peor_cromosoma] = poblacion[mejor_cromosoma];

        }


        //Copiamos a poblacion lo que hay en padres seleccionados

        poblacion = padres_cruzados;

    }



    //Imprimimos los datos del mejor cromosoma de la poblacion

    float mejor_funcion_objetivo = numeric_limits<float>::max();
    float mejor_desv;
    int mejor_cromosoma, mejor_inf;

    for (int i=0; i<poblacion.size(); i++){

        if (poblacion[i].funcion < mejor_funcion_objetivo){
            mejor_funcion_objetivo = poblacion[i].funcion;
            mejor_desv = poblacion[i].desviacion;
            mejor_inf = poblacion[i].infeasibility;
            mejor_cromosoma = i;
        }
    }

    cout << endl;
    cout << "Mejor cromosoma: " << endl;

    for (int i=0; i<poblacion[mejor_cromosoma].elementos.size(); i++)
        cout << poblacion[mejor_cromosoma].elementos[i] << " ";

    cout << endl;
    
    cout << "Desviacion : " << mejor_desv << endl;
    cout << "infactibilidad: " << mejor_inf << endl;
    cout << "Funcion objetivo: " << mejor_funcion_objetivo << endl;
}


//--------------------------------------------------------------------------------------------

void procesaAGE(vector<Cromosoma> poblacion, vector<Centroide> centroides, vector<list<float>> dataset, vector<Restriccion> restricciones, int num_caracteristicas, const float LAMBDA, int tipo_cruce){

    int evaluaciones_funcion = 50;
    int padre_aleatorio1, padre_aleatorio2;
    float funcion_padre1, funcion_padre2;
    int numero_esperado_cruces;
    float numero_mutaciones_esperadas;
    int gen_aleatorio, nuevo_indice_aleatorio;
    int sumador_contador;
    vector<int> indices_vacios;
    vector<Cromosoma> padres_cruzados;
    vector<Cromosoma> padres_seleccionados;
    float desv;
    int inf;


    for (int i=0; i<poblacion.size(); i++){

        for (int j=0; j<centroides.size(); j++){
            rellenaCentroide(centroides[j],poblacion[i].elementos);
            actualizaCentroide(centroides[j], dataset, num_caracteristicas);
        }

        poblacion[i].funcion = calculaFuncionObjetivo(poblacion[i].elementos,centroides,dataset,restricciones,LAMBDA,desv,inf);
        poblacion[i].desviacion = desv;
        poblacion[i].infeasibility = inf;
    }


    while (evaluaciones_funcion < 100000){

        //Primero seleccionamos los 50 hijos con torneos binarios       
        padres_seleccionados.clear();

        
        for (int i=0; i<2; i++){

            padre_aleatorio1 = Randint(0,49);
            padre_aleatorio2 = Randint(0,49);

            funcion_padre1 = poblacion[padre_aleatorio1].funcion;
            funcion_padre2 = poblacion[padre_aleatorio2].funcion;


            if (funcion_padre1 < funcion_padre2){
                padres_seleccionados.push_back(poblacion[padre_aleatorio1]);
            }

            else{
                padres_seleccionados.push_back(poblacion[padre_aleatorio2]);
            }

        }

        numero_esperado_cruces = 1; 

        //Ya hemos seleccionado los 50 padres. Procedemos al cruce
        if (tipo_cruce == 0)        //Si cruzamos los padres de la forma uniforme
            padres_cruzados = funcion_cruce_uniforme(padres_seleccionados, numero_esperado_cruces, centroides.size(),sumador_contador,centroides,dataset,restricciones,LAMBDA,num_caracteristicas);

        else                        //Si cruzamos los padres de la forma de segmento fijo
            padres_cruzados = funcion_cruce_segmento_fijo(padres_seleccionados, numero_esperado_cruces, centroides.size(),sumador_contador,centroides,dataset,restricciones,LAMBDA,num_caracteristicas);

        evaluaciones_funcion += sumador_contador;
        
        //Ahora pasamos a mutar

        numero_mutaciones_esperadas = dataset.size()*0.001;

        for (int j = 0; j<padres_cruzados.size(); j++){

            float probabilidad_mutacion = Randfloat(0,1);


            if (probabilidad_mutacion < numero_mutaciones_esperadas){
                vector<int> aux;

                do{
                    aux = padres_cruzados[j].elementos;
                    gen_aleatorio = Randint(0,dataset.size()-2);
                    nuevo_indice_aleatorio = Randint(0, centroides.size()-1);

                    aux[gen_aleatorio] = nuevo_indice_aleatorio;

                }while(!compruebaIndicesCorrectos(aux,centroides.size(),indices_vacios) && (nuevo_indice_aleatorio != (padres_cruzados[j].elementos)[gen_aleatorio]));

                (padres_cruzados[j].elementos)[gen_aleatorio] = nuevo_indice_aleatorio;

                for (int k=0; k<centroides.size(); k++){
                    rellenaCentroide(centroides[k],padres_cruzados[j].elementos);
                    actualizaCentroide(centroides[k], dataset, num_caracteristicas);
                }

                padres_cruzados[j].funcion = calculaFuncionObjetivo(padres_cruzados[j].elementos,centroides,dataset,restricciones,LAMBDA,desv,inf);
                padres_cruzados[j].desviacion = desv;
                padres_cruzados[j].infeasibility = inf;
                evaluaciones_funcion++;
            }
        }

        //Sustituimos la poblacion anterior por la nueva

        float peor_funcion_objetivo = 0;
        int peor_cromosoma, cromosoma_anterior;

        for (int i=0; i<poblacion.size(); i++){

            if (poblacion[i].funcion > peor_funcion_objetivo){
                peor_funcion_objetivo = poblacion[i].funcion;
                cromosoma_anterior = peor_cromosoma;
                peor_cromosoma = i;
            }
        }


        int mejor_cromosoma, otro_cromosoma;

        if (padres_cruzados[0].funcion <= padres_cruzados[1].funcion){
            mejor_cromosoma = 0;
            otro_cromosoma = 1;
        }
        else{
            mejor_cromosoma = 1;  
            otro_cromosoma = 0;
        }      


        if (padres_cruzados[mejor_cromosoma].funcion < poblacion[peor_cromosoma].funcion)
            poblacion[peor_cromosoma] = padres_cruzados[mejor_cromosoma];

        if (padres_cruzados[otro_cromosoma].funcion < poblacion[cromosoma_anterior].funcion)
            poblacion[cromosoma_anterior] = padres_cruzados[otro_cromosoma];    
        

    }


    //Imprimimos los datos del mejor cromosoma de la poblacion

    float mejor_funcion_objetivo = numeric_limits<float>::max();
    float mejor_desv;
    int mejor_cromosoma, mejor_inf;

    for (int i=0; i<poblacion.size(); i++){

        if (poblacion[i].funcion < mejor_funcion_objetivo){
            mejor_funcion_objetivo = poblacion[i].funcion;
            mejor_desv = poblacion[i].desviacion;
            mejor_inf = poblacion[i].infeasibility;
            mejor_cromosoma = i;
        }
    }

    cout << endl;
    cout << "Mejor cromosoma: " << endl;

    for (int i=0; i<poblacion[mejor_cromosoma].elementos.size(); i++)
        cout << poblacion[mejor_cromosoma].elementos[i] << " ";

    cout << endl;

    cout << "Desviacion : " << mejor_desv << endl;
    cout << "infactibilidad: " << mejor_inf << endl;
    cout << "Funcion objetivo: " << mejor_funcion_objetivo << endl;
}

//-------------------------------------------------------------------------------------------- 

/*Funcion que asigna a un centroide los indices de un vector que se le pasa como parametro*/
void rellenaCentroide(Centroide &centroide, vector<int> indices){
    centroide.vaciaDatosAsignados();

    for (int i=0; i<indices.size(); i++){
        if (indices[i] == centroide.getNumero())
            centroide.addDato(i);
    }
}


//--------------------------------------------------------------------------------------------

/*Funcion que calcula la distancia maxima entre los datos de un dataset*/
float distanciaMaxima(vector<list<float>> data_set){
    float distancia, distancia_maxima = 0;

    for (int i=0; i<data_set.size(); i++){
        for (int j=i+1; j<data_set.size(); j++){
            vector<float> valores_punto1{begin(data_set[i]), end(data_set[i])};
            vector<float> valores_punto2{begin(data_set[j]), end(data_set[j])};
            vector<float> res;
            distancia = 0.0;

            for (int k=0; k<valores_punto1.size(); k++){
                res.push_back((valores_punto2[k] - valores_punto1[k])*(valores_punto2[k] - valores_punto1[k]));
            }

            for (int k=0; k<res.size(); k++)
                distancia+=res[k];

            distancia = sqrt(distancia);

        
            if (distancia > distancia_maxima)
                distancia_maxima = distancia;
        }
    }

    return distancia_maxima;
}


//--------------------------------------------------------------------------------------------

/*F = Desviacion + (Infeasibility * lambda)
Desviacion = Distancia de cada punto a su centroide y se suma el resultado de todos los centroides
Infeasibility es el numero de restricciones NO cumplidas
Lambda es una constante*/

float calculaFuncionObjetivo(vector<int> indices, vector<Centroide> centroides, vector<list<float>> dataset, vector<Restriccion> restrictions, const float lambda, float &desv, int &inf){

    float funcion_objetivo = 0.0;
    float desviacion = 0.0;
    int infeasibility = 0;

    desviacion = calculaDesviacion(centroides,dataset);
    infeasibility = calculaInfeasibility(indices,restrictions);

    desv = desviacion;
    inf = infeasibility;
     
    funcion_objetivo = desviacion + (infeasibility * lambda);

    return funcion_objetivo;  
}

//--------------------------------------------------------------------------------------------

/*Funcion que comprueba si los indices de un vecindario son correctos y no dejan ningun
centroide vacio*/
bool compruebaIndicesCorrectos(vector<int> indices, int num_centroides, vector<int> &clusters_vacios){

    bool son_correctos = true;
    vector<bool> comprobaciones;
    clusters_vacios.clear();

    for (int i=0; i<num_centroides; i++)
        comprobaciones.push_back(false);

    for (int i=0; i<indices.size(); i++)
        comprobaciones[indices[i]] = true;
    
    for (int i=0; i<num_centroides; i++){

        if (comprobaciones[i] == false){
            clusters_vacios.push_back(i);
            son_correctos = false;
        }
    }

    return son_correctos;

}

//--------------------------------------------------------------------------------------------
/*Funcion para calcular la desviacion de un conjunto de centroides*/

float calculaDesviacion(vector<Centroide> centroides, vector<list<float>> dataset){

    float distancia_euclidea = 0.0, desviacion = 0.0, sumatoria = 0.0;

    for (int i=0; i<centroides.size(); i++){
        sumatoria = 0.0;

        for (int j=0; j<(centroides[i].getDatosAsignados().size()); j++){

            distancia_euclidea = 0.0;

            vector<float> aux{begin(dataset[(centroides[i].getDatosAsignados())[j]]), end(dataset[(centroides[i].getDatosAsignados())[j]])};

            for (int k=0; k<aux.size(); k++)
                distancia_euclidea += ((aux[k]-centroides[i].getindice(k)) * (aux[k]-centroides[i].getindice(k)));

            distancia_euclidea = sqrt(distancia_euclidea);
            sumatoria += distancia_euclidea;
        }

        sumatoria = sumatoria / (float)((centroides[i].getDatosAsignados()).size());
        desviacion += sumatoria;
    }

    desviacion = desviacion / (float)(centroides.size());

    return desviacion;
}

//--------------------------------------------------------------------------------------------
/*Funcion que calcula la infactibilidad de un conjunto de indices*/

int calculaInfeasibility(vector<int> indices, vector<Restriccion> restrictions){

    int infeasibility = 0;

    for (int i=0; i<restrictions.size(); i++){

        if (restrictions[i].valor == 1 && indices[restrictions[i].indice1] != indices[restrictions[i].indice2])
            infeasibility++;


        if (restrictions[i].valor == -1 && indices[restrictions[i].indice1] == indices[restrictions[i].indice2])
            infeasibility++;
    }

    return infeasibility;
}

//--------------------------------------------------------------------------------------------
/*Funcion que realiza un cruce uniforme sobre un conjunto de padres seleccionados*/
vector<Cromosoma> funcion_cruce_uniforme(vector<Cromosoma> padres, int cruces_esperados, int num_centroides, int &cambios_f_objetivo,vector<Centroide> &centroides, vector<list<float>> dataset, vector<Restriccion> restrictions,const float lambda, int num_caracteristicas){

    int tamanio_cromosoma = padres[0].elementos.size();
    int indice_a_cruzar = Randint(0, tamanio_cromosoma-1);
    int contador = 0;
    int elemento_aleatorio;
    vector<Cromosoma> padres_cruzados;
    vector<int> hijo_cruzado;
    vector<bool> indices_que_cruzan;
    vector<int> clusters_vacios;
    float desv;
    int inf;

    cambios_f_objetivo = 0;

    indices_que_cruzan.assign(tamanio_cromosoma-1, false);

    for (int i=0; i<cruces_esperados; i++){

        //Generamos los n/2 indices aleatorios
        for(int n=0; n<2; n++){ //Son dos hijos por pareja

            hijo_cruzado.clear();

            for (int j=0; j<tamanio_cromosoma / 2; j++){

                do{
                    indice_a_cruzar = Randint(0, tamanio_cromosoma-1);
                }while(indices_que_cruzan[indice_a_cruzar]);

                indices_que_cruzan[indice_a_cruzar] = true;
            }


            for (int j=0; j<tamanio_cromosoma; j++){
                if (indices_que_cruzan[j]){
                    hijo_cruzado.push_back((padres[contador].elementos)[j]);
                }
                else{
                    hijo_cruzado.push_back((padres[contador+1].elementos)[j]);
                }

            }

            //Comprobamos que el hijo generado es correcto y, en caso contrario, lo reparamos

            while (!compruebaIndicesCorrectos(hijo_cruzado,num_centroides,clusters_vacios)){
                //Reparamos el hijo

                for (int k=0; k<clusters_vacios.size(); k++){
                    elemento_aleatorio = Randint(0, tamanio_cromosoma-1);
                    hijo_cruzado[elemento_aleatorio] = clusters_vacios[k];
                }
            }

            for (int j=0; j<centroides.size(); j++){
                rellenaCentroide(centroides[j],hijo_cruzado);
                actualizaCentroide(centroides[j], dataset, num_caracteristicas);
            }

            Cromosoma aux;
            aux.elementos = hijo_cruzado;
            aux.funcion = calculaFuncionObjetivo(hijo_cruzado,centroides,dataset,restrictions,lambda,desv,inf);
            aux.desviacion = desv;
            aux.infeasibility = inf;
            padres_cruzados.push_back(aux);
            cambios_f_objetivo++;

            indices_que_cruzan.clear();
            indices_que_cruzan.assign(tamanio_cromosoma-1, false);

        }

        contador+=2;
    }




    for (int i=contador; i<padres.size(); i++)
        padres_cruzados.push_back(padres[i]);


    return padres_cruzados;
    

}


//--------------------------------------------------------------------------------------------
/*Funcion que realiza un cruce uniforme sobre un conjunto de padres seleccionados*/
vector<Cromosoma> funcion_cruce_segmento_fijo(vector<Cromosoma> padres, int cruces_esperados, int num_centroides, int &cambios_f_objetivo,vector<Centroide> &centroides, vector<list<float>> dataset, vector<Restriccion> restrictions,const float lambda, int num_caracteristicas){

    int tamanio_cromosoma = padres[0].elementos.size();
    int indice_a_cruzar = Randint(0, tamanio_cromosoma-1);
    int contador = 0;
    int elemento_aleatorio;
    vector<Cromosoma> padres_cruzados;
    vector<int> hijo_cruzado;
    vector<bool> indices_que_cruzan, indices_que_cruzan_fijos;
    vector<int> clusters_vacios;
    int inicio_segmento, tamanio_segmento;
    float desv;
    int inf;

    cambios_f_objetivo = 0;

    indices_que_cruzan.assign(tamanio_cromosoma-1, false);
    indices_que_cruzan_fijos.assign(tamanio_cromosoma-1,false);


    for (int i=0; i<cruces_esperados; i++){

        //Generamos los n/2 indices aleatorios
        for(int n=0; n<2; n++){ //Son dos hijos por pareja

            hijo_cruzado.clear();
            //Parte fija del hijo
            inicio_segmento = Randint(0,tamanio_cromosoma-1);
            tamanio_segmento = Randint(0,tamanio_cromosoma-1);

            if ( ((inicio_segmento + tamanio_segmento) % tamanio_cromosoma) > inicio_segmento){

                for (int j=inicio_segmento; j < ((inicio_segmento + tamanio_segmento) % tamanio_cromosoma); j++)
                    indices_que_cruzan_fijos[j] = true;
            }
            else{

                for (int j=inicio_segmento; j < tamanio_cromosoma; j++)
                    indices_que_cruzan_fijos[j] = true;

                for (int j=0; j < ((inicio_segmento + tamanio_segmento) % tamanio_cromosoma); j++)
                    indices_que_cruzan_fijos[j] = true;
            }

            //Parte restante del hijo
            for (int j=0; j<tamanio_cromosoma / 2; j++){

                do{
                    indice_a_cruzar = Randint(0, tamanio_cromosoma-1);
                }while(indices_que_cruzan[indice_a_cruzar] && indices_que_cruzan_fijos[indice_a_cruzar]);

                indices_que_cruzan[indice_a_cruzar] = true;
            }


            for (int j=0; j<tamanio_cromosoma; j++){

                if (indices_que_cruzan[j] || indices_que_cruzan_fijos[j]){
                    hijo_cruzado.push_back((padres[contador].elementos)[j]);
                }
                else if (!indices_que_cruzan[j] && !indices_que_cruzan_fijos[j]){
                    hijo_cruzado.push_back((padres[contador+1].elementos)[j]);
                }

            }


            //Comprobamos que el hijo generado es correcto y, en caso contrario, lo reparamos

            while (!compruebaIndicesCorrectos(hijo_cruzado,num_centroides,clusters_vacios)){
                //Reparamos el hijo

                for (int k=0; k<clusters_vacios.size(); k++){
                    elemento_aleatorio = Randint(0, tamanio_cromosoma-1);
                    hijo_cruzado[elemento_aleatorio] = clusters_vacios[k];
                }
            }


            for (int j=0; j<centroides.size(); j++){
                rellenaCentroide(centroides[j],hijo_cruzado);
                actualizaCentroide(centroides[j], dataset, num_caracteristicas);
            }


            Cromosoma aux;
            aux.elementos = hijo_cruzado;
            aux.funcion = calculaFuncionObjetivo(hijo_cruzado,centroides,dataset,restrictions,lambda,desv,inf);
            aux.desviacion = desv;
            aux.infeasibility = inf;
            padres_cruzados.push_back(aux);
            cambios_f_objetivo++;


            indices_que_cruzan.clear();
            indices_que_cruzan_fijos.clear();
            indices_que_cruzan.assign(tamanio_cromosoma-1, false);
            indices_que_cruzan_fijos.assign(tamanio_cromosoma-1,false);


        }

        contador+=2;
    }

    for (int i=contador; i<padres.size(); i++)
        padres_cruzados.push_back(padres[i]);


    return padres_cruzados;
    

}