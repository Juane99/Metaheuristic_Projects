//############################################################################################
// Nombre: ILS.cpp                                                                           #
// Autor: Juan Emilio Martinez Manjon                                                        #
// Asignatura: Metaheuristica                                                                #
// Algoritmo: Busqueda Local Iterativa                                                       #
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
using namespace std;

//-------------------STRUCTS Y CLASES-------------------

/*Este struct nos permite pasar la matriz de restricciones a una lista,
donde cada elemento es una instancia de este struct.*/

struct Restriccion{
    int indice1;
    int indice2;
    int valor;
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
bool compruebaIndicesCorrectos(vector<int> indices,int num_centroides);
float calculaDesviacion(vector<Centroide> centroides, vector<list<float>> dataset);
float calculaFuncionObjetivo(vector<int> indices, vector<Centroide> centroides, vector<list<float>> dataset, vector<Restriccion> restrictions,  const float lambda, float &desviacion, int &infeasibility);
float distanciaMaxima(vector<list<float>> data_set);
void actualizaCentroide(Centroide &a_actualizar, vector<list<float>> data_set, int tipo);
void rellenaDatos(string file_name, vector<list<float>> &data_set, int dimension, float &max, float &min);
void rellenaRestricciones(string file_name, vector<Restriccion> &restrictions, int tam);
void rellenaCentroide(Centroide &centroides, vector<int> indices);
void procesa(vector<list<float>> data_set, vector<int> &indices, vector<Restriccion> restrictions, vector<Centroide> &centroides, int problema, const float lambda, int semilla, int &mejor_infeas, float &mejor_valor, float &mejor_desv);
void procesaILS(vector<list<float>> data_set, vector<int> &indices, vector<Restriccion> restrictions, vector<Centroide> &centroides, int problema, const float lambda, int semilla);
vector<int> mutaIndices(vector<int> indices, int num_centroides);
float enfriaTemperatura(float temperatura, float t_inicial, float t_final, int tam_dataset);
float getTempInicial(float coste_s0);


//-------------------PROGRAMA PRINCIPAL-------------------


int main(int argc, char** argv){

    if (argc != 3){
        cerr << "Introduzca un numero para inicializar la seed y el porcentaje de restricciones" << endl;
        exit(1);
    }

    //-------------------DECLARACION VARIABLES Y CONSTANTES-------------------

    //Inicializamos la semilla con el argumento de entrada
    Set_random(atoi(argv[1]));

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
    restrictions_20_rand, restrictions_10_ecoli, restrictions_20_ecoli, restrictions_10_newthyroid, 
    restrictions_20_newthyroid;

    float max_iris = 0, min_iris = numeric_limits<float>::max();
    float max_rand = 0, min_rand = numeric_limits<float>::max();
    float max_ecoli = 0, min_ecoli = numeric_limits<float>::max();
    float max_newthyroid = 0, min_newthyroid = numeric_limits<float>::max();

    vector<Centroide> centroides_iris, centroides_rand, centroides_ecoli, centroides_newthyroid;

    vector<int> indices_iris, indices_rand, indices_ecoli, indices_newthyroid;

    //Para calcular el tiempo de ejecucion
    unsigned t0_iris, t0_rand, t0_ecoli, t1_iris, t1_rand, t1_ecoli, t0_newthyroid, t1_newthyroid;
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
        

    //-------------------INICIALIZACION VECTORES-------------------


    do{
        indices_iris.clear();
            

        for (int i=0; i<150; i++){
            indices_iris.push_back(Randint(0,2));
        }
        
    }while(!compruebaIndicesCorrectos(indices_iris,3));


    do{
        indices_rand.clear();
            

        for (int i=0; i<150; i++){
            indices_rand.push_back(Randint(0,2));
        }
        
    }while(!compruebaIndicesCorrectos(indices_rand,3));


    do{
        indices_ecoli.clear();
            

        for (int i=0; i<336; i++){
            indices_ecoli.push_back(Randint(0,7));
        }
        
    }while(!compruebaIndicesCorrectos(indices_ecoli,8));


    do{
        indices_newthyroid.clear();
            

        for (int i=0; i<215; i++){
            indices_newthyroid.push_back(Randint(0,2));
        }
        
    }while(!compruebaIndicesCorrectos(indices_newthyroid,3));

    
    
    //Comenzamos procesamiento datasets

    if (atoi(argv[2]) == 10){

        cout << "-------------------ILS-ES-------------------" << endl;


        cout << "IRIS" << endl;

        t0_iris = clock();
        procesaILS(data_set_iris, indices_iris, restrictions_10_iris,centroides_iris,4,LAMBDA_IRIS_10,atoi(argv[1]));
        t1_iris = clock();
        tiempo_iris = (double(t1_iris-t0_iris)/CLOCKS_PER_SEC);
        cout << "Tiempo Iris: " << tiempo_iris << endl << endl;

        cout << "RAND" << endl;

        t0_rand = clock();
        procesaILS(data_set_rand, indices_rand, restrictions_10_rand,centroides_rand,2,LAMBDA_RAND_10,atoi(argv[1]));
        t1_rand = clock();
        tiempo_rand = (double(t1_rand-t0_rand)/CLOCKS_PER_SEC);
        cout << "Tiempo Rand: " << tiempo_rand << endl << endl;

        cout << "ECOLI" << endl;

        t0_ecoli = clock();
        procesaILS(data_set_ecoli, indices_ecoli, restrictions_10_ecoli,centroides_ecoli,7,LAMBDA_ECOLI_10,atoi(argv[1]));
        t1_ecoli = clock();
        tiempo_ecoli = (double(t1_ecoli-t0_ecoli)/CLOCKS_PER_SEC);
        cout << "Tiempo Ecoli: " << tiempo_ecoli << endl << endl;

        cout << "NEWTHYROID" << endl;

        t0_newthyroid = clock();
        procesaILS(data_set_newthyroid, indices_newthyroid, restrictions_10_newthyroid,centroides_newthyroid,5,LAMBDA_NEWTHYROID_10,atoi(argv[1]));
        t1_newthyroid = clock();
        tiempo_newthyroid = (double(t1_newthyroid-t0_newthyroid)/CLOCKS_PER_SEC);
        cout << "Tiempo Newthyroid: " << tiempo_newthyroid << endl << endl; 

    }
    else if (atoi(argv[2]) == 20){

        cout << "-------------------ILS-ES-------------------" << endl;


        cout << "IRIS" << endl;

        t0_iris = clock();
        procesaILS(data_set_iris, indices_iris, restrictions_20_iris,centroides_iris,4,LAMBDA_IRIS_20,atoi(argv[1]));
        t1_iris = clock();
        tiempo_iris = (double(t1_iris-t0_iris)/CLOCKS_PER_SEC);
        cout << "Tiempo Iris: " << tiempo_iris << endl << endl;

        cout << "RAND" << endl;

        t0_rand = clock();
        procesaILS(data_set_rand, indices_rand, restrictions_20_rand,centroides_rand,2,LAMBDA_RAND_20,atoi(argv[1]));
        t1_rand = clock();
        tiempo_rand = (double(t1_rand-t0_rand)/CLOCKS_PER_SEC);
        cout << "Tiempo Rand: " << tiempo_rand << endl << endl;

        cout << "ECOLI" << endl;

        t0_ecoli = clock();
        procesaILS(data_set_ecoli, indices_ecoli, restrictions_20_ecoli,centroides_ecoli,7,LAMBDA_ECOLI_20,atoi(argv[1]));
        t1_ecoli = clock();
        tiempo_ecoli = (double(t1_ecoli-t0_ecoli)/CLOCKS_PER_SEC);
        cout << "Tiempo Ecoli: " << tiempo_ecoli << endl << endl;

        cout << "NEWTHYROID" << endl;

        t0_newthyroid = clock();
        procesaILS(data_set_newthyroid, indices_newthyroid, restrictions_20_newthyroid,centroides_newthyroid,5,LAMBDA_NEWTHYROID_20,atoi(argv[1]));
        t1_newthyroid = clock();
        tiempo_newthyroid = (double(t1_newthyroid-t0_newthyroid)/CLOCKS_PER_SEC);
        cout << "Tiempo Newthyroid: " << tiempo_newthyroid << endl << endl; 
       

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

    float medias[tipo];

    for (int i=0; i<tipo; i++)
        medias[i] = 0.0;

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

/*Funcion principal que realiza el algoritmo de enfriamiento s*/

void procesa(vector<list<float>> data_set, vector<int> &indices, vector<Restriccion> restrictions, vector<Centroide> &centroides, int problema, const float lambda, int semilla, int &mejor_infeas, float &mejor_valor, float &mejor_desv){

    float la_desv;
    int evaluaciones = 0, la_infeas;
    bool puede_calcular_funcion_objetivo = false;
    vector<int> mejor_solucion;
    int vecinos_generados = 0, vecinos_aceptados = 1;

    int generaciones_maximas = (10*data_set.size());
    int aceptaciones_maximas = (0.1*generaciones_maximas);


    //Inicializamos los centroides con los datos de los indices generados aleatoriamente
    for (int j=0; j<centroides.size(); j++){
        rellenaCentroide(centroides[j],indices);
        actualizaCentroide(centroides[j], data_set, problema);
    }

    mejor_valor = calculaFuncionObjetivo(indices,centroides,data_set,restrictions,lambda, la_desv, la_infeas);
    evaluaciones++;
    mejor_desv = la_desv;
    mejor_infeas = la_infeas;

    float actual = mejor_valor;
    int infactibilidad_actual = la_infeas;    
    float temperatura_inicial = getTempInicial(mejor_valor);
    float temperatura = temperatura_inicial;


    while ((evaluaciones < 10000) && vecinos_aceptados > 0 && temperatura > 0.001){
        vecinos_aceptados = 0;
        vecinos_generados = 0;

        while(vecinos_aceptados < aceptaciones_maximas && vecinos_generados < generaciones_maximas){

            int elemento_a_cambiar = Randint(0, data_set.size()-2);

            int centroide_anterior = indices[elemento_a_cambiar];            
            int elemento_cambiado;

            do{

                elemento_cambiado = Randint(0,centroides.size()-1);
            }while(indices[elemento_a_cambiar] == elemento_cambiado);

            vecinos_generados++;

            puede_calcular_funcion_objetivo = false;

            vector<int> aux = indices;
            aux[elemento_a_cambiar] = elemento_cambiado;

            if(compruebaIndicesCorrectos(aux,centroides.size()))
                puede_calcular_funcion_objetivo = true;

            if (puede_calcular_funcion_objetivo){
                vector<Centroide> vector_auxiliar = centroides;

                for (int j=0; j<centroides.size(); j++){
                    rellenaCentroide(centroides[j],aux);
                    actualizaCentroide(centroides[j], data_set, problema);
                }

                float a_comparar = (calculaFuncionObjetivo(aux,centroides,data_set,restrictions,lambda, la_desv, la_infeas));
                evaluaciones++;
                float aleatorio = Randfloat(0,1);
                float diferencia_funciones = a_comparar - actual;

                if (diferencia_funciones < 0 || aleatorio <= (exp(-diferencia_funciones/temperatura))){           

                    vecinos_aceptados++;
                   
                    indices = aux;

                    if (a_comparar < mejor_valor){
                        mejor_desv = la_desv;
                        mejor_infeas = la_infeas;
                        mejor_valor = a_comparar;
                        mejor_solucion = indices;
                    }

                    actual = a_comparar;
                    infactibilidad_actual = la_infeas;

                }
                else
                    centroides = vector_auxiliar;

            }


        }
        temperatura = enfriaTemperatura(temperatura,temperatura_inicial,0.001,data_set.size());
        //float alfa = Randfloat(0.9,0.99);
        //temperatura = temperatura*alfa;

    }


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

float calculaFuncionObjetivo(vector<int> indices, vector<Centroide> centroides, vector<list<float>> dataset, vector<Restriccion> restrictions, const float lambda, float &desviacion, int &infeasibility){

    float funcion_objetivo = 0.0;
    desviacion = 0.0;
    infeasibility = 0;

    desviacion = calculaDesviacion(centroides,dataset);
    infeasibility = calculaInfeasibility(indices,restrictions);
     
    funcion_objetivo = desviacion + (infeasibility * lambda);

    return funcion_objetivo;  
}

//--------------------------------------------------------------------------------------------

/*Funcion que comprueba si los indices de un vecindario son correctos y no dejan ningun
centroide vacio*/
bool compruebaIndicesCorrectos(vector<int> indices, int num_centroides){

    bool son_correctos = true;
    bool comprobaciones[num_centroides];

    for (int i=0; i<num_centroides; i++)
        comprobaciones[i] = false;

    for (int i=0; i<indices.size(); i++)
        comprobaciones[indices[i]] = true;
    
    for (int i=0; i<num_centroides; i++){

        if (comprobaciones[i] == false)
            son_correctos = false;
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
/*Funcion que se encarga de ejecutar la Busqueda Multiarranque*/

void procesaILS(vector<list<float>> data_set, vector<int> &indices, vector<Restriccion> restrictions, vector<Centroide> &centroides, int problema, const float lambda, int semilla){

    vector<int> mejor_resultado;
    int mejor_infactibilidad = numeric_limits<int>::max();
    float mejor_desviacion = numeric_limits<float>::max(), mejor_funcion = numeric_limits<float>::max();

    int inf;
    float desv, fun;

    for (int j=0; j<centroides.size(); j++){
        rellenaCentroide(centroides[j],indices);
        actualizaCentroide(centroides[j], data_set, problema);
    }

    procesa(data_set,indices,restrictions,centroides,problema,lambda,semilla,inf,fun,desv);

    vector<int> S = indices; //S
    vector<int> S_prima;
    float funcion_S = fun;
    int inf_S = inf;
    float desv_S = desv;
   
   
    mejor_funcion = funcion_S;
    mejor_desviacion = desv_S;
    mejor_infactibilidad = inf_S;
    mejor_resultado = S;
    

    //Realizamos las 9 llamadas restantes a la BL
    for (int i=0; i<9; i++){

        S_prima = mutaIndices(S,centroides.size()); //S'
        procesa(data_set,S_prima,restrictions,centroides,problema,lambda,semilla,inf,fun,desv); //S''

        if (fun < funcion_S){
            S = S_prima;
            funcion_S = fun; 
            desv_S = desv;
            inf_S = inf;
        }

        if (funcion_S < mejor_funcion){
            mejor_funcion = funcion_S;
            mejor_desviacion = desv_S;
            mejor_infactibilidad = inf_S;
            mejor_resultado = S;
        }
    }


    cout << "Funcion Objetivo: " << mejor_funcion << endl;
    cout << "Desviacion: " << mejor_desviacion << endl;
    cout << "Infeasibility: " << mejor_infactibilidad << endl;
    
    for (int i=0; i<mejor_resultado.size(); i++){
        cout << mejor_resultado[i] << " ";
    }

    cout << endl;
}

//--------------------------------------------------------------------------------------------
/*Funcion que aplique la mutacion al vector de indices*/

vector<int> mutaIndices(vector<int> indices, int num_centroides){

    vector<int> mutado;
    int inicio_segmento = Randint(0,indices.size()-3);
    int v = 0.1*indices.size();

    vector<bool> necesita_cambiar;
    necesita_cambiar.assign(indices.size()-1,false);

    if ( ((inicio_segmento + v) % indices.size()-1) > inicio_segmento){

        for (int j=inicio_segmento; j < ((inicio_segmento + v) % indices.size()-1); j++)
            necesita_cambiar[j] = true;
    }
    else{

        for (int j=inicio_segmento; j < indices.size()-1; j++)
            necesita_cambiar[j] = true;

        for (int j=0; j < ((inicio_segmento + v) % indices.size()-1); j++)
            necesita_cambiar[j] = true;
    }

    //Cambiamos los indices que deben cambiar

    for (int i=0; i<indices.size(); i++){
        vector<int> aux;
        if (necesita_cambiar[i]){

            do{
                aux = indices;
                int aleatorio = Randint(0,num_centroides-1);
                aux[i] = aleatorio;

            }while(!compruebaIndicesCorrectos(aux,num_centroides));

            indices = aux;
        }

    }

    mutado = indices;

    return mutado;

}

//--------------------------------------------------------------------------------------------
/*Funcion que calcula la temperatura inicial*/

float getTempInicial(float coste_s0){
    float numerador = 0.3*coste_s0;
    float denominador = -log(0.3);

    return (float)(numerador/denominador);
}

//--------------------------------------------------------------------------------------------
/*Funcion que calcula la nueva temperatura al enfriar*/

float enfriaTemperatura(float temperatura, float t_inicial, float t_final, int tam_dataset){
    int maximos_vecinos = (10*tam_dataset);
    float M = 10000/maximos_vecinos;

    float Beta = (t_inicial - t_final) / (M*t_inicial*t_final);

    float temperatura_enfriada = temperatura / (1 + Beta*temperatura);

    return temperatura_enfriada;

}