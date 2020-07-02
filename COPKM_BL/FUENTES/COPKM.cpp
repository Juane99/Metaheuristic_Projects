//############################################################################################
// Nombre: COPKM.cpp                                                                         #
// Autor: Juan Emilio Martinez Manjon                                                        #
// Asignatura: Metaheuristica                                                                #
// Algoritmo: Greedy                                                                         #
//############################################################################################

//-------------------EXPLICACION PROGRAMA-------------------
/*

Vamos a trabajar con tres conjuntos de datos (Iris, Rand y Ecoli) de los que sabemos lo siguiente:

    - Iris tiene 150 datos en 4 dimensiones y debe ir en 3 clusters
    - Rand tiene 150 datos en 2 dimensiones y debe ir en 3 clusters
    - Ecoli tiene 336 datos en 7 dimensiones y debe ir en 8 clusters


Tenemos que asignarle a cada cluster un conjunto de datos n de entre los datos del dataset.
Esto lo haremos con un algoritmo puramente greedy, donde meteremos un indice en el cluster cuya
infactibilidad sea menor. En el caso de que haya un empate en infactibilidades, se eligira aquel
cuya distancia al indice sea menor.

Este proceso lo repetiremos hasta que al evaluar TODOS los indices no cambiemos ninguna asignacion
de centroides.

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

    int getNumero() const{
        return numero;
    }

    void addDato(int indice){
        datos_asignados.push_back(indice);
    }

    void delDato(){
        datos_asignados.pop_back();
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

    void eliminaDatosAsignados(){
        datos_asignados.clear();
    }

    void setDatosAsignados(vector<int> datos){
        datos_asignados = datos;
    }

};

//-------------------CABECERAS FUNCIONES-------------------

void actualizaCentroide(Centroide &a_actualizar, vector<list<float>> data_set, int tipo);
void rellenaDatos(string file_name, vector<list<float>> &data_set, int dimension, float &max, float &min);
void rellenaRestricciones(string file_name, vector<vector<int>> &restrictions, int tam);
void procesa(vector<list<float>> data_set, vector<int> indices, vector<vector<int>> restrictions, vector<Centroide> &centroides, int problema, const float lambda);
double distanciaCentroide(list<float> caracteristicas, Centroide centroide);
float calculaDesviacion(vector<Centroide> centroides, vector<list<float>> dataset);
bool esIgual(vector<Centroide> cent1 ,vector<Centroide> cent2);
int calculaInfeasibility(vector<Centroide> centroides, vector<vector<int>> restrictions);
float distanciaMaxima(vector<list<float>> data_set);


//-------------------PROGRAMA PRINCIPAL-------------------

int main(int argc, char** argv){

    if (argc != 3){
        cerr << "Introduzca un numero para inicializar la seed y el porcentaje de restricciones" << endl;
        exit(1);
    }

    //-------------------DECLARACION VARIABLES Y CONSTANTES-------------------

    //Inicializamos la semilla con el argumento de flujo_entrada
    Set_random(atoi(argv[1]));


    //Numero de restricciones de cada dataset

    const float RESTR_IRIS_10 = 1117.0;
    const float RESTR_IRIS_20 = 2235.0;
    const float RESTR_RAND_10 = 1117.0;
    const float RESTR_RAND_20 = 2235.0;
    const float RESTR_ECOLI_10 = 5628.0;
    const float RESTR_ECOLI_20 = 11256.0;


    vector<list<float>> data_set_iris, data_set_rand, data_set_ecoli;
    vector<vector<int>> restrictions_10_iris, restrictions_20_iris, restrictions_10_rand,
    restrictions_20_rand, restrictions_10_ecoli, restrictions_20_ecoli;

    float max_iris = 0, min_iris = numeric_limits<float>::max();
    float max_rand = 0, min_rand = numeric_limits<float>::max();
    float max_ecoli = 0, min_ecoli = numeric_limits<float>::max();

    vector<Centroide> centroides_iris, centroides_rand, centroides_ecoli;

    vector<int> indices_iris, indices_rand, indices_ecoli;

    unsigned t0_iris, t0_rand, t0_ecoli, t1_iris, t1_rand, t1_ecoli;
    double tiempo_iris, tiempo_rand, tiempo_ecoli;


    //-------------------RELLENADO DE TDA-------------------

    //Vamos a llenar los data_sets con los datos
    rellenaDatos("iris_set.dat",data_set_iris,4,max_iris,min_iris);
    rellenaDatos("rand_set.dat",data_set_rand,2,max_rand,min_rand);
    rellenaDatos("ecoli_set.dat",data_set_ecoli,7,max_ecoli,min_ecoli);

    //Vamos a rellenar el 10% de restricciones de los datos
    rellenaRestricciones("iris_set_const_10.const",restrictions_10_iris,150);
    rellenaRestricciones("rand_set_const_10.const",restrictions_10_rand,150);
    rellenaRestricciones("ecoli_set_const_10.const",restrictions_10_ecoli,336);

    //Vamos a rellenar el 20% de restricciones de los datos
    rellenaRestricciones("iris_set_const_20.const",restrictions_20_iris,150);
    rellenaRestricciones("rand_set_const_20.const",restrictions_20_rand,150);
    rellenaRestricciones("ecoli_set_const_20.const",restrictions_20_ecoli,336);


    const float LAMBDA_IRIS_10 = distanciaMaxima(data_set_iris) / RESTR_IRIS_10;
    const float LAMBDA_IRIS_20 = distanciaMaxima(data_set_iris) / RESTR_IRIS_20;
    const float LAMBDA_RAND_10 = distanciaMaxima(data_set_rand) / RESTR_RAND_10;
    const float LAMBDA_RAND_20 = distanciaMaxima(data_set_rand) / RESTR_RAND_20;
    const float LAMBDA_ECOLI_10 = distanciaMaxima(data_set_ecoli) / RESTR_ECOLI_10;
    const float LAMBDA_ECOLI_20 = distanciaMaxima(data_set_ecoli) / RESTR_ECOLI_20;


    //Generamos los centroides de los conjuntos de datos

    for (int i=0; i<3; i++){
        Centroide aux_iris(max_iris,min_iris,i);
        Centroide aux_rand(max_rand,min_rand,i);
        
        centroides_iris.push_back(aux_iris);
        centroides_rand.push_back(aux_rand);
    }

    for (int i=0; i<8; i++){
        Centroide aux_ecoli(max_ecoli,min_ecoli,i);
        
        centroides_ecoli.push_back(aux_ecoli);
    }

    //Inicializamos los vectores de indices

    for (int i=0; i<150; i++){
        indices_iris.push_back(i);
        indices_rand.push_back(i);
    }

    for (int j=0; j<336; j++)
        indices_ecoli.push_back(j);


    //-------------------INICIALIZACION VECTORES-------------------


    //Mezclamos los conjuntos de datos de forma aleatoria

    shuffle (indices_iris.begin(), indices_iris.end(),default_random_engine(atoi(argv[1])));
    shuffle (indices_rand.begin(), indices_rand.end(),default_random_engine(atoi(argv[1])));
    shuffle (indices_ecoli.begin(), indices_ecoli.end(),default_random_engine(atoi(argv[1])));


    //Comenzamos procesamiento datasets

    if (atoi(argv[2]) == 10){

        cout << "IRIS" << endl;

        t0_iris = clock();
        procesa(data_set_iris, indices_iris, restrictions_10_iris,centroides_iris,4,LAMBDA_IRIS_10);
        t1_iris = clock();
        tiempo_iris = (double(t1_iris-t0_iris)/CLOCKS_PER_SEC);
        cout << "Tiempo Iris: " << tiempo_iris << endl << endl;

        cout << "RAND" << endl;

        t0_rand = clock();
        procesa(data_set_rand, indices_rand, restrictions_10_rand,centroides_rand,2,LAMBDA_RAND_10);
        t1_rand = clock();
        tiempo_rand = (double(t1_rand-t0_rand)/CLOCKS_PER_SEC);
        cout << "Tiempo Rand: " << tiempo_rand << endl << endl;

        cout << "ECOLI" << endl;

        t0_ecoli = clock();
        procesa(data_set_ecoli, indices_ecoli, restrictions_10_ecoli,centroides_ecoli,7,LAMBDA_ECOLI_10);
        t1_ecoli = clock();
        tiempo_ecoli = (double(t1_ecoli-t0_ecoli)/CLOCKS_PER_SEC);
        cout << "Tiempo Ecoli: " << tiempo_ecoli << endl << endl;

    }
    else if (atoi(argv[2]) == 20){
       
        cout << "IRIS" << endl;

        t0_iris = clock();
        procesa(data_set_iris, indices_iris, restrictions_20_iris,centroides_iris,4,LAMBDA_IRIS_20);
        t1_iris = clock();
        tiempo_iris = (double(t1_iris-t0_iris)/CLOCKS_PER_SEC);
        cout << "Tiempo Iris: " << tiempo_iris << endl << endl;

        cout << "RAND" << endl;

        t0_rand = clock();
        procesa(data_set_rand, indices_rand, restrictions_20_rand,centroides_rand,2,LAMBDA_RAND_20);
        t1_rand = clock();
        tiempo_rand = (double(t1_rand-t0_rand)/CLOCKS_PER_SEC);
        cout << "Tiempo Rand: " << tiempo_rand << endl << endl;

        cout << "ECOLI" << endl;

        t0_ecoli = clock();
        procesa(data_set_ecoli, indices_ecoli, restrictions_20_ecoli,centroides_ecoli,7,LAMBDA_ECOLI_20);
        t1_ecoli = clock();
        tiempo_ecoli = (double(t1_ecoli-t0_ecoli)/CLOCKS_PER_SEC);
        cout << "Tiempo Ecoli: " << tiempo_ecoli << endl << endl;

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
void rellenaRestricciones(string file_name, vector<vector<int>> &restrictions, int tam){

    ifstream flujo_entrada(file_name);

    if (!flujo_entrada){
        cerr << "No se pudo abrir el archivo deseado";
        exit(1);
    }

    int dato = 0;
    char coma = ' ';

    for (int i=0; i<tam; i++){
        vector<int> aux;
        for (int j=0; j<tam; j++){
            flujo_entrada >> dato;

            if (j < tam-1)
                flujo_entrada >> coma;

            aux.push_back(dato);
        }
        restrictions.push_back(aux);
    }           
    

}

//--------------------------------------------------------------------------------------------

/*Funcion que calcula la distancia de un indice a su centroide*/
double distanciaCentroide(list<float> caracteristicas, Centroide centroide){

    float distancia = 0.0;
    vector<float> vec_carac{begin(caracteristicas), end(caracteristicas)};

    for (int i=0; i<vec_carac.size(); i++)
        distancia += ((vec_carac[i] - centroide.getindice(i))*(vec_carac[i] - centroide.getindice(i) ));

    distancia = sqrt(distancia);


    return distancia;    
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

/*Funcion principal que realiza el algoritmo greedy*/
void procesa(vector<list<float>> data_set, vector<int> indices, vector<vector<int>> restrictions, vector<Centroide> &centroides, int problema, const float lambda){
    bool hay_cambio = true, es_necesario_borrar = false;
    int infactibilidad = 0, mejor_infactibilidad = numeric_limits<int>::max(), mejor_cluster;
    float mejor_distancia = numeric_limits<float>::max(), distancia = 0.0, desviacion;
    vector<Centroide> antes = centroides;

    while (hay_cambio){
        hay_cambio = false;
        
        for (int i=0; i<indices.size(); i++){
            mejor_infactibilidad = numeric_limits<int>::max();
            mejor_distancia = numeric_limits<float>::max();
            mejor_cluster = -1;
            for (int j=0; j<centroides.size(); j++){
                infactibilidad = 0;
                distancia = distanciaCentroide(data_set[indices[i]],centroides[j]);

                for (int k=0; k<antes[j].getDatosAsignados().size(); k++){
                    if (restrictions[indices[i]][(antes[j].getDatosAsignados())[k]] == -1)
                        infactibilidad++;
                }

                for (int k=j+1; k<centroides.size(); k++){
                    for (int l=0; l<antes[k].getDatosAsignados().size(); l++){
                        if (restrictions[indices[i]][(antes[k].getDatosAsignados())[l]] == 1)
                            infactibilidad++;
                    }
                }

                if (infactibilidad < mejor_infactibilidad || (infactibilidad == mejor_infactibilidad &&
                distancia < mejor_distancia)){
                    mejor_infactibilidad = infactibilidad;
                    mejor_distancia = distancia;
                    mejor_cluster = centroides[j].getNumero();
                }
            }

            if (mejor_cluster != -1)
                centroides[mejor_cluster].addDato(indices[i]);
            
        }

        if (!esIgual(centroides,antes))
            hay_cambio = true;

        //Comprobamos si ha habido algun cambio con la asignacion anterior
        //If yes, hay cambio = true

        if (hay_cambio){
            for (int i=0; i<centroides.size(); i++){
                actualizaCentroide(centroides[i],data_set,problema);
            }
        }

        antes = centroides;

        for (int k=0;k<centroides.size(); k++)
            centroides[k].eliminaDatosAsignados();
    }

    centroides = antes;

    //Imprimimos la desviacion del conjunto

    desviacion = calculaDesviacion(centroides,data_set);
    infactibilidad = calculaInfeasibility(centroides,restrictions);

    cout << "Desviacion: " << desviacion << endl;
    cout << "infactibilidad " << infactibilidad << endl;
    cout << "Funcion Objetivo: " << desviacion + (infactibilidad*lambda) << endl;
}

//--------------------------------------------------------------------------------------------

/*Funcion que comprueba si dos vectores de centroides son iguales*/
bool esIgual(vector<Centroide> cent1 ,vector<Centroide> cent2){
    bool es_igual = true;

    for (int i=0; i<cent1.size() && es_igual; i++){
        if (cent1[i].getDatosAsignados().size() != cent2[i].getDatosAsignados().size())
            es_igual = false;
        
        else{

            for (int j=0; j<cent1[i].getDatosAsignados().size(); j++){

                if ((cent1[i].getDatosAsignados())[j] != (cent2[i].getDatosAsignados())[j])
                    es_igual = false;
            }

        }
    }

    return es_igual;
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
/*Funcion para calcular la infactibilidad de un conjunto de centroides*/

int calculaInfeasibility(vector<Centroide> centroides, vector<vector<int>> restrictions){
    int infactibilidad = 0;

    for (int i=0; i<centroides.size(); i++){

        for (int j=0; j<centroides[i].getDatosAsignados().size(); j++){

            for (int k=j+1; k<centroides[i].getDatosAsignados().size(); k++){
                if (restrictions[(centroides[i].getDatosAsignados())[j]][(centroides[i].getDatosAsignados())[k]] == -1)
                    infactibilidad++;
            }

            for (int k=i+1; k<centroides.size(); k++){
                for (int l=0; l<centroides[k].getDatosAsignados().size(); l++){
                    if (restrictions[(centroides[i].getDatosAsignados())[j]][(centroides[k].getDatosAsignados())[l]] == 1)
                        infactibilidad++;
                }
            }
        }
    }

    return infactibilidad;
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
