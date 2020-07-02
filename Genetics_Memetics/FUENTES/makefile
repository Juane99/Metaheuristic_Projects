#Nombre: makefile
#Autor: Juan Emilio Martínez Manjón
#Función: Compilar los dos ficheros de la práctica 2 de Metaheurística

all: clean ../BIN/AG ../BIN/AM

../BIN/AG: AG.cpp random.cpp
	g++ -std=c++11 AG.cpp random.cpp -o ../BIN/AG -O3
	
../BIN/AM: AM.cpp random.cpp
	g++ -std=c++11 AM.cpp random.cpp -o ../BIN/AM -O3
	
	
clean:
	rm -f ../BIN/AG ../BIN/AM
