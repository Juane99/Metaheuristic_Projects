#Nombre: makefile
#Autor: Juan Emilio Martínez Manjón
#Función: Compilar los cuatro ficheros de la práctica 3 de Metaheurística

all: clean ../BIN/BMB ../BIN/ILS ../BIN/ES ../BIN/ILS-ES

../BIN/BMB: BMB.cpp random.cpp
	g++ -std=c++11 BMB.cpp random.cpp -o ../BIN/BMB -O3
	
../BIN/ILS: ILS.cpp random.cpp
	g++ -std=c++11 ILS.cpp random.cpp -o ../BIN/ILS -O3
	
../BIN/ES: ES.cpp random.cpp
	g++ -std=c++11 ES.cpp random.cpp -o ../BIN/ES -O3
	
../BIN/ILS-ES: ILS-ES.cpp random.cpp
	g++ -std=c++11 ILS-ES.cpp random.cpp -o ../BIN/ILS-ES -O3
	
clean:
	rm -f ../BIN/BMB ../BIN/ILS ../BIN/ILS-ES ../BIN/ES
