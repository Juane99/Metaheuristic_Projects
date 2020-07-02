#Nombre: makefile
#Autor: Juan Emilio Martínez Manjón
#Función: Compilar los dos ficheros de la práctica 1 de Metaheurística

all: clean ../BIN/COPKM ../BIN/BL

../BIN/COPKM: COPKM.cpp random.cpp
	g++ -std=c++11 COPKM.cpp random.cpp -o ../BIN/COPKM
	
../BIN/BL: BL.cpp random.cpp
	g++ -std=c++11 BL.cpp random.cpp -o ../BIN/BL
	
	
clean:
	rm -f ../BIN/COPKM ../BIN/BL
