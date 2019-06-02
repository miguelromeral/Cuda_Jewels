#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "string"
#include <time.h>
#include <windows.h>
#include <iostream>
using namespace std;

//Máximo número de ancho de la tesela, puesto que el máximo está en 32*32 = 1024
#define MAX 32

//Tablero del juego.
int * matriz;
//Modo de ejecución
char * tipoEjecucion;
//Variable del dispositivo.
int * matriz_d;
//Dificultad del juego (1, 2, 3)
//Rango: 4, 6, 8
//Filas, columnas y la multiplicación de éstas, el tamaño.
int dificultad, rango, filas, columnas, tam;

//Inicializa la matriz en funcion del rango de valores.
void rellenaMatriz(int * mat, int size, int rango)
{
	for (int i = 0; i < size; i++) {
		mat[i] = (rand() % rango) + 1;
	}
}

//Cambia el color de la salida de la consola, "colorea el diamante"
void changeUnderlined(int num) {
	SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), num);
}

//Pone por defecto la fuente a la típica salida de consola.
void clearUnderlined() {
	SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 7);
}

//Elimina la columna número colum (empezando desde 0 hasta nc - 1)
void eliminaColumna(int * mat, int rango, int nf, int nc, int colum)
{
	//Dejamos libre la columna 0, movemos de izquierda a derecha.
	for (int i = colum; i > 0; i--) {
		for (int j = 0; j < nf; j++) {
			mat[j*nc + i] = mat[j*nc + i - 1];
		}
	}
	//La columna 0, la de la izquierda, generamos valores random.
	for (int i = 0; i < nf; i++) {
		mat[i*nc] = (rand() % rango) + 1;
	}
}

//Elimina la fila número fila (empezando desde 0 hasta nf - 1)
void eliminaFila(int * mat, int rango, int nf, int nc, int fila)
{
	//Dejamos libre la columna 0, bajamos todas las filas.
	for (int i = fila; i > 0; i--) {
		for (int j = 0; j < nc; j++) {
			mat[i*nc + j] = mat[(i - 1)*nc + j];
		}
	}
	//Generamos números al azar en la primera fila.
	for (int i = 0; i < nc; i++) {
		mat[i] = (rand() % rango) + 1;
	}
}

//Devuelve un número natural (mayor que 0) indicando la cantidad de diamantes que se pueden eliminar (en horizontal y vertical)
int comportamientoCelda(int * mat, int nf, int nc, int celda) {
	//Y es la fila, X es la columna.
	int y = celda / nf, x = celda % nc;
	int color = mat[celda];
	int i;
	//Almacenamos en un array el número de 
	int seguidos[4]; //0 arriba, 1 derecha, 2 abajo, 3 izquierda.
	seguidos[0] = 0;
	seguidos[1] = 0;
	seguidos[2] = 0;
	seguidos[3] = 0;
	//Continuar indica que siga mirando en uno de los cuatro laterales.
	boolean continuar = true;
	i = 1;
	//Miramos inicialmente hacia arriba. (0)
	while (continuar) {
		//Si no nos salimos de la matriz y el color es el mismo...
		if (y - i >= 0 && color == mat[celda - nc * i]) {
			i++;
			seguidos[0]++;
		}
		else {
			continuar = false;
		}
	}
	//Miramos hacia la derecha (1)
	i = 1;
	continuar = true;
	while (continuar) {
		//Si no nos salimos de la matriz y el color es el mismo...
		if (x + i < nc && color == mat[celda + i]) {
			i++;
			seguidos[1]++;
		}
		else {
			continuar = false;
		}
	}
	i = 1;
	//Miramos hacia abajo (2)
	continuar = true;
	while (continuar) {
		//Si no nos salimos de la matriz y el color es el mismo...
		if (y + i < nf && color == mat[celda + nc * i]) {
			i++;
			seguidos[2]++;
		}
		else {
			continuar = false;
		}
	}
	continuar = true;
	//Miramos hacia la izquierda (3)
	i = 1;
	while (continuar) {
		//Si no nos salimos de la matriz y el color es el mismo...
		if (x - i >= 0 && color == mat[celda - i]) {
			i++;
			seguidos[3]++;
		}
		else {
			continuar = false;
		}
	}
	int ret = 0, aux;
	//Buscamos cual de las dos dimensiones (arriba y abajo o derehca y izquierda) a ver cual casa más.
	for (int k = 0; k < 2; k++) {
		aux = seguidos[k] + seguidos[k + 2];
		//Si tenemos más de un coincidente (1-1, 2-0, 0-2 en adelante) obtenemos la mayor.
		if (aux >= 2) {
			if (aux > ret) {
				ret = aux;
			}
		}
	}
	//Si no hay diamantes a eliminar, entonces devuelve 0.
	return ret;
}

//Elige el color del diamante para cambiar la salida de la consola.
void cambiarColorEnFuncionDeDiamante(int n) {
	/*
	Colores de diamantes:

	Solo fondo	Texto Blanco	Texto Negro
	1 - Azul			153			159				144
	2 - Rojo			204			207				192
	3 - Naranja (Rosa)	221			223				208
	4 - Verde			170			175				160
	5 - Marrón (granate)68			79				64
	6 - Amarillo		238			239				224
	7 - Negro			0			15				-
	8 - Blanco			256			-				240
	*/
	switch (n) {
	case 1: changeUnderlined(/*153*//*159*/144);
		break;
	case 2: changeUnderlined(/*204*/207/*192*/);
		break;
	case 3: changeUnderlined(/*221*//*223*/208); //CAMBIADO DE NARANJA A ROSA.
		break;
	case 4: changeUnderlined(/*170*//*175*/160);
		break;
	case 5: changeUnderlined(/*68*/79/*64*/); //CAMBIADO DE MARRÓN A GRANATE
		break;
	case 6: changeUnderlined(/*238*//*239*/224);
		break;
	case 7: changeUnderlined(/*0*/15/*-*/);
		break;
	default:
		changeUnderlined(/*256*//*-*/240);
	}
}

//Imprime por pantalla la matriz con los diamantes coloreados.
void imprimeMatriz(int * mat, int fil, int col) {
	int tam = fil * col;
	for (int i = 0; i < tam; i++) {
		cambiarColorEnFuncionDeDiamante(mat[i]);
		printf(" %d ", mat[i]);
		if ((i + 1) % col == 0) {
			clearUnderlined();
			printf("\n");
		}
	}
	clearUnderlined();
}

//Elimina el diamante de la celda "pos", baja los de arriba y el de arriba se genera aleatoriamente.
void eliminarDiamante(int * mat, int nc, int pos, int rango) {
	int filas = pos / nc;
	int col = pos % nc;
	//Bajamos las celdas de esa columna.
	for (int i = filas; i > 0; i--) {
		mat[i*nc + col] = mat[(i - 1)*nc + col];

	}
	//Generamos la celda de arriba.
	mat[col] = (rand() % rango) + 1;
}

//Calcula el rango en función de su dificultad. 
int calcularRango(int dif) {
	int r;
	switch (dificultad) {
	case 1:
		r = 4;
		break;
	case 2:
		r = 6;
		break;
	case 3:
		r = 8;
		break;
	}
	return r;
}

//Comprueba toda la matriz y elimina todos los movimientos que estén actualmente.
//El entero saltarEliminacion si esta a 0 solo comprueba si hay movimientos. 0 para saltarsela.
//Devuelve 1 se hay movimientos posibles y 0 sino.
int comprobarYEliminar(int * mat, int nf, int nc, int rango, int saltarEliminacion) {
	//Matriz que indica en cada celda cuantos diamantes tiene seguidos.
	int * aEliminar;
	aEliminar = (int*)malloc(nf*nc * sizeof(int));
	//Indicamos el mejor valor a eliminar.
	int mejor = 0;
	//Indicamos el índice de donde está la celda que tiene la mejor opción.
	int indMejor = -1;
	for (int i = 0; i < nf*nc; i++) {
		//Obtenemos cada valor.
		aEliminar[i] = comportamientoCelda(mat, nf, nc, i);
		//Si la combinación actual es mejor que la que tenemos actualmente, actualizamos el valor y el índice.
		if (aEliminar[i] > mejor) {
			mejor = aEliminar[i];
			indMejor = i;
		}
	}
	//Si el indice del mejor es mayor de -1 indica que hay al menos un movimiento posible.
	if (indMejor > -1) {
		//Si no se salta la eliminación...
		if (saltarEliminacion == 0) {
			//Eliminamos cada diamante empezando por arriba, para que no perdamos índices de diamantes que si se deben eliminar.
			for (int i = 0; i < nf*nc; i++) {
				if (aEliminar[i] > 0) {
					eliminarDiamante(mat, nc, i, rango);
				}
			}
		}
		//Si hay movimientos posibles retornamos 1
		return 1;
	}
	else {
		//Si no hay movimientos posibles retornamos 0
		return 0;
	}
}

//Deberá devolver 1 si se realiza un cambio para eliminar diamantes, 0 sino.
//Intercambia entre dos valores, el de la posición fil, col, el movimiento mov.
//Si esta automatico (!= 0), solo se permuta, sin hacer comprobaciones
//Si está borrarSolo, eliminará los diamantes de todo el tablero.
int permutar(int * mat, int nf, int nc, int fi, int col, char mov, int rango, int automatico, int borrarSolo) {
	int aux;
	switch (mov) {
	case 'u':
	case 'U':
		if (fi > 0) {
			//Permutación.
			aux = mat[(fi - 1)*nc + col];
			mat[(fi - 1)*nc + col] = mat[fi*nc + col];
			mat[fi*nc + col] = aux;

			//Si está en modo manual, si el movimiento no genera una combinación, se deshace.
			if (automatico == 0) {
				if (comportamientoCelda(mat, nf, nc, (fi - 1)*nc + col) > 0 ||
					comportamientoCelda(mat, nf, nc, fi*nc + col) > 0) {
					//Si genera la combinación y se desean eliminar todos los posibles, se eliminan.
					if (borrarSolo == 1) {
						comprobarYEliminar(mat, nf, nc, rango, automatico);

					}return 1;
				}
				else {
					//Se deshace el cambio.
					aux = mat[(fi - 1)*nc + col];
					mat[(fi - 1)*nc + col] = mat[fi*nc + col];
					mat[fi*nc + col] = aux;
				}
			}

		}
		break;
	case 'r':
	case 'R':
		//De igual manera ocurre en los demás posibles movimientos.
		if (col < nc - 1) {
			aux = mat[fi*nc + col + 1];
			mat[fi*nc + col + 1] = mat[fi*nc + col];
			mat[fi*nc + col] = aux;


			if (automatico == 0) {
				if (comportamientoCelda(mat, nf, nc, fi*nc + col + 1) > 0 ||
					comportamientoCelda(mat, nf, nc, fi*nc + col) > 0) {
					if (borrarSolo == 1) {
						comprobarYEliminar(mat, nf, nc, rango, automatico);

					}return 1;
				}

				else {
					aux = mat[fi*nc + col + 1];
					mat[fi*nc + col + 1] = mat[fi*nc + col];
					mat[fi*nc + col] = aux;
				}
			}
		}
		break;
	case 'd':
	case 'D':
		if (fi < nf - 1) {
			aux = mat[(fi + 1)*nc + col];
			mat[(fi + 1)*nc + col] = mat[fi*nc + col];
			mat[fi*nc + col] = aux;

			if (automatico == 0) {
				if (comportamientoCelda(mat, nf, nc, (fi + 1)*nc + col) > 0 ||
					comportamientoCelda(mat, nf, nc, fi*nc + col) > 0) {
					if (borrarSolo == 1) {
						comprobarYEliminar(mat, nf, nc, rango, automatico);

					}return 1;
				}

				else {
					aux = mat[(fi + 1)*nc + col];
					mat[(fi + 1)*nc + col] = mat[fi*nc + col];
					mat[fi*nc + col] = aux;
				}
			}
		}
		break;
	case 'l':
	case 'L':
		if (col > 0) {
			aux = mat[fi*nc + col - 1];
			mat[fi*nc + col - 1] = mat[fi*nc + col];
			mat[fi*nc + col] = aux;

			if (automatico == 0) {
				if (comportamientoCelda(mat, nf, nc, fi*nc + col - 1) > 0 ||
					comportamientoCelda(mat, nf, nc, fi*nc + col) > 0) {
					if (borrarSolo == 1) {
						comprobarYEliminar(mat, nf, nc, rango, automatico);

					}return 1;
				}

				else {
					aux = mat[fi*nc + col - 1];
					mat[fi*nc + col - 1] = mat[fi*nc + col];
					mat[fi*nc + col] = aux;
				}
			}
		}
		break;
	}
	return 0;
}

//ATENCIÓN: Método muy poco eficiente. Mejor utilizar la de la CPU.
//NO TIENE UTILIDAD EN EL FLUJO DE EJECUCIÓN.
__global__ void permutar_MG(int * mat, int nf, int nc, int fi, int co, char mov) {
	//Obtenemos la columna
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int fil = blockIdx.y * blockDim.y + threadIdx.y;
	//Y la fila
	int dim = fil * nc + col;
	//Color del diamante que vamos a cambiar.
	int aCambiar;
	//Color inicial de la celda actual
	int inicial = mat[dim];
	//Colcam y filcam son las columnas y filas que se van a cambiar con la seleccionada (dada por fi, co y el movimiento)
	//Si colcam y filcam coinciden con la celda inicial, significa que esa celda, y solamente esa, se va a cambiar con la inicial.
	int colcam = col;
	int filcam = fil;
	//Obtenemos solo la cruz de los diamantes más cercanos a la celda a cambiar (los diamantes de arriba, derecha, abajo e izquierda).
	if ((col < co + 2 && col > co - 2) || (fil < fi + 2 && fil > fi - 2)) {
		switch (mov) {
		case 'u':
		case 'U':
			if (fi > 0) {
				//Si la celda que se ha seleccionado ha pulsado arriba, la posicion con la que cambia varía (- 1 fila)
				if (co == col && fi == fil) {
					filcam--;
				}
				//En cambio, si está ejecutando este flujo y no es la celda que se quiere eliminar, es una de sus vecinas,
				//El efecto es el contrario, significa que se tendrá que cambiar con la fila de abajo. (+ 1 fila)
				else {
					filcam++;
				}
			}
			break;
		case 'r':
		case 'R':
			//De igual manera ocurre en los demás posibles movimientos.
			if (col < nc - 1) {
				if (co == col && fi == fil) {
					colcam++;
				}
				else {
					colcam--;
				}
			}
			break;
		case 'd':
		case 'D':
			if (fi < nf - 1) {
				if (co == col && fi == fil) {
					filcam++;
				}
				else {
					filcam--;
				}
			}
			break;
		case 'l':
		case 'L':
			if (col > 0) {
				if (co == col && fi == fil) {
					colcam--;
				}
				else {
					colcam++;
				}
			}
			break;
		}
		//Si las columnas cambiadas y las filas cambiadas coinciden con la original (solo dos lo harán, que serán 
		//la celda que se quiere cambiar y la celda con la que se cambiará), entonces se permutan.
		if ((filcam == fi && colcam == co) || (fi == fil && co == col)) {
			//Si estamos en la celda seleccionada, se coge la que se ha querido hacer el movimiento.
			if (fi == fil && co == col) {
				aCambiar = mat[filcam*nc + colcam];
			}
			//Si no es la celda seleccionada, es la celda con la que la seleccionada se quiere cambiar, cogemos el valor
			//de la otra.
			else {
				aCambiar = mat[fi*nc + co];
			}
			__syncthreads();
			//Hacemos el cambio.
			mat[fil * nc + col] = aCambiar;
		}
	}
}

//Muestra al usuario la manera de ejecutar el programa por consola
void usage() {
	cout << "Usage: CudaLegends.exe <tipo_de_ejecucion> " <<
		"<dificultad> <columnas> <filas>" << endl
		<< "Valores posibles: " << endl
		<< "tipo_de_ejecucion: -a (automatico) | -m (manual)" << endl
		<< "dificultad: valores del 1 al 3." << endl
		<< "columnas y filas un valor entero." << endl;
	getchar();
}

//Obtiene el número del stream del archivo hasta encontrarse con ';'
//Sirve para cargar datos del archivo como las filas, columnas...
int cargarNumeroFile(FILE * archivo) {
	int num = -1;
	char caracter;
	bool continuar = true;
	while (continuar) {
		caracter = fgetc(archivo);
		if (caracter != ';') {
			//Si es la primera vez, entonces procedemos con las unidades.
			if (num == -1) {
				//Obtenemos el código ASCII del carácter
				num = (int)caracter;
				//Le restamos 48 para poder tener el valor decimal.
				num -= 48;
			}
			//sino, hay que multiplicarlo por diez para guardar las nuevas unidades.
			else {
				num *= 10;
				num += (int)caracter - 48;
			}
		}
		else
			continuar = false;
	}
	return num;
}

//Carga los datos de la anterior partida en la actual.
//Formato del archivo:
//<modo>;<columnas>;<filas>;<Cadena enteros de filas * columnas
//Retorna 1 si se ha podido cargar con éxito.
int cargar() {
	FILE *archivo;
	archivo = fopen("data.cuda", "r");
	if (archivo != NULL)
	{
		*tipoEjecucion = fgetc(archivo);
		fgetc(archivo);
		dificultad = (int)fgetc(archivo) - 48;
		//Nos quitamos el ; que está entre medias.
		fgetc(archivo);
		columnas = cargarNumeroFile(archivo);
		filas = cargarNumeroFile(archivo);
		tam = filas * columnas;
		matriz = (int*)malloc(tam * sizeof(int));
		int i = 0;
		//Asignamos los valores en toda la matriz.
		while (feof(archivo) == 0)
		{
			matriz[i] = (int)fgetc(archivo) - 48;
			i++;
		}
		fclose(archivo);
		rango = calcularRango(dificultad);
		cout << endl << "--------------------------------------------------------------" << endl <<
			"Datos cargados con exito: Ejec.: " << tipoEjecucion << ", Dif.: " << dificultad << ", Filas: " << filas << ", Colum.: " << columnas << "." << endl <<
			"--------------------------------------------------------------" << endl << endl;
		return 1;
	}
	return 0;
}

//Envia a un fichero los datos de la partida en el momento actual.
void guardar(int * mat, int dif, int nc, int nf) {
	FILE *fp;
	//Sobreescribimos el archivo si es que ya existiera, sino, se crea.
	fp = fopen("data.cuda", "w");
	//Se almacena el contenido de los datos del juego.
	fprintf(fp, "%c;%d;%d;%d;", *tipoEjecucion, dif, nc, nf);
	//Y cada celda del tablero.
	for (int i = 0; i < nc*nf; i++) {
		fprintf(fp, "%d", mat[i]);
	}
	fclose(fp);
	cout << endl << "--------------------------------------------------------------" << endl <<
		"Datos guardados con exito." << endl <<
		"--------------------------------------------------------------" << endl << endl;
}

//Obtiene la columna y la fila que se encuentra en la cadena "cad", empezando desde "inicio"
//Sirve para obtener las coordenadas de los movimientos del usuario: cad: "-2,4 u" columna 2, fila 4
//Devuelve array de dos posiciones: columna y fila (desde 1 hasta #Columnas,#Filas)
int * getCoordenadas(char * cad, int inicio) {
	//Aquí guardaremos las coordenadas en memoria dinámica.
	int * coordenadas;
	coordenadas = (int*)malloc(2 * sizeof(int));
	//0 son las columnas, 1 las filas.
	coordenadas[0] = -1;
	coordenadas[1] = -1;
	int length = strlen(cad);
	int i = inicio;
	bool encontrado;
	char * car;
	car = (char*)malloc(sizeof(char));
	//Ejecutamos el ciclo dos veces, para las columnas y las filas.
	for (int j = 0; j < 2; j++) {
		encontrado = false;
		while (!encontrado) {
			*car = cad[i];
			//Si el caracter es un número...
			if (*car >= '0' && *car <= '9') {
				//Si no tenemos aún ningún valor, ponemos las unidades directamente.
				if (coordenadas[j] == -1) {
					coordenadas[j] = atoi(car);
				}
				else {
					//Sino, desplazamos a la izquierda y ponemos las nuevas unidades.
					coordenadas[j] *= 10;
					coordenadas[j] += atoi(car);
				}
			}
			else {
				//Sino, ya hemos llegado al cambio de coordenada.
				encontrado = true;
			}
			i++;
		}
	}
	free(car);
	return coordenadas;
}

//Obtiene un número que se encuentra a partir del carácter número "inicio" en la cadena "cad"
int getNum(char * cad, int inicio) {
	int num = -1;
	int length = strlen(cad);
	int i = inicio;
	char * car;
	car = (char*)malloc(sizeof(char));
	while (i < length) {
		*car = cad[i];
		if (*car >= '0' && *car <= '9') {
			if (num == -1) {
				num = atoi(car);
			}
			else {
				num *= 10;
				num += atoi(car);
			}
		}
		else {
			break;
		}
		i++;
	}
	return num;
}

//Muestra todas las entradas que puede hacer el usuario.
void mostrarAyuda() {
	cout << endl << "--------------------------------------------------------------" << endl <<
		"-C,F M: donde C es la columna, F es la fila y M esta en {u,r,d,l} para permutar la posicion F,C con el de arriba, derecha, abajo e izquierda respectivamente." << endl <<
		"9 1 X: para eliminar la columna numero X" << endl;
	if (dificultad >= 2) {
		cout << "9 2 X: para eliminar la fila numero X (A partir del nivel 2)" << endl;
	}
	if (dificultad >= 3) {
		cout << "9 3: para girar diamantes (A partir del nivel 3)" << endl;
	}
	cout << "cargar: para reanudar una partida guardada" << endl <<
		"guardar: para salvar la partida actual" << endl <<
		"salir: para cerrar la aplicacion" << endl <<
		"ayuda: para mostrar todos los comandos posibles" << endl <<
		"--------------------------------------------------------------" << endl << endl;
}

//Elimina los diamantes que sean coincidentes a una única celda. SÓLO se elimina 1 combinación,
//al contrario que en comprobarYEliminar.
void eliminarCoincidentes(int * mat, int nf, int nc, int celda, int rango) {
	int x = celda % nc; //Columnas
	int y = celda / nc; //Filas
	int i;
	int color = mat[celda];
	//Eliminamos el diamante de la celda.
	eliminarDiamante(mat, nc, celda, rango);
	//Miramos hacia arriba. Solo tenemos que mirar la posición de la celda actual, porque los demás han bajado.
	//Seguimos eliminando hasta que ya no esté de ese color.
	while (color == mat[y*nc + x]) {
		eliminarDiamante(mat, nc, y*nc + x, rango);
	}
	//Miramos hacia derecha hasta que deje de ser del color.
	for (i = x + 1; i < nc; i++) {
		if (color == mat[y*nc + i]) {
			eliminarDiamante(mat, nc, y*nc + i, rango);
		}
		else {
			break;
		}
	}
	//Miramos hacia abajo.
	for (i = y + 1; i > 0; i++) {
		if (color == mat[i*nc + x]) {
			eliminarDiamante(mat, nc, i*nc + x, rango);
		}
		else {
			break;
		}
	}
	//Miramos hacia izquierda.
	for (i = x - 1; i > 0; i--) {
		if (color == mat[y*nc + i]) {
			eliminarDiamante(mat, nc, y*nc + i, rango);
		}
		else {
			break;
		}
	}
}

//Elimina los diamantes en la GPU que involucran el movimientos. Realiza las comprobaciones por si hay que eliminarlas
//en las filas o columnas ÚNICAMENTE que son afectadas por el usuario. Si es un movimiento horizontal, involucra a
//dos columnas y una fila y si es vertial una columna y dos filas. Mediante foc indicamos si hay que comprobar
//otra fila o columna (0,1) y con aux indicamos qué fila o columna tenemos que comprobar.
__global__ void eliminarDiamante_MG(int * mat, int nc, int nf, int colsel, int filsel, int foc, int aux, int rango) {
	//Columna de la matriz.
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	//Fila de la matriz.
	int fil = blockIdx.y * blockDim.y + threadIdx.y;
	//Valor en el vector.
	int dim = fil * nc + col;
	//Si el hilo está en la columna o fila indicada (incluso en la auxiliar) se pasa a comprobar.
	if (col == colsel || fil == filsel || (foc == 0 && fil == aux) || (foc == 1 && col == aux)) {
		//Realiza las comprobaciones igual que lo hacía en la CPU.
		int color = mat[dim];
		int i;
		int seguidos[4]; //0 arriba, 1 derecha, 2 abajo, 3 izquierda.
		seguidos[0] = 0;
		seguidos[1] = 0;
		seguidos[2] = 0;
		seguidos[3] = 0;
		boolean continuar = true;
		i = 1;
		while (continuar) {
			//Miramos hacia arriba --> 0
			//Si seguimos dentro del tablero y el color de arriba es igual al de la celda se le añade uno, sino, se deja de comprobar.
			if (fil - i >= 0 && color == mat[dim - nc * i]) {
				i++;
				seguidos[0]++;
			}
			else {
				continuar = false;
			}
		}
		i = 1;
		continuar = true;
		while (continuar) {
			//Miramos hacia la derecha --> 1
			if (col + i < nc && color == mat[dim + i]) {
				i++;
				seguidos[1]++;
			}
			else {
				continuar = false;
			}
		}
		i = 1;
		continuar = true;
		while (continuar) {
			//Miramos hacia la abajo --> 2
			if (fil + i < nf && color == mat[dim + nc * i]) {
				i++;
				seguidos[2]++;
			}
			else {
				continuar = false;
			}
		}
		continuar = true;
		i = 1;
		while (continuar) {
			//Miramos hacia la izquierda --> 3
			if (col - i >= 0 && color == mat[dim - i]) {
				i++;
				seguidos[3]++;
			}
			else {
				continuar = false;
			}
		}
		//Indicador de si hay movimientos posibles.
		int ret = 0;
		for (int k = 0; k < 2; k++) {
			//Se mira la cruz, arriba y abajo y derecha e izquierda.
			if (seguidos[k] + seguidos[k + 2] >= 2) {
				ret = 1;
			}
		}
		//Tras hacer las comprobaciones de todos, se eliminan, Y NO ANTES.
		__syncthreads();
		if (ret == 1) {
			//Indicamos que lo hemos eliminado como 0, la CPU se encargará de actualizar las posiciones.
			mat[dim] = 0;
		}
	}
}

//Elimina una columna "colum" (bomba 1) mediante GPU y las demás se desplazan a la derecha.
__global__ void eliminaColumna_MG(int * mat, int rango, int nf, int nc, int colum) {
	//Obtenemos la columna y la fila respectivamente.
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int fil = blockIdx.y * blockDim.y + threadIdx.y;
	//Dimensión dentro del vector
	int dim = fil * nc + col;
	//Valor nuevo a poner en la celda actual.
	int nuevo;
	//Comprobamos solo las columnas que serán movidas.
	if (col <= colum) {
		//Si no es la columna de la izquierda del todo, se coge el valor de su derecha, las columnas se mueven a la izuqierda.
		if (col != 0) {
			nuevo = mat[dim - 1];
		}
		else {
			//Si estamos en la columna de la izquierda, lo ponemos a 0 (eliminado para la GPU)
			nuevo = 0;
		}
		//Después de que todos hayan hecho las comprobaciones, se ponen los nuevos valores en la matriz.
		__syncthreads();
		mat[dim] = nuevo;
	}
}

//Elimina una fila "fila" (bomba 2) mediante GPU
__global__ void eliminaFila_MG(int * mat, int rango, int nf, int nc, int fila)
{
	//Obtenemos la columna y la fila respectivamente.
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int fil = blockIdx.y * blockDim.y + threadIdx.y;
	//Dimensión dentro del vector
	int dim = fil * nc + col;
	int nuevo;
	//Si estamos en las filas que se moverán/eliminarán.
	if (fil <= fila) {
		if (fil != 0) {
			//Movemos las filas que no sean la primera.
			nuevo = mat[dim - nc];
		}
		else {
			//"Eliminamos" la primera fila.
			nuevo = 0;
		}
		//Una vez obtenidos los valores nuevos por parte de todos, se actualizan.
		__syncthreads();
		mat[dim] = nuevo;
	}
}

//Genera números aleatorios cuando la matriz tiene ceros creados por parte de la GPU.
void actualizarCeros(int * mat, int nc, int tam, int rango) {
	//Tomará el valor de la celda actual, ya que si tiene un 0 pasará la columna a bajar, por lo que tendrá nuevo valor.
	int aux;
	//Tiene el valor de cuantas filas se desplaza arriba, ya que si hay una columna con muchos 0, el nuevo valor
	//de la celda no puede ser 0, debe ser el de más arriba o, si se llega al final, un nuevo valor.
	int j;
	//Comprobamos desde el final, para evitar saltarnos comprobaciones en las celdas.
	for (int i = tam; i >= 0; i--) {
		aux = i;
		j = 1;
		//Seguimos realizando operaciones hasta que la celda actual sea diferente de 0.
		while (mat[aux] == 0) {
			//Si estamos en la última fila, entonces generamos un número al azar.
			if (aux < nc) {
				mat[aux] = (rand() % rango) + 1;
			}
			else {
				//Si no nos pasamos del tablero (la celda es mayor o igual que 0), cogemos el valor de arriba.
				if (aux - nc * j >= 0) {
					//Se coge el valor de arriba.
					mat[aux] = mat[aux - nc * j];
					mat[aux - nc * j] = 0;
					//Pasamos a mirar una fila más arriba por si vuelve a ser 0.
					j++;
				}
				//Sino, si estamos en la última columna, generamos un número aleatorio.
				else {
					mat[aux] = (rand() % rango) + 1;
					break;
				}
			}
		}
	}
}

//Giramos cuadrados en la GPU (bomba 3)
__global__ void giraCuadrados_MG(int * mat, int nf, int nc) {
	//Giros posibles horizontales
	int gph = nf / 3;
	//Giros posibles verticales
	int gpv = nc / 3;
	//Obtenemos la columna y la fila respectivamente.
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int fil = blockIdx.y * blockDim.y + threadIdx.y;
	//Dimensión dentro del vector
	int dim = fil * nc + col;

	//Si nos encontramos en una columna o fila que se puede rotar...
	if (col < gpv * 3 && fil < gph * 3) {
		int nuevo;
		//CG y FG son los índices de la columna girada (CG) y la fila girada (FG)
		//(0,0)(1,0)(2,0)
		//(0,1)(1,1)(2,1)
		//(0,2)(1,2)(2,2)
		//Se hace así para que todos los subcuadrados 3x3 tengan el mismo código.
		int cg, fg;
		//Se obtiene la posición del subcuadrado.
		cg = col % 3;
		fg = fil % 3;
		//Se permutan las caras como si fuera la cara de un cubo de rubik: esquinas con esquinas y aristas con aristas en sentido horario.
		switch (cg) {
		case 0:
			switch (fg) {
			case 0:
				nuevo = mat[dim + nc * 2];
				break;
			case 1:
				nuevo = mat[dim + nc + 1];
				break;
			case 2:
				nuevo = mat[dim + 2];
				break;
			}
			break;
		case 1:
			switch (fg) {
			case 0:
				nuevo = mat[dim + nc - 1];
				break;
			case 1:
				nuevo = mat[dim];
				break;
			case 2:
				nuevo = mat[dim - nc + 1];
				break;
			}
			break;
		case 2:
			switch (fg) {
			case 0:
				nuevo = mat[dim - 2];
				break;
			case 1:
				nuevo = mat[dim - nc - 1];
				break;
			case 2:
				nuevo = mat[dim - nc * 2];
				break;
			}
			break;
		}
		//Cuando todas las celdas tengan sus nuevos valores, se actualizan, no antes.
		__syncthreads();
		mat[dim] = nuevo;
	}
}

//Devuelve 1 se el movimiento se ha llevado a cabo un movimiento, 0 sino.
int ia(int * mat, int nc, int nf, int rango) {
	//Contadores.
	int i, j;
	//columna, fila y número de diamantes que tiene alrededor de igual color.
	int col, fil, iguales;
	char mov[4] = { 'U', 'R', 'D', 'L' };
	//Matriz con los mejores posibles movimientos con los 4 combinaciones.
	int * mejor;
	//Este será el mejor movimiento entre arriba, derecha, abajo, izquierda.
	char movMejor = 'n';
	mejor = (int*)malloc(nf*nc * sizeof(int));
	//Inicializamos a 0 cada mejor combinación.
	for (i = 0; i < nc*nf; i++) {
		mejor[i] = 0;
	}
	//Índice de la mejor posibilidad.
	int ind = -1;
	//Para cada celda...
	for (i = 0; i < nc * nf; i++) {
		//Obtenemos la columna y fila
		col = i % nc;
		fil = i / nc;
		//Para cada uno de los cuatro movimientos...
		for (j = 0; j < 4; j++) {
			//Permutamos sin que se deshaga el movimiento (1) y sin que se borre solo (0)
			permutar(mat, nf, nc, fil, col, mov[j], rango, 1, 0);
			//Nos quedamos con la combinación
			iguales = comportamientoCelda(mat, nf, nc, i);
			//Deshacemos el cambio (backtracking)
			permutar(mat, nf, nc, fil, col, mov[j], rango, 1, 0);
			//Si es mejor combinación que otro movimiento sobre la misma celda lo actualizamos.
			if (iguales > mejor[i]) {
				mejor[i] = iguales;
				if (ind == -1 || iguales > mejor[ind]) {
					ind = i;
					movMejor = mov[j];
				}
			}
		}
	}
	//Si el índice es -1, no hay posibles combinaciones con ningún movimiento.
	if (ind != -1) {
		//Sino, podemos hacer el cambio y eliminar únicamente esa combinación.
		permutar(mat, nf, nc, ind / nc, ind % nc, movMejor, rango, 1, 0);
		eliminarCoincidentes(mat, nf, nc, ind, rango);
		//Avisamos al usuario de la jugada.
		printf("Movimiento: X: %d, Y: %d --> %c\n", (ind % nc) + 1, (ind / nc) + 1, movMejor);
		return 1;
	}
	else {
		return 0;
	}
}

//El main: nada main que añadir
int main(int argc, char* argv[]) {
	//Si se ha iniciado mal el programa, avisa al usuario.
	if (argc != 5) {
		usage();
		return 0;
	}
	//Semilla para obtener números aleatorios
	srand(time(0));
	//Recogemos los argumentos pasados en su ejecución.
	tipoEjecucion = argv[1];
	if (strcmp(tipoEjecucion, "-m") != 0 && strcmp(tipoEjecucion, "-a") != 0) {
		usage();
		return 0;
	}
	//Convertimos con atoi() el char a int.
	dificultad = atoi(argv[2]);
	if (dificultad < 1 || dificultad > 3) {
		usage();
		return 0;
	}
	columnas = atoi(argv[3]);
	filas = atoi(argv[4]);
	tam = columnas * filas;
	//La dimensión del grid será 1 solo bloque.
	dim3 dimGrid(1, 1);
	//Configuramos la distribución de hilos en función de las propiedades de la tarjeta.
	dim3 dimBlock(columnas, filas);
	//En función de la dificultad,el rango de valores varía.
	rango = calcularRango(dificultad);
	//Obtenemos memoria suficiente para la matriz.
	matriz = (int*)malloc(tam * sizeof(int));
	//Rellenamos la matriz de valores.
	rellenaMatriz(matriz, tam, rango);
	//Obtenemos espacio para la matriz del dispositivo.
	cudaMalloc((void**)&matriz_d, columnas * filas * sizeof(int));
	//Si la ejecución es automática.
	if (strcmp(tipoEjecucion, "-a") == 0) {
		//Cogemos la bomba número 1 al principio.
		int bomba = 1;
		//Número de intentos, para no tener todo el rato las bombas y que pueda acabarse el juego.
		int intento = 1;
		do {
			imprimeMatriz(matriz, filas, columnas);
			//Si la IA da como resultado 0 es que no hay movimientos posibles, momento de usar una bomba.
			if (ia(matriz, columnas, filas, rango) == 0) {
				switch (bomba) {
					//Si se han superado los intentos (ponemos como límite el número de columnas, para que haya fin).
					//Cogemos otra bomba.
				case 1: if (intento > columnas) {
					printf("Cambio a la bomba 2\n");
					bomba++;
					intento = 1;
				}
						else {
							printf("Uso la bomba 1 en el intento %d\n", intento);
							//Usamos la primera bomba.
							cudaMemcpy(matriz_d, matriz, tam * sizeof(int), cudaMemcpyHostToDevice);
							eliminaColumna_MG << <dimGrid, dimBlock >> > (matriz_d, rango, columnas, filas, intento - 1);
							cudaMemcpy(matriz, matriz_d, columnas * filas * sizeof(int), cudaMemcpyDeviceToHost);
							actualizarCeros(matriz, columnas, filas * columnas, rango);
							intento++;
						}
						break;
				case 2:
					//De igual manera comprobamos las diferentes bombas en función del nivel.
					if (dificultad < 2) {
						printf("No puedo usar la bomba 2 porque estoy en el nivel 1.\n");
						bomba = 4;
						break;
					}
					if (intento > filas) {
						printf("Cambio a la bomba 3\n");
						bomba++;
						intento = 1;
					}
					else {
						printf("Uso la bomba 2 en el intento %d\n", intento);
						cudaMemcpy(matriz_d, matriz, tam * sizeof(int), cudaMemcpyHostToDevice);
						eliminaFila_MG << <dimGrid, dimBlock >> > (matriz_d, rango, columnas, filas, intento - 1);
						cudaMemcpy(matriz, matriz_d, columnas * filas * sizeof(int), cudaMemcpyDeviceToHost);
						actualizarCeros(matriz, columnas, filas * columnas, rango);
						intento++;
					}
					break;
				case 3: if (dificultad < 3) {
					printf("No puedo usar la bomba 3 porque estoy en el nivel %d.\n", dificultad);
					bomba = 4;
					break;
				}
						if (intento > 8) {
							bomba++;
						}
						else {
							printf("Uso la bomba 3 en el intento %d\n", intento);
							cudaMemcpy(matriz_d, matriz, tam * sizeof(int), cudaMemcpyHostToDevice);
							giraCuadrados_MG << <dimGrid, dimBlock >> > (matriz_d, filas, columnas);
							cudaMemcpy(matriz, matriz_d, columnas * filas * sizeof(int), cudaMemcpyDeviceToHost);
							actualizarCeros(matriz, columnas, filas * columnas, rango);
							intento++;
						}
						break;
				}
			}
		} while (bomba <= 3);
		printf("Fin de la partida\n");
	}
	else {
		//Recordamos al usuario que tiene un juego guardado anteriormente.
		if (fopen("data", "r") != NULL) {
			cout << "---------------------------------------------------------" << endl;
			cout << "** RECUERDE que tiene un juego guardado anteriormente. **" << endl;
			cout << "---------------------------------------------------------" << endl;
		}
		//Movimiento y variable para obtener el numero seleccionado por el usuario.
		int mov = 0, num;
		//Mantiene el bucle principal.
		int bucle = 1;
		bool ordenErronea = false;
		//Acción introducida por el usuario.
		char* acc;
		//Coordenadas X e Y (columna y fila)
		int * coord;
		acc = (char*)malloc(40 * sizeof(char*));
		while (bucle >= 0) {
			cout << endl << "-----------------------------------------------------" << endl << "Movimiento: " << mov << endl;
			cout << endl;
			//Imprimimos la matriz.
			imprimeMatriz(matriz, filas, columnas);
			//Repetimos mientras la orden sea erronea.
			do {
				cout << endl << ">";
				cin.getline(acc, 40);
				cout << endl;
				//Comprobamos si la accion es una orden correcta.
				if (strcmp(acc, "salir") == 0 || strncmp(acc, "9 1 ", 4) == 0 || strncmp(acc, "9 2 ", 4) == 0 ||
					strcmp(acc, "9 3") == 0 || strncmp(acc, "-", 1) == 0 ||
					strcmp(acc, "cargar") == 0 || strcmp(acc, "guardar") == 0 || strcmp(acc, "ayuda") == 0) {
					//Si estabe en orden erronea se cambia.
					if (ordenErronea)
						ordenErronea = false;
					//Acaba con el bucle principal.
					if (strcmp(acc, "salir") == 0) {
						bucle = -1;
					}
					//Permuta las celdas seleccionadas.
					if (strncmp(acc, "-", 1) == 0) {
						//X guarda la columna
						//Y guarda la fila.
						int x, y;
						//Se obtiene espacio y luego se parsea la entrada buscando las coordenadas.
						coord = (int*)malloc(2 * sizeof(int));
						coord = getCoordenadas(acc, 1);
						x = coord[0]; //Columnas
						y = coord[1]; //Filas
									  //Miraremos además en otra fila o columna, que es la que sufrirá el cambio. Ejemplo:
									  //Si estamos en la fila 3 y movemos arriba, además de mirar esa columna, tenemos que
									  //mirar la fila 2 porque sufre el cambio.
						int filaocolumna; //0 fila, 1 columna.
										  //Guarda el valor de la fila o columna auxiliar, la que sufre el cambio.
						int aux;
						//Si las coordenadas están en el tablero sin salirse de él
						if (x > 0 && x <= columnas) {
							if (y > 0 && y <= filas) {
								//Se obtiene el movimiento, presumiblemente el último carácter
								char mov = acc[strlen(acc) - 1];
								//Si está entre los movimientos válidos.
								if (mov == 'u' || mov == 'U' || mov == 'r' || mov == 'R' ||
									mov == 'd' || mov == 'D' || mov == 'l' || mov == 'L') {
									//Indica si el movimiento ha sido posible.
									int posible;
									//Permuta las celdas deshaciendo el cambio si no tiene combinaciones posibles y no borra
									//los diamantes.
									posible = permutar(matriz, filas, columnas, y - 1, x - 1, mov, rango, 0, 0);
									//Obtenemos la fila y columna  en función del movimiento (contando de 0 a nc-1 y 0 a nf-1).
									//Además, se añade el valor de la fila o columna auxiliar.
									int colsel, filsel;
									if (mov == 'u' || mov == 'U') {
										colsel = x - 1;
										filsel = y - 2;
										filaocolumna = 0;
										aux = y - 1;
									}
									if (mov == 'r' || mov == 'R') {
										colsel = x;
										filsel = y - 1;
										filaocolumna = 1;
										aux = x - 1;
									}
									if (mov == 'd' || mov == 'D') {
										colsel = x - 1;
										filsel = y;
										filaocolumna = 0;
										aux = y - 1;
									}
									if (mov == 'l' || mov == 'L') {
										colsel = x - 2;
										filsel = y - 1;
										filaocolumna = 1;
										aux = x - 1;
									}
									cudaMemcpy(matriz_d, matriz, tam * sizeof(int), cudaMemcpyHostToDevice);
									eliminarDiamante_MG << <dimGrid, dimBlock >> > (matriz_d, columnas, filas, colsel, filsel, filaocolumna, aux, rango);
									cudaMemcpy(matriz, matriz_d, columnas * filas * sizeof(int), cudaMemcpyDeviceToHost);
									actualizarCeros(matriz, columnas, filas * columnas, rango);
								}
								else {
									cout << "Movimiento no valido. El caracter final debe ser:" << endl <<
										"u: cambio con el de arriba." << endl <<
										"r: cambio con el de la derecha." << endl <<
										"d: cambio con el de abajo." << endl <<
										"l: cambio con el de la izquierda." << endl;
									ordenErronea = true;
								}
							}
							else {
								cout << "La fila " << y << " no existe. Elija entre 1 y " << filas << ".";
								ordenErronea = true;
							}
						}
						else {
							cout << "La columna " << x << " no existe. Elija entre 1 y " << columnas << ".";
							ordenErronea = true;
						}
					}
					//Si el usuario selecciona la primera bomba
					if (strncmp(acc, "9 1 ", 4) == 0) {
						//Obtenemos el número de columna que eliminará.
						num = getNum(acc, 4);
						//Comrpobamos que esté en el tablero.
						if (num > columnas || num <= 0) {
							cout << "La columna " << num << " no existe. Hay un maximo de " << columnas << "." << endl;
							ordenErronea = true;
						}
						else {
							cudaMemcpy(matriz_d, matriz, tam * sizeof(int), cudaMemcpyHostToDevice);
							eliminaColumna_MG << <dimGrid, dimBlock >> > (matriz_d, rango, filas, columnas, num - 1);
							cudaMemcpy(matriz, matriz_d, columnas * filas * sizeof(int), cudaMemcpyDeviceToHost);
							actualizarCeros(matriz, columnas, filas * columnas, rango);
						}

					}
					//Si el usuario selecciona la bomba 2.
					if (strncmp(acc, "9 2 ", 4) == 0) {
						//Comprobamos que la dificultad sea adecuada para utilizar la bomba.
						if (dificultad >= 2) {
							//Cogemos el número de fila a eliminar.
							num = getNum(acc, 4);
							//Comprobamos que esté entre las filas de la matriz.
							if (num > filas || num == 0) {
								cout << "La fila " << num << " no existe. Elija de 1 a " << filas << "." << endl;
								ordenErronea = true;
							}
							else {
								cudaMemcpy(matriz_d, matriz, tam * sizeof(int), cudaMemcpyHostToDevice);
								eliminaFila_MG << <dimGrid, dimBlock >> > (matriz_d, rango, filas, columnas, num - 1);
								cudaMemcpy(matriz, matriz_d, columnas * filas * sizeof(int), cudaMemcpyDeviceToHost);
								actualizarCeros(matriz, columnas, filas * columnas, rango);
							}
						}
						else {
							cout << "Esta bomba no pertenece a este nivel." << endl;
							ordenErronea = true;
						}
					}
					//Usamos la bomba número 3
					if (strcmp(acc, "9 3") == 0) {
						//Si la dificultad es la tercera.
						if (dificultad == 3) {
							cudaMemcpy(matriz_d, matriz, tam * sizeof(int), cudaMemcpyHostToDevice);
							giraCuadrados_MG << <dimGrid, dimBlock >> > (matriz_d, filas, columnas);
							cudaMemcpy(matriz, matriz_d, columnas * filas * sizeof(int), cudaMemcpyDeviceToHost);
							actualizarCeros(matriz, columnas, filas * columnas, rango);
						}
						else {
							cout << "Esta bomba no pertenece a este nivel." << endl;
							ordenErronea = true;
						}

					}
					//El usuario carga el juego anteriormente guardado
					if (strcmp(acc, "cargar") == 0) {
						//Si se ha podido cargar con éxito
						if (cargar() == 1) {
							//Liberamos la matriz del dispositivo.
							cudaFree(matriz_d);
							//Le asignamos nuevo espacio.
							cudaMalloc((void**)&matriz_d, columnas * filas * sizeof(int));
							mov = 0;
							bucle = 1;
							system("PAUSE");
							system("cls");
						}
						else {
							//Si avisa al usuario de que no hay juego guardado.
							cout << "Actualmente no existen datos de un juego guardado anteriormente." << endl;
						}
					}
					//El usuario decide guardar el juego.
					if (strcmp(acc, "guardar") == 0) {
						//Guarda la matriz.
						guardar(matriz, dificultad, columnas, filas);
						cout << "¿Desea salir del juego (si/no)?" << endl;
						//Pregunta si se desea salir.
						cin.getline(acc, 40);
						if (strncmp(acc, "si", 2) == 0) {
							bucle = -1;
						}
					}
					//Muestra las diferentes movimientos.
					if (strcmp(acc, "ayuda") == 0) {
						mostrarAyuda();
						system("PAUSE");
					}
				}
				else {
					//Si la orden ha sido incorrecta.
					cout << "Esa orden no existe. Prueba con alguna de estas: " << endl;
					mostrarAyuda();
					system("PAUSE");
					ordenErronea = true;
				}
			} while (ordenErronea);
			mov++;
		}
	}
	//Liberamos la memoria de la matriz
	cudaFree(matriz_d);
	free(matriz);
	//getchar(); //NO ME FUNCIONA EL GETCHAR, LO HE CAMBIADO POR SYSTEM PAUSE.

	system("PAUSE");
	return 0;
}