#include <iostream>
#include <string>
#include "LoadMap.h"
#include "dijkstra.h"
using namespace std;

int main(int argc, char **argv){

	if (argc != 2){
		cout << "No input file" << endl;
	}
	else{
		string filein = argv[1];

		Grid hi(filein);
		hi.PrintGrid();
		cout << endl;

		dijkstra(filein);
	}

	return 0;
}