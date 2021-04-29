#include <iostream>
#include <string>
#include "LoadMap.h"
#include "dijkstra.h"
#include "a_star.h"
using namespace std;

int main(int argc, char **argv){

	if (argc != 2){
		cout << "No input file" << endl;
	}
	else{
		string filein = argv[1];

		Grid map(filein);
		//map.PrintGrid();
		//cout << endl;

		dijkstra(filein);
		a_star(filein);
	}

	return 0;
}