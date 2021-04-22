#ifndef MAKE_GRID
#define MAKE_GRID

#include <iostream>
#include <fstream>
#include <string>
#include <sstream> 
#include <vector>
using namespace std;

/**************************************************************
						AGENT CLASS
**************************************************************/

class Agent{
public:
	// variables
	int sx;
	int sy;
	int dx;
	int dy;
	string name;

	// constructors
	Agent();
	Agent(const Agent& obj); // copy constructor
	Agent(int start_x, int start_y, int end_x, int end_y);
	Agent(int start_x, int start_y, int end_x, int end_y, string in_name);
};

Agent::Agent(){
	sx = 0;
	sy = 0;
	dx = 0;
	dy = 0;
	name = "default";
}

Agent::Agent(const Agent& obj){
	sx = obj.sx;
	sy = obj.sy;
	dx = obj.dx;
	dy = obj.dy;
	name = obj.name;
}

Agent::Agent(int start_x, int start_y, int end_x, int end_y){
	sx = start_x;
	sy = start_y;
	dx = end_x;
	dy = end_y;
	name = "default";
}

Agent::Agent(int start_x, int start_y, int end_x, int end_y, string in_name){
	sx = start_x;
	sy = start_y;
	dx = end_x;
	dy = end_y;
	name = in_name;
}


/**************************************************************
						GRID CLASS
**************************************************************/

class Grid{
public:
	// variables
	int num_agents;
	vector<Agent> agents;
	int size;
	int* matrix;

	// constructors
	Grid(); // default
	Grid(string filename);

	// member functions
	int PrintGrid();
};

Grid::Grid(){
	num_agents = 0;
	size = 0;
	matrix = (int*) calloc(size * size, sizeof(int));
}

Grid::Grid(string filename){
	// Reads from YAML file to initialize grid
	int i, j, mult;
	int x, y, aso_x, aso_y, ades_x, ades_y;
	string line, token, aname;
	fstream infile;

	num_agents = 0;

	infile.open(filename);
	if (!infile){
		cout << "Cannot open file" << endl;
		exit(1);
	}

	// Line Parser/load data
	try{
		getline(infile, line); // agents

		// handle agent info
		while (getline(infile, line)){
			if (line.compare(0, 4, "map:") == 0)
				break;

			// increment agent count
			num_agents++;
			
			// read in goal
			stringstream s1(line);
			getline(s1, token, '['); // remove padding
			getline(s1, token, ','); // first digit
			ades_x = stoi(token);
			getline(s1, token, ']'); // second digit
			ades_y = stoi(token);

			// read in name
			getline(infile, line);
			stringstream s2(line);
			getline(s2, token, ':'); // remove padding
			getline(s2, token, ' '); // remove padding
			getline(s2, token); // name
			aname = token;

			// read in start
			getline(infile, line);
			stringstream s3(line);
			getline(s3, token, '['); // remove padding
			getline(s3, token, ','); // first digit
			aso_x = stoi(token);
			getline(s3, token, ']'); // second digit
			aso_y = stoi(token);

			// initialize agent
			Agent bot(aso_x, aso_y, ades_x, ades_y, aname);
			agents.push_back(bot);
		}

		// get dimensions
		getline(infile, line);
		stringstream ss(line);
		getline(ss, token, '['); // remove padding
		getline(ss, token, ','); // first digit
		size = stoi(token);
		getline(ss, token, ']'); // second digit

		// initialize matrix w/ 0s
		matrix = (int*) calloc(size * size, sizeof(int));
		for (i = 0; i < size; i++){
			mult = i*size;
			for (j = 0; j < size; j++){
				matrix[mult + j] = 0;
			}
		}

		getline(infile, line); // remove padding

		// Load obstacle info into matrix
		while (getline(infile, line)){
			stringstream ss(line);
			getline(ss, token, '['); // remove padding
			getline(ss, token, ','); // first digit
			x = stoi(token);
			getline(ss, token, ']'); // second digit
			y = stoi(token);

			matrix[x + y*size] = -1; // obstacle to -1
		}
	}
	catch (...){
		cout << "Cannot parse file..." << endl << "SHUTTING DOWN" << endl;
		infile.close();
		exit(1);
	}

	infile.close();
}

int Grid::PrintGrid(){
	int i, j, mult;

	for (i = 0; i < size; i++){
		mult = i*size;
		for (j = 0; j < size; j++){
			cout << matrix[mult + j] << ",\t";
		}
		cout << endl;
	}

	return 0;
}

#endif