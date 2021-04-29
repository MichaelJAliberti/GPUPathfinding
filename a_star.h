// Michael Aliberti w/ Collaborator Dmitry Zimin

#ifndef A_STAR_
#define A_STAR_

#include <stdlib.h>
#include <math.h>
#include "myHeap.h"  
#define LARGE 9999999

using namespace std;


struct arc{
  struct arc *next;
  int length;
  int end;
  };

typedef struct node{
   int id; /* first arc in linked list */
   int dist;  // The number of the vertex in this node 
   int key;  /* Distance estimate, named key to reuse heap code*/
   int Pre;  /* Predecessor node in shortest path */
   int position;  /* Position of node in heap, from 0 to Nm, where 0 is best */
   } nodeitem;
   
int fdiff(int first, int second, int side_size)
{
	return (abs(first%side_size - second%side_size) + abs(first/side_size - second/side_size));
}

int a_star(string grid_name)
{
	
	//Grid Declaration & Initialization
	Grid map(grid_name);
	
	//Size variables Declaration & Initialization
	int side_size = map.size;
	int size = side_size * side_size;
	
	//A * Star variables Declaration
	int src;
	int dst;
	bool sptSet[size];
	int counter;
	
	//Agent Declaration & Initialization
	Agent iter = map.agents[0];

	// Prepare output file
	stringstream s1(grid_name);
	string token;
	getline(s1, token, '.');
	token += "_astar.yaml";
	ofstream outfile(token);
	outfile << "paths:" << endl;

	// loop through agents with a_star
	for (auto k = map.agents.begin(); k != map.agents.end(); ++k){
		
		// Agent declaration
		iter = *k;

		// Source and destination from the Agent
		src = iter.sx + iter.sy * side_size;
		dst = iter.dx + iter.dy * side_size;
		
		// Heap node Initialization
		nodeitem* Nodes = (nodeitem*) calloc(size, sizeof(nodeitem));
		for (int i=0;i<size;i++){  
			Nodes[i].id = i;
			Nodes[i].dist = LARGE;
			Nodes[i].key = LARGE + fdiff(i, dst, side_size);
			Nodes[i].Pre = -1;
			Nodes[i].position = -1;
			sptSet[i] = false;
		}
		
		// Heap Declaration & Initialization
		Heap<nodeitem> *thisHeap;
		nodeitem *u;
		int v;
		
		// Insert all nodes into Heap
		thisHeap = new Heap<nodeitem>;
		for (int i = 0; i <size; i++) 
		{
			thisHeap->insert(&Nodes[i]);
		}
		
		// Source Initialization and key decrease in Heap
		Nodes[src].key = fdiff(src, dst, side_size);
		Nodes[src].dist = 0;
		thisHeap->decreaseKey(Nodes[src].position, Nodes[src].key);

		// write to output file
		stringstream s2(iter.name);
		getline(s2, token, '\n');
		outfile << "-   name: "<< token << endl;
		outfile << "    nodes:" << endl;
		counter = 0;

		// Loop until the Heap is Empty
		while(!thisHeap->IsEmpty())  
		{
			u = thisHeap->remove_min();
			sptSet[u->id] = true;

			counter++;
	        outfile << "    - [" << (u->id)%side_size << ", " << (u->id)/side_size << "]" << endl;
			
			// If current node is dst then break
			if (u->id == dst)
				break;		

			// The node to the RIGHT
			v = u->id + 1;
			if (v < size && v >= 0)
				if (!(v % (side_size) == 0) && !sptSet[v] && map.matrix[v] == 0  
					&& Nodes[v].dist > (1 + u->dist))
					{
						Nodes[v].dist = 1 + u->dist;
						Nodes[v].key = (1 + u->dist) + fdiff(v,dst, side_size);
						Nodes[v].Pre = u->id;
						thisHeap->decreaseKey(Nodes[v].position, Nodes[v].key);  
					}
			// The node to the LEFT
			v = u->id - 1;
			if (v < size && v >= 0)
				if (!(v % (side_size) == side_size - 1) && !sptSet[v] && map.matrix[v] == 0 
					&& Nodes[v].dist > (1 + u->dist))
					{
						Nodes[v].dist = 1 + u->dist;
						Nodes[v].key = (1 + u->dist) + fdiff(v,dst, side_size);
						Nodes[v].Pre = u->id;
						thisHeap->decreaseKey(Nodes[v].position, Nodes[v].key);  
					}
			// The node to the TOP
			v = u->id - side_size;
			if (v < size && v >= 0)
				if ((v > -1)  && map.matrix[v] == 0 && !sptSet[v] 
					&& Nodes[v].dist > (1 + u->dist))
					{
						Nodes[v].dist = 1 + u->dist;
						Nodes[v].key = (1 + u->dist) + fdiff(v,dst, side_size);
						Nodes[v].Pre = u->id;
						thisHeap->decreaseKey(Nodes[v].position, Nodes[v].key);  
					}
			// The node to the BOTTOM
			v = u->id + side_size;
			if (v < size && v >= 0 )
				if (v < size && !(u->id / (side_size) == side_size - 1) &&!sptSet[v] && map.matrix[v] == 0 
					&& Nodes[v].dist > (1 + u->dist))
					{
						Nodes[v].dist = 1 + u->dist;
						Nodes[v].key = (1 + u->dist) + fdiff(v,dst, side_size);
						Nodes[v].Pre = u->id;

						thisHeap->decreaseKey(Nodes[v].position, Nodes[v].key);  
					}
		}

		outfile << "    number: " << counter << endl;
		
		int preditor = dst;
		int counter = 0; // length of path
		vector<int> path;
		path.push_back(preditor);
		while (preditor != src && counter < 50)
		{
			preditor = Nodes[preditor].Pre;		
			path.push_back(preditor);
			counter++;
		}

		outfile << "    " << "path:" << endl;
		for (auto i = path.begin(); i != path.end(); ++i){
			outfile << "    ";
	        outfile << "- [" << (*i)%side_size << ", " << (*i)/side_size << "]" << endl;
		}
		outfile << "    " << "length: " << counter << endl;
		
		free(thisHeap);
		free(Nodes);
	}

	return 0;
}

#endif
