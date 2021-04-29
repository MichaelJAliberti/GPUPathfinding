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
//cout << "dst [" << (dst)%side_size << ", " << (dst)/side_size << "]" << endl;

vector<int> a_star(string grid_name, int num_agent)
{
	
	//Grid Initialization
	Grid map(grid_name);
	
	map.PrintGrid();
	
	//size variables
	int side_size = map.size;
	int size = side_size * side_size;
	
	//dijkstra variables Declaration
	//int dist[size];
	//bool sptSet[size];
	nodeitem Nodes[20000]; // The vertices of the graph
	
	//vector<int> pred(size, -1);
	//vector<int> path;
	//int preditor;
	//int counter;
	int src;
	int dst;
	
	bool sptSet[size];
	
	//Agent Initialization
	Agent iter;
	iter = map.agents[num_agent];
	
	//source and destination from the Agent
	src = iter.sx + iter.sy * side_size;
	dst = iter.dx + iter.dy * side_size;
	

	for (int i=0;i<size;i++){  // Initialize nodes
		Nodes[i].id = i;
		Nodes[i].dist = LARGE;
		Nodes[i].key = LARGE + fdiff(i, dst, side_size);
		Nodes[i].Pre = -1;
		Nodes[i].position = -1;
		sptSet[i] = false;
	}
	
	
	
	//Dijkstra's source Initialization
	//pred[src] = src;
	//dist[src] = 0;
	//int min = INF;
	//int u;



	Heap<nodeitem> *thisHeap;
	thisHeap = new Heap<nodeitem>;
	//struct arc *edge;
	nodeitem *u;
	int v;

	

	for (int i = 0; i <size; i++) //insert all nodes into Heap
	{
	 thisHeap->insert(&Nodes[i]);
	}
	
	Nodes[src].key = fdiff(src, dst, side_size);
	cout << "Nodes[src].key: " << Nodes[src].key << endl;
	cout << "Nodes[src].id: " << (Nodes[src].id)%side_size << ", " << (Nodes[src].id)/side_size << "]" << endl;
	cout << "Nodes[dst].key: " << Nodes[dst].key << endl;
	Nodes[src].dist = 0;
	//sptSet[src] = true;
	thisHeap->decreaseKey(Nodes[src].position, Nodes[src].key);
	
	cout << "dst [" << (dst)%side_size << ", " << (dst)/side_size << "]" << endl;
	cout << "src [" << (src)%side_size << ", " << (src)/side_size << "]" << endl;
	


	while(!thisHeap->IsEmpty())  //until heap is empty
	{
		u = thisHeap->remove_min();
		
		cout << "u - [" << (u->id)%side_size << ", " << (u->id)/side_size << "]" << endl;
		
		sptSet[u->id] = true;
		if (u->id == dst)
			break;
			
		
		
		
		//The node to the RIGHT
			v = u->id + 1;
			if (v < size && v >= 0)
				if (!(v % (side_size) == 0) && !sptSet[v] && map.matrix[v] == 0  
					&& Nodes[v].dist > (1 + u->dist))
					{
						Nodes[v].dist = 1 + u->dist;
						Nodes[v].key = (1 + u->dist) + fdiff(v,dst, side_size);
						Nodes[v].Pre = u->id;
						cout << "- [" << (v)%side_size << ", " << (v)/side_size << "] Right" << endl;
						thisHeap->decreaseKey(Nodes[v].position, Nodes[v].key);  
					}
			//The node to the LEFT
			v = u->id - 1;
			if (v < size && v >= 0)
				if (!(v % (side_size) == side_size - 1) && !sptSet[v] && map.matrix[v] == 0 
					&& Nodes[v].dist > (1 + u->dist))
					{
						Nodes[v].dist = 1 + u->dist;
						Nodes[v].key = (1 + u->dist) + fdiff(v,dst, side_size);
						Nodes[v].Pre = u->id;
						cout << "- [" << (v)%side_size << ", " << (v)/side_size << "] Left" << endl;
						thisHeap->decreaseKey(Nodes[v].position, Nodes[v].key);  
					}
			//The node to the TOP
			v = u->id - side_size;
			if (v < size && v >= 0)
				if ((v > -1)  && map.matrix[v] == 0 && !sptSet[v] 
					&& Nodes[v].dist > (1 + u->dist))
					{
						Nodes[v].dist = 1 + u->dist;
						Nodes[v].key = (1 + u->dist) + fdiff(v,dst, side_size);
						Nodes[v].Pre = u->id;
						cout << "- [" << (v)%side_size << ", " << (v)/side_size << "] Top" << endl;
						thisHeap->decreaseKey(Nodes[v].position, Nodes[v].key);  
					}
			//The node to the BOTTOM
			v = u->id + side_size;
			if (v < size && v >= 0 )
				if (v < size && !(u->id / (side_size) == side_size - 1) &&!sptSet[v] && map.matrix[v] == 0 
					&& Nodes[v].dist > (1 + u->dist))
					{
						Nodes[v].dist = 1 + u->dist;
						Nodes[v].key = (1 + u->dist) + fdiff(v,dst, side_size);
						Nodes[v].Pre = u->id;
						cout << "- [" << (v)%side_size << ", " << (v)/side_size << "] Down" << endl;
						//cout << "id: " << Nodes[v].id << endl;
						//cout << "key: " << Nodes[v].key << endl;
						thisHeap->decreaseKey(Nodes[v].position, Nodes[v].key);  
					}
			cout << "v: " << v << endl;
		
		
		
	 /*edge = u->first;
	 while(edge != NULL)
	 {
	   
	   //v = edge->end;
	   
	   if (Nodes[v].dist > (1 + u->dist))  //if dist is 
	   {
		Nodes[v].dist = 1 + u->key;
		Nodes[v].key = (1 + u->key) + fdiff(v,dst);
		Nodes[v].Pre = u->id;
		thisHeap->decreaseKey(Nodes[v].position, Nodes[v].key);   
	   }
	   edge = edge->next;

	 }*/
	 
	 
	}
	
	cout << "Printing: " << endl;
	
	int preditor = dst;
	int counter = 0; // length of path
	vector<int> path;
	path.push_back(preditor);
	while (preditor != src && counter < 50)
	{
		preditor = Nodes[preditor].Pre;		
		path.push_back(preditor);
		counter++;
		//cout << preditor << " -> " << endl;
		cout << "- [" << (preditor)%side_size << ", " << (preditor)/side_size << "]" << endl;
	}

	
	free(thisHeap);

	return path;
} /* end DijkstraHeap */ 
#endif
