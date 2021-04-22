#ifndef DIJKSTRA_H_
#define DIJKSTRA_H_

#include <queue>

#define INF 9999999

using namespace std;

vector<int> dijkstra(string grid_name, int num_agent)
{
	
	//Grid Initialization
	Grid map(grid_name);
	
	//size variables
	int side_size = map.size;
	int size = side_size * side_size;
	
	//dijkstra variables Declaration
	int dist[size];
	bool sptSet[size];
	
	vector<int> pred(size, -1);
	vector<int> path(size, -1);
	int preditor;
	int counter;
	int src;
	int dst;
	
	//agent Declaration
	Agent iter;
	

	//dijkstra's variables Initialization
	for (int i = 0; i < size; i++)
	{
		dist[i] = INF;
		sptSet[i] = false;
		pred[i] = -1;
	}
	
	//Agent Initialization
	iter = map.agents[num_agent];
	
	//source and destination from the Agent
	src = iter.sx + iter.sy * side_size;
	dst = iter.dx + iter.dy * side_size;
	
	//Dijkstra's source Initialization
	pred[src] = src;
	dist[src] = 0;
	int min = INF;
	int u;
	
	
	for (int count = 0; count < size - 1; count++) {

		//Find the minimum element so far
		min = INF;
		for (int v = 0; v < size; v++)
			if (sptSet[v] == false && dist[v] <= min)
			{
				min = dist[v];
				u = v;
			}
	  
		sptSet[u] = true;
		

  
		//Since the data structure is a grid, there are only 4 adjacent nodes to check
		//The node to the RIGHT
		int v = u + 1;
		if (!sptSet[v] && !(v % (side_size) == 0) && map.matrix[v] == 0 && dist[u] != INF 
			&& dist[u] + map.matrix[u*side_size+v] < dist[v])
			{
			dist[v] = dist[u] + 1;
			pred[v] = u;
			//cout << "right" << endl;
			}
		//The node to the LEFT
		v = u - 1;
		if (!sptSet[v] && !(v % (side_size) == side_size - 1) && map.matrix[v] == 0 && dist[u] != INF 
			&& dist[u] + map.matrix[u*side_size+v] < dist[v])
			{
			dist[v] = dist[u] + 1;
			pred[v] = u;
			//cout << "left" << endl;
			}
		//The node to the TOP
		v = u - side_size;
		if (!sptSet[v] && (v > -1) && map.matrix[v] == 0 && dist[u] != INF 
			&& dist[u] + map.matrix[u*side_size+v] < dist[v])
			{
			dist[v] = dist[u] + 1;
			pred[v] = u;
			//cout << "up" << endl;
			}
		//The node to the BOTTOM
		v = u + side_size;
		if (!sptSet[v] && !(u / (side_size) == side_size - 1) && map.matrix[v] == 0 && dist[u] != INF 
			&& dist[u] + map.matrix[u*side_size+v] < dist[v])
			{
			dist[v] = dist[u] + 1;
			pred[v] = u;
			//cout << "down" << endl;
			}
	}
	
	
	//Find the shortest path from the pred[]
	preditor = dst;
	counter = 0;
	while (preditor != src)
	{
		//cout << preditor << " -> ";
		preditor = pred[preditor];		
		path[counter] = preditor;
		counter++;
		
	}
	/*cout << preditor << endl;
	for (auto i = path.begin(); i != path.end(); ++i)
		std::cout << *i << ' ';
	*/

	

	return path;

} 



#endif
