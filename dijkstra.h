#ifndef DIJKSTRA_H_
#define DIJKSTRA_H_

#include <queue>

#define INF 9999999

using namespace std;

void dijkstra(string grid_name, int src, int dst)
{
	
	grid map(grid_name);
	
	int side_size = map.size;
	int size = side_size * side_size;
	
	int dist[size];
	bool sptSet[size];
	
	int pred[size];
	int path[size*map.numagents];
	int preditor;
	int counter;
	int src;
	
	agent iter;
	
	
	
	for(int i = 0; i < map.numagents; i++)
	{
		for (int i = 0; i < size; i++)
		{
			dist[i] = INF;
			sptSet[i] = false;
			pred[i] = -1;
		}
		
		iter = map.agents[i];
		
		src = iter.sx + iter.sy;
		
		pred[src] = src;
		dist[src] = 0;
		int min = INF;
		int u;
		for (int count = 0; count < V - 1; count++) {

			
			min = INF;
			for (int v = 0; v < size; v++)
				if (sptSet[v] == false && dist[v] <= min)
				{
					min = dist[v];
					u = v;
				}
		  
			sptSet[u] = true;
	  
			for (int v = 0; v < size; v++)
	  
				if (!sptSet[v] && map.matrix[u*side_size+v] != -1 && dist[u] != INF
					&& dist[u] + map.matrix[u*side_size+v] < dist[v])
					{
					dist[v] = dist[u] + map.matrix[u*side_size+v];
					pred[u] = v;
					}
		}
		
		preditor = dst;
		path[i*size] = dst;
		counter = i*size+1;
		while (dst != src)
		{
			path[counter] = pred[preditor];
			preditor = pred[preditor];
			counter++;
		}

	}	
	
	

} 



#endif
