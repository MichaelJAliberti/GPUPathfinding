#ifndef DIJKSTRA_H_
#define DIJKSTRA_H_

#include <queue>

#define INF 9999999

using namespace std;

void dijkstra(string grid_name)
{
	
	Grid map(grid_name);
	
	int side_size = map.size;
	int size = side_size * side_size;
	
	int dist[size];
	bool sptSet[size];
	
	int pred[size];
	int path[size*map.num_agents];
	int preditor;
	int counter;
	int src;
	int dst;
	
	Agent iter;
	
	cout << "Passed: Initialization" << endl;
	cout << map.num_agents << endl;
	
		for(int i = 0; i < map.num_agents; i++)
		{
			for (int i = 0; i < size; i++)
			{
				dist[i] = INF;
				sptSet[i] = false;
				pred[i] = -1;
			}
			
			cout << "Passed: Initialization variables per Agent" << endl;
			
			iter = map.agents[i];
			cout << "Passed: Agent Initialization" << endl;
			
			src = iter.sx + iter.sy;
			dst = iter.dx + iter.dy;
			
			cout << "Passed: Agent Setup" << endl;
			
			pred[src] = src;
			dist[src] = 0;
			int min = INF;
			int u;
			
			cout << "Passed: Load data per Agent" << endl;
			
			for (int count = 0; count < size - 1; count++) {

				
				min = INF;
				for (int v = 0; v < size; v++)
					if (sptSet[v] == false && dist[v] <= min)
					{
						min = dist[v];
						u = v;
					}
			  
				sptSet[u] = true;
				
				cout << "Passed: Find Min" << endl;
		  
				//for (int v = 0; v < size; v++) // map.matrix[(u / side_size)*side_size + (u % side_size)] map.matrix[(u / side_size)*side_size + (u % side_size)]
					int v = u + 1;
					if (!sptSet[v] && !(v % side_size == 0) && map.matrix[v] && dist[u] != INF 
						&& dist[u] + map.matrix[u*side_size+v] < dist[v])
						{
						dist[v] = dist[u] + map.matrix[u*side_size+v];
						pred[u] = v;
						}
					v = u - 1;
					if (!sptSet[v] && !(v % side_size == side_size - 1) && map.matrix[v] && dist[u] != INF 
						&& dist[u] + map.matrix[u*side_size+v] < dist[v])
						{
						dist[v] = dist[u] + map.matrix[u*side_size+v];
						pred[u] = v;
						}
					v = u + side_size;
					if (!sptSet[v] && !(v / side_size == side_size - 1) && map.matrix[v] && dist[u] != INF 
						&& dist[u] + map.matrix[u*side_size+v] < dist[v])
						{
						dist[v] = dist[u] + map.matrix[u*side_size+v];
						pred[u] = v;
						}
					v = u - side_size;
					if (!sptSet[v] && !(v < 0) && map.matrix[v] && dist[u] != INF 
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
				cout << path[counter] << endl;
			}
			cout << endl;

		}	
	

	

} 



#endif
