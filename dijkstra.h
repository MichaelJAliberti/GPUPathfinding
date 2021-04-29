#ifndef DIJKSTRA_H_
#define DIJKSTRA_H_

#include <queue>
#include <iostream>
#include <fstream>

#define INF 9999999

using namespace std;

int dijkstra(string grid_name)
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
	int preditor;
	int counter;
	int src;
	int dst;
	
	//agent Declaration
	Agent iter;

	// Prepare output file
	stringstream s1(grid_name);
	string token;
	getline(s1, token, '.');
	token += "_dijkstra.yaml";
	ofstream outfile(token);

	// loop through agents with dijkstras
	for (auto k = map.agents.begin(); k != map.agents.end(); ++k){
		
		//dijkstra's variables Initialization
		for (int i = 0; i < size; i++)
		{
			dist[i] = INF;
			sptSet[i] = false;
			pred[i] = -1;
		}
		
		//Agent Initialization
		iter = *k;
		
		//source and destination from the Agent
		src = iter.sx + iter.sy * side_size;
		dst = iter.dx + iter.dy * side_size;
		
		//Dijkstra's source Initialization
		pred[src] = src;
		dist[src] = 0;
		int u;
		int v;

		// begin queue at source
		queue<int> q;
		q.push(src);
		sptSet[src] = true;

		// write to output file
		token = iter.name.erase(iter.name.length()-1, 1);
		outfile << token << ":" << endl;
		outfile << "    nodes:" << endl;
		counter = 0;

		while (!q.empty()) {

			u = q.front();
			q.pop();
			
			counter++;
	        outfile << "    - [" << u%side_size << ", " << u/side_size << "]" << endl;

	        if (u == dst) break;
	  
			//Since the data structure is a grid, there are only 4 adjacent nodes to check

			//The node to the RIGHT
			v = u + 1;
			if (v < size && v >= 0)
				if (!(v % (side_size) == 0) && !sptSet[v] && map.matrix[v] == 0 && dist[u] != INF 
					&& dist[u] + 1 < dist[v])
					{
						dist[v] = dist[u] + 1;
						pred[v] = u;
						sptSet[v] = true;
						q.push(v);
					}
			//The node to the LEFT
			v = u - 1;
			if (v < size && v >= 0)
				if (!(v % (side_size) == side_size - 1) && !sptSet[v] && map.matrix[v] == 0 && dist[u] != INF 
					&& dist[u] + 1 < dist[v])
					{
						dist[v] = dist[u] + 1;
						pred[v] = u;
						sptSet[v] = true;
						q.push(v);
					}
			//The node to the TOP
			v = u - side_size;
			if (v < size && v >= 0)
				if ((v > -1) && !sptSet[v] && map.matrix[v] == 0 && dist[u] != INF 
					&& dist[u] + 1 < dist[v])
					{
						dist[v] = dist[u] + 1;
						pred[v] = u;
						sptSet[v] = true;
						q.push(v);
					}
			//The node to the BOTTOM
			v = u + side_size;
			if (v < size && v >= 0)
				if (v < size && !sptSet[v] && !(u / (side_size) == side_size - 1) && map.matrix[v] == 0 && dist[u] != INF 
					&& dist[u] + 1 < dist[v])
					{
						dist[v] = dist[u] + 1;
						pred[v] = u;
						sptSet[v] = true;
						q.push(v);
					}
		}

		outfile << "    number: " << counter << endl;
		
		//Find the shortest path from the pred[]
		preditor = dst;
		counter = 0; // length of path
		vector<int> path;
		path.push_back(preditor);
		while (preditor != src)
		{
			preditor = pred[preditor];		
			path.push_back(preditor);
			counter++;
		}

		outfile << "    " << "path:" << endl;
		for (auto i = path.begin(); i != path.end(); ++i){
			outfile << "    ";
	        outfile << "- [" << (*i)%side_size << ", " << (*i)/side_size << "]" << endl;
		}
		outfile << "    " << "length: " << counter << endl;
	}

    outfile.close();

    return 0;
} 

#endif
