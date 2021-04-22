#include "LoadMap.h"
#include "dijkstra.h"

using namespace std;

int main()
{
	cout << "Hello, World!" << endl;
	
	//Grid map("SampleMaps/8x8_obst12/map_8by8_obst12_agents2_ex0.yaml");
	
	//cout << map.a
	
	dijkstra("SampleMaps/8x8_obst12/map_8by8_obst12_agents2_ex0.yaml", 0);
	
	return 0;
}