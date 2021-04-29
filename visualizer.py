###############################################
# In this file, a blank map is first created by initializing
# a 2D grid of zeros. To represent different states of any
# potential grid spot, the following keys are used:
#  0 : Completely unvisited space (White)
#  1 : Obstacle (Black)
#  2 : Visited Node (Orange)
#  3 : Shortest Path (Red)
#  4 : Source/Destination (Blue)
###############################################

from matplotlib import pyplot
from matplotlib import colors
from matplotlib.ticker import MultipleLocator
import numpy as np
import yaml



# Reading in agent and obstacle data for graph initialization
with open(r'D:\OneDrive\Documents\School\EC504\GPUPathfinding\Visualizer_Examples\map_32by32_obst204_agents100_ex99.yaml') as initMap:
    data = yaml.safe_load(initMap)
with open(r'D:\OneDrive\Documents\School\EC504\GPUPathfinding\Visualizer_Examples\map_32by32_obst204_agents100_ex99_astar.yaml') as pathMap:
    astar_pathData = yaml.safe_load(pathMap)
with open(r'D:\OneDrive\Documents\School\EC504\GPUPathfinding\Visualizer_Examples\map_32by32_obst204_agents100_ex99_dijkstra.yaml') as pathMap:
    dijkstra_pathData = yaml.safe_load(pathMap)

num_agents = len(data['agents'])
# Establish arrays to read reformatted data into

obstacles = []
tempPath = []
astar_nodes = dict()
dijkstra_nodes = dict()
astar_paths = dict()
dijkstra_paths = dict()

pathDict = dict()
goalDict = dict()
# Read in maze size and number of agents

size = max(data['map']['dimensions'])
# creating blank map of zeros
astar_maze = np.full((size,size), 0)
dijkstra_maze = np.full((size,size), 0)


colorTracker = (num_agents)/4
# Read in the start and end goal of agents
for agent in data['agents']:
    goalDict[agent['name']] = [agent['start'],agent['goal']]
# Read in the obstacles
for obstacle in data['map']['obstacles']:
    obstacles.append(obstacle)
    astar_maze[obstacle[0]][obstacle[1]] = 1 #adding obstacle points to maze
    dijkstra_maze[obstacle[0]][obstacle[1]] = 1  # adding obstacle points to maze

# Reading in astar search data
for agent in astar_pathData['paths']:
    name = agent['name']
    astar_nodes[name] = agent['nodes']
    astar_paths[name] = agent['path']

# Reading in dijkstra search data
for agent in dijkstra_pathData['paths']:
    name = agent['name']
    dijkstra_nodes[name] = agent['nodes']
    dijkstra_paths[name] = agent['path']


astar_template = np.copy(astar_maze)
dijkstra_template = np.copy(dijkstra_maze)

# Plot initialization
cmap = colors.ListedColormap(['white', 'black' ,'orange', 'red', 'blue'])
boundaries = [0,1,2,3,4,5]
norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

fig = pyplot.figure(figsize =(20,10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ast = ax1.imshow(astar_maze, aspect='auto', cmap=cmap, norm= norm, extent = [0,size,size,0])
dij = ax2.imshow(dijkstra_maze, aspect='auto', cmap=cmap, norm= norm, extent = [0,size,size,0])
ax1.grid(which='major', axis='both')
ax1.grid(color='black', linestyle='-', linewidth=1)
ax2.grid(which='major', axis='both')
ax2.grid(color='black', linestyle='-', linewidth=1)
majorLocator = MultipleLocator(1)
ax1.xaxis.set_major_locator(majorLocator)
ax1.yaxis.set_major_locator(majorLocator)
ax2.xaxis.set_major_locator(majorLocator)
ax2.yaxis.set_major_locator(majorLocator)
ax1.set_title('A*    |    Nodes Visited: ' + str(0) + '    |    Path Length: N/A')
ax2.set_title('Dijkstra    |    Nodes Visited: ' + str(0) + '    |    Path Length: N/A')

pathIter = 0 # tracks which paths from the yaml file we have done
pathTracker = 0 # tracks the position in the current path we are displaying

astar_pathlength = 'N/A'
dijkstra_pathlength = 'N/A'
astar_nodeTracker = 1
dijkstra_nodeTracker = 1

astar_flag = False # will become true when all astar agents have finished
dijkstra_flag = False # will become true when all dijkstra agents have finished

for agent in range(0,num_agents):
    # Initializing the graphs to only show obstacles, start, and end
    agentName = "agent"+str(agent)
    s = goalDict[agentName][0]
    d = goalDict[agentName][1]
    astar_maze[s[0], s[1]] = 4
    astar_maze[d[0], d[1]] = 4
    dijkstra_maze[s[0], s[1]] = 4
    dijkstra_maze[d[0], d[1]] = 4
    ast.set_data(astar_maze)
    dij.set_data(dijkstra_maze)
    fig.canvas.draw_idle()
    pyplot.pause(1)
    while astar_flag == False or dijkstra_flag == False:
        if not astar_flag:
            nextSpot = astar_nodes[agentName][astar_nodeTracker]
            astar_maze[nextSpot[0],nextSpot[1]] = 2
            astar_nodeTracker += 1


        if not dijkstra_flag:
            nextSpot = dijkstra_nodes[agentName][dijkstra_nodeTracker]
            dijkstra_maze[nextSpot[0], nextSpot[1]] = 2
            dijkstra_nodeTracker += 1

        if astar_nodeTracker >= len(astar_nodes[agentName]):
            astar_flag = True
            for spot in astar_paths[agentName]:
                astar_maze[spot[0], spot[1]] = 3
            astar_pathlength = len(astar_paths[agentName])
        if dijkstra_nodeTracker >= len(dijkstra_nodes[agentName]):
            dijkstra_flag = True
            for spot in dijkstra_paths[agentName]:
                dijkstra_maze[spot[0], spot[1]] = 3
            dijkstra_pathlength = len(dijkstra_paths[agentName])
        ax1.set_title('A*    |    Nodes Visited: ' + str(astar_nodeTracker) + '    |    Path Length: ' + str(astar_pathlength))
        ax2.set_title('Dijkstra    |    Nodes Visited: ' + str(dijkstra_nodeTracker) + '    |    Path Length: ' + str(dijkstra_pathlength))
        #updating graphs
        ast.set_data(astar_maze)
        dij.set_data(dijkstra_maze)
        fig.canvas.draw_idle()
        pyplot.pause(.1)

    # Both algorithms have finished their path finding, now we display the shortest path they found
    pyplot.pause(1)

    # Now we reset the mazes for the next agent
    astar_maze = np.copy(astar_template)
    dijkstra_maze = np.copy(dijkstra_template)
    astar_flag = False
    dijkstra_flag = False
    astar_nodeTracker = 1
    dijkstra_nodeTracker = 1
    astar_pathlength = 'N/A'
    dijkstra_pathlength = 'N/A'
