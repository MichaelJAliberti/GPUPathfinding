###############################################
# In this file, a blank map is first created by initializing
# a 2D grid of zeros. To represent different states of any
# potential grid spot, the following keys are used:
#  100 : Completely unvisited space (White)
#  1 : Obstacle (Black)
#  2 : Start spot (Red)
#  3 : Path spot (Orange)
#  4 : Traversed space from agent's prior, corrected path (Yellow)
#  5 : Destination (Blue)
#  6 : Finished path (Green)
###############################################

from matplotlib import pyplot
from matplotlib import colors
import numpy as np
import yaml

def updateGraph(figure, maze):
    currentFrame.set_data(maze)
    figure.canvas.draw_idle()
    pyplot.pause(.1)


# Reading in agent and obstacle data for graph initialization
with open(r'D:\OneDrive\Documents\School\EC504\GPUPathfinding\SampleMaps\32x32_obst204\map_32by32_obst204_agents10_ex90.yaml') as initMap:
    data = yaml.safe_load(initMap)
with open(r'D:\OneDrive\Documents\School\EC504\GPUPathfinding\SolvedPaths\map_32by32_obst204_agents10_ex90_path_file.yaml') as pathMap:
    pathData = yaml.safe_load(pathMap)

# Establish arrays to read reformatted data into
obstacles = []
tempPath = []
pathDict = dict()
goalDict = dict()
colorDict = dict()
# Read in maze size and number of agents
num_agents = len(data['agents'])
size = max(data['map']['dimensions'])
# creating blank map of zeros
maze = np.full((size,size), num_agents+(num_agents/4))

colorTracker = (num_agents)/4
# Read in the start and end goal of agents
for agent in data['agents']:
    goalDict[agent['name']] = [agent['start'],agent['goal']]
    colorDict[agent['name']] = colorTracker
    maze[agent['start'][0]][agent['start'][1]] = colorTracker #adding agent startpoints to maze
    maze[agent['goal'][0]][agent['goal'][1]] = colorTracker #adding agent end points to maze
    colorTracker = colorTracker + 1
# Read in the obstacles
for obstacle in data['map']['obstacles']:
    obstacles.append(obstacle)
    maze[obstacle[0]][obstacle[1]] = 0 #adding obstacle points to maze


# Plot initialization
cmap = colors.ListedColormap(['white','black','red','orange','yellow','blue','green'])
bounds = [0,1,2,3,4,5,6,7]
norm = colors.BoundaryNorm(bounds, cmap.N)
fig = pyplot.figure(figsize =(10,10))
# currentFrame = pyplot.imshow(maze, aspect='auto', cmap='nipy_spectral', norm=norm, vmin = 0, vmax = 100)
currentFrame = pyplot.imshow(maze, aspect='auto', cmap='nipy_spectral', vmin = 0, vmax = num_agents+(num_agents/4))
fps = 30
sec = 30
pyplot.grid(which='major', axis='both')
pyplot.grid(color='black', linestyle='-', linewidth=1)
pyplot.xticks(np.arange(-.5, size, 1));
pyplot.yticks(np.arange(-.5, size, 1));

pathIter = 0 # tracks which paths from the yaml file we have done
pathTracker = 0 # tracks the position in the current path we are displaying

# Read in paths, display new spots as they are read in
for agent in pathData['paths']:
    agentName = agent['name']

    # if an agent is encountered that we already have a shortest path for, delete it from maze and reset the start spot
    if agentName in pathDict.keys():
        print("Found better path for",agentName,". Deleting old path.")
        oldPath = pathDict[agentName]
        start = oldPath[0]
        dest = oldPath[-1]
        for spot in oldPath:
            maze[spot[0], spot[1]] = 0  # deleting old visited spots
        maze[start[0], start[1]] = colorDict[agentName]  # resetting start point
        maze[dest[0], dest[1]] = colorDict[agentName]  # resetting start point

    # Ensuring start and end point are shown on the map
    s = goalDict[agentName][0]
    d = goalDict[agentName][1]
    #print(s,d)
    maze[s[0], s[1]] = colorDict[agentName]
    maze[d[0], d[1]] = colorDict[agentName]

    # for each spot in this new path, add it to tempPath to eventually update the dictionary, update its maze value, and redraw the frame
    print("\nDrawing next path for", agentName)
    for path in agent['path']:
        print(path, "added.")
        tempPath.append(path)
        maze[path[0],path[1]] = colorDict[agentName]
        # currentFrame.set_data(maze)
        # fig.canvas.draw_idle()
        # pyplot.pause(.25)
        updateGraph(fig, maze)

    pathDict[agentName] = tempPath
    print("New path dictionary:", pathDict)

    # Changing color of the finished path to green to indicate it is done
    # for spot in tempPath:
    #     maze[spot[0], spot[1]] = 6
    tempPath = []



pyplot.show()
