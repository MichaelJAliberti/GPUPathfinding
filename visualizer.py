###############################################
# In this file, a blank map is first created by initializing
# a 2D grid of zeros. To represent different states of any
# potential grid spot, the following keys are used:
#  0 : Completely unvisited space (White)
#  1 : Obstacle (Black)
#  2 : Currently occupied space (Red)
#  3 : Recently traversed space on the same path (Orange)
#  4 : Traversed space from agent's prior, corrected path (Yellow)
#  5 : Destination (Blue)
#  6 : Finished path (Green)
###############################################

import matplotlib as mpl
from matplotlib import pyplot
from matplotlib import colors
import matplotlib.animation as animation
import numpy as np
import yaml
import sys

print("hi")
# Reading in agent and obstacle data for graph initialization
with open(r'D:\OneDrive\Documents\School\EC504\GPUPathfinding\SampleMaps\8x8_obst12\map_8by8_obst12_agents4_ex0.yaml') as f:
    data = yaml.safe_load(f)

# Establish arrays to read reformatted data into
starts = []
goals = []
obstacles = []
# Read in maze size and number of agents
num_agents = len(data['agents'])
size = max(data['map']['dimensions'])
# creating blank map of zeros
maze = np.zeros((size,size))

print(maze)

# Read in the start and end goal of agents
for agent in data['agents']:
    starts.append(agent['start'])
    maze[agent['start'][0]][agent['start'][1]] = 2 #adding agent startpoints to maze
    goals.append(agent['goal'])
    maze[agent['goal'][0]][agent['goal'][1]] = 5 #adding agent end points to maze

# Read in the obstacles
for obstacle in data['map']['obstacles']:
    obstacles.append(obstacle)
    maze[obstacle[0]][obstacle[1]] = 1 #adding obstacle points to maze

cmap = colors.ListedColormap(['white','black','red','orange','yellow','blue','green'])
bounds = [0,1,2,3,4,5,6,7]
norm = colors.BoundaryNorm(bounds, cmap.N)

pyplot.figure(figsize =(10,10))
currentFrame = pyplot.imshow(maze, cmap=cmap, norm=norm)

fps = 30
sec = 30

pyplot.grid(which='major', axis='both')
pyplot.grid(color='black', linestyle='-', linewidth=1)
pyplot.xticks(np.arange(-.5, size, 1));
pyplot.yticks(np.arange(-.5, size, 1));

def animate(newMaze):
    currentFrame.set_data(newMaze)
    return [currentFrame]

# anim = animation.FuncAnimation(fig, animate, frames = sec*fps, interval = 1000/fps)

# anim.save('test_anim.mp4', fps=fps)

pyplot.show()
