import argparse
import os

# Take in command line arguments
parser = argparse.ArgumentParser(prog='VISUALIZE',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('in_file', help='YAML file for grid')
args = parser.parse_args()

# Find files based on input
graph_file = args.in_file
tokens = graph_file.split('.')
if len(tokens) < 2:
    exit()
elif tokens[1] != "yaml":
	exit()
dijkstra_file = tokens[0] + "_dijkstra.yaml"
astar_file = tokens[0] + "_astar.yaml"

# Verify files
if not os.path.isfile(graph_file):
	# Check if file 1 exists
	exit()
if not os.path.isfile(dijkstra_file):
	# Check if file 1 exists
	exit()
if not os.path.isfile(astar_file):
	# Check if file 1 exists
	exit()

# Remove generated files
os.remove(dijkstra_file)
os.remove(astar_file)