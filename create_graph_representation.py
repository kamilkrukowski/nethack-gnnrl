import gym
import nle
import numpy as np

env = gym.make("NetHackScore-v0")
obs = env.reset()  # each reset generates a new dungeon
# print("obs.keys()",obs.keys())
env.render()

# look in the west, northwest, north, and north east directions to see if we should add new edges to previously seen nodes
def check_w_nw_n_ne(i:int, j:int, this_node_id, glyphs, node_ids, edge_index):
    add_edge_index(i,j-1, this_node_id, glyphs, node_ids, edge_index) # add an edge to the west
    add_edge_index(i-1,j-1, this_node_id, glyphs, node_ids, edge_index) # add an edge to the northwest
    add_edge_index(i-1,j, this_node_id, glyphs, node_ids, edge_index) # add an edge to the north
    add_edge_index(i-1,j+1, this_node_id, glyphs, node_ids, edge_index) # add an edge to the northeast
    

def add_edge_index(i:int, j:int, this_node_id, glyphs, node_ids, edge_index):
    if(i>=0 and j>=0 and i<glyphs.shape[0] and j<glyphs.shape[1]): # if the i and j indices are valid
        value = glyphs[i,j] # check the value at that coordinate
        if value != 2359: #ignore empty spaces
            neighbor_node_id = node_ids[i,j] # get the id of the neighbor
            edge_index.append([this_node_id, neighbor_node_id]) # append the new edge

# double for loop over the glyphys to make a graph representation
nodes = []
edge_index = []
glyphs = obs["glyphs"]
node_ids = np.zeros_like(glyphs) # track locations of node ids
for i in range(glyphs.shape[0]): # loop over the rows
    for j in range(glyphs.shape[1]): # loop over the columns
        value = glyphs[i,j] # get the value at that coordinate
        if value != 2359: #ignore empty spaces
            this_node_id = len(nodes) + 1 # assign this node a new id
            node_ids[i,j] = this_node_id # set the node id in its position in node_ids
            nodes.append(value) # add this character to the list of nodes
            check_w_nw_n_ne(i,j, this_node_id, glyphs, node_ids, edge_index) # check some of the node's neighbors to make edges

# ints to one hot encoding https://stackoverflow.com/a/29831596
nodes = np.array(nodes)
one_hot_nodes = np.zeros((nodes.size, nodes.max() + 1))
one_hot_nodes[np.arange(nodes.size), nodes] = 1

print("one_hot_nodes", one_hot_nodes.shape, one_hot_nodes)
print("edge_index", edge_index)
