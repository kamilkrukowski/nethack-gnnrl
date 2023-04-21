from abc import ABC, abstractmethod, abstractproperty

import gym  # type: ignore
import nle  # type: ignore
import numpy as np  # type: ignore
from torch_geometric.data import Data, Batch # type: ignore


class WorldGraph(ABC):
    """Base class for NetHack RL GNN representations."""

    @abstractmethod
    def update(self, obs):
        """Given OpenAI NLE Observation,\
            update the graph-representation state."""
        pass

    @abstractproperty
    def edge_index(self):
        pass

    @abstractproperty
    def nodes(self):
        pass

    def pyg(self):
        """Generate Pytorch-Geometric Data Representation of current state."""
        return Data(x=self.nodes, edge_index=self.edge_index)


class TileGraphV1(WorldGraph):
    """
    GNN Representation treating each tile as vertex, \
        with connections to 8 neighbors.
    """

    def __init__(self):
        super().__init__()
        self._edge_index = []
        self._nodes = []

    def update(self, obs):

        glyphs = obs['glyphs']
        self._nodes = []
        node_ids = np.zeros_like(glyphs)  # track locations of node ids
        # double for loop over the glyphys to make a graph representation
        for i in range(glyphs.shape[0]):  # loop over the rows
            for j in range(glyphs.shape[1]):  # loop over the columns
                value = glyphs[i, j]  # get the value at that coordinate
                if value != 2359:  # ignore empty spaces
                    # assign this node a new id
                    this_node_id = len(self._nodes) + 1
                    # set the node id in its position in node_ids
                    node_ids[i, j] = this_node_id
                    # add this character to the list of nodes
                    self._nodes.append(value)
                    # check some of the node's neighbors to make edges
                    self._check_w_nw_n_ne(
                        i, j, this_node_id, glyphs, node_ids)

    @property
    def edge_index(self):
        return np.array(self._edge_index)

    @property
    def nodes(self):
        # ints to one hot encoding https://stackoverflow.com/a/29831596
        nodes = np.array(self._nodes)
        one_hot_nodes = np.zeros((nodes.size, nodes.max() + 1))
        one_hot_nodes[np.arange(nodes.size), nodes] = 1
        return one_hot_nodes

    # look in the west, northwest, north, and north east directions to see if
    # we should add new edges to previously seen nodes
    def _check_w_nw_n_ne(self,
                         i: int, j: int, this_node_id, glyphs, node_ids):
        self._add_edge_index(i, j - 1, this_node_id, glyphs,
                             node_ids)  # add an edge to the west
        # add an edge to the northwest
        self._add_edge_index(i - 1, j - 1, this_node_id,
                             glyphs, node_ids)
        self._add_edge_index(i - 1, j, this_node_id, glyphs,
                             node_ids)  # add an edge to the north
        # add an edge to the northeast
        self._add_edge_index(i - 1, j + 1, this_node_id,
                             glyphs, node_ids)

    def _add_edge_index(self, i: int, j: int, this_node_id, glyphs, node_ids):
        # if the i and j indices are valid
        if (i >= 0 and j >= 0 and i < glyphs.shape[0] and j < glyphs.shape[1]):
            value = glyphs[i, j]  # check the value at that coordinate
            if value != 2359:  # ignore empty spaces
                neighbor_node_id = node_ids[i, j]  # get the id of the neighbor
                # append the new edge
                self._edge_index.append([this_node_id, neighbor_node_id])

class TileGraphV2(WorldGraph):
    """
    GNN Representation treating each tile as vertex, \
        with connections to 8 neighbors.
    """

    def __init__(self):
        super().__init__()
        self._edge_index = []
        self._nodes = []
        self._prev_obs = None

    def update(self, obs):

        if obs == self._prev_obs:
            return
        else:
            self._prev_obs = obs

        glyphs = obs['glyphs']
        self._nodes = []
        node_ids = np.zeros_like(glyphs)  # track locations of node ids
        # double for loop over the glyphys to make a graph representation
        for i in range(glyphs.shape[0]):  # loop over the rows
            for j in range(glyphs.shape[1]):  # loop over the columns
                value = glyphs[i, j]  # get the value at that coordinate
                if value != 2359:  # ignore empty spaces
                    # assign this node a new id
                    this_node_id = len(self._nodes) + 1
                    # set the node id in its position in node_ids
                    node_ids[i, j] = this_node_id
                    # add this character to the list of nodes
                    self._nodes.append(value)
                    # check some of the node's neighbors to make edges
                    self._check_w_nw_n_ne(
                        i, j, this_node_id, glyphs, node_ids)

    @property
    def edge_index(self):
        return np.array(self._edge_index)

    @property
    def nodes(self):
        # ints to one hot encoding https://stackoverflow.com/a/29831596
        nodes = np.array(self._nodes)
        one_hot_nodes = np.zeros((nodes.size, nodes.max() + 1))
        one_hot_nodes[np.arange(nodes.size), nodes] = 1
        return one_hot_nodes

    # look in the west, northwest, north, and north east directions to see if
    # we should add new edges to previously seen nodes
    def _check_w_nw_n_ne(self,
                         i: int, j: int, this_node_id, glyphs, node_ids):
        self._add_edge_index(i, j - 1, this_node_id, glyphs,
                             node_ids)  # add an edge to the west
        # add an edge to the northwest
        self._add_edge_index(i - 1, j - 1, this_node_id,
                             glyphs, node_ids)
        self._add_edge_index(i - 1, j, this_node_id, glyphs,
                             node_ids)  # add an edge to the north
        # add an edge to the northeast
        self._add_edge_index(i - 1, j + 1, this_node_id,
                             glyphs, node_ids)

    def _add_edge_index(self, i: int, j: int, this_node_id, glyphs, node_ids):
        # if the i and j indices are valid
        if (i >= 0 and j >= 0 and i < glyphs.shape[0] and j < glyphs.shape[1]):
            value = glyphs[i, j]  # check the value at that coordinate
            if value != 2359:  # ignore empty spaces
                neighbor_node_id = node_ids[i, j]  # get the id of the neighbor
                # append the new edge
                self._edge_index.append([this_node_id, neighbor_node_id])


TileGraph = TileGraphV2


if __name__ == '__main__':

    # Set up Environment

    env = gym.make("NetHackScore-v0")
    obs = env.reset()  # each reset generates a new dungeon
    # print("obs.keys()",obs.keys())
    env.render()

    TestGraph = TileGraph()

    TestGraph.update(obs)
    pygData = TestGraph.pyg()

    nodes, edge_index = pygData.x, pygData.edge_index
    print(f"Node shape is {nodes.shape}")
    print(f"Edge Index shape is {edge_index.shape}")
