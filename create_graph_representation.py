from abc import ABC, abstractmethod, abstractproperty

import gym  # type: ignore
import nle  # type: ignore
import numpy as np  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch import is_tensor, Tensor, LongTensor, from_numpy


class NodeBufferOverflow(Exception):
    pass


class EdgeBufferOverflow(Exception):
    pass


class WorldGraph(ABC):
    """Base class for NetHack RL GNN representations."""

    def __init__(self, max_nodes=None, max_edges=None):
        self.max_nodes = max_nodes
        self.max_edges = max_edges

    @abstractmethod
    def update(self, obs):
        """Given OpenAI NLE Observation,\
            update the graph-representation state."""
        pass

    @abstractmethod
    def reset(self):
        """Initialize self as new instance."""
        pass

    @abstractproperty
    def edge_index(self):
        pass

    @abstractproperty
    def nodes(self):
        pass

    def pyg(self):
        """Generate Pytorch-Geometric Data Representation of current state."""
        return Data(x=from_numpy(self.nodes).int(),
                    edge_index=from_numpy(self.edge_index).int().reshape(2, -1)
                    )


class TileGraph(WorldGraph):
    """
    GNN Representation treating each tile as vertex, \
        with connections to 8 neighbors.
    """

    def __init__(self, max_nodes=None, max_edges=None):
        super().__init__(max_nodes=max_nodes, max_edges=max_edges)
        self._edge_index = []
        self._nodes = np.array([333])  # Monk Player
        self._prev_obs = None

    def update(self, obs):

        glyphs = obs['glyphs']
        if is_tensor(glyphs):
            glyphs = glyphs.detach().numpy()
        glyphs = glyphs.reshape(21, 79)

        """
        if np.array_equal(glyphs, self._prev_obs):
            print('equal')
            print(glyphs)
            print(self._prev_obs)
            return
        else:
            self.reset()
            self._prev_obs = glyphs
        """
        self.reset()
        self._nodes = []

        node_ids = np.zeros_like(glyphs)  # track locations of node ids
        # double for loop over the glyphys to make a graph representation
        for i in range(glyphs.shape[0]):  # loop over the rows
            for j in range(glyphs.shape[1]):  # loop over the columns
                value = glyphs[i, j]  # get the value at that coordinate
                if value != 2359:  # ignore empty spaces
                    # assign this node a new id
                    this_node_id = len(self._nodes)
                    # set the node id in its position in node_ids
                    node_ids[i, j] = this_node_id
                    # add this character to the list of nodes
                    self._nodes.append(value)
                    # check some of the node's neighbors to make edges
                    self._check_w_nw_n_ne(
                        i, j, this_node_id, glyphs, node_ids)

        self._nodes = np.array(self._nodes)
        n_nodes = len(self._nodes)
        n_edges = len(self._edge_index)
        if n_nodes > self.max_nodes:
            raise NodeBufferOverflow(f"WorldGraph generated {n_nodes} nodes, "
                                     f"exceeding mp buffer of {self.max_nodes}"
                                     )
        if n_edges > self.max_edges:
            raise EdgeBufferOverflow(f"WorldGraph generated {n_edges} edges, "
                                     f"exceeding mp buffer of {self.max_edges}"
                                     )

    def reset(self):
        self._nodes = np.array([333])  # Monk Player
        self._edge_index = []
        self._prev_obs = None

    @property
    def edge_index(self):
        return np.array(self._edge_index)

    @property
    def nodes(self):
        return self._nodes

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
