from abc import ABC, abstractmethod, abstractproperty
import re

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
    def setup(self, obs):
        """Set up new world graph from freshly-reset environment"""
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

    @property
    def node_feature_dim(self):
        return self.nodes.shape[1:]

    def pyg(self):
        """Generate Pytorch-Geometric Data Representation of current state."""
        return Data(x=from_numpy(self.nodes).int(),
                    edge_index=from_numpy(self.edge_index).int().reshape(2, -1)
                    )


class_to_id = {
    "archeologist": 327,
    "barbarian": 328,
    "caveman": 329,
    "cavewoman": 330,
    "healer": 331,
    "knight": 332,
    "monk": 333,
    "priest": 334,
    "priestess": 335,
    "ranger": 336,
    "rogue": 337,
    "samurai": 338,
    "tourist": 339,
    "valkyrie": 340,
    "wizard": 341
}
classes = list(class_to_id.keys())


def get_message(obs):
    return bytes(obs['message']).decode('utf-8').strip('\x00')


def get_class(first_obs):
    """
    Obtain character class and associated glyph from nethack starting message.
    -----------
    Output
    -------
    (class_id: int, class_name: str) : Tuple(int, str)
    """
    return 333, 'monk'
    start_message = get_message(first_obs).lower()
    #  Find string match for class
    for class_ in classes:
        if re.search(class_, start_message) is not None:
            return class_to_id[class_], class_
    print(start_message)
    raise RuntimeError('Started Environment without valid Character Class')


class EmptyGraph(WorldGraph):
    """WorldGraph consisting of only 1 static node. No observation info."""
    def __init__(self, max_nodes=800, max_edges=4000):
        super().__init__(max_nodes=max_nodes, max_edges=max_edges)
        self._edge_index = np.array([[0, 0]])
        self._nodes = np.array([333])
        self._pyg = Data(x=from_numpy(self.nodes).int(),
                         edge_index=from_numpy(
                             self.edge_index).int().reshape(2, -1)
                         )

    def update(self, obs):
        return

    def setup(obs):
        return

    @property
    def edge_index(self):
        return self._edge_index

    @abstractproperty
    def nodes(self):
        return self._nodes

    @property
    def node_feature_dim(self):
        return 1

    def pyg(self):
        return self._pyg


class TileGraphPosEnc(WorldGraph):
    """
    GNN Representation treating each tile as vertex, \
        with connections to 8 neighbors.
    """

    def __init__(self, max_nodes=800, max_edges=4000):
        super().__init__(max_nodes=max_nodes, max_edges=max_edges)

        self._edge_index = None

        self._nodes = None
        self._node_pos = -1 * np.ones((21, 79), dtype=int)
        self.agent_pos = np.array([0, 0], dtype=int)

        self._prev_obs = None
        self._class = None

        self.max_nodes = max_nodes
        self.max_edges = max_edges

        self._class = None
        self._class_id = -1

    def set_agent_pos(self, x, y):

        x2, y2 = self.agent_pos[0], self.agent_pos[1]
        old_node_id = self._node_pos[x2, y2]
        if old_node_id != -1:
            self._nodes[old_node_id][3] = 0

        new_node_id = self._node_pos[x, y]
        self._nodes[new_node_id][3] = 1
        self.agent_pos = np.array([x, y])

    def _build_from_scratch(self, glyphs):

        self._nodes = []
        self._edge_index = []

        node_ids = np.zeros_like(glyphs)  # track locations of node ids
        # double for loop over the glyphs to make a graph representation
        for i in range(glyphs.shape[0]):  # loop over the rows
            for j in range(glyphs.shape[1]):  # loop over the columns
                value = glyphs[i, j]  # get the value at that coordinate
                if value != 2359:  # ignore empty spaces
                    # assign this node a new id
                    this_node_id = len(self._nodes)
                    self._node_pos[i, j] = this_node_id
                    # add this character to the list of nodes
                    features = [value, i, j, 0]
                    self._nodes.append(features)
                    # check some of the node's neighbors to make edges
                    self._check_w_nw_n_ne(
                        i, j, this_node_id, glyphs, node_ids)

        self._nodes = np.array(self._nodes)

    def update(self, obs):

        glyphs = obs['glyphs']
        blstats = obs["blstats"]

        if is_tensor(blstats):
            blstats = blstats.detach().numpy()
        if is_tensor(glyphs):
            glyphs = glyphs.detach().numpy()
        glyphs = glyphs.reshape(21, 79)

        if np.array_equal(glyphs, self._prev_obs):
            return  # all done!

        self._prev_obs = glyphs.copy()

        self.reset()
        self._build_from_scratch(glyphs)

        coordinates = blstats.reshape(-1)[:2]  # agent x-y coordinates
        x, y = coordinates[1], coordinates[0]
        self.set_agent_pos(x, y)

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

    def setup(self, obs):
        (self._class_id, self._class) = get_class(obs)
        self.reset()
        self.update(obs)

    def reset(self):
        self._nodes = np.array([self._class_id])
        self._edge_index = []
        self._prev_obs = None
        self._node_pos = -1 * np.ones((21, 79), dtype=int)
        self.agent_pos = np.array([0, 0])

    @property
    def edge_index(self):
        return np.array(self._edge_index)

    @property
    def nodes(self):
        return self._nodes

    @property
    def node_feature_dim(self):
        return 4

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


class TileGraph(WorldGraph):
    """
    GNN Representation treating each tile as vertex, \
        with connections to 8 neighbors.
    """

    def __init__(self, max_nodes=800, max_edges=4000):
        super().__init__(max_nodes=max_nodes, max_edges=max_edges)

        self._edge_index = None

        self._nodes = None

        self._prev_obs = None
        self._class = None

        self.max_nodes = max_nodes
        self.max_edges = max_edges

        self._class = None
        self._class_id = -1

    def _build_from_scratch(self, glyphs):

        self._nodes = []
        self._edge_index = []

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
        self._node_ages = np.zeros_like(self._nodes)

    def update(self, obs):

        glyphs = obs['glyphs']

        if is_tensor(glyphs):
            glyphs = glyphs.detach().numpy()
        glyphs = glyphs.reshape(21, 79)

        if np.array_equal(glyphs, self._prev_obs):
            return  # all done!

        self._prev_obs = glyphs.copy()

        self._build_from_scratch(glyphs)

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

    def setup(self, obs):
        (self._class_id, self._class) = get_class(obs)
        self.reset()
        self.update(obs)

    def reset(self):
        self._nodes = np.array([self._class_id])
        self._edge_index = []
        self._prev_obs = None

    @property
    def edge_index(self):
        return np.array(self._edge_index)

    @property
    def nodes(self):
        return self._nodes

    @property
    def node_feature_dim(self):
        return 1

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
