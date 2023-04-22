import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool, Sequential
from torch.nn import Linear, ReLU, Dropout, Embedding


class GNN(torch.nn.Module):

    def __init__(self, in_dim, out_dim, hid_dim=16,
                 num_layers=15, layer_type='GCNConv'):
        """
        Initialize a GNN model for graph classification.
        args:
        """
        super().__init__()

        GCNLayer = globals()[layer_type]

        # Maximum number of glyphs in simulation + 1 hardcoded is 5977
        _layers = [
            (Embedding(in_dim, hid_dim), 'x -> x'),
            (Dropout(p=0.5), 'x -> x'),
            (GCNLayer(hid_dim, hid_dim), 'x, edge_index -> x1')
        ]

        for _ in range(num_layers - 1):
            _layers.append(ReLU(inplace=True))
            _layers.append((Dropout(p=0.5), 'x1 -> x1'))
            _layers.append(
                (GCNLayer(hid_dim, hid_dim), 'x1, edge_index -> x1'))

        _layers = _layers + [
            (global_mean_pool, 'x1, batch -> x2'),
            Linear(hid_dim, out_dim)
        ]

        self.layers = Sequential('x, edge_index, batch', _layers)

    def forward(self, x, edge_index, batch):
        """
        Implement the GNN calculation. The output should be logits for graph labels in the batch.

        args:
            x: a Tensor of shape [n, num_features], node features
            edge_index: a Tensor of shape [2, num_edges], each column is a tuple contains a pair `(sender, receiver)`. Here `sender` and `receiver`
            batch: the indicator vector indicating different graphs
        """
#        print(f"{x.shape}, {edge_list.shape}")
        out = self.layers(x, edge_index, batch)
        return out
