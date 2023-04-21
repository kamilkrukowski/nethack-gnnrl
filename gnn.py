import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, GCNConv, global_mean_pool


class GNN(torch.nn.Module):
    def __init__(self, num_features, 
            out_dim, 
            hid_dim=64, 
            num_layers=30, layer_type='GCNConv'): 
        """
        Initialize a GNN model for graph classification. 
        args: 
        """

        super(GNN, self).__init__()

        Layer = globals()[layer_type]

        self.conv1 = Layer(num_features, hid_dim)
        self.middle_layers = []
        for i in range(num_layers - 1):
            self.middle_layers.append(Layer(hid_dim, hid_dim))
        self.pool = global_mean_pool
        self.lin = Linear(hid_dim, out_dim)
        

    def forward(self, x, edge_list, batch):
        """
        Implement the GNN calculation. The output should be logits for graph labels in the batch.    

        args: 
            x: a Tensor of shape [n, num_features], node features
            edge_list: a Tensor of shape [2, num_edges], each column is a tuple contains a pair `(sender, receiver)`. Here `sender` and `receiver`
            batch: the indicator vector indicating different graphs    
        """
        # print("x",x.shape)
        # print("edge_list",edge_list.shape)
        # print("batch",batch.shape)
        out = self.conv1(x, edge_list)
        out = F.relu(out)
        for hidden in self.middle_layers:
            out = hidden(out, edge_list)
            out = F.relu(out)
        out = self.pool(out, batch)
        out = self.lin(out)
        return out

