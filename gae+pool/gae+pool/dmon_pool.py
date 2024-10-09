import torch
from math import ceil

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import DenseGraphConv, GCNConv, DMoNPooling
import torch_geometric.utils as utils
from torch_geometric.utils import to_dense_adj, to_dense_batch

class dmon_net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, max_nodes, number_of_unique_types,type_embedding_size, hidden_channels=32):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        num_nodes = 4
        self.pool1 = DMoNPooling([hidden_channels, hidden_channels], num_nodes)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

        self.embedding = torch.nn.Embedding(number_of_unique_types, type_embedding_size)

        self.lin3 = torch.nn.Linear(type_embedding_size + out_channels, out_channels)

    def forward(self, x, edge_index, node_types, one_hot=True):
        adj = utils.to_dense_adj(edge_index)
        x = self.embedding(node_types)
        x = self.conv1(x, edge_index).relu()

        s, s_x, adj, sp1, o1, c1 = self.pool1(x, adj)

        # breakpoint()
        s_x = self.lin1(s_x).relu()
        s_x = self.lin2(s_x)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        # breakpoint()
        if one_hot:
            s = torch.nn.functional.one_hot(s.argmax(dim=-1), num_classes=4)
            indices = s.argmax(dim=-1)
            if len(indices.shape) < 2:
                indices = indices.unsqueeze(0)

            # breakpoint()
            result_tensor = torch.empty(1, indices.shape[1], s_x.shape[-1])
            for i in range(indices.shape[1]):
                selected_index = indices[0, i]
                result_tensor[0, i, :] = s_x[0, selected_index, :]
        else:
            result_tensor = s@s_x
        # breakpoint()
        x = torch.cat((x, result_tensor), dim=2)
        x = self.lin3(x)
        # breakpoint()
        return x