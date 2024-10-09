import torch
from math import ceil

import torch.nn.functional as F
import torch_geometric.utils as utils
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool



class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, self.conv1(x0, adj, mask).relu())
        x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        x3 = self.bn(3, self.conv3(x2, adj, mask).relu())

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()

        return x


class diff_pool_net(torch.nn.Module):
    def __init__(self,in_channels,out_channels,max_nodes,
                 number_of_unique_types,type_embedding_size):
        super().__init__()

        num_nodes =4
        self.gnn1_pool = GNN(in_channels, 64, num_nodes)
        self.gnn1_embed = GNN(in_channels, 64, 64, lin=False)

        num_nodes = ceil(0.25 * num_nodes)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, out_channels)

        self.embed_lin = torch.nn.Linear(192, 32)

        self.lin3 = torch.nn.Linear(32 + out_channels, out_channels)

        self.embedding = torch.nn.Embedding(number_of_unique_types, type_embedding_size)

    def forward(self, x, edge_index, node_types, mask=None,one_hot=True):

        adj = utils.to_dense_adj(edge_index)

        x = self.embedding(node_types)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        if len(adj.shape) == 2:
            adj = adj.unsqueeze(0)

        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x_embed_linear = self.embed_lin(x)

        #print(s.shape)
        s_shape = s.shape
        #print(s_shape)

        s_x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        print(s_x.shape)
        print(adj.shape)
        s_x = self.lin1(s_x).relu()
        s_x = self.lin2(s_x)

        if one_hot:
            s = torch.nn.functional.one_hot(s.argmax(dim=-1), num_classes=4)
            indices = s.argmax(dim=-1)
            result_tensor = torch.empty(1, indices.shape[1], s_x.shape[-1])
            for i in range(indices.shape[1]):
                selected_index = indices[0, i]
                result_tensor[0, i, :] = s_x[0, selected_index, :]
        else:
            result_tensor = s@s_x

        x = torch.cat((x_embed_linear, result_tensor), dim=2)
        # breakpoint()
        x = self.lin3(x)
        # TODO: s is the assignment matrix, use it to determine the cluster.  
        return x
    

