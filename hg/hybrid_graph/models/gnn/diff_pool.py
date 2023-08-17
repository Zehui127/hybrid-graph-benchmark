
from math import ceil
import torch
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def forward(self, x, adj):

        x0 = x
        x1 = self.conv1(x0, adj).relu()
        x2 = self.conv2(x1, adj).relu()
        x3 = self.conv3(x2, adj).relu()

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()

        return x


class DiffPool(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        super().__init__()
        self.is_regression = info["is_regression"]
        if info["is_regression"]:
            num_nodes = 30
        else:
            num_nodes = 10
        self.gnn1_pool = GNN(info["num_node_features"], 32, num_nodes)
        self.gnn1_embed = GNN(info["num_node_features"], 32, 32, lin=False)

        num_nodes = info["num_node"]
        self.gnn2_pool = GNN(3 * 32, 32, num_nodes)
        self.gnn2_embed = GNN(3 * 32, 32, 32, lin=False)

        self.gnn3_embed = GNN(3 * 32, 32, 32, lin=False)

        self.lin1 = torch.nn.Linear(3 * 32, 32)
        if info["is_regression"]:
            self.lin2 = torch.nn.Linear(32, 1)
        else:
            self.lin2 = torch.nn.Linear(32, info["num_classes"])

    def forward(self, data, *args, **kargs):
        x, adj = data.x, data.edge_index
        adj = to_dense_adj(adj)[0]
        # print(f"X before pool0: {x.shape}")
        s = self.gnn1_pool(x, adj)
        x = self.gnn1_embed(x, adj)
        # print(f"X before pool: {x.shape}")
        # print(f"s before pool: {s.shape}")
        x, adj, l1, e1 = dense_diff_pool(x, adj, s)
        # print(f"X after pool: {x.shape}")
        # print(f"s after pool: {s.shape}")
        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        # print(f"X before pool2: {x.shape}")
        # print(f"s before pool2: {s.shape}")

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        # print(f"X after pool: {x.shape}")
        # print(f"s after pool: {s.shape}")
        x = self.gnn3_embed(x, adj)

        # x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        # return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2
        # print(f"X final shape: {x.shape}")
        if self.is_regression:
            return x.squeeze()
        return F.log_softmax(x, dim=1).squeeze(0)
