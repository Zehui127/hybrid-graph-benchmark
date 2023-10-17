import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv,HypergraphConv


class GCN(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        """
        This model is used to replicate the performance of obg
        GCN require the following normalization for improve the performance
            # Pre-compute GCN normalization.
            adj_t = data.adj_t.set_diag()
            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
            data.adj_t = adj_t
        """
        super(GCN, self).__init__()

        self.dropout = 0.5 #info["dropout"]
        num_layers = 3 #info['num_layers']
        hidden_channels = 256 #info['hidden_channels']
        in_channels = info["num_node_features"]
        out_channels = info["num_classes"]
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels,normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, *args, **kargs):
        x, adj_t = data.x, data.adj_t
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


class SAGE(torch.nn.Module):
    """
    No normalisation is needed according to
    https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/gnn.py
    """
    def __init__(self, info, *args, **kwargs):
        super(SAGE, self).__init__()

        dropout = 0.5 #info["dropout"]
        num_layers = 3 #info['num_layers']
        hidden_channels = 256 #info['hidden_channels']
        in_channels = info["num_node_features"]
        out_channels = info["num_classes"]

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, *args, **kargs):
        x, adj_t = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


class HyperSAGE(torch.nn.Module):
    """
    No normalisation is needed according to
    https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/gnn.py
    """
    def __init__(self, info, *args, **kwargs):
        super(HyperSAGE, self).__init__()

        dropout = 0.5 #info["dropout"]
        num_layers = 3 #info['num_layers']
        hidden_channels = 256 #info['hidden_channels']
        in_channels = info["num_node_features"]
        out_channels = info["num_classes"]

        self.convs = torch.nn.ModuleList()
        self.hyper_convs = torch.nn.ModuleList()

        self.hyper_convs.append(HypergraphConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers-2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        for _ in range(num_layers):
            self.hyper_convs.append(HypergraphConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.hyper_convs.append(HypergraphConv(hidden_channels, out_channels))

        self.lp = torch.nn.Linear(2*info["num_classes"],info["num_classes"])

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, *args, **kargs):
        x, adj_t, hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_simple = self.convs[-1](x, adj_t)
        x = data.x
        for conv in self.hyper_convs[:-1]:
            x = conv(x, hyperedge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.hyper_convs[-1](x, hyperedge_index)
        x = self.lp(torch.cat([x_simple.unsqueeze(0),x.unsqueeze(0)],dim=2)).squeeze(0)
        return torch.log_softmax(x, dim=-1)
