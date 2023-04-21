import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv


class GCNNet(torch.nn.Module):
    def __init__(
            self, info, *args, **kwargs):
        super().__init__()
        self.conv1 = GCNConv(info["num_node_features"], 16)
        self.is_regression = info["is_regression"]
        if info["is_regression"]:
            self.conv2 = GCNConv(16, 16)
            self.head = nn.Linear(16, 1)
        else:
            self.conv2 = GCNConv(16, info["num_classes"])
    def forward(self, data, *args, **kargs):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        if self.is_regression:
            x = self.head(x).squeeze()
        else:
            x = F.log_softmax(x, dim=1)
        return x


class SAGENet(torch.nn.Module):
    def __init__(
            self, info, mixture_cls=None, load_config=None, *args, **kwargs):
        super(SAGENet, self).__init__()
        self.is_regression = info["is_regression"]
        if self.is_regression:
            self.conv2 = SAGEConv(16, 16, normalize=False)
            self.head = nn.Linear(16, 1)
        else:
            self.conv2 = SAGEConv(16, info["num_classes"], normalize=False)

        self.conv1 = SAGEConv(
            info["num_node_features"], 16, normalize=False)

    def forward(self, data, *args, **kargs):
        x, edge_index = data.x, data.edge_index
        x = F.relu(
            self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        if self.is_regression:
            x = self.head(x).squeeze()
        else:
            x = F.log_softmax(x, dim=1)
        return x
