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
        self.conv2 = GCNConv(16, info["num_classes"])

    def forward(self, data, *args, **kargs):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class SAGENet(torch.nn.Module):
    def __init__(
            self, info, mixture_cls=None, load_config=None, *args, **kwargs):
        super(SAGENet, self).__init__()
        self.conv1 = SAGEConv(
            info["num_node_features"], 16, normalize=False)
        self.conv2 = SAGEConv(
            16, info["num_classes"], normalize=False)

    def forward(self, data, *args, **kargs):
        x, edge_index = data.x, data.edge_index
        x = F.relu(
            self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
