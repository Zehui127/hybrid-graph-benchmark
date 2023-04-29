import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import HypergraphConv

"""
for baseline method:
# self.hyperGCN
# x2 = hyperGCN(x,hyperedge_index)
# x,x2
# potential 2 impelmentations
"""
class HyperGCN(torch.nn.Module):
    def __init__(
            self, info, *args, **kwargs):
        super().__init__()
        dim = 32
        self.conv1 = HypergraphConv(info["num_node_features"], dim)
        self.is_regression = info["is_regression"]
        self.is_edge_pred = info["is_edge_pred"]
        if info["is_regression"]:
            self.conv2 = HypergraphConv(dim, dim, )
            self.head = nn.Linear(dim, 1)
        else:
            self.conv2 = HypergraphConv(dim, info["num_classes"])

    def forward(self, data, *args, **kargs):
        x, edge_index = data.x, data.hyperedge_index
        if self.is_edge_pred:
            edge_index = args[0][0] # the message passing edge index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)


        if self.is_regression:
            x = self.head(x).squeeze()
        elif self.is_edge_pred:
            # args[1] is the edge_label_index
            edge_label_index = args[0][1]
            x = x[edge_label_index[0]]*x[edge_label_index[1]]
            x = torch.sum(x,-1)
        else:
            x = F.log_softmax(x, dim=1)
        return x


class HyperGAT(torch.nn.Module):
    def __init__(
            self, info, mixture_cls=None, load_config=None, *args, **kwargs):
        super().__init__()
        dim = 32
        self.is_regression = info["is_regression"]
        self.is_edge_pred = info["is_edge_pred"]
        if self.is_regression:
            self.conv2 = HypergraphConv(dim, dim, use_attention=True)
            self.head = nn.Linear(dim, 1)
        else:
            self.conv2 = HypergraphConv(dim, info["num_classes"], use_attention=True)

        self.conv1 = HypergraphConv(
            info["num_node_features"], dim, use_attention=True)

    def forward(self, data, *args, **kargs):
        x, edge_index = data.x, data.hyperedge_index
        if self.is_edge_pred:
            edge_index = args[0][0] # the message passing edge index
        x = F.relu(
            self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        if self.is_regression:
            x = self.head(x).squeeze()
        elif self.is_edge_pred:
            # args[1] is the edge_label_index
            edge_label_index = args[0][1]
            x = x[edge_label_index[0]]*x[edge_label_index[1]]
            x = torch.sum(x,-1)
        else:
            x = F.log_softmax(x, dim=1)
        return x
