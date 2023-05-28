import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GAT

"""
for baseline method:
# self.hyperGCN
# x2 = hyperGCN(x,hyperedge_index)
# x,x2
# potential 2 impelmentations
"""
class GATNet(torch.nn.Module):
    def __init__(
            self, info, *args, **kwargs):
        super().__init__()
        dim = 32
        self.conv1 = GAT(info["num_node_features"], dim, num_layers=1)
        self.is_regression = info["is_regression"]
        self.is_edge_pred = info["is_edge_pred"]
        if info["is_regression"]:
            self.conv2 = GAT(dim, dim,num_layers=1 )
            self.head = nn.Linear(dim, 1)
        elif info["is_edge_pred"]:
            self.conv2 = GAT(dim, dim,num_layers=1 )
        else:
            self.conv2 = GAT(dim, info["num_classes"], num_layers=1)

    def forward(self, data, *args, **kargs):
        x, edge_index = data.x, data.edge_index
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


class GATV2Net(torch.nn.Module):
    def __init__(
            self, info, mixture_cls=None, load_config=None, *args, **kwargs):
        super().__init__()
        dim = 32
        self.is_regression = info["is_regression"]
        self.is_edge_pred = info["is_edge_pred"]
        if self.is_regression:
            self.conv2 = GAT(dim, dim, v2=True,num_layers=1)
            self.head = nn.Linear(dim, 1)
        elif info["is_edge_pred"]:
            self.conv2 = GAT(dim, dim, v2=True,num_layers=1)
        else:
            self.conv2 = GAT(dim, info["num_classes"], v2=True,num_layers=1)

        self.conv1 = GAT(
            info["num_node_features"], dim, v2=True, num_layers=1)

    def forward(self, data, *args, **kargs):
        x, edge_index = data.x, data.edge_index
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
