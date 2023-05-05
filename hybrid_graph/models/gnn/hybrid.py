import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, HypergraphConv
from .attention import Attention


class HybridGCN(torch.nn.Module):
    def __init__(
            self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.hyper1 = HypergraphConv(info["num_node_features"], dim)
        self.conv1 = GCNConv(info["num_node_features"], dim)
        self.attn1 = Attention(dim) # TODO: cross attention between q = hyper1(x) and k = conv1(x)
        self.is_regression = info["is_regression"]
        self.is_edge_pred = info["is_edge_pred"]
        if info["is_regression"]:
            self.hyper2 = HypergraphConv(dim, dim)
            self.conv2 = GCNConv(dim, dim)
            self.attn2 = Attention(dim) # TODO: cross attention between q = hyper2(x) and k = conv2(x)
            self.head = nn.Linear(dim, 1)
        elif info["is_edge_pred"]:
            self.hyper2 = HypergraphConv(dim, dim)
            self.conv2 = GCNConv(dim, dim)
            self.attn2 = Attention(dim) # TODO: cross attention between q = hyper2(x) and k = conv2(x)
        else:
            self.conv2 = GCNConv(dim, info["num_classes"])
            self.hyper2 = HypergraphConv(dim, info["num_classes"])
            self.attn2 = Attention(info["num_classes"],head=1) # TODO: cross attention between q = hyper2(x) and k = conv2(x)
    def forward(self, data, *args, **kargs):
        x, edge_index,hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        if self.is_edge_pred:
            edge_index = args[0][0] # the message passing edge index
        x_gcn = F.relu(self.conv1(x, edge_index))
        #print(f"!!!!!!x shape is:{x.shape}")
        x_hyper = F.relu(self.hyper1(x, hyperedge_index))
        x = self.attn1(x_gcn.unsqueeze(0), x_hyper.unsqueeze(0)).squeeze(0)
        x = F.dropout(x, training=self.training)
        x_gcn = self.conv2(x, edge_index)
        x_hyper = self.hyper2(x_hyper, hyperedge_index)
        x = self.attn2(x_gcn.unsqueeze(0), x_hyper.unsqueeze(0)).squeeze(0)

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

class HybridSAGE(torch.nn.Module):
    def __init__(
            self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.hyper1 = HypergraphConv(info["num_node_features"], dim)
        self.conv1 = SAGEConv(info["num_node_features"], dim, normalize=False)
        self.attn1 = Attention(dim) # TODO: cross attention between q = hyper1(x) and k = conv1(x)
        self.is_regression = info["is_regression"]
        self.is_edge_pred = info["is_edge_pred"]
        if info["is_regression"]:
            self.hyper2 = HypergraphConv(dim, dim)
            self.conv2 = SAGEConv(dim, dim, normalize=False)
            self.attn2 = Attention(dim) # TODO: cross attention between q = hyper2(x) and k = conv2(x)
            self.head = nn.Linear(dim, 1)
        elif info["is_edge_pred"]:
            self.hyper2 = HypergraphConv(dim, dim)
            self.conv2 = SAGEConv(dim, dim, normalize=False)
            self.attn2 = Attention(dim) # TODO: cross attention between q = hyper2(x) and k = conv2(x)
        else:
            self.conv2 = SAGEConv(dim, info["num_classes"], normalize=False)
            self.hyper2 = HypergraphConv(dim, info["num_classes"])
            self.attn2 = Attention(info["num_classes"],head=1) # TODO: cross attention between q = hyper2(x) and k = conv2(x)
    def forward(self, data, *args, **kargs):
        x, edge_index,hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        if self.is_edge_pred:
            edge_index = args[0][0] # the message passing edge index
        x_sage = F.relu(self.conv1(x, edge_index))
        x_hyper = F.relu(self.hyper1(x, hyperedge_index))
        x = self.attn1(x_sage.unsqueeze(0), x_hyper.unsqueeze(0)).squeeze(0)
        x = F.dropout(x, training=self.training)
        x_sage = self.conv2(x, edge_index)
        x_hyper = self.hyper2(x_hyper, hyperedge_index)
        x = self.attn2(x_sage.unsqueeze(0), x_hyper.unsqueeze(0)).squeeze(0)

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
