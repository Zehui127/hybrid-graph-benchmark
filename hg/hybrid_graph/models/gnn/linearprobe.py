import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, HypergraphConv, GAT


class LPGCNHyperGCN(torch.nn.Module):
    def __init__(
            self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.hyper1 = HypergraphConv(info["num_node_features"], dim)
        self.conv1 = GCNConv(info["num_node_features"], dim)
        self.is_regression = info["is_regression"]
        if info["is_regression"]:
            self.hyper2 = HypergraphConv(dim, dim)
            self.conv2 = GCNConv(dim, dim)
            self.lp = nn.Linear(2*dim,dim) # TODO: cross attention between q = hyper2(x) and k = conv2(x)
            self.head = nn.Linear(dim, 1)
        else:
            self.conv2 = GCNConv(dim, info["num_classes"])
            self.hyper2 = HypergraphConv(dim, info["num_classes"])
            self.lp = nn.Linear(2*info["num_classes"],info["num_classes"])
    def forward(self, data, *args, **kargs):
        x, edge_index,hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        x_gcn = F.relu(self.conv1(x, edge_index))
        x_hyper = F.relu(self.hyper1(x, hyperedge_index))
        x_gcn = F.dropout(x_gcn, training=self.training)
        x_hyper = F.dropout(x_hyper, training=self.training)
        x_gcn = self.conv2(x_gcn, edge_index)
        x_hyper = self.hyper2(x_hyper, hyperedge_index)
        x = self.lp(torch.cat([x_gcn.unsqueeze(0),x_hyper.unsqueeze(0)],dim=2)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        else:
            x = F.log_softmax(x, dim=1)
        return x
    

class LPGATHyperGCN(torch.nn.Module):
    def __init__(
            self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.hyper1 = HypergraphConv(info["num_node_features"], dim)
        self.conv1 = GAT(info["num_node_features"], dim,num_layers=1)
        self.is_regression = info["is_regression"]
        if info["is_regression"]:
            self.hyper2 = HypergraphConv(dim, dim)
            self.conv2 =GAT(dim, dim,num_layers=1)
            self.lp = nn.Linear(2*dim,dim) # TODO: cross attention between q = hyper2(x) and k = conv2(x)
            self.head = nn.Linear(dim, 1)
        else:
            self.conv2 = GCNConv(dim, info["num_classes"])
            self.hyper2 = GAT(dim, info["num_classes"],num_layers=1)
            self.lp = nn.Linear(2*info["num_classes"],info["num_classes"])
    def forward(self, data, *args, **kargs):
        x, edge_index,hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        x_gcn = F.relu(self.conv1(x, edge_index))
        x_hyper = F.relu(self.hyper1(x, hyperedge_index))
        x_gcn = F.dropout(x_gcn, training=self.training)
        x_hyper = F.dropout(x_hyper, training=self.training)
        x_gcn = self.conv2(x_gcn, edge_index)
        x_hyper = self.hyper2(x_hyper, hyperedge_index)
        x = self.lp(torch.cat([x_gcn.unsqueeze(0),x_hyper.unsqueeze(0)],dim=2)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        else:
            x = F.log_softmax(x, dim=1)
        return x


class LPGGATGCN(torch.nn.Module):
    def __init__(
            self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.hyper1 = GAT(info["num_node_features"], dim,num_layers=1)
        self.conv1 = GCNConv(info["num_node_features"], dim)
        self.is_regression = info["is_regression"]
        if info["is_regression"]:
            self.hyper2 = GAT(dim, dim,num_layers=1 )
            self.conv2 = GCNConv(dim, dim)
            self.lp = nn.Linear(2*dim,dim) # TODO: cross attention between q = hyper2(x) and k = conv2(x)
            self.head = nn.Linear(dim, 1)
        else:
            self.conv2 = GCNConv(dim, info["num_classes"])
            self.hyper2 = GAT(dim, info["num_classes"],num_layers=1 )
            self.lp = nn.Linear(2*info["num_classes"],info["num_classes"])
    def forward(self, data, *args, **kargs):
        x, edge_index,hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        x_gcn = F.relu(self.conv1(x, edge_index))
        x_hyper = F.relu(self.hyper1(x, edge_index))
        x_gcn = F.dropout(x_gcn, training=self.training)
        x_hyper = F.dropout(x_hyper, training=self.training)
        x_gcn = self.conv2(x_gcn, edge_index)
        x_hyper = self.hyper2(x_hyper, edge_index)
        x = self.lp(torch.cat([x_gcn.unsqueeze(0),x_hyper.unsqueeze(0)],dim=2)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        else:
            x = F.log_softmax(x, dim=1)
        return x
    

class LPGGATGAT(torch.nn.Module):
    def __init__(
            self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.hyper1 = GAT(info["num_node_features"], dim,num_layers=1)
        self.conv1 =  GAT(info["num_node_features"], dim,num_layers=1)
        self.is_regression = info["is_regression"]
        if info["is_regression"]:
            self.hyper2 = GAT(dim, dim,num_layers=1 )
            self.conv2 =  GAT(dim, dim,num_layers=1 )
            self.lp = nn.Linear(2*dim,dim) # TODO: cross attention between q = hyper2(x) and k = conv2(x)
            self.head = nn.Linear(dim, 1)
        else:
            self.conv2 = GAT(dim, info["num_classes"],num_layers=1 )
            self.hyper2 = GAT(dim, info["num_classes"],num_layers=1 )
            self.lp = nn.Linear(2*info["num_classes"],info["num_classes"])
    def forward(self, data, *args, **kargs):
        x, edge_index,hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        x_gcn = F.relu(self.conv1(x, edge_index))
        x_hyper = F.relu(self.hyper1(x, edge_index))
        x_gcn = F.dropout(x_gcn, training=self.training)
        x_hyper = F.dropout(x_hyper, training=self.training)
        x_gcn = self.conv2(x_gcn, edge_index)
        x_hyper = self.hyper2(x_hyper, edge_index)
        x = self.lp(torch.cat([x_gcn.unsqueeze(0),x_hyper.unsqueeze(0)],dim=2)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        else:
            x = F.log_softmax(x, dim=1)
        return x
    

class LPGCNGCN(torch.nn.Module):
    def __init__(
            self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.hyper1 = GCNConv(info["num_node_features"], dim)
        self.conv1 = GCNConv(info["num_node_features"], dim)
        self.is_regression = info["is_regression"]
        if info["is_regression"]:
            self.hyper2 = GCNConv(dim, dim)
            self.conv2 =  GCNConv(dim, dim)
            self.lp = nn.Linear(2*dim,dim) # TODO: cross attention between q = hyper2(x) and k = conv2(x)
            self.head = nn.Linear(dim, 1)
        else:
            self.hyper2 = GCNConv(dim, info["num_classes"])
            self.conv2 = GCNConv(dim, info["num_classes"])
            self.lp = nn.Linear(2*info["num_classes"],info["num_classes"])
    def forward(self, data, *args, **kargs):
        x, edge_index,hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        x_gcn = F.relu(self.conv1(x, edge_index))
        x_hyper = F.relu(self.hyper1(x, edge_index))
        x_gcn = F.dropout(x_gcn, training=self.training)
        x_hyper = F.dropout(x_hyper, training=self.training)
        x_gcn = self.conv2(x_gcn, edge_index)
        x_hyper = self.hyper2(x_hyper, edge_index)
        x = self.lp(torch.cat([x_gcn.unsqueeze(0),x_hyper.unsqueeze(0)],dim=2)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        else:
            x = F.log_softmax(x, dim=1)
        return x
    

class LPHYPERHYPER(torch.nn.Module):
    def __init__(
            self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.hyper1 = HypergraphConv(info["num_node_features"], dim)
        self.conv1 = HypergraphConv(info["num_node_features"], dim)
        self.is_regression = info["is_regression"]
        if info["is_regression"]:
            self.hyper2 = HypergraphConv(dim, dim)
            self.conv2 =HypergraphConv(dim, dim)
            self.lp = nn.Linear(2*dim,dim) # TODO: cross attention between q = hyper2(x) and k = conv2(x)
            self.head = nn.Linear(dim, 1)
        else:
            self.conv2 = HypergraphConv(dim, info["num_classes"])
            self.hyper2 = HypergraphConv(dim, info["num_classes"])
            self.lp = nn.Linear(2*info["num_classes"],info["num_classes"])
    def forward(self, data, *args, **kargs):
        x, edge_index,hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        x_gcn = F.relu(self.conv1(x, hyperedge_index))
        x_hyper = F.relu(self.hyper1(x, hyperedge_index))
        x_gcn = F.dropout(x_gcn, training=self.training)
        x_hyper = F.dropout(x_hyper, training=self.training)
        x_gcn = self.conv2(x_gcn, hyperedge_index)
        x_hyper = self.hyper2(x_hyper, hyperedge_index)
        x = self.lp(torch.cat([x_gcn.unsqueeze(0),x_hyper.unsqueeze(0)],dim=2)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        else:
            x = F.log_softmax(x, dim=1)
        return x
