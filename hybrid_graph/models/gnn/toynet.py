import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ToyNet(torch.nn.Module):
    def __init__(self, info):
        super(ToyNet, self).__init__()
        self.conv1 = GCNConv(info['num_node_features'], 16)
        self.conv2 = GCNConv(16, info['num_classes'])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        logit = F.log_softmax(x, dim=1)
        return logit
