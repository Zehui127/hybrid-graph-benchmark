import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ToyNet(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        super(ToyNet, self).__init__()
        self.linear = torch.nn.Linear(info['num_node_features'], 16)
        self.conv1 = GCNConv(16, 16)
        self.post_conv = torch.nn.Linear(16, info['num_classes'])

    def forward(self, data, *args, **kargs):
        x, edge_index = data.x, data.edge_index
        x = self.linear(x)
        x = F.relu(x)
        x = F.dropout(x ,p=0.7,training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x ,p=0.7,training=self.training)
        x = self.post_conv(x)
        logit = F.log_softmax(x, dim=1)
        return logit
