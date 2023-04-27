import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, HypergraphConv
"""
for baseline method:
# self.hyperGCN
# x2 = hyperGCN(x,hyperedge_index)
# x,x2
# potential 2 impelmentations
"""
class MultiHeadAttention(nn.Module):
    #TODO add dropout for attention and fc layer
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = torch.tensor(self.depth, dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)

        output = torch.matmul(attention_weights, v)
        return output, attention_weights

    def forward(self, q, k, v):
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)

        scaled_attention = scaled_attention.transpose(1, 2).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        output = self.fc(concat_attention)
        return output, attention_weights

class Attention(torch.nn.Module):
    # TODO: add layer norm and dropout; and add fc layer and residual connection
    def __init__(sef, dim):
        super().__init__()

    def forward(self,x,x_hyper):
        pass


class HybridGCN(torch.nn.Module):
    def __init__(
            self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.hyper1 = HypergraphConv(info["num_node_features"], dim)
        self.conv1 = GCNConv(info["num_node_features"], dim)
        self.attn1 = Attention() # TODO: cross attention between q = hyper1(x) and k = conv1(x)
        self.is_regression = info["is_regression"]
        self.is_edge_pred = info["is_edge_pred"]
        if info["is_regression"]:
            self.hyper2 = HypergraphConv(dim, dim)
            self.conv2 = GCNConv(dim, dim)
            self.attn2 = Attention(dim) # TODO: cross attention between q = hyper2(x) and k = conv2(x)
            self.head = nn.Linear(dim, 1)
        else:
            self.conv2 = GCNConv(dim, info["num_classes"])
            self.hyper2 = HypergraphConv(dim, info["num_classes"])
            self.attn2 = Attention(dim) # TODO: cross attention between q = hyper2(x) and k = conv2(x)
    def forward(self, data, *args, **kargs):
        x, edge_index,hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        if self.is_edge_pred:
            edge_index = args[0][0] # the message passing edge index
        x = F.relu(self.conv1(x, edge_index))
        x_hyper = F.relu(self.hyper1(x, hyperedge_index))
        x = self.attn1(x, x_hyper)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x_hyper = self.hyper2(x_hyper, hyperedge_index)
        x = self.attn2(x, x_hyper)

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


class SAGENet(torch.nn.Module):
    def __init__(
            self, info, mixture_cls=None, load_config=None, *args, **kwargs):
        super(SAGENet, self).__init__()
        dim = 16
        self.is_regression = info["is_regression"]
        self.is_edge_pred = info["is_edge_pred"]
        if self.is_regression:
            self.conv2 = SAGEConv(dim, dim, normalize=False)
            self.head = nn.Linear(dim, 1)
        else:
            self.conv2 = SAGEConv(dim, info["num_classes"], normalize=False)

        self.conv1 = SAGEConv(
            info["num_node_features"], dim, normalize=False)

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
