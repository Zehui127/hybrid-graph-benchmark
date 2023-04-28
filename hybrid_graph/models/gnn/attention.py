import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(attn_dropout)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = torch.tensor(self.depth, dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
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

class AttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, attn_dropout, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mha = MultiHeadAttention(dim, num_heads=num_heads, dropout_rate=attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm_hyper = nn.LayerNorm(dim)
    def forward(self, x, x_hyper):
        x = self.norm(x)
        x_hyper = self.norm_hyper(x_hyper)
        #TODO: potential other design choices e.g. mha(x, x_hyper, x_hyper)
        attn_output, _ = self.mha(x, x_hyper, x)
        return self.dropout(attn_output)


class Attention(torch.nn.Module):
    def __init__(self, dim, head, dropout_rate=0.2, attn_dropout=0.2,depth=1):
        super().__init__()
        transformer = []
        for _ in range(depth):
            transformer.append(
                Residual(AttentionLayer(dim, head, attn_dropout, dropout_rate)),
                Residual(
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, dim * 2),
                        nn.Dropout(dropout_rate),
                        nn.ReLU(),
                        nn.Linear(dim * 2, dim),
                        nn.Dropout(dropout_rate),
                    )
                ),
            )
        self.transformer = nn.Sequential(*transformer)

    def forward(self,x,x_hyper):
        return self.transformer(x,x_hyper)
