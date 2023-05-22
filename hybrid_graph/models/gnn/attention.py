import math
import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import trunc_normal_


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args):
        return self.fn(x, *args) + x


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
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Sequential):
            for submodule in m.children():
                self._init_weights(submodule)
        elif isinstance(m, nn.ModuleList):
            for submodule in m:
                self._init_weights(submodule)

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
        self.mha = MultiHeadAttention(dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm_hyper = nn.LayerNorm(dim)
    def forward(self, x, x_hyper):
        x = self.norm(x)
        x_hyper = self.norm_hyper(x_hyper)
        #TODO: potential other design choices e.g. mha(x, x_hyper, x_hyper)
        attn_output, _ = self.mha(x_hyper, x, x)
        return self.dropout(attn_output)

class FeedForwardLayer(nn.Module):
    def __init__(self, dim, dropout_rate):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim * 2, dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Sequential):
            for submodule in m.children():
                self._init_weights(submodule)
        elif isinstance(m, nn.ModuleList):
            for submodule in m:
                self._init_weights(submodule)

    def forward(self, x):
        x_norm = self.norm(x)
        out = self.linear1(x_norm)
        out = self.dropout1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return self.dropout2(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, head, attn_dropout, dropout_rate):
        super().__init__()
        self.attention = Residual(AttentionLayer(dim, head, attn_dropout, dropout_rate))
        self.feedforward = Residual(FeedForwardLayer(dim, dropout_rate))

    def forward(self, x, x_hyper):
        x = self.attention(x, x_hyper)
        x = self.feedforward(x)
        return x


class Attention(torch.nn.Module):
    def __init__(self, dim, head=8, dropout_rate=0.5, attn_dropout=0.5, depth=1):
        super().__init__()
        self.transformer = nn.Sequential(*[TransformerBlock(dim, head, attn_dropout, dropout_rate) for _ in range(depth)])

    def forward(self, x, x_hyper):
        for layer in self.transformer:
            x = layer(x, x_hyper)
        return x
