import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import attention


# TODO: init 0.4

class AttentionSublayer(nn.Module):
    def __init__(self, size, n_heads, dropout):
        super().__init__()

        self.attention = attention.MultiHeadAttention(size, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, x, states, mask=None):
        identity = x

        x = self.attention(x, states, mask)
        x = self.dropout(x)
        x = self.layer_norm(identity + x)

        return x


class SelfAttentionSublayer(AttentionSublayer):
    def forward(self, input, mask=None):
        return super().forward(input, input, mask)


class FeedForwardSublayer(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()

        self.linear_1 = nn.Linear(size, size * 4)
        self.relu = nn.ReLU(inplace=True)
        self.linear_2 = nn.Linear(4 * size, size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(size)

        init.xavier_normal_(self.linear_1.weight)
        init.xavier_normal_(self.linear_2.weight)

    def forward(self, input):
        identity = input

        input = self.linear_1(input)
        input = self.relu(input)
        input = self.linear_2(input)
        input = self.dropout(input)
        input = self.layer_norm(identity + input)

        return input
