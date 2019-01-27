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
    def forward(self, x, mask=None):
        return super().forward(x, x, mask)


class FeedForwardSublayer(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()

        self.fc1 = nn.Linear(size, size * 4)
        self.fc2 = nn.Linear(4 * size, size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(size)

        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        identity = x

        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.layer_norm(identity + x)

        return x

# # TODO:
# # TODO: check states are switched between train and eval
# class LayerNorm(nn.Module):
#     def __init__(self, size, eps=1e-6):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.ones(size).unsqueeze(0).unsqueeze(0))
#         self.beta = nn.Parameter(torch.zeros(size).unsqueeze(0).unsqueeze(0))
#         self.eps = eps
#
#     def forward(self, x):
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#
#         return self.gamma * (x - mean) / (std + self.eps) + self.beta
