import torch
import torch.nn as nn
import torch.nn.functional as F
import attention


class AttentionSublayer(nn.Module):
  def __init__(self, size, n_heads):
    super().__init__()

    self.attention = attention.MultiHeadAttention(size, n_heads)
    self.layer_norm = LayerNorm(size)

  def forward(self, x, states, mask=None):
    saved = x

    x = self.attention(x, states, mask)
    x = self.layer_norm(saved + x)

    return x


class SelfAttentionSublayer(AttentionSublayer):
  def forward(self, x, mask=None):
    return super().forward(x, x, mask)


class FeedForwardSublayer(nn.Module):
  def __init__(self, size):
    super().__init__()

    # TODO: check if bias needed
    self.fc1 = nn.Linear(size, size * 4)
    self.fc2 = nn.Linear(4 * size, size)
    self.layer_norm = LayerNorm(size)

  def forward(self, x):
    saved = x

    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = self.layer_norm(saved + x)

    return x


class LayerNorm(nn.Module):
  # TODO: train and test states

  def __init__(self, size, eps=1e-6):
    super().__init__()
    self.gamma = nn.Parameter(torch.ones(size).unsqueeze(0).unsqueeze(0))
    self.beta = nn.Parameter(torch.zeros(size).unsqueeze(0).unsqueeze(0))
    self.eps = eps

  def forward(self, x):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)

    return self.gamma * (x - mean) / (std + self.eps) + self.beta
