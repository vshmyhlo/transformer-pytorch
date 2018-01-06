import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
  # TODO: vectorize this

  def __init__(self, size, n_heads):
    super().__init__()

    self.attentions = nn.ModuleList([Attention(size) for _ in range(n_heads)])
    self.projection = nn.Linear(size * n_heads, size)

  def forward(self, x, states):
    xs = [attention(x, states) for attention in self.attentions]
    x = torch.cat(xs, -1)
    x = self.projection(x)
    return x


class Attention(nn.Module):
  def __init__(self, size):
    super().__init__()

    self.ql = nn.Linear(size, size, bias=False)
    self.kl = nn.Linear(size, size, bias=False)
    self.vl = nn.Linear(size, size, bias=False)
    self.attention = LuongAttention(size)

  def forward(self, x, states):
    q = self.ql(x)
    k = self.kl(states)
    v = self.vl(states)

    scores = self.attention(q, k)
    scores = scores.unsqueeze(-1)
    v = v.unsqueeze(-3)
    attended = v * scores
    context = attended.sum(-2)

    return context


class LuongAttention(nn.Module):
  # TODO: check if this is correct

  def __init__(self, size):
    super().__init__()
    self.fc = nn.Linear(size, size, bias=False)

  def forward(self, q, k):
    wk = self.fc(k)
    wk = wk.transpose(2, 1)
    scores = torch.bmm(q, wk)
    scores = F.softmax(scores, -1)

    return scores
