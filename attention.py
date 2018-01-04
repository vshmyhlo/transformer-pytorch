import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
  def __init__(self, size):
    super().__init__()

    self.ql = nn.Linear(size, size)
    self.kl = nn.Linear(size, size)
    self.vl = nn.Linear(size, size)
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
    self.fc = nn.Linear(size, size)

  def forward(self, q, k):
    wk = self.fc(k)
    wk = wk.transpose(2, 1)
    scores = torch.bmm(q, wk)
    scores = F.softmax(scores, -1)

    return scores
