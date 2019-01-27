import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# TODO: refactor attention (no luong, no bahd)

# TODO: vectorize this
# TODO: should project after concatenation?
class MultiHeadAttention(nn.Module):
    def __init__(self, size, n_heads):
        super().__init__()

        self.attentions = nn.ModuleList([Attention(size) for _ in range(n_heads)])
        self.projection = nn.Linear(size * n_heads, size)

        init.xavier_normal_(self.projection.weight)

    def forward(self, x, states, mask):
        xs = [attention(x, states, mask) for attention in self.attentions]
        x = torch.cat(xs, -1)
        x = self.projection(x)

        return x


# TODO: init
class Attention(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.ql = nn.Linear(size, size)
        self.kl = nn.Linear(size, size)
        self.vl = nn.Linear(size, size)

        init.xavier_normal_(self.ql.weight)
        init.xavier_normal_(self.kl.weight)
        init.xavier_normal_(self.vl.weight)

    def forward(self, input, states, mask):
        q = self.ql(input)
        k = self.kl(states)
        v = self.vl(states)

        # TODO: check shapes
        # TODO: scores to weights
        assert q.size(-1) == k.size(-1)
        scores = torch.bmm(q, k.transpose(2, 1)) / math.sqrt(k.size(-1))
        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))

        # print()
        scores = scores.unsqueeze(3)
        v = v.unsqueeze(1)
        # print(scores.size())
        # print(v.size())
        scores = scores.softmax(2)
        context = (v * scores).sum(2)
        # print(context.size())

        return context
