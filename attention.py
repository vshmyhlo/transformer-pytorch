import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# TODO: vectorize this
class MultiHeadAttention(nn.Module):
    def __init__(self, size, n_heads, attention_type):
        super().__init__()

        self.attentions = nn.ModuleList([
            Attention(size, attention_type=attention_type) for _ in range(n_heads)
        ])
        self.projection = nn.Linear(size * n_heads, size, bias=False)

        init.xavier_normal(self.projection.weight)

    def forward(self, x, states, mask):
        xs = [attention(x, states, mask) for attention in self.attentions]
        x = torch.cat(xs, -1)
        x = self.projection(x)
        return x


class Attention(nn.Module):
    def __init__(self, size, attention_type):
        super().__init__()

        self.ql = nn.Linear(size, size, bias=False)
        self.kl = nn.Linear(size, size, bias=False)
        self.vl = nn.Linear(size, size, bias=False)

        if attention_type == 'luong':
            self.attention = LuongAttention(size)
        elif attention_type == 'scaled_dot_product':
            self.attention = ScaledDotProductAttention()

        init.xavier_normal(self.ql.weight)
        init.xavier_normal(self.kl.weight)
        init.xavier_normal(self.vl.weight)

    def forward(self, x, states, mask):
        q = self.ql(x)
        k = self.kl(states)
        v = self.vl(states)

        scores = self.attention(q, k, mask)
        scores = scores.unsqueeze(-1)
        v = v.unsqueeze(-3)
        attended = v * scores
        context = attended.sum(-2)

        return context


class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, mask):
        assert q.size(-1) == k.size(-1)
        scores = torch.bmm(q, k.transpose(2, 1)) / k.size(-1)**0.5
        if mask is not None:
            # TODO: mask variable or its data?
            scores.masked_fill_(mask == 0, float('-inf'))
        scores = F.softmax(scores, -1)

        return scores


class LuongAttention(nn.Module):
    # TODO: check everything is correct
    def __init__(self, size):
        super().__init__()
        self.fc = nn.Linear(size, size, bias=False)

        init.xavier_normal(self.fc.weight)

    def forward(self, q, k, mask):
        wk = self.fc(k)
        wk = wk.transpose(2, 1)
        scores = torch.bmm(q, wk)
        if mask is not None:
            # TODO: mask variable or its data?
            scores.masked_fill_(mask == 0, float('-inf'))
        scores = F.softmax(scores, -1)

        return scores
