import torch
import torch.nn as nn
import torch.nn.functional as F


class Tranformer(nn.Module):
  def __init__(self, num_source_embeddings, num_target_embeddings, size):
    super().__init__()

    self.encoder = Encoder(num_source_embeddings, size)
    self.decoder = Decoder(num_target_embeddings, size)

  def forward(self, x, y_bottom):
    encoder_states = self.encoder(x)
    y_top = self.decoder(encoder_states, y_bottom)
    return y_top


class Encoder(nn.Module):
  def __init__(self, num_embeddings, size):
    super().__init__()

    self.embedding = nn.Embedding(
        num_embeddings=num_embeddings, embedding_dim=size)
    self.el1 = EncoderLayer(size)
    self.el2 = EncoderLayer(size)

  def forward(self, x):
    x = self.embedding(x)
    x = self.el1(x)
    x = self.el2(x)

    return x


class Decoder(nn.Module):
  def __init__(self, num_embeddings, size):
    super().__init__()

    self.embedding = nn.Embedding(
        num_embeddings=num_embeddings, embedding_dim=size)
    self.dl1 = DecoderLayer(size)
    self.dl2 = DecoderLayer(size)

  def forward(self, states, y_bottom):
    y_bottom = self.embedding(y_bottom)
    y_bottom = self.dl1(y_bottom)
    y_bottom = self.dl2(y_bottom)

    return y_bottom


class EncoderLayer(nn.Module):
  def __init__(self, size):
    super().__init__()

    self.self_attention = SelfAttention(size)

  def forward(self, x):
    x = self.self_attention(x)
    return x


class DecoderLayer(nn.Module):
  def __init__(self, size):
    super().__init__()

    self.self_attention = SelfAttention(size)

  def forward(self, x):
    x = self.self_attention(x)
    return x


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
    v = v.unsqueeze(-2)
    attended = v * scores
    context = attended.sum(-2)

    return context


class SelfAttention(Attention):
  def forward(self, x):
    return super().forward(x, x)


class LuongAttention(nn.Module):
  # TODO: check this is correct

  def __init__(self, size):
    super().__init__()

    self.fc = nn.Linear(size, size)

  def forward(self, q, k):
    wk = self.fc(k)
    wk = wk.transpose(2, 1)
    scores = torch.bmm(q, wk)
    scores = F.softmax(scores, -1)

    return scores
