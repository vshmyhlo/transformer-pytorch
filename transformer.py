import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import attention


class Tranformer(nn.Module):
  # TODO: multihead attention
  def __init__(self, source_vocab_size, target_vocab_size, size, n_layers,
               n_heads, dropout, padding_idx):
    super().__init__()

    self.encoder = Encoder(source_vocab_size, size, n_layers, n_heads, dropout,
                           padding_idx)
    self.decoder = Decoder(target_vocab_size, size, n_layers, n_heads, dropout,
                           padding_idx)
    self.projection = nn.Linear(size, target_vocab_size)

  def forward(self, x, y_bottom):
    encoder_states = self.encoder(x)
    y_top = self.decoder(y_bottom, encoder_states)
    y_top = self.projection(y_top)
    return y_top


class Encoder(nn.Module):
  def __init__(self, num_embeddings, size, n_layers, n_heads, dropout,
               padding_idx):
    super().__init__()

    self.n_layers = n_layers
    self.embedding = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=size,
        padding_idx=padding_idx)
    self.positional_encoding = PositionalEncoding()
    self.dropout = nn.Dropout(dropout)
    self.encoder_layers = nn.ModuleList(
        [EncoderLayer(size, n_heads) for _ in range(self.n_layers)])

  def forward(self, x):
    x = self.embedding(x)
    x = self.positional_encoding(x)
    x = self.dropout(x)

    for layer in self.encoder_layers:
      x = layer(x)

    x /= self.n_layers

    return x


class Decoder(nn.Module):
  # TODO: check train() and eval() sets state to layers

  def __init__(self, num_embeddings, size, n_layers, n_heads, dropout,
               padding_idx):
    super().__init__()

    self.n_layers = n_layers
    self.embedding = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=size,
        padding_idx=padding_idx)
    self.positional_encoding = PositionalEncoding()
    self.dropout = nn.Dropout(dropout)
    self.decoder_layers = nn.ModuleList(
        [DecoderLayer(size, n_heads) for _ in range(self.n_layers)])

  def forward(self, y_bottom, states):
    y_bottom = self.embedding(y_bottom)
    y_bottom = self.positional_encoding(y_bottom)
    y_bottom = self.dropout(y_bottom)

    for layer in self.decoder_layers:
      y_bottom = layer(y_bottom, states)

    return y_bottom


class EncoderLayer(nn.Module):
  def __init__(self, size, n_heads):
    super().__init__()

    self.self_attention = SelfAttentionSublayer(size, n_heads)
    self.feed_forward = FeedForwardSublayer(size)

  def forward(self, x):
    x = self.self_attention(x)
    x = self.feed_forward(x)

    return x


class DecoderLayer(nn.Module):
  def __init__(self, size, n_heads):
    super().__init__()

    self.self_attention = SelfAttentionSublayer(size, n_heads)
    self.encoder_attention = AttentionSublayer(size, n_heads)
    self.feed_forward = FeedForwardSublayer(size)

  def forward(self, x, states):
    x = self.self_attention(x)
    x = self.encoder_attention(x, states)
    x = self.feed_forward(x)

    return x


class PositionalEncoding(nn.Module):
  def forward(self, x):
    size = x.size()

    pos = torch.arange(size[1]).unsqueeze(0).unsqueeze(-1)
    dim = torch.arange(size[2]).unsqueeze(0).unsqueeze(0)
    # TODO: find good multiplier
    # encoding = torch.sin(pos / 10000**(2 * dim / size[-1]))
    encoding = torch.sin(pos / 10000**(1 * dim / size[-1]))
    # encoding = torch.sin(pos / 10000**(0.75 * dim / size[-1]))

    # import matplotlib.pyplot as plt
    # plt.plot(encoding[0, :, 0].numpy())
    # plt.plot(encoding[0, :, 15].numpy())
    # plt.plot(encoding[0, :, -15].numpy())
    # plt.plot(encoding[0, :, -1].numpy())
    # plt.show()
    # plt.plot(encoding[0, 0, :].numpy())
    # plt.plot(encoding[0, 40, :].numpy())
    # plt.plot(encoding[0, -40, :].numpy())
    # plt.plot(encoding[0, -1, :].numpy())
    # plt.show()
    # fail

    encoding = Variable(encoding)

    return x + encoding


class AttentionSublayer(nn.Module):
  def __init__(self, size, n_heads):
    super().__init__()

    self.attention = attention.MultiHeadAttention(size, n_heads)
    self.layer_norm = LayerNorm(size)

  def forward(self, x, states):
    saved = x

    x = self.attention(x, states)
    x = self.layer_norm(saved + x)

    return x


class SelfAttentionSublayer(AttentionSublayer):
  def forward(self, x):
    return super().forward(x, x)


class FeedForwardSublayer(nn.Module):
  def __init__(self, size):
    super().__init__()

    # TODO: check for bias
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
  # TODO: check if this is correct
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


# def loss(y_top, y):
#   # TODO: check y_top.size(-1) == dataset.vocab_size (73)
#   mask = torch.ones(y_top.size(-1)).index_add_(
#       0,
#       torch.LongTensor([dataset.pad]),
#       torch.FloatTensor([-1]),
#   )
#
#   loss = F.cross_entropy(
#       y_top.view(-1, dataset.vocab_size), y.contiguous().view(-1), weight=mask)
#
#   return loss


def loss(y_top, y):
  loss = F.cross_entropy(
      y_top.view(-1, y_top.size(-1)), y.contiguous().view(-1), reduce=False)
  loss = loss.view(y.size())
  mask = (y != 0).float()
  loss = loss * mask
  loss = loss.sum(1) / mask.sum(1)
  loss = loss.mean()

  return loss
