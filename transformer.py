import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sublayers


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
    self.positional_encoding = PositionalEncoding(size)
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
    self.positional_encoding = PositionalEncoding(size)
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

    self.self_attention = sublayers.SelfAttentionSublayer(size, n_heads)
    self.feed_forward = sublayers.FeedForwardSublayer(size)

  def forward(self, x):
    x = self.self_attention(x)
    x = self.feed_forward(x)

    return x


class DecoderLayer(nn.Module):
  def __init__(self, size, n_heads):
    super().__init__()

    self.self_attention = sublayers.SelfAttentionSublayer(size, n_heads)
    self.encoder_attention = sublayers.AttentionSublayer(size, n_heads)
    self.feed_forward = sublayers.FeedForwardSublayer(size)

  def forward(self, x, states):
    x = self.self_attention(x)
    x = self.encoder_attention(x, states)
    x = self.feed_forward(x)

    return x


class PositionalEncoding(nn.Module):
  def __init__(self, size):
    super().__init__()

    # self.projection = nn.Linear(size * 2, size, bias=False)

  def forward(self, x):
    size = x.size()

    # TODO: cuda
    pos = torch.arange(0, size[1], 1).unsqueeze(0).unsqueeze(-1).cuda()
    dim = torch.arange(0, size[2], 2).unsqueeze(0).unsqueeze(0).cuda()
    encoding = pos / 10000**(2 * dim / size[-1])
    encoding = Variable(encoding)
    encoding_sin = torch.sin(encoding)
    encoding_cos = torch.cos(encoding)

    # import matplotlib.pyplot as plt
    # for i in range(size[1]):
    #   plt.plot(encoding[0, i, :].data.numpy())
    #   plt.title('size')
    # plt.show()
    # for i in range(size[2]):
    #   plt.plot(encoding[0, :, i].data.numpy())
    #   plt.title('time')
    # plt.show()
    # fail

    x = torch.cat([
        x[:, :, 0::2] + encoding_sin,
        x[:, :, 1::2] + encoding_cos,
    ], -1)

    # x = torch.cat([
    #     x,
    #     encoding_sin.repeat(size[0], 1, 1),
    #     encoding_cos.repeat(size[0], 1, 1),
    # ], -1)
    #
    # x = self.projection(x)

    return x


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


def accuracy(y_top, y):
  y_top = y_top.max(-1)[1]
  mask = (y != 0).float()
  eq = (y_top == y).float()
  eq = eq * mask
  acc = eq.sum() / mask.sum()

  return acc
