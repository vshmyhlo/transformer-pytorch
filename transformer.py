import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sublayers


class Tranformer(nn.Module):
  def __init__(self, source_vocab_size, target_vocab_size, size, n_layers,
               n_heads, pe_type, dropout, padding_idx):
    super().__init__()

    self.encoder = Encoder(source_vocab_size, size, n_layers, n_heads, pe_type,
                           dropout, padding_idx)
    self.decoder = Decoder(target_vocab_size, size, n_layers, n_heads, pe_type,
                           dropout, padding_idx)
    self.projection = nn.Linear(size, target_vocab_size)

  def forward(self, x, y_bottom):
    encoder_states = self.encoder(x)
    y_top = self.decoder(y_bottom, encoder_states)
    y_top = self.projection(y_top)
    return y_top


class Encoder(nn.Module):
  def __init__(self, num_embeddings, size, n_layers, n_heads, pe_type, dropout,
               padding_idx):
    super().__init__()

    self.n_layers = n_layers
    self.n_heads = n_heads
    self.embedding = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=size,
        padding_idx=padding_idx)
    self.positional_encoding = PositionalEncoding(size, pe_type=pe_type)
    self.dropout = nn.Dropout(dropout)
    self.encoder_layers = nn.ModuleList(
        [EncoderLayer(size, n_heads) for _ in range(self.n_layers)])

  def forward(self, x):
    x = self.embedding(x)
    x = self.positional_encoding(x)
    x = self.dropout(x)

    for layer in self.encoder_layers:
      x = layer(x)

    x /= (self.n_heads**2 * self.n_layers)

    return x


class Decoder(nn.Module):
  def __init__(self, num_embeddings, size, n_layers, n_heads, pe_type, dropout,
               padding_idx):
    super().__init__()

    self.n_layers = n_layers
    self.n_heads = n_heads
    self.embedding = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=size,
        padding_idx=padding_idx)
    self.positional_encoding = PositionalEncoding(size, pe_type=pe_type)
    self.dropout = nn.Dropout(dropout)
    self.decoder_layers = nn.ModuleList(
        [DecoderLayer(size, n_heads) for _ in range(self.n_layers)])

  def forward(self, y_bottom, states):
    y_bottom = self.embedding(y_bottom)
    y_bottom = self.positional_encoding(y_bottom)
    y_bottom = self.dropout(y_bottom)

    for layer in self.decoder_layers:
      y_bottom = layer(y_bottom, states)

    y_bottom /= self.n_heads

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
  def __init__(self, size, pe_type):
    super().__init__()

    self.pe_type = pe_type

    if self.pe_type == 'projection':
      self.projection = nn.Linear(size * 2, size, bias=False)

  def forward(self, x):
    size = x.size()

    if self.pe_type == 'projection':
      # TODO: search for parameter
      # k = 2
      k = 0.75
      # k = 0.5
    elif self.pe_type == 'addition':
      k = 2

    pos = torch.arange(0, size[1], 1).unsqueeze(0).unsqueeze(-1).type_as(
        x.data)
    dim = torch.arange(0, size[2], 2).unsqueeze(0).unsqueeze(0).type_as(x.data)
    encoding = pos / 10000**(k * dim / size[-1])
    encoding = Variable(encoding)
    encoding_sin = torch.sin(encoding)
    encoding_cos = torch.cos(encoding)

    if self.pe_type == 'projection':
      x = torch.cat([
          x,
          encoding_sin.repeat(size[0], 1, 1),
          encoding_cos.repeat(size[0], 1, 1),
      ], -1)

      x = self.projection(x)

      return x
    elif self.pe_type == 'addition':
      x = torch.cat([
          x[:, :, 0::2] + encoding_sin,
          x[:, :, 1::2] + encoding_cos,
      ], -1)

      return x


def loss(y_top, y, padding_idx, reduce=True):
  not_padding = y != padding_idx
  loss = F.cross_entropy(y_top[not_padding], y[not_padding], reduce=reduce)
  return loss


def accuracy(y_top, y, padding_idx, reduce=True):
  _, y_top = y_top.max(-1)
  not_padding = y != padding_idx
  eq = y_top[not_padding] == y[not_padding]

  if reduce:
    return eq.float().mean()
  else:
    return eq.float()


def infer(model, x, sos_idx, max_len):
  from torch.autograd import Variable
  y_bottom = Variable(torch.LongTensor([[1]]) * sos_idx)

  while y_bottom.size(1) < max_len:
    y_top = model(x, y_bottom)
    y_top = y_top.max(-1)[1]
    y_bottom = torch.cat([y_bottom, y_top[:, -1:]], -1)

  return y_top
