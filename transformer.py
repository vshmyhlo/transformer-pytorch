import torch
from torch.autograd import Variable
import torch.nn as nn
import sublayers
import numpy as np


def get_attn_subsequent_mask(seq):
  # TODO: check how this works
  ''' Get an attention mask to avoid using the subsequent info.'''
  assert seq.dim() == 2
  attn_shape = (1, seq.size(1), seq.size(1))
  # TODO: check this
  subsequent_mask = np.tril(np.ones(attn_shape), k=0).astype('uint8')
  subsequent_mask = torch.from_numpy(subsequent_mask)
  if seq.is_cuda:
    subsequent_mask = subsequent_mask.cuda()
  return subsequent_mask


class Tranformer(nn.Module):
  def __init__(self, source_vocab_size, target_vocab_size, size, n_layers,
               n_heads, pe_type, dropout, padding_idx, attention_type):
    super().__init__()

    self.encoder = Encoder(
        source_vocab_size,
        size,
        n_layers,
        n_heads,
        pe_type,
        dropout,
        padding_idx,
        attention_type=attention_type)
    self.decoder = Decoder(
        target_vocab_size,
        size,
        n_layers,
        n_heads,
        pe_type,
        dropout,
        padding_idx,
        attention_type=attention_type)
    self.projection = nn.Linear(size, target_vocab_size)

  def forward(self, x, y_bottom):
    encoder_states = self.encoder(x)
    y_top = self.decoder(y_bottom, encoder_states)
    y_top = self.projection(y_top)
    return y_top


class Encoder(nn.Module):
  def __init__(self, num_embeddings, size, n_layers, n_heads, pe_type, dropout,
               padding_idx, attention_type):
    super().__init__()

    self.n_layers = n_layers
    self.n_heads = n_heads
    self.embedding = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=size,
        padding_idx=padding_idx)
    self.positional_encoding = PositionalEncoding(size, pe_type=pe_type)
    self.dropout = nn.Dropout(dropout)
    self.encoder_layers = nn.ModuleList([
        EncoderLayer(size, n_heads, attention_type=attention_type)
        for _ in range(self.n_layers)
    ])

  def forward(self, x):
    x = self.embedding(x)
    x = self.positional_encoding(x)
    x = self.dropout(x)

    for layer in self.encoder_layers:
      x = layer(x)

    # x /= (self.n_heads**2 * self.n_layers)

    return x


class Decoder(nn.Module):
  def __init__(self, num_embeddings, size, n_layers, n_heads, pe_type, dropout,
               padding_idx, attention_type):
    super().__init__()

    self.n_layers = n_layers
    self.n_heads = n_heads
    self.embedding = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=size,
        padding_idx=padding_idx)
    self.positional_encoding = PositionalEncoding(size, pe_type=pe_type)
    self.dropout = nn.Dropout(dropout)
    self.decoder_layers = nn.ModuleList([
        DecoderLayer(size, n_heads, attention_type=attention_type)
        for _ in range(self.n_layers)
    ])

  def forward(self, y_bottom, states):
    self_attention_mask = get_attn_subsequent_mask(y_bottom)
    self_attention_mask = Variable(self_attention_mask)

    y_bottom = self.embedding(y_bottom)
    y_bottom = self.positional_encoding(y_bottom)
    y_bottom = self.dropout(y_bottom)

    for layer in self.decoder_layers:
      y_bottom = layer(
          y_bottom, states, self_attention_mask=self_attention_mask)

    # y_bottom /= self.n_heads

    return y_bottom


class EncoderLayer(nn.Module):
  def __init__(self, size, n_heads, attention_type):
    super().__init__()

    self.self_attention = sublayers.SelfAttentionSublayer(
        size, n_heads, attention_type=attention_type)
    self.feed_forward = sublayers.FeedForwardSublayer(size)

  def forward(self, x):
    x = self.self_attention(x)
    x = self.feed_forward(x)

    return x


class DecoderLayer(nn.Module):
  def __init__(self, size, n_heads, attention_type):
    super().__init__()

    self.self_attention = sublayers.SelfAttentionSublayer(
        size, n_heads, attention_type=attention_type)
    self.encoder_attention = sublayers.AttentionSublayer(
        size, n_heads, attention_type=attention_type)
    self.feed_forward = sublayers.FeedForwardSublayer(size)

  def forward(self, x, states, self_attention_mask):
    x = self.self_attention(x, self_attention_mask)
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
      k = 0.75
    elif self.pe_type == 'addition':
      k = 2

    pos = torch.arange(0, size[1], 1).unsqueeze(0).unsqueeze(-1)
    dim = torch.arange(0, size[2], 2).unsqueeze(0).unsqueeze(0)
    if x.is_cuda:
      pos, dim = pos.cuda(), dim.cuda()
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
      # encoding = torch.cat([encoding_sin, encoding_cos], -1)
      # x += encoding

      x = torch.cat([
          x[:, :, 0::2] + encoding_sin,
          x[:, :, 1::2] + encoding_cos,
      ], -1)

      return x
