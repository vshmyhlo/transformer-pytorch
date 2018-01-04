import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import attention


class Tranformer(nn.Module):
  # TODO: add biases
  # TODO: layer normalization
  # TODO: multihead attention
  def __init__(self, source_vocab_size, target_vocab_size, size, num_layers):
    super().__init__()

    self.encoder = Encoder(source_vocab_size, size, num_layers)
    self.decoder = Decoder(target_vocab_size, size, num_layers)
    self.projection = nn.Linear(size, target_vocab_size)

  def forward(self, x, y_bottom):
    encoder_states = self.encoder(x)
    y_top = self.decoder(y_bottom, encoder_states)
    y_top = self.projection(y_top)
    return y_top


class Encoder(nn.Module):
  # TODO: check train() and eval() sets state to layers

  def __init__(self, num_embeddings, size, num_layers):
    super().__init__()

    self.embedding = nn.Embedding(
        num_embeddings=num_embeddings, embedding_dim=size)
    self.positional_encoding = PositionalEncoding()
    self.encoder_layers = [
        EncoderLayer(size) for _ in range(1, num_layers + 1)
    ]

  def forward(self, x):
    x = self.embedding(x)
    x = self.positional_encoding(x)

    for layer in self.encoder_layers:
      x = layer(x)

    return x


class Decoder(nn.Module):
  # TODO: check train() and eval() sets state to layers

  def __init__(self, num_embeddings, size, num_layers):
    super().__init__()

    self.embedding = nn.Embedding(
        num_embeddings=num_embeddings, embedding_dim=size)
    self.positional_encoding = PositionalEncoding()
    self.decoder_layers = [
        DecoderLayer(size) for _ in range(1, num_layers + 1)
    ]

  def forward(self, y_bottom, states):
    y_bottom = self.embedding(y_bottom)
    y_bottom = self.positional_encoding(y_bottom)

    for layer in self.decoder_layers:
      y_bottom = layer(y_bottom, states)

    return y_bottom


class EncoderLayer(nn.Module):
  def __init__(self, size):
    super().__init__()

    self.self_attention = SelfAttentionSublayer(size)
    self.feed_forward = FeedForwardSublayer(size)

  def forward(self, x):
    x = self.self_attention(x)
    x = self.feed_forward(x)

    return x


class DecoderLayer(nn.Module):
  def __init__(self, size):
    super().__init__()

    self.self_attention = SelfAttentionSublayer(size)
    self.encoder_attention = AttentionSublayer(size)
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
    encoding = torch.sin(pos / 10000**(2 * dim / size[-1]))
    encoding = Variable(encoding)

    return x + encoding


class AttentionSublayer(nn.Module):
  def __init__(self, size):
    super().__init__()

    self.attention = attention.Attention(size)
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
  # TODO: check inner-layer size

  def __init__(self, size):
    super().__init__()

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

  def __init__(self, size, eps=1e-6):
    super().__init__()
    self.gamma = nn.Parameter(torch.ones(size).unsqueeze(0).unsqueeze(0))
    self.beta = nn.Parameter(torch.zeros(size).unsqueeze(0).unsqueeze(0))
    self.eps = eps

  def forward(self, x):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)

    return self.gamma * (x - mean) / (std + self.eps) + self.beta
