import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import attention


class Tranformer(nn.Module):
  # TODO: add biases
  # TODO: layer normalization
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
  def __init__(self, num_embeddings, size, num_layers):
    super().__init__()

    self.embedding = nn.Embedding(
        num_embeddings=num_embeddings, embedding_dim=size)
    self.positional_encoding = PositionalEncoding()

    for i in range(1, num_layers + 1):
      layer = EncoderLayer(size)
      setattr(self, 'el{}'.format(i), layer)

  def forward(self, x):
    x = self.embedding(x)
    x = self.positional_encoding(x)
    x = self.el1(x)
    x = self.el2(x)

    return x


class Decoder(nn.Module):
  def __init__(self, num_embeddings, size, num_layers):
    super().__init__()

    self.embedding = nn.Embedding(
        num_embeddings=num_embeddings, embedding_dim=size)
    self.positional_encoding = PositionalEncoding()

    for i in range(1, num_layers + 1):
      layer = DecoderLayer(size)
      setattr(self, 'dl{}'.format(i), layer)

  def forward(self, y_bottom, states):
    y_bottom = self.embedding(y_bottom)
    y_bottom = self.positional_encoding(y_bottom)
    y_bottom = self.dl1(y_bottom, states)
    y_bottom = self.dl2(y_bottom, states)

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

  def forward(self, x, states):
    saved = x

    x = self.attention(x, states)
    x = saved + x
    x = F.relu(x)

    return x


class SelfAttentionSublayer(AttentionSublayer):
  def forward(self, x):
    return super().forward(x, x)


class FeedForwardSublayer(nn.Module):
  # TODO: batch normalization
  # TODO: check inner-layer size

  def __init__(self, size):
    super().__init__()

    self.fc1 = nn.Linear(size, size)
    # self.bn1 = nn.Linear(size, size)
    self.fc2 = nn.Linear(size, size)
    # self.bn2 = nn.Linear(size, size)

  def forward(self, x):
    saved = x

    x = self.fc1(x)
    # x = self.bn1(x)
    x = F.relu(x)

    x = self.fc2(x)
    # x = self.bn2(x)
    x = saved + x
    x = F.relu(x)

    return x
