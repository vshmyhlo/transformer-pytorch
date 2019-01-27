import numpy as np
import torch
import torch.nn as nn
import sublayers


# TODO: test this
def compute_attention_padding_mask(seq_q, seq_k, padding_idx=0):
    assert seq_q.dim() == 2 and seq_k.dim() == 2

    pad_attn_mask = (seq_k.data != padding_idx).unsqueeze(1)
    return pad_attn_mask


# TODO: test this
def compute_attention_subsequent_mask(seq):
    assert seq.dim() == 2

    attention_shape = (1, seq.size(1), seq.size(1))
    # TODO: check typecasting
    subsequent_mask = np.tril(np.ones(attention_shape), k=0).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask


class Tranformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, size, n_layers, n_heads, dropout, share_embedding):
        super().__init__()

        self.encoder = Encoder(source_vocab_size, size, n_layers, n_heads, dropout)
        self.decoder = Decoder(target_vocab_size, size, n_layers, n_heads, dropout)
        self.projection = nn.Linear(size, target_vocab_size)

        if share_embedding:
            self.projection.weight = self.decoder.embedding.weight

    def forward(self, x, y_bottom):
        encoder_self_attention_mask = compute_attention_padding_mask(x, x)
        decoder_self_attention_mask = compute_attention_subsequent_mask(y_bottom)
        decoder_encoder_attention_mask = compute_attention_padding_mask(y_bottom, x)

        encoder_states = self.encoder(x, self_attention_mask=encoder_self_attention_mask)
        logits = self.decoder(
            y_bottom,
            encoder_states,
            self_attention_mask=decoder_self_attention_mask,
            encoder_attention_mask=decoder_encoder_attention_mask)
        logits = self.projection(logits)

        return logits


class Encoder(nn.Module):
    def __init__(self, num_embeddings, size, n_layers, n_heads, dropout):
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=size)
        self.positional_encoding = PositionalEncoding()
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(size, n_heads, dropout=dropout)
            for _ in range(self.n_layers)])

    def forward(self, x, self_attention_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # TODO: sequential
        for layer in self.encoder_layers:
            x = layer(x, self_attention_mask=self_attention_mask)

        return x


class Decoder(nn.Module):
    def __init__(self, num_embeddings, size, n_layers, n_heads, dropout):
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=size)
        self.positional_encoding = PositionalEncoding()
        self.dropout = nn.Dropout(dropout)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(size, n_heads, dropout=dropout)
            for _ in range(self.n_layers)])

    def forward(self, y_bottom, states, self_attention_mask, encoder_attention_mask):
        y_bottom = self.embedding(y_bottom)
        y_bottom = self.positional_encoding(y_bottom)
        y_bottom = self.dropout(y_bottom)

        for layer in self.decoder_layers:
            y_bottom = layer(
                y_bottom,
                states,
                self_attention_mask=self_attention_mask,
                encoder_attention_mask=encoder_attention_mask)

        return y_bottom


class EncoderLayer(nn.Module):
    def __init__(self, size, n_heads, dropout):
        super().__init__()

        self.self_attention = sublayers.SelfAttentionSublayer(size, n_heads, dropout=dropout)
        self.feed_forward = sublayers.FeedForwardSublayer(size, dropout=dropout)

    def forward(self, input, self_attention_mask):
        input = self.self_attention(input, mask=self_attention_mask)
        input = self.feed_forward(input)

        return input


class DecoderLayer(nn.Module):
    def __init__(self, size, n_heads, dropout):
        super().__init__()

        self.self_attention = sublayers.SelfAttentionSublayer(size, n_heads, dropout=dropout)
        self.encoder_attention = sublayers.AttentionSublayer(size, n_heads, dropout=dropout)
        self.feed_forward = sublayers.FeedForwardSublayer(size, dropout=dropout)

    def forward(self, input, states, self_attention_mask, encoder_attention_mask):
        input = self.self_attention(input, mask=self_attention_mask)
        input = self.encoder_attention(input, states, mask=encoder_attention_mask)
        input = self.feed_forward(input)

        return input


# TODO: test
class PositionalEncoding(nn.Module):
    def forward(self, input):
        d_model = input.size(2)
        # TODO: start from 0 or 1?
        pos = torch.arange(0, input.size(1)).unsqueeze(-1).float()
        i = torch.arange(0, input.size(2)).unsqueeze(0).float()
        encoding = pos / 10000**(2 * i / d_model)  # TODO: i // 2?
        encoding[:, 0::2] = torch.sin(encoding[:, 0::2])
        encoding[:, 1::2] = torch.cos(encoding[:, 1::2])

        encoding = encoding.unsqueeze(0)
        input += encoding.to(input.device)

        return input
