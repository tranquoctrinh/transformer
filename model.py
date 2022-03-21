import torch
import torch.nn as nn
import numpy as np

# Embedding the input sequence
class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

# The positional encoding vector
class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_len):
        super(PositionalEncoder, self).__init__()
        # Compute the positional encoding vector
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding vector to the embedding vector
        x = x + self.pe[:, :x.size(1), :]
        return x

# Self-attention layer
class SelfAttention(object):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        # The number of heads
        self.num_heads = num_heads
        # The dimension of each head
        self.dim_per_head = embedding_dim // num_heads
        # The linear projections
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # Apply the linear projections
        batch_size = query.size(0)
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        # Reshape the input
        query = query.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2)
        # Calculate the attention
        attention = torch.bmm(query, key.transpose(2, 1)) / np.sqrt(dim_per_head)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        attention = self.dropout(torch.softmax(attention, dim=-1))
        # Apply the attention to the value
        output = torch.bmm(attention, value)
        return output

# Multi-head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_attention = SelfAttention(embedding_dim, num_heads, dropout)
        # The number of heads
        self.num_heads = num_heads
        # The dimension of each head
        self.dim_per_head = embedding_dim // num_heads
        # The linear projections
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value, mask=None):
        # Apply the linear projections
        batch_size = query.size(0)
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        # Reshape the input
        query = query.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        # Calculate the attention
        scores = self.self_attention(query, key, value, mask)
        # Reshape the output
        output = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        # Apply the linear projection
        output = self.out(output)
        return output

# Norm layer
class Norm(nn.Module):
    def __init__(self, embedding_dim):
        super(Norm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.norm(x)


# Transformer encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        self.norm1 = Norm(embedding_dim)
        self.norm2 = Norm(embedding_dim)

    def forward(self, x, mask=None):
        # Self attention
        x = self.self_attention(x, x, x, mask)
        # Dropout and residual connection
        x = self.norm1(x + x)
        # Feed forward
        x = self.feed_forward(x)
        # Dropout and residual connection
        x = self.norm2(x + x)
        return x

# Transformer decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.encoder_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        self.norm1 = Norm(embedding_dim)
        self.norm2 = Norm(embedding_dim)
        self.norm3 = Norm(embedding_dim)

    def forward(self, x, memory, src_mask, tgt_mask):
        # Self attention
        x = self.self_attention(x, x, x, tgt_mask)
        # Dropout and residual connection
        x = self.norm1(x + x)
        # Encoder attention
        x = self.encoder_attention(x, memory, memory, src_mask)
        # Dropout and residual connection
        x = self.norm2(x + x)
        # Feed forward
        x = self.feed_forward(x)
        # Dropout and residual connection
        x = self.norm3(x + x)
        return x

# Transformers
class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, embedding_dim, num_layers, num_heads, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.encoder = nn.Sequential(
            nn.Embedding(source_vocab_size, embedding_dim),
            EncoderLayer(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        )
        self.decoder = nn.Sequential(
            nn.Embedding(target_vocab_size, embedding_dim),
            DecoderLayer(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        )
        self.fc = nn.Linear(embedding_dim, target_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encoder(src)
        x = self.decoder(tgt, memory, src_mask, tgt_mask)
        x = self.fc(x)
        return x