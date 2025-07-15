# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # masking out (setting to −inf) all values in the input of the softmax which correspond to illegal connections
            # to prevent leftward information flow in the decoder to preserve the auto-regressive property
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))   # query vector Q
        K = self.split_heads(self.W_k(K))   # key vector K
        V = self.split_heads(self.W_v(V))   # value vector V
        
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attention_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        n = 10000.0
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(n) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)    # sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)    # cosine to odd indices
        
        self.register_buffer('pe', pe.unsqueeze(0))     # add a batch dimension to the positional encoding and register the positional encoding as a buffer
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, source_mask, target_mask):
        attention_output = self.self_attention(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(attention_output))
        attention_output = self.cross_attention(x, encoder_output, encoder_output, source_mask)
        x = self.norm2(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(source_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, source, target):
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2)
        target_mask = (target != 0).unsqueeze(1).unsqueeze(3)
        seq_length = target.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        target_mask = target_mask & nopeak_mask
        return source_mask, target_mask

    def forward(self, source, target):
        source_mask, target_mask = self.generate_mask(source, target)
        source_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(source)))
        target_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(target)))

        encoder_output = source_embedded
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, source_mask)

        decoder_output = target_embedded
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, source_mask, target_mask)

        output = self.fc(decoder_output)
        return output

# input parameters

source_vocab_size = 1000
target_vocab_size = 1000
d_model = 512               # all sub-layers in the model, as well as the embedding layers, produce outputs of this dimension
num_heads = 8               # number of parallel attention layers (or heads)
num_layers = 6              # stack of identical layers
d_ff = 2048                 # hidden layer size of the feed-forward network
max_seq_length = 100        # maximum sequence length for positional encoding
dropout = 0.1

transformer = Transformer(source_vocab_size, target_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# generate random dataset
source_data = torch.randint(1, source_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
target_data = torch.randint(1, target_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(10):
    optimizer.zero_grad()
    output = transformer(source_data, target_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, target_vocab_size), target_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

