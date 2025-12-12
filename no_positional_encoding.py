# -*- coding: utf-8 -*-

import numpy as np

# The embedding of each element
embeddings = {
    "gene1": np.array([3]),
    "gene2": np.array([2]),
    "gene3": np.array([1])
}

# Creating matrices for transforming the embeddings into Q, K, and V vectors
W_q = W_k = W_v = np.array([[1]])

# Three sequences
sequence1 = ["gene1", "gene2", "gene3"]
sequence2 = ["gene2", "gene1", "gene3"]
sequence3 = ["gene3", "gene1", "gene2"]

def calculate_attention(sequence, embeddings, W_q, W_k, W_v):
    # Transform elements into Q, K, V
    Q = np.stack([embeddings[element] @ W_q for element in sequence])
    K = np.stack([embeddings[element] @ W_k for element in sequence])
    V = np.stack([embeddings[element] @ W_v for element in sequence])
    
    # The attention scores calculation
    attention_scores = Q @ K.T
    
    # Scaling factor (sqrt of key dimension) as in the original transformer
    scale_factor = np.sqrt(K.shape[1])
    scaled_attention_scores = attention_scores / scale_factor
    
    # Then we apply the softmax to get the attention weights
    attention_weights = np.exp(scaled_attention_scores) / np.sum(np.exp(scaled_attention_scores), axis=1, keepdims=True)
    
    # The weighted sum of the Value vectors
    weighted_sum = attention_weights @ V
    
    return weighted_sum

# The attention for every sequence
attention_sequence1 = calculate_attention(sequence1, embeddings, W_q, W_k, W_v)
attention_sequence2 = calculate_attention(sequence2, embeddings, W_q, W_k, W_v)
attention_sequence3 = calculate_attention(sequence3, embeddings, W_q, W_k, W_v)

print("Attention for first sequence:", attention_sequence1)
print("Attention for second sequence:", attention_sequence2)
print("Attention for third sequence:", attention_sequence3)

