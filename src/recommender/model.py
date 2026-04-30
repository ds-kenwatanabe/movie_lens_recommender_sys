"""
Matrix factorization model.

The model learns latent embeddings and bias terms for users and movies, then
predicts a rating from the global mean, user bias, movie bias, and dot product
of the matching user and movie vectors.
"""

import torch
from torch import nn


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size, global_mean=0.0):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        self.register_buffer("global_mean", torch.tensor(float(global_mean)))

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, user_idx, movie_idx):
        user_emb = self.user_embedding(user_idx)
        movie_emb = self.movie_embedding(movie_idx)
        user_bias = self.user_bias(user_idx).squeeze(-1)
        movie_bias = self.movie_bias(movie_idx).squeeze(-1)
        rating = (
            self.global_mean
            + user_bias
            + movie_bias
            + torch.sum(user_emb * movie_emb, dim=1)
        )
        return rating
