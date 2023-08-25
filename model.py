"""
Matrix factorization.

In machine learning, an embedding is a way of representing a high-dimensional data point in a lower-dimensional space.
This is done by finding a mapping from the high-dimensional space to the
lower-dimensional space that preserves the relationships between the data points.

The matrix factorization technique factorizes a matrix that represents the relationships between words or phrases.
The factors of the matrix are then used to represent the words or phrases as vectors.
In this case dimensionality is not reduced and the model is trained to find relationships between the words
"""
import torch
from torch import nn


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)

    def forward(self, user_idx, movie_idx):
        user_emb = self.user_embedding(user_idx)
        movie_emb = self.movie_embedding(movie_idx)
        rating = torch.sum(user_emb * movie_emb, dim=1)
        return rating
