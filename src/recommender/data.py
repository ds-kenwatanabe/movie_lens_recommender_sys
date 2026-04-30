import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from recommender.config import DEFAULT_RATINGS_PATH


class MovieLens(Dataset):
    def __init__(self, file_path=DEFAULT_RATINGS_PATH):
        self.data = pd.read_csv(file_path)
        self.users = np.unique(self.data["userId"])
        self.movies = np.unique(self.data["movieId"])
        self.user_map = {id: np.where(self.users == id)[0][0] for id in self.users}
        self.movie_map = {id: np.where(self.movies == id)[0][0] for id in self.movies}
        self.data["normalized_user_id"] = self.data["userId"].map(self.user_map.get)
        self.data["normalized_movie_id"] = self.data["movieId"].map(self.movie_map.get)
        self.global_mean = float(self.data["rating"].mean())
        self.size = len(self.users), len(self.movies)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user_id = torch.LongTensor([self.data.iloc[index]["normalized_user_id"]])
        movie_id = torch.LongTensor([self.data.iloc[index]["normalized_movie_id"]])
        rating = torch.FloatTensor([self.data.iloc[index]["rating"]])

        return user_id, movie_id, rating


def parse_args():
    parser = argparse.ArgumentParser(description="Preview MovieLens ratings data.")
    parser.add_argument(
        "--ratings-path", default=DEFAULT_RATINGS_PATH, help="Path to ratings.csv."
    )
    parser.add_argument(
        "--rows", type=int, default=10, help="Number of rows to display."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = MovieLens(args.ratings_path)
    for i in range(min(args.rows, len(data))):
        print(data[i])


if __name__ == "__main__":
    main()
