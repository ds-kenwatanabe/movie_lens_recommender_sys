import argparse

from recommender.config import DEFAULT_MODEL_PATH, DEFAULT_MOVIES_PATH, DEFAULT_RATINGS_PATH


DEFAULT_SAMPLE_MOVIE_IDS = [
    116797,  # The Imitation Game (2014)
    7153,  # Lord of the Rings: The Return of the King, The (2003)
    2959,  # Fight Club (1999)
    6377,  # Finding Nemo (2003)
    858,  # Godfather, The (1972)
    5349,  # Spider-Man (2002)
    109487,  # Interstellar (2014)
]


class MovieRecommender:
    def __init__(self, ratings_path=DEFAULT_RATINGS_PATH, movies_path=DEFAULT_MOVIES_PATH, model_path=DEFAULT_MODEL_PATH):
        import pandas as pd

        from recommender.data import MovieLens
        from recommender.io import load_model
        from recommender.model import MatrixFactorization

        movielens_dataset = MovieLens(ratings_path)
        num_users, num_movies = movielens_dataset.size
        embedding_size = 200
        self.movielens = movielens_dataset
        self.model = MatrixFactorization(num_users, num_movies, embedding_size)

        load_model(self.model, model_path)
        self.movies = pd.read_csv(movies_path)

    def get_similar(self, target_movie_id, top_n=5):
        from torch.nn.functional import pairwise_distance

        target_movie_id = self.movielens.movie_map[target_movie_id]
        movie_embeddings = self.model.movie_embedding.weight.data
        target_embedding = movie_embeddings[target_movie_id].reshape(1, -1)

        similarities = pairwise_distance(target_embedding.unsqueeze(0), movie_embeddings)

        similar_movie_indices = similarities.argsort(dim=1, descending=False).squeeze()[1:top_n + 1]

        similar_movie_indices = [id.item() for id in similar_movie_indices]
        self.display_similar_movies(target_movie_id, similar_movie_indices)

    def movie_info(self, movie_id):
        movie_id = self.movielens.movies[movie_id]
        return {
            "Movie ID": [movie_id],
            "Title": self.movies[self.movies["movieId"] == movie_id]["title"].values[0],
            "Genre": self.movies[self.movies["movieId"] == movie_id]["genres"].values[0],
        }

    def display_similar_movies(self, movie_id, similar_ids):
        main_title = self.movie_info(movie_id)
        print(f'Top {len(similar_ids)} most similar movies to {main_title["Title"]} [{main_title["Genre"]}]')

        for id in similar_ids:
            title = self.movie_info(id)
            print(f'- [{title["Movie ID"][0]}] {title["Title"]} [{title["Genre"]}]')


SimilarMovies = MovieRecommender


def parse_args():
    parser = argparse.ArgumentParser(description="Find movies with similar embeddings.")
    parser.add_argument("--ratings-path", default=DEFAULT_RATINGS_PATH, help="Path to ratings.csv.")
    parser.add_argument("--movies-path", default=DEFAULT_MOVIES_PATH, help="Path to movies.csv.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to recommender_model.pth.")
    parser.add_argument("--movie-id", type=int, nargs="*", default=DEFAULT_SAMPLE_MOVIE_IDS)
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    finder = MovieRecommender(args.ratings_path, args.movies_path, args.model_path)

    for target_id_movie in args.movie_id:
        finder.get_similar(target_id_movie, top_n=args.top_n)


def preview_movie_ids_main():
    parser = argparse.ArgumentParser(description="Preview MovieLens movie IDs.")
    parser.add_argument("--movies-path", default=DEFAULT_MOVIES_PATH, help="Path to movies.csv.")
    parser.add_argument("--rows", type=int, default=5, help="Number of rows to display.")
    args = parser.parse_args()

    import pandas as pd

    data = pd.read_csv(args.movies_path)
    print(data.head(args.rows))


if __name__ == "__main__":
    main()
