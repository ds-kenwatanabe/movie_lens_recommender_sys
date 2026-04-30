import argparse

from recommender.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MOVIES_PATH,
    DEFAULT_RATINGS_PATH,
)

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
    def __init__(
        self,
        ratings_path=DEFAULT_RATINGS_PATH,
        movies_path=DEFAULT_MOVIES_PATH,
        model_path=DEFAULT_MODEL_PATH,
    ):
        import pandas as pd
        import torch

        from recommender.data import MovieLens
        from recommender.io import load_model_artifact
        from recommender.model import MatrixFactorization

        movielens_dataset = MovieLens(ratings_path)
        checkpoint = load_model_artifact(model_path)
        model_state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        checkpoint_config = (
            checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        )
        checkpoint_user_map = (
            checkpoint.get("user_map") if isinstance(checkpoint, dict) else None
        )
        checkpoint_movie_map = (
            checkpoint.get("movie_map") if isinstance(checkpoint, dict) else None
        )

        self.movielens = movielens_dataset
        if checkpoint_user_map:
            self.movielens.user_map = checkpoint_user_map
            self.movielens.users = self._ordered_ids_from_map(checkpoint_user_map)
        if checkpoint_movie_map:
            self.movielens.movie_map = checkpoint_movie_map
            self.movielens.movies = self._ordered_ids_from_map(checkpoint_movie_map)
        self.movielens.data["normalized_user_id"] = self.movielens.data["userId"].map(
            self.movielens.user_map.get
        )
        self.movielens.data["normalized_movie_id"] = self.movielens.data["movieId"].map(
            self.movielens.movie_map.get
        )
        self.movielens.size = len(self.movielens.users), len(self.movielens.movies)

        num_users, num_movies = self.movielens.size
        embedding_size = checkpoint_config.get(
            "embedding_size", model_state_dict["movie_embedding.weight"].shape[1]
        )
        global_mean = model_state_dict.get("global_mean", torch.tensor(0.0)).item()
        self.model = MatrixFactorization(
            num_users,
            num_movies,
            embedding_size,
            global_mean=global_mean,
        )

        self.model.load_state_dict(model_state_dict)
        self.movies = pd.read_csv(movies_path)
        self.model.eval()
        self.movie_embeddings = torch.nn.functional.normalize(
            self.model.movie_embedding.weight.detach(), p=2, dim=1
        )
        self.movie_popularity_scores = self._build_movie_popularity_scores()
        self._annoy_index = None

    @staticmethod
    def _ordered_ids_from_map(id_map):
        ordered_ids = [None] * len(id_map)
        for original_id, normalized_id in id_map.items():
            ordered_ids[int(normalized_id)] = original_id
        return ordered_ids

    def _build_movie_popularity_scores(self):
        stats = (
            self.movielens.data.groupby("normalized_movie_id")["rating"]
            .agg(["count", "mean"])
            .to_dict("index")
        )
        max_count = max((item["count"] for item in stats.values()), default=1)
        return {
            int(movie_index): (values["count"] / max_count) * float(values["mean"])
            for movie_index, values in stats.items()
        }

    def _genre_matches(self, movie_index, genre):
        if genre is None:
            return True
        movie_id = self.movielens.movies[movie_index]
        genres = self.movies[self.movies["movieId"] == movie_id]["genres"].values
        if len(genres) == 0:
            return False
        return genre.lower() in genres[0].lower().split("|")

    def _exact_cosine_neighbors(self, target_movie_index, top_n, genre=None):
        import torch

        target_embedding = self.movie_embeddings[target_movie_index].reshape(1, -1)
        similarities = torch.matmul(self.movie_embeddings, target_embedding.squeeze())
        ranked_indices = similarities.argsort(descending=True).tolist()
        return [
            movie_index
            for movie_index in ranked_indices
            if movie_index != target_movie_index
            and self._genre_matches(movie_index, genre)
        ][:top_n]

    def _annoy_neighbors(self, target_movie_index, top_n, search_k=-1, genre=None):
        try:
            from annoy import AnnoyIndex
        except ImportError:
            return None

        embeddings = self.movie_embeddings.cpu().numpy()
        if self._annoy_index is None:
            self._annoy_index = AnnoyIndex(embeddings.shape[1], "angular")
            for movie_index, embedding in enumerate(embeddings):
                self._annoy_index.add_item(movie_index, embedding.tolist())
            self._annoy_index.build(10)

        candidate_count = min(len(embeddings), max(top_n * 20, top_n + 1))
        candidates = self._annoy_index.get_nns_by_item(
            target_movie_index, candidate_count, search_k=search_k
        )
        return [
            movie_index
            for movie_index in candidates
            if movie_index != target_movie_index
            and self._genre_matches(movie_index, genre)
        ][:top_n]

    def get_similar(
        self, target_movie_id, top_n=5, genre=None, use_annoy=False, search_k=-1
    ):
        target_movie_index = self.movielens.movie_map[target_movie_id]
        similar_movie_indices = None
        if use_annoy:
            similar_movie_indices = self._annoy_neighbors(
                target_movie_index, top_n, search_k=search_k, genre=genre
            )
        if similar_movie_indices is None:
            similar_movie_indices = self._exact_cosine_neighbors(
                target_movie_index, top_n, genre=genre
            )

        self.display_similar_movies(target_movie_index, similar_movie_indices)

    def recommend_for_user(self, user_id, top_k=10, genre=None):
        import torch

        if user_id not in self.movielens.user_map:
            recommendations = self.recommend_cold_start(top_k=top_k, genre=genre)
            self.display_user_recommendations(user_id, recommendations, cold_start=True)
            return recommendations

        normalized_user_id = self.movielens.user_map[user_id]
        interacted_movies = set(
            self.movielens.data[
                self.movielens.data["normalized_user_id"] == normalized_user_id
            ]["normalized_movie_id"].tolist()
        )
        candidate_movie_indices = [
            movie_index
            for movie_index in range(len(self.movielens.movies))
            if movie_index not in interacted_movies
            and self._genre_matches(movie_index, genre)
        ]
        if not candidate_movie_indices:
            return []

        movie_ids = torch.LongTensor(candidate_movie_indices)
        user_ids = torch.full(
            (len(candidate_movie_indices),), normalized_user_id, dtype=torch.long
        )
        with torch.no_grad():
            scores = self.model(user_ids, movie_ids)

        ranked_positions = scores.argsort(descending=True).tolist()
        recommendations = [
            (candidate_movie_indices[position], scores[position].item())
            for position in ranked_positions[:top_k]
        ]
        self.display_user_recommendations(user_id, recommendations)
        return recommendations

    def recommend_cold_start(self, top_k=10, genre=None):
        candidate_movie_indices = [
            movie_index
            for movie_index in range(len(self.movielens.movies))
            if self._genre_matches(movie_index, genre)
        ]
        ranked_movies = sorted(
            candidate_movie_indices,
            key=lambda movie_index: self.movie_popularity_scores.get(movie_index, 0.0),
            reverse=True,
        )
        return [
            (movie_index, self.movie_popularity_scores.get(movie_index, 0.0))
            for movie_index in ranked_movies[:top_k]
        ]

    def movie_info(self, movie_id):
        movie_id = self.movielens.movies[movie_id]
        return {
            "Movie ID": [movie_id],
            "Title": self.movies[self.movies["movieId"] == movie_id]["title"].values[0],
            "Genre": self.movies[self.movies["movieId"] == movie_id]["genres"].values[
                0
            ],
        }

    def display_similar_movies(self, movie_id, similar_ids):
        main_title = self.movie_info(movie_id)
        print(
            f'Top {len(similar_ids)} most similar movies to {main_title["Title"]} [{main_title["Genre"]}]'
        )

        for id in similar_ids:
            title = self.movie_info(id)
            print(f'- [{title["Movie ID"][0]}] {title["Title"]} [{title["Genre"]}]')

    def display_user_recommendations(self, user_id, recommendations, cold_start=False):
        label = "cold-start recommendations" if cold_start else "recommendations"
        print(f"Top {len(recommendations)} {label} for user {user_id}")

        for movie_index, score in recommendations:
            title = self.movie_info(movie_index)
            print(
                f'- [{title["Movie ID"][0]}] {title["Title"]} [{title["Genre"]}] score={score:.4f}'
            )


SimilarMovies = MovieRecommender


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recommend movies or find similar movies."
    )
    parser.add_argument(
        "--ratings-path", default=DEFAULT_RATINGS_PATH, help="Path to ratings.csv."
    )
    parser.add_argument(
        "--movies-path", default=DEFAULT_MOVIES_PATH, help="Path to movies.csv."
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to recommender_model.pth.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="User ID to generate top-K recommendations for.",
    )
    parser.add_argument("--movie-id", type=int, nargs="*", default=None)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--genre", default=None, help="Only return movies containing this genre."
    )
    parser.add_argument(
        "--use-annoy",
        action="store_true",
        help="Use Annoy approximate nearest-neighbor search.",
    )
    parser.add_argument(
        "--search-k",
        type=int,
        default=-1,
        help="Annoy search_k value; -1 uses Annoy default.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    finder = MovieRecommender(args.ratings_path, args.movies_path, args.model_path)

    if args.user_id is not None:
        finder.recommend_for_user(args.user_id, top_k=args.top_k, genre=args.genre)
        return

    for target_id_movie in args.movie_id or DEFAULT_SAMPLE_MOVIE_IDS:
        finder.get_similar(
            target_id_movie,
            top_n=args.top_n,
            genre=args.genre,
            use_annoy=args.use_annoy,
            search_k=args.search_k,
        )


def preview_movie_ids_main():
    parser = argparse.ArgumentParser(description="Preview MovieLens movie IDs.")
    parser.add_argument(
        "--movies-path", default=DEFAULT_MOVIES_PATH, help="Path to movies.csv."
    )
    parser.add_argument(
        "--rows", type=int, default=5, help="Number of rows to display."
    )
    args = parser.parse_args()

    import pandas as pd

    data = pd.read_csv(args.movies_path)
    print(data.head(args.rows))


if __name__ == "__main__":
    main()
