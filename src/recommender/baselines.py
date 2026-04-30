import argparse
import math
from collections import defaultdict

from recommender.config import DEFAULT_RATINGS_PATH
from recommender.train import build_implicit_feedback_samples, temporal_split_indices


class GlobalMeanBaseline:
    supports_rating_metrics = True

    def fit(self, data, num_users, num_movies):
        self.global_mean = float(data["rating"].mean())
        return self

    def score(self, user_id, movie_id):
        return self.global_mean


class UserMeanBaseline:
    supports_rating_metrics = True

    def fit(self, data, num_users, num_movies):
        self.global_mean = float(data["rating"].mean())
        self.user_means = data.groupby("normalized_user_id")["rating"].mean().to_dict()
        return self

    def score(self, user_id, movie_id):
        return float(self.user_means.get(user_id, self.global_mean))


class MovieMeanBaseline:
    supports_rating_metrics = True

    def fit(self, data, num_users, num_movies):
        self.global_mean = float(data["rating"].mean())
        self.movie_means = data.groupby("normalized_movie_id")["rating"].mean().to_dict()
        return self

    def score(self, user_id, movie_id):
        return float(self.movie_means.get(movie_id, self.global_mean))


class PopularityBaseline:
    supports_rating_metrics = False

    def fit(self, data, num_users, num_movies):
        counts = data.groupby("normalized_movie_id").size().to_dict()
        max_count = max(counts.values()) if counts else 1
        self.popularity = {int(movie_id): count / max_count for movie_id, count in counts.items()}
        return self

    def score(self, user_id, movie_id):
        return float(self.popularity.get(movie_id, 0.0))


class ItemItemCosineBaseline:
    supports_rating_metrics = False

    def fit(self, data, num_users, num_movies):
        self.user_movies = defaultdict(set)
        self.movie_users = defaultdict(set)
        for user_id, movie_id in zip(data["normalized_user_id"], data["normalized_movie_id"]):
            user_id = int(user_id)
            movie_id = int(movie_id)
            self.user_movies[user_id].add(movie_id)
            self.movie_users[movie_id].add(user_id)
        return self

    def _cosine(self, left_movie_id, right_movie_id):
        left_users = self.movie_users.get(left_movie_id, set())
        right_users = self.movie_users.get(right_movie_id, set())
        if not left_users or not right_users:
            return 0.0
        return len(left_users & right_users) / math.sqrt(len(left_users) * len(right_users))

    def score(self, user_id, movie_id):
        interacted_movies = self.user_movies.get(user_id, set())
        if not interacted_movies:
            return 0.0
        return max(self._cosine(movie_id, interacted_movie_id) for interacted_movie_id in interacted_movies)


class SVDBaseline:
    supports_rating_metrics = True

    def __init__(self, factors=50):
        self.factors = factors

    def fit(self, data, num_users, num_movies):
        import numpy as np

        self.global_mean = float(data["rating"].mean())
        matrix = np.full((num_users, num_movies), self.global_mean, dtype=float)
        for row in data.itertuples(index=False):
            matrix[int(row.normalized_user_id), int(row.normalized_movie_id)] = float(row.rating)

        centered = matrix - self.global_mean
        u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        factors = min(self.factors, len(singular_values))
        self.predictions = self.global_mean + (u[:, :factors] * singular_values[:factors]) @ vt[:factors, :]
        return self

    def score(self, user_id, movie_id):
        return float(self.predictions[user_id, movie_id])


def _ndcg_at_k(recommended_movies, relevant_movies, k):
    dcg = 0.0
    for rank, movie_id in enumerate(recommended_movies[:k], start=1):
        if movie_id in relevant_movies:
            dcg += 1.0 / math.log2(rank + 1.0)

    ideal_hits = min(len(relevant_movies), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def evaluate_baseline(model, validation_data, validation_samples, num_movies, k):
    if k < 1:
        raise ValueError("k must be at least 1")

    metrics = {}
    if model.supports_rating_metrics:
        errors = [
            model.score(int(row.normalized_user_id), int(row.normalized_movie_id)) - float(row.rating)
            for row in validation_data.itertuples(index=False)
        ]
        metrics["mae"] = sum(abs(error) for error in errors) / len(errors)
        metrics["rmse"] = math.sqrt(sum(error ** 2 for error in errors) / len(errors))
    else:
        metrics["mae"] = None
        metrics["rmse"] = None

    candidates_by_user = defaultdict(set)
    relevant_by_user = defaultdict(set)
    for user_id, movie_id, label in validation_samples:
        candidates_by_user[user_id].add(movie_id)
        if label >= 1.0:
            relevant_by_user[user_id].add(movie_id)

    recommended_movies = set()
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    hit_scores = []
    for user_id, relevant_movies in relevant_by_user.items():
        candidates = sorted(candidates_by_user[user_id])
        ranked_movies = sorted(candidates, key=lambda movie_id: model.score(user_id, movie_id), reverse=True)
        top_movies = ranked_movies[:k]
        recommended_movies.update(top_movies)
        hits = len(set(top_movies) & relevant_movies)

        precision_scores.append(hits / min(k, len(candidates)))
        recall_scores.append(hits / len(relevant_movies))
        ndcg_scores.append(_ndcg_at_k(top_movies, relevant_movies, k))
        hit_scores.append(1.0 if hits else 0.0)

    metrics.update({
        f"precision@{k}": sum(precision_scores) / len(precision_scores) if precision_scores else 0.0,
        f"recall@{k}": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        f"ndcg@{k}": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        f"hitrate@{k}": sum(hit_scores) / len(hit_scores) if hit_scores else 0.0,
        "coverage": len(recommended_movies) / num_movies if num_movies else 0.0,
    })
    return metrics


def compare_baselines(data, val_ratio=0.2, k=10, relevance_threshold=4.0, negatives_per_positive=4, seed=42, svd_factors=50):
    train_indices, val_indices = temporal_split_indices(data["timestamp"].tolist(), val_ratio)
    train_data = data.iloc[train_indices]
    validation_data = data.iloc[val_indices]
    num_users = int(data["normalized_user_id"].max()) + 1
    num_movies = int(data["normalized_movie_id"].max()) + 1
    validation_samples = build_implicit_feedback_samples(
        data,
        val_indices,
        num_movies,
        relevance_threshold,
        negatives_per_positive,
        seed + 1,
    )

    baselines = {
        "global_mean": GlobalMeanBaseline(),
        "user_mean": UserMeanBaseline(),
        "movie_mean": MovieMeanBaseline(),
        "popularity": PopularityBaseline(),
        "item_item_cosine": ItemItemCosineBaseline(),
        "svd": SVDBaseline(factors=svd_factors),
    }

    results = {}
    for name, baseline in baselines.items():
        baseline.fit(train_data, num_users, num_movies)
        results[name] = evaluate_baseline(baseline, validation_data, validation_samples, num_movies, k)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Compare recommender baselines on a temporal split.")
    parser.add_argument("--ratings-path", default=DEFAULT_RATINGS_PATH, help="Path to ratings.csv.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--relevance-threshold", type=float, default=4.0)
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svd-factors", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()

    import pandas as pd

    data = pd.read_csv(args.ratings_path)
    if "timestamp" not in data.columns:
        raise ValueError("Baseline comparison requires a timestamp column in the ratings data")

    users = sorted(data["userId"].unique())
    movies = sorted(data["movieId"].unique())
    user_map = {user_id: index for index, user_id in enumerate(users)}
    movie_map = {movie_id: index for index, movie_id in enumerate(movies)}
    data["normalized_user_id"] = data["userId"].map(user_map)
    data["normalized_movie_id"] = data["movieId"].map(movie_map)

    results = compare_baselines(
        data,
        val_ratio=args.val_ratio,
        k=args.eval_k,
        relevance_threshold=args.relevance_threshold,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
        svd_factors=args.svd_factors,
    )
    for name, metrics in results.items():
        formatted_metrics = []
        for metric, value in metrics.items():
            formatted_metrics.append(f"{metric}: n/a" if value is None else f"{metric}: {value:.4f}")
        print(f"{name}: " + ", ".join(formatted_metrics))


if __name__ == "__main__":
    main()

