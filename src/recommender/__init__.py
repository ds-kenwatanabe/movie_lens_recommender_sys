"""MovieLens recommender package."""

__all__ = [
    "MatrixFactorization",
    "MovieLens",
    "MovieRecommender",
    "SimilarMovies",
    "compare_baselines",
]


def __getattr__(name):
    if name == "MatrixFactorization":
        from recommender.model import MatrixFactorization

        return MatrixFactorization
    if name == "MovieLens":
        from recommender.data import MovieLens

        return MovieLens
    if name in {"MovieRecommender", "SimilarMovies"}:
        from recommender.recommend import MovieRecommender, SimilarMovies

        return {"MovieRecommender": MovieRecommender, "SimilarMovies": SimilarMovies}[name]
    if name == "compare_baselines":
        from recommender.baselines import compare_baselines

        return compare_baselines
    raise AttributeError(f"module 'recommender' has no attribute {name!r}")
