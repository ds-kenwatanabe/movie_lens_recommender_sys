import argparse
from pathlib import Path

from recommender.config import DEFAULT_MODEL_PATH, DEFAULT_RATINGS_PATH


def _load_pyplot():
    import matplotlib.pyplot as plt

    return plt


def plot_training_history(checkpoint_path, output_dir):
    import torch

    plt = _load_pyplot()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history = (
        checkpoint.get("training_history", []) if isinstance(checkpoint, dict) else []
    )
    if not history:
        return None

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    validation_mae = [item["validation_metrics"]["mae"] for item in history]

    output_path = Path(output_dir) / "training_loss_vs_validation_mae.png"
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Training loss")
    plt.plot(epochs, validation_mae, marker="o", label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / MAE")
    plt.title("Training Loss vs Validation MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_rating_distribution(data, output_dir):
    plt = _load_pyplot()
    output_path = Path(output_dir) / "rating_distribution.png"
    plt.figure(figsize=(8, 5))
    data["rating"].hist(bins=20)
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.title("Rating Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_ratings_per_user(data, output_dir):
    plt = _load_pyplot()
    output_path = Path(output_dir) / "ratings_per_user.png"
    plt.figure(figsize=(8, 5))
    data.groupby("userId").size().hist(bins=50)
    plt.xlabel("Ratings per user")
    plt.ylabel("User count")
    plt.title("Number of Ratings per User")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_ratings_per_movie(data, output_dir):
    plt = _load_pyplot()
    output_path = Path(output_dir) / "ratings_per_movie.png"
    plt.figure(figsize=(8, 5))
    data.groupby("movieId").size().hist(bins=50)
    plt.xlabel("Ratings per movie")
    plt.ylabel("Movie count")
    plt.title("Number of Ratings per Movie")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def _load_movie_embeddings(checkpoint_path):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if "movie_embedding.weight" not in state_dict:
        raise ValueError("Checkpoint does not contain movie embeddings")
    return state_dict["movie_embedding.weight"].detach().cpu().numpy()


def _reduce_embeddings(embeddings, method, max_points, seed):
    if len(embeddings) > max_points:
        embeddings = embeddings[:max_points]

    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "Install umap-learn to use --embedding-method umap"
            ) from exc

        reducer = umap.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(embeddings)

    from sklearn.manifold import TSNE

    perplexity = min(30, max(2, len(embeddings) - 1))
    reducer = TSNE(
        n_components=2, random_state=seed, init="random", perplexity=perplexity
    )
    return reducer.fit_transform(embeddings)


def plot_embedding_visualization(
    checkpoint_path, output_dir, method="tsne", max_points=5000, seed=42
):
    plt = _load_pyplot()
    embeddings = _load_movie_embeddings(checkpoint_path)
    points = _reduce_embeddings(embeddings, method, max_points, seed)

    output_path = Path(output_dir) / f"movie_embedding_{method}.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], s=5, alpha=0.6)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(f"Movie Embedding Visualization ({method.upper()})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def create_plots(
    ratings_path,
    output_dir,
    model_path=None,
    embedding_method="tsne",
    max_points=5000,
    seed=42,
):
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ratings_path)

    outputs = [
        plot_rating_distribution(data, output_dir),
        plot_ratings_per_user(data, output_dir),
        plot_ratings_per_movie(data, output_dir),
    ]
    if model_path:
        outputs.append(plot_training_history(model_path, output_dir))
        outputs.append(
            plot_embedding_visualization(
                model_path, output_dir, embedding_method, max_points, seed
            )
        )
    return [output for output in outputs if output is not None]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate recommender system diagnostic plots."
    )
    parser.add_argument(
        "--ratings-path", default=DEFAULT_RATINGS_PATH, help="Path to ratings.csv."
    )
    parser.add_argument(
        "--model-path", default=DEFAULT_MODEL_PATH, help="Path to training checkpoint."
    )
    parser.add_argument(
        "--output-dir", default="plots", help="Directory where plots are written."
    )
    parser.add_argument("--embedding-method", choices=["tsne", "umap"], default="tsne")
    parser.add_argument(
        "--max-points", type=int, default=5000, help="Maximum embeddings to visualize."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-model-plots", action="store_true", help="Skip loss and embedding plots."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = None if args.skip_model_plots else args.model_path
    outputs = create_plots(
        args.ratings_path,
        args.output_dir,
        model_path=model_path,
        embedding_method=args.embedding_method,
        max_points=args.max_points,
        seed=args.seed,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
