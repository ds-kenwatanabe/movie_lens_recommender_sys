import argparse
import random

from recommender.config import DEFAULT_MODEL_PATH, DEFAULT_RATINGS_PATH


def build_user_interactions(data):
    interactions = {}
    for user_id, movie_id in zip(data["normalized_user_id"], data["normalized_movie_id"]):
        interactions.setdefault(int(user_id), set()).add(int(movie_id))
    return interactions


def build_implicit_feedback_samples(
    data,
    positive_indices,
    num_movies,
    relevance_threshold,
    negatives_per_positive,
    seed,
):
    if negatives_per_positive < 1:
        raise ValueError("--negatives-per-positive must be at least 1")

    rng = random.Random(seed)
    all_movie_ids = set(range(num_movies))
    interactions_by_user = build_user_interactions(data)
    samples = []

    for index in positive_indices:
        row = data.iloc[index]
        if row["rating"] < relevance_threshold:
            continue

        user_id = int(row["normalized_user_id"])
        movie_id = int(row["normalized_movie_id"])
        samples.append((user_id, movie_id, 1.0))

        candidate_negatives = list(all_movie_ids - interactions_by_user.get(user_id, set()))
        if not candidate_negatives:
            continue

        replace = len(candidate_negatives) < negatives_per_positive
        for _ in range(negatives_per_positive):
            negative_movie_id = rng.choice(candidate_negatives)
            if not replace:
                candidate_negatives.remove(negative_movie_id)
            samples.append((user_id, negative_movie_id, 0.0))

    if not samples:
        raise ValueError("No implicit feedback samples were created. Lower --relevance-threshold or check the data.")

    return samples


def temporal_split_indices(timestamps, val_ratio):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    total_length = len(timestamps)
    if total_length < 2:
        raise ValueError("Temporal split requires at least two ratings")

    train_length = int((1.0 - val_ratio) * total_length)
    if train_length == 0 or train_length == total_length:
        raise ValueError("--val-ratio leaves an empty train or validation split")

    sorted_indices = sorted(range(total_length), key=lambda index: timestamps[index])
    return sorted_indices[:train_length], sorted_indices[train_length:]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the MovieLens recommender.")
    parser.add_argument("--ratings-path", default=DEFAULT_RATINGS_PATH, help="Path to ratings.csv.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Where to save the trained model.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--embedding-size", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20, help="Might take a day or more depending on hardware.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--eval-k", type=int, default=10, help="K for ranking metrics such as Precision@K.")
    parser.add_argument("--relevance-threshold", type=float, default=4.0, help="Minimum rating treated as relevant.")
    parser.add_argument("--negatives-per-positive", type=int, default=4, help="Negative samples per positive rating.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.eval_k < 1:
        raise ValueError("--eval-k must be at least 1")

    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    from recommender.data import MovieLens
    from recommender.evaluate import evaluate_model
    from recommender.io import save_model
    from recommender.model import MatrixFactorization

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    movielens = MovieLens(args.ratings_path)

    if "timestamp" not in movielens.data.columns:
        raise ValueError("Temporal split requires a timestamp column in the ratings data")

    train_indices, val_indices = temporal_split_indices(movielens.data["timestamp"].tolist(), args.val_ratio)
    num_users, num_movies = movielens.size
    train_samples = build_implicit_feedback_samples(
        movielens.data,
        train_indices,
        num_movies,
        args.relevance_threshold,
        args.negatives_per_positive,
        args.seed,
    )
    val_samples = build_implicit_feedback_samples(
        movielens.data,
        val_indices,
        num_movies,
        args.relevance_threshold,
        args.negatives_per_positive,
        args.seed + 1,
    )
    train_users, train_movies, train_labels = zip(*train_samples)
    val_users, val_movies, val_labels = zip(*val_samples)
    train_dataset = TensorDataset(
        torch.LongTensor(train_users),
        torch.LongTensor(train_movies),
        torch.FloatTensor(train_labels),
    )
    val_dataset = TensorDataset(
        torch.LongTensor(val_users),
        torch.LongTensor(val_movies),
        torch.FloatTensor(val_labels),
    )
    shuffle_generator = torch.Generator().manual_seed(args.seed)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=shuffle_generator)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = MatrixFactorization(
        num_users=num_users,
        num_movies=num_movies,
        embedding_size=args.embedding_size,
        global_mean=0.0,
    ).to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    min_val_loss = np.inf

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}\n")
        running_loss = 0.0

        model.train()
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (user_id, movies_id, ratings) in train_loop:
            user_id = user_id.view(-1).to(device)
            movies_id = movies_id.view(-1).to(device)
            ratings = ratings.view(-1).to(device)

            preds = model(user_id, movies_id)
            loss = loss_fn(preds, ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loop.set_postfix(loss=running_loss / (i + 1))

        running_loss = running_loss / len(train_dataloader)
        print(f"Train Ranking Loss: {running_loss:.2f}")

        metrics = evaluate_model(
            model,
            val_dataloader,
            device,
            k=args.eval_k,
            relevance_threshold=0.5,
            implicit_feedback=True,
        )
        print(
            "Validation Metrics: "
            + ", ".join(f"{metric}: {value:.4f}" for metric, value in metrics.items())
        )

        if metrics["mae"] < min_val_loss:
            save_model(model, args.model_path)
            min_val_loss = metrics["mae"]

    print("Training finished.")


if __name__ == "__main__":
    main()
