import argparse

from recommender.config import DEFAULT_MODEL_PATH, DEFAULT_RATINGS_PATH


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
    return parser.parse_args()


def main():
    args = parse_args()
    if args.eval_k < 1:
        raise ValueError("--eval-k must be at least 1")

    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader, Subset
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
    train_dataset = Subset(movielens, train_indices)
    val_dataset = Subset(movielens, val_indices)
    shuffle_generator = torch.Generator().manual_seed(args.seed)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=shuffle_generator)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    num_users, num_movies = movielens.size
    model = MatrixFactorization(
        num_users=num_users,
        num_movies=num_movies,
        embedding_size=args.embedding_size,
        global_mean=movielens.global_mean,
    ).to(device)

    loss_fn = torch.nn.L1Loss().to(device)
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
        print(f"Train Loss: {running_loss:.2f}")

        metrics = evaluate_model(
            model,
            val_dataloader,
            device,
            k=args.eval_k,
            relevance_threshold=args.relevance_threshold,
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
