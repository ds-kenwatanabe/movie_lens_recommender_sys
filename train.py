import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, random_split
from data import MovieLens
from model import MatrixFactorization
from paths import DEFAULT_MODEL_PATH, DEFAULT_RATINGS_PATH
from utils import save_model


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
    return parser.parse_args()


def main():
    args = parse_args()

    # Set device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Training on the dataset
    movielens = MovieLens(args.ratings_path)

    # Setting up validation and test dataset ratios
    train_ratio = 1.0 - args.val_ratio

    # Checking lengths
    total_length = len(movielens)
    train_length = int(train_ratio * total_length)
    val_length = int(args.val_ratio * total_length)

    # Split into train and validation datasets.
    train_dataset, val_dataset = random_split(movielens,
                                              lengths=[train_length, val_length],
                                              generator=torch.Generator().manual_seed(args.seed))

    # Creating DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Creating MatrixFactorization model
    num_users, num_movies = movielens.size
    model = MatrixFactorization(num_users=num_users,
                                num_movies=num_movies,
                                embedding_size=args.embedding_size).to(device)

    loss_fn = torch.nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    min_val_loss = np.inf

    # Loop
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}\n')
        running_loss = 0.0

        model.train()
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (user_id, movies_id, ratings) in train_loop:
            user_id = user_id.squeeze().to(device)
            movies_id = movies_id.squeeze().to(device)
            ratings = ratings.squeeze().to(device)

            # Forward pass
            preds = model(user_id, movies_id)
            # Loss and accuracy
            loss = loss_fn(preds, ratings)
            # Zero the gradients
            optimizer.zero_grad()
            # Backpropagation
            loss.backward()
            # Optimization
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()
            train_loop.set_postfix(loss=running_loss / (i + 1))

        running_loss = running_loss / len(train_dataloader)
        print(f"Train Loss: {running_loss:.2f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for user_id, movies_id, ratings in val_dataloader:
                user_id = user_id.squeeze().to(device)
                movies_id = movies_id.squeeze().to(device)
                ratings = ratings.squeeze().to(device)

                # Forward pass
                val_preds = model(user_id, movies_id)
                # Calculate accumulate the loss
                loss = loss_fn(val_preds, ratings)
                val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)
        print(f"Validation Loss: {val_loss:.2f}")

        if val_loss < min_val_loss:
            save_model(model, args.model_path)
            min_val_loss = val_loss

    print('Training finished.')


if __name__ == "__main__":
    main()
